"""上一任交接 §7.6：170 批外部技能中文化完成性验收。

不重新调用 Codex。只有历史输入、批次清单、prompt、schema、170 个结果文件与
170 个 QC 文件全部可验证且 source ID 一一守恒时，才写 formal_pass manifest。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from config.paths import get_project_paths

EXPECTED_BATCHES = 170
EXPECTED_TOTAL = 169 * 100 + 14


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_field(obj, names: tuple[str, ...]):
    """在历史结果可能的浅/嵌套对象中确定性查找字段。"""
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] not in (None, ""):
                return obj[name]
        for key in sorted(obj):
            value = _find_field(obj[key], names)
            if value not in (None, ""):
                return value
    elif isinstance(obj, list):
        for value in obj:
            found = _find_field(value, names)
            if found not in (None, ""):
                return found
    return None


def _source_id(rec: dict) -> str:
    value = _find_field(
        rec, ("source_skill_id", "skill_id", "source_id")
    )
    return str(value or "").strip()


def _canonical_en(rec: dict) -> str:
    value = _find_field(rec, ("canonical_en",))
    return str(value or "").strip().casefold()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name}:{line_no} JSON 无效"
            ) from exc
        if not isinstance(rec, dict):
            raise ValueError(
                f"{path.name}:{line_no} 必须是 JSON object"
            )
        rows.append(rec)
    return rows


def _input_records(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(
            f"完整翻译输入 {len(rows)} != {EXPECTED_TOTAL}"
        )
    ids = [_source_id(r) for r in rows]
    if any(not sid for sid in ids):
        raise ValueError("完整翻译输入存在空 source skill id")
    if len(set(ids)) != len(ids):
        raise ValueError("完整翻译输入 source skill id 不唯一")
    return rows


def _resolve_batch_ids(
    result_rows: list[dict],
    expected_input: list[dict],
    batch: int,
) -> tuple[list[str], bool, float]:
    """显式ID优先；缺ID时按交接要求做>=98%顺序证据回填。"""
    if len(result_rows) != len(expected_input):
        raise ValueError(
            f"batch={batch} 结果行数 {len(result_rows)} "
            f"!= 输入 {len(expected_input)}"
        )
    expected_ids = [_source_id(r) for r in expected_input]
    result_ids = [_source_id(r) for r in result_rows]
    explicit = [bool(x) for x in result_ids]

    for i, (has_id, rid, eid) in enumerate(
        zip(explicit, result_ids, expected_ids)
    ):
        if has_id and rid != eid:
            raise ValueError(
                f"batch={batch} line={i+1} 显式 source id "
                f"{rid!r} != 输入 {eid!r}"
            )

    if all(explicit):
        return result_ids, False, 1.0

    comparable = matched = 0
    for result, source in zip(result_rows, expected_input):
        r_name = _canonical_en(result)
        s_name = _canonical_en(source)
        if r_name and s_name:
            comparable += 1
            matched += int(r_name == s_name)
    ratio = matched / comparable if comparable else 0.0
    if ratio < 0.98:
        raise ValueError(
            f"batch={batch} 缺 source id 且 canonical_en 顺序匹配率 "
            f"{ratio:.2%} < 98%，禁止按顺序回填"
        )
    resolved = [
        rid if rid else eid
        for rid, eid in zip(result_ids, expected_ids)
    ]
    return resolved, True, ratio


def _qc_state(obj) -> tuple[bool, bool]:
    """返回(has_pass, has_fail)；任意显式 fail 优先于 pass。"""
    has_pass = has_fail = False
    if isinstance(obj, dict):
        for key in ("passed", "valid", "ok"):
            value = obj.get(key)
            if isinstance(value, bool):
                has_pass |= value
                has_fail |= not value
        if "status" in obj:
            status = str(obj["status"]).strip().lower()
            has_pass |= status in {
                "pass", "passed", "success", "valid",
                "completed", "complete", "ok",
            }
            has_fail |= status in {
                "fail", "failed", "invalid", "error", "blocked",
            }
        for value in obj.values():
            p, f = _qc_state(value)
            has_pass |= p
            has_fail |= f
    elif isinstance(obj, list):
        for value in obj:
            p, f = _qc_state(value)
            has_pass |= p
            has_fail |= f
    return has_pass, has_fail


def _qc_passed(obj) -> bool:
    passed, failed = _qc_state(obj)
    return passed and not failed


def verify(
    results_dir: Path,
    result_pattern: str,
    qc_dir: Path,
    qc_pattern: str,
    input_jsonl: Path,
    batch_manifest: Path,
    prompt_file: Path,
    schema_file: Path,
) -> Path:
    fixed_inputs = [
        input_jsonl, batch_manifest, prompt_file, schema_file,
    ]
    missing = [str(p) for p in fixed_inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "中文化验收缺少上游固定件:\n" + "\n".join(missing)
        )

    input_records = _input_records(input_jsonl)
    input_ids = {_source_id(r) for r in input_records}
    records = []
    resolved_batches: list[tuple[int, Path, list[dict], list[str]]] = []
    global_ids: set[str] = set()
    total = 0

    for batch in range(1, EXPECTED_BATCHES + 1):
        expected = 14 if batch == EXPECTED_BATCHES else 100
        result_path = results_dir / result_pattern.format(batch=batch)
        qc_path = qc_dir / qc_pattern.format(batch=batch)
        if not result_path.exists():
            raise FileNotFoundError(
                f"缺少翻译批次 {batch}: {result_path}"
            )
        if not qc_path.exists():
            raise FileNotFoundError(
                f"缺少翻译 QC {batch}: {qc_path}"
            )
        result_rows = _read_jsonl(result_path)
        lo = (batch - 1) * 100
        expected_input = input_records[lo:lo + expected]
        resolved_ids, backfilled, order_ratio = _resolve_batch_ids(
            result_rows, expected_input, batch
        )
        ids = set(resolved_ids)
        if len(ids) != expected:
            raise ValueError(
                f"batch={batch} 回填后 source id 不唯一"
            )
        overlap = global_ids & ids
        if overlap:
            raise ValueError(
                f"跨批 source skill id 重复，batch={batch}: "
                f"{sorted(overlap)[:5]}"
            )
        try:
            qc_obj = json.loads(qc_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"QC JSON 无效 batch={batch}: {qc_path}"
            ) from exc
        if not _qc_passed(qc_obj):
            raise ValueError(
                f"QC 未显式通过或含失败信号 batch={batch}: {qc_path}"
            )

        global_ids |= ids
        total += len(ids)
        resolved_batches.append(
            (batch, result_path, result_rows, resolved_ids)
        )
        records.append({
            "batch": batch,
            "result_file": str(result_path.resolve()),
            "result_rows": len(ids),
            "result_sha256": _sha(result_path),
            "qc_file": str(qc_path.resolve()),
            "qc_sha256": _sha(qc_path),
            "ids_backfilled_by_order": backfilled,
            "canonical_en_order_match_rate": order_ratio,
        })

    if total != EXPECTED_TOTAL or len(global_ids) != EXPECTED_TOTAL:
        raise ValueError(
            f"170 批总量不守恒: rows={total}, unique_ids={len(global_ids)}, "
            f"expected={EXPECTED_TOTAL}"
        )
    missing_outputs = input_ids - global_ids
    extra_outputs = global_ids - input_ids
    if missing_outputs or extra_outputs:
        raise ValueError(
            "翻译输入/输出 source ID 不守恒: "
            f"missing={len(missing_outputs)} extra={len(extra_outputs)}"
        )

    out_dir = get_project_paths().output_dir / "dictionary"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "external_translation_log_v1.jsonl"
    rec_by_batch = {int(r["batch"]): r for r in records}
    with log_path.open("w", encoding="utf-8") as log_fh:
        for batch, batch_path, result_rows, resolved_ids in resolved_batches:
            batch_rec = rec_by_batch[batch]
            for line_no, (payload, sid) in enumerate(
                zip(result_rows, resolved_ids), 1
            ):
                log_fh.write(json.dumps({
                    "batch": batch,
                    "line_no": line_no,
                    "source_skill_id": sid,
                    "source_result_file": str(batch_path),
                    "source_result_sha256": batch_rec["result_sha256"],
                    "qc_file": batch_rec["qc_file"],
                    "qc_sha256": batch_rec["qc_sha256"],
                    "result": payload,
                }, ensure_ascii=False) + "\n")

    manifest = {
        "status": "formal_pass",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "expected_batches": EXPECTED_BATCHES,
        "observed_batches": len(records),
        "expected_contextual_records": EXPECTED_TOTAL,
        "unique_source_skill_ids": len(global_ids),
        "result_pattern": result_pattern,
        "qc_pattern": qc_pattern,
        "fixed_input_sha256": {
            "translation_input": _sha(input_jsonl),
            "batch_manifest": _sha(batch_manifest),
            "prompt": _sha(prompt_file),
            "schema": _sha(schema_file),
        },
        "batches": records,
        "external_translation_log": str(log_path),
        "external_translation_log_sha256": _sha(log_path),
    }
    out = (
        out_dir / "external_translation_completion_manifest_v1.json"
    )
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="170 批 Codex 中文化完整性 + QC 验收"
    )
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--qc-dir", type=Path, required=True)
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--batch-manifest", type=Path, required=True)
    ap.add_argument("--prompt-file", type=Path, required=True)
    ap.add_argument("--schema-file", type=Path, required=True)
    ap.add_argument(
        "--result-pattern",
        default="contextual_translation_v1_{batch:04d}.result.jsonl",
    )
    ap.add_argument(
        "--qc-pattern",
        default="contextual_translation_v1_{batch:04d}.qc.json",
    )
    args = ap.parse_args()
    verify(
        args.results_dir,
        args.result_pattern,
        args.qc_dir,
        args.qc_pattern,
        args.input_jsonl,
        args.batch_manifest,
        args.prompt_file,
        args.schema_file,
    )


if __name__ == "__main__":
    main()
