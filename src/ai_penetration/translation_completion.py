"""上一任交接 §7.6：170 批外部技能中文化完成性验收。

不重新调用 Codex。只有历史输入、批次清单、prompt、schema、170 个结果文件与
170 个 QC 文件全部可验证且 source ID 一一守恒时，才写 formal_pass manifest。
"""
from __future__ import annotations

import argparse
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


def _source_id(rec: dict) -> str:
    for key in ("source_skill_id", "skill_id", "source_id"):
        value = str(rec.get(key, "")).strip()
        if value:
            return value
    return ""


def _read_jsonl_ids(path: Path, expected_rows: int | None = None) -> set[str]:
    ids: set[str] = set()
    rows = 0
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no} JSON 无效") from exc
        sid = _source_id(rec)
        if not sid:
            raise ValueError(f"{path.name}:{line_no} 缺 source skill id")
        if sid in ids:
            raise ValueError(f"{path.name} 内 source skill id 重复: {sid}")
        ids.add(sid)
        rows += 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(
            f"{path.name} 行数 {rows} != 预期 {expected_rows}"
        )
    return ids


def _qc_passed(obj) -> bool:
    """只接受显式 pass 信号；未知 QC schema fail-closed。"""
    if isinstance(obj, dict):
        for key in ("passed", "valid", "ok"):
            if key in obj and isinstance(obj[key], bool):
                if obj[key] is True:
                    return True
        if "status" in obj:
            status = str(obj["status"]).strip().lower()
            if status in {
                "pass", "passed", "success", "valid",
                "completed", "complete", "ok",
            }:
                return True
            if status in {
                "fail", "failed", "invalid", "error", "blocked",
            }:
                return False
        return any(_qc_passed(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_qc_passed(v) for v in obj)
    return False


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

    input_ids = _read_jsonl_ids(input_jsonl, EXPECTED_TOTAL)
    records = []
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
        ids = _read_jsonl_ids(result_path, expected)
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
                f"QC 未找到显式通过状态 batch={batch}: {qc_path}"
            )

        global_ids |= ids
        total += len(ids)
        records.append({
            "batch": batch,
            "result_file": str(result_path.resolve()),
            "result_rows": len(ids),
            "result_sha256": _sha(result_path),
            "qc_file": str(qc_path.resolve()),
            "qc_sha256": _sha(qc_path),
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
    with log_path.open("w", encoding="utf-8") as log_fh:
        for batch_rec in records:
            batch = int(batch_rec["batch"])
            batch_path = Path(batch_rec["result_file"])
            for line_no, line in enumerate(
                batch_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                payload = json.loads(line)
                log_fh.write(json.dumps({
                    "batch": batch,
                    "line_no": line_no,
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
