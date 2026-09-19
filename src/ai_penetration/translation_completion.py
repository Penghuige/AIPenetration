"""上一任交接 §7.6：170 批外部技能中文化完成性验收。

不重新调用 Codex；只验证历史批次结果是否齐全、无重复、可解析，并生成
供正式发布读取的 completion manifest。
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


def _read_batch(path: Path, expected_rows: int) -> tuple[int, set[str]]:
    rows = 0
    ids: set[str] = set()
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no} JSON 无效") from exc
        sid = str(rec.get("source_skill_id", "")).strip()
        if not sid:
            raise ValueError(f"{path.name}:{line_no} 缺 source_skill_id")
        if sid in ids:
            raise ValueError(f"{path.name} 内 source_skill_id 重复: {sid}")
        ids.add(sid)
        rows += 1
    if rows != expected_rows:
        raise ValueError(
            f"{path.name} 行数 {rows} != 预期 {expected_rows}"
        )
    return rows, ids


def verify(results_dir: Path, pattern: str) -> Path:
    records = []
    global_ids: set[str] = set()
    total = 0
    for batch in range(1, EXPECTED_BATCHES + 1):
        expected = 14 if batch == EXPECTED_BATCHES else 100
        filename = pattern.format(batch=batch)
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少翻译批次 {batch}: {path}")
        rows, ids = _read_batch(path, expected)
        overlap = global_ids & ids
        if overlap:
            raise ValueError(
                f"跨批 source_skill_id 重复，batch={batch}: "
                f"{sorted(overlap)[:5]}"
            )
        global_ids |= ids
        total += rows
        records.append({
            "batch": batch,
            "file": str(path.resolve()),
            "rows": rows,
            "sha256": _sha(path),
        })

    if total != EXPECTED_TOTAL or len(global_ids) != EXPECTED_TOTAL:
        raise ValueError(
            f"170 批总量不守恒: rows={total}, unique_ids={len(global_ids)}, "
            f"expected={EXPECTED_TOTAL}"
        )

    out_dir = get_project_paths().output_dir / "dictionary"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "external_translation_log_v1.jsonl"
    with log_path.open("w", encoding="utf-8") as log_fh:
        for batch_rec in records:
            batch = int(batch_rec["batch"])
            batch_path = Path(batch_rec["file"])
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
                    "source_result_sha256": batch_rec["sha256"],
                    "result": payload,
                }, ensure_ascii=False) + "\n")

    manifest = {
        "status": "formal_pass",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "expected_batches": EXPECTED_BATCHES,
        "observed_batches": len(records),
        "expected_contextual_records": EXPECTED_TOTAL,
        "unique_source_skill_ids": len(global_ids),
        "pattern": pattern,
        "batches": records,
        "external_translation_log": str(log_path),
        "external_translation_log_sha256": _sha(log_path),
    }
    out = (
        get_project_paths().output_dir
        / "dictionary"
        / "external_translation_completion_manifest_v1.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="170 批 Codex 中文化完成性验收")
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument(
        "--pattern",
        default="contextual_translation_v1_{batch:04d}.result.jsonl",
        help="可用 {batch:04d} 占位",
    )
    args = ap.parse_args()
    verify(args.results_dir, args.pattern)


if __name__ == "__main__":
    main()
