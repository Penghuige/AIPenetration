"""panel_v2 M4-b：§18 发布物组装（11 件 + 同名 metadata.json）。

把 counts/relevance/scoring 产物与词典、锚点表汇总到 release/panel_v2/，
逐文件生成 §18.1 十二字段 metadata（含 sha256、run_id、config_hash）。
默认拒绝覆盖已存在发布文件（--force-release 显式覆盖，§4 版本纪律）。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.export_release --run-id 20260907_v2a
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

from config.paths import get_project_paths

from ..common import eps_conn_params, setup_logging
from .anchors import anchor_dictionary_rows

logger = logging.getLogger("ai_penetration.panel_v2.export")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash() -> str:
    root = get_project_paths().config_dir
    h = hashlib.sha256()
    for f in sorted(root.glob("*.yaml")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _meta(p: Path, run_id: str, primary_key: str, source_files: str,
          anchor_version: str = "-") -> None:
    import pyarrow.parquet as _pq
    if p.suffix == ".parquet":
        schema = str(_pq.read_schema(p))
        n = _pq.ParquetFile(p).metadata.num_rows
    else:
        text = p.read_text(encoding="utf-8-sig")
        schema = "csv"
        n = max(text.count("\n") - 1, 0)
    meta = {
        "file_name": p.name, "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": n, "column_schema": schema, "primary_key": primary_key,
        "source_files": source_files,
        "dictionary_version": "bilingual_a_frozen_v1.1+legacy_union_v1",
        "anchor_version": anchor_version,
        "config_hash": _config_hash(), "sha256": _sha(p),
    }
    p.with_suffix(p.suffix + ".metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def export_dictionaries(rel: Path) -> None:
    """词典三件 + 锚点表（§18）。"""
    conn = psycopg2.connect(**eps_conn_params())
    try:
        concepts = pd.read_sql(
            "SELECT skill_id, canonical_zh, canonical_en, skill_type, "
            "skill_category, confidence_tier, translation_status, dictionary_version "
            "FROM ai_dict.skill_concepts", conn)
        aliases = pd.read_sql(
            "SELECT alias_id, skill_id, alias, alias_normalized, language, "
            "is_active, activation_reason FROM ai_dict.skill_aliases "
            "WHERE is_active='1'", conn)
    finally:
        conn.close()
    concepts.to_parquet(rel / "skill_concept_v1.parquet", index=False)
    aliases.to_parquet(rel / "skill_alias_v1.parquet", index=False)
    # D 级候选：union 词表 legacy 空间（A 级未收录、由自建表贡献的词）
    vocab_path = get_project_paths().output_dir / "panel_v2" / "pass2" / "skill_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    legacy = sorted(s[len("legacy:"):] for s in vocab
                    if s.startswith("legacy:"))
    pd.DataFrame({"term": legacy}).assign(
        grade_candidate="D_legacy_union_contrib"
    ).to_parquet(rel / "skill_candidate_d_v1.parquet", index=False)
    rows = anchor_dictionary_rows()
    pd.DataFrame(rows).to_csv(rel / "ai_anchor_dictionary_v1.csv",
                              index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="panel_v2 §18 发布组装")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--force-release", action="store_true")
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_export.log")
    rel = paths.output_dir / "release" / "panel_v2"
    rel.mkdir(parents=True, exist_ok=True)
    if not args.force_release and (rel / "quality_control_report.md").exists():
        raise SystemExit("发布目录已有本次结果（--force-release 覆盖，或改 run_id）")

    export_dictionaries(rel)
    # job_ai_score 以 18 单元 dataset 产出（B1 内存纪律），发布前流式合并
    score_dir = rel / "job_ai_score"
    score_single = rel / "job_ai_score.parquet"
    if score_dir.is_dir() and not score_single.exists():
        import pyarrow.parquet as _pq
        import pyarrow as _pa
        files = sorted(score_dir.glob("*.parquet"))
        schema = _pq.read_schema(files[0])
        with _pq.ParquetWriter(score_single, schema,
                               compression="zstd") as w:
            for f in files:
                for tbl in _pq.ParquetFile(f).iter_batches(batch_size=2_000_000):
                    w.write_table(tbl)
        logger.info("job_ai_score 合并 %d 单元 -> 单文件", len(files))
    specs = [
        ("skill_concept_v1.parquet", "skill_id", "ai_dict.skill_concepts"),
        ("skill_alias_v1.parquet", "alias_id", "ai_dict.skill_aliases(active)"),
        ("skill_candidate_d_v1.parquet", "term", "pass2/skill_vocab.json"),
        ("ai_anchor_dictionary_v1.csv", "anchor_version+keyword", "panel_v2/anchors.py"),
        ("job_anchor_flag.parquet", "job_id", "panel_v2/scan.py"),
        ("job_skill_long.parquet", "job_id+skill_code", "panel_v2/scan.py"),
        ("job_firm.parquet", "job_id", "panel_v2/scan.py"),
        ("skill_ai_counts.parquet", "skill+ver+win+year", "panel_v2/counts.py"),
        ("skill_ai_relevance.parquet", "skill+ver+win+year", "panel_v2/relevance.py"),
        ("job_ai_score.parquet", "job+ver+win+stype", "panel_v2/scoring.py"),
        ("job_ai_classification.parquet", "job_id", "panel_v2/scoring.py"),
        ("job_ai_score_loo.parquet", "job_id", "panel_v2/scoring.py(§14.3)"),
        ("quality_control_report.md", "-", "panel_v2/quality.py"),
    ]
    made = 0
    for name, pk, src in specs:
        p = rel / name
        if not p.exists():
            logger.warning("缺文件跳过: %s", name)
            continue
        _meta(p, args.run_id, pk, src,
              anchor_version="main,cn_paper,babina")
        made += 1
    print(f"发布组装完成: {made} 件 + metadata @ {rel}")


if __name__ == "__main__":
    main()
