"""v2h：锚点规则 b 全链一致版（重扫 + 重算，替代 v2f 的规则 a 文本层）。

动因：v2f 的岗位锚点标记来自规则 a（20260908）扫描，而现行锚点词典/代码为
规则 b（20260909，LLM 复数与 TRANS 尾界修订）。本代际用规则 b 重扫 pass2
（`--out-tag b` → output/panel_v2/pass2b、release/panel_v2b），并从匹配层起
全量重算：D 级过滤 → counts（flag 变了必须重算）→ relevance → scoring →
quality → 装配。词表仍为 v1.3（legacy_grade_v2），主口径仍为指南 §2.4（005）。

产物：release/panel_v2h（15 件 + metadata + run_manifest.json，
run_id 20260910_v2h）。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.v2h [--run-id 20260910_v2h]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from . import quality, scoring
from .counts import compute_counts
from .export_release import _meta
from .relevance import _load_tier_map, compute_relevance, decode_skill_ids
from .reproducibility import write_run_manifest

logger = logging.getLogger("ai_penetration.panel_v2.v2h")

LEX_VERSION = "bilingual_a_frozen_v1.1+legacy_grade_v2"


def filter_longs(rel2: Path, codes: np.ndarray,
                 grade_by_sid: dict[str, str]) -> None:
    """按最终 B/C/D 分级过滤并补齐指南 §11.3 的正式长表字段。"""
    target = rel2 / "job_skill_long.parquet"
    tmp = rel2 / "_job_skill_long_filtered.parquet"
    src = pq.ParquetFile(target)
    required = {
        "job_id", "year", "skill_code", "skill_id", "surface_form",
        "match_start", "match_end", "mention_count", "match_method",
        "ambiguity_flag", "span_verified",
    }
    missing = required - set(src.schema_arrow.names)
    if missing:
        src.close()
        raise RuntimeError(
            "pass2 仍是旧版无证据长表，必须按 handoff-compliant scan 重扫: "
            + ", ".join(sorted(missing))
        )
    keep = drop = 0
    writer = None
    try:
        for rb in src.iter_batches(batch_size=2_000_000):
            t = pa.Table.from_batches([rb])
            sc = t["skill_code"].to_numpy()
            m = ~np.isin(sc, codes)
            keep += int(m.sum())
            drop += int((~m).sum())
            t = t.filter(pa.array(m))
            if len(t) == 0:
                continue
            names = [
                "start" if n == "match_start" else
                "end" if n == "match_end" else n
                for n in t.column_names
            ]
            t = t.rename_columns(names)
            sids = t["skill_id"].to_pylist()
            tiers = [grade_by_sid.get(str(sid), "A") for sid in sids]
            t = t.append_column("confidence_tier", pa.array(tiers, pa.string()))
            t = t.append_column(
                "dictionary_version",
                pa.array([LEX_VERSION] * len(t), pa.string()),
            )
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
            writer.write_table(t)
    finally:
        src.close()
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("D 级过滤后 job_skill_long 为空，拒绝发布")
    import gc
    import time
    gc.collect()
    for attempt in range(5):
        try:
            target.unlink(missing_ok=True)
            tmp.rename(target)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(3)
    logger.info("longs 过滤/证据补齐: 保留 %d，剔除 D 级 %d", keep, drop)

def main() -> None:
    ap = argparse.ArgumentParser(description="v2h 规则 b 一致版重算")
    ap.add_argument("--run-id", default="20260910_v2h")
    args = ap.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_v2h.log")
    t0 = datetime.now()
    rel_src = paths.output_dir / "release" / "panel_v2"
    rel2 = paths.output_dir / "release" / "panel_v2b"   # 扫描合并产物（输入）
    rel3 = paths.output_dir / "release" / "panel_v2h"   # 本代际发布（输出）
    rel3.mkdir(parents=True, exist_ok=True)

    # 1) 输入件从 v2b 移入 v2h（flag/long/firm 为规则 b 扫描产物）
    scan_inputs = ("job_anchor_flag.parquet", "job_skill_long.parquet",
                   "job_firm.parquet")
    for f in scan_inputs:
        shutil.copy2(rel2 / f, rel3 / f)

    # 2) v1.3 D 级过滤
    gcsv = paths.output_dir / "dictionary" / "skill_legacy_graded_BCD_v2.csv"
    grade = pd.read_csv(gcsv, encoding="utf-8-sig")
    pass2b = paths.output_dir / "panel_v2" / "pass2b"
    vocab_path = pass2b / "skill_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    d_ids = set(grade[grade.final_grade == "D"].skill_id.astype(str))
    grade_by_sid = dict(zip(grade.skill_id.astype(str), grade.final_grade.astype(str)))
    codes = np.array(sorted(vocab[s] for s in d_ids if s in vocab), np.int32)
    logger.info("D 级排除 %d 码", len(codes))
    filter_longs(rel3, codes, grade_by_sid)

    # 3) counts（新 flag）→ relevance
    counts = compute_counts(rel3)
    counts.to_parquet(rel3 / "skill_ai_counts.parquet", index=False)
    logger.info("skill_ai_counts: %d 行", len(counts))
    rel_df = decode_skill_ids(
        compute_relevance(counts, _load_tier_map(vocab_path)), vocab_path
    )
    rel_df.to_parquet(rel3 / "skill_ai_relevance.parquet", index=False)
    logger.info("skill_ai_relevance: %d 行", len(rel_df))

    # 4) scoring + quality
    scoring.run(rel3)
    quality.run(rel3)

    # 5) 装配：字典/锚点/分级件 + metadata
    copied_release_inputs = (
        "skill_concept_v1.parquet", "skill_alias_v1.parquet",
        "skill_candidate_d_v1.parquet", "skill_legacy_v1.parquet",
        "ai_anchor_dictionary_v1.csv",
    )
    for f in copied_release_inputs:
        shutil.copy2(rel_src / f, rel3 / f)
    shutil.copy2(gcsv, rel3 / gcsv.name)

    single = rel3 / "job_ai_score.parquet"
    files = sorted((rel3 / "job_ai_score").glob("*.parquet"))
    if not files:
        raise RuntimeError("job_ai_score 分区为空，拒绝装配 v2h release")
    with pq.ParquetWriter(single, pq.read_schema(files[0]),
                          compression="zstd") as w:
        for f in files:
            for rb in pq.ParquetFile(f).iter_batches(batch_size=2_000_000):
                w.write_table(pa.Table.from_batches([rb]))

    specs = [
        ("skill_concept_v1.parquet", "skill_id", "ai_dict.skill_concepts"),
        ("skill_alias_v1.parquet", "alias_id", "ai_dict.skill_aliases(active)"),
        ("skill_candidate_d_v1.parquet", "term", "pass2b/skill_vocab.json"),
        ("skill_legacy_v1.parquet", "term", "panel_v2/lexicon.py(union构建处置)"),
        ("skill_legacy_graded_BCD_v2.csv", "skill_id",
         "panel_v2/lexicon_llm.py(T1∧T2 合并分级)"),
        ("ai_anchor_dictionary_v1.csv", "anchor_version+keyword",
         "panel_v2/anchors.py(规则 20260909_b)"),
        ("job_anchor_flag.parquet", "job_id", "panel_v2/scan.py(规则 b 重扫)"),
        ("job_skill_long.parquet", "job_id+skill_code", "panel_v2/v2h.py(D 级过滤)"),
        ("job_firm.parquet", "job_id", "panel_v2/scan.py(规则 b 重扫)"),
        ("skill_ai_counts.parquet", "skill+ver+win+year", "panel_v2/counts.py(规则 b)"),
        ("skill_ai_relevance.parquet", "skill+ver+win+year", "panel_v2/relevance.py(规则 b)"),
        ("job_ai_score.parquet", "job+ver+win+stype", "panel_v2/scoring.py(v2h)"),
        ("job_ai_classification.parquet", "job_id", "panel_v2/scoring.py(v2h)"),
        ("job_ai_score_loo.parquet", "job_id", "panel_v2/scoring.py(§14.3 v2h)"),
        ("quality_control_report.md", "-", "panel_v2/quality.py(v2h)"),
    ]

    missing = [name for name, _, _ in specs if not (rel3 / name).exists()]
    if missing:
        raise RuntimeError(
            "指南 §18 必需发布件缺失，拒绝生成正式 v2h release: "
            + ", ".join(missing)
        )

    for name, pk, src in specs:
        _meta(
            rel3 / name, args.run_id, pk, src,
            anchor_version="main,cn_paper,babina",
            dictionary_version=LEX_VERSION,
        )

    # 6) 指南 §4.2：把本次正式运行的代码/配置、输入、输出绑定成一个总账。
    manifest_inputs = [rel2 / f for f in scan_inputs]
    manifest_inputs += [rel_src / f for f in copied_release_inputs]
    manifest_inputs += [gcsv, vocab_path]
    manifest_outputs = [rel3 / name for name, _, _ in specs]
    manifest = write_run_manifest(
        rel3,
        run_id=args.run_id,
        started_at=t0,
        input_paths=manifest_inputs,
        output_paths=manifest_outputs,
        dictionary_version=LEX_VERSION,
        anchor_version="main,cn_paper,babina@20260909_b",
    )

    print(f"panel_v2h 发布: {len(specs)} 件 + metadata + {manifest.name} @ {rel3}，用时 "
          f"{(datetime.now() - t0).total_seconds() / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
