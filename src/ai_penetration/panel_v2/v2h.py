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
from .anchors import anchor_dictionary_rows
from .export_release import _meta, export_final_dictionaries, export_legacy_disposition
from .lexicon import load_formal_legacy_spec
from .relevance import compute_relevance, decode_skill_ids
from .reproducibility import write_run_manifest

logger = logging.getLogger("ai_penetration.panel_v2.v2h")

LEX_VERSION = "bilingual_a_frozen_v1.1+legacy_grade_v2"


def filter_longs(rel2: Path, codes: np.ndarray) -> None:
    """按 v1.3 D 级码流式过滤 job_skill_long（B1 内存纪律）。

    注意（自锁 bug 修复 2026-09-10）：pyarrow 的 ParquetFile 会持续持有读
    句柄——必须先 close() + gc 释放，才能对同一路径做删除/改名（此前两次
    "文件被占用" PermissionError 即此自锁，非外部干扰）。
    """
    target = rel2 / "job_skill_long.parquet"
    tmp = rel2 / "_job_skill_long_filtered.parquet"
    src = pq.ParquetFile(target)
    keep = drop = 0
    try:
        with pq.ParquetWriter(tmp, src.schema_arrow, compression="zstd") as w:
            for rb in src.iter_batches(batch_size=2_000_000):
                t = pa.Table.from_batches([rb])
                sc = t["skill_code"].to_numpy()
                m = ~np.isin(sc, codes)
                keep += int(m.sum())
                drop += int((~m).sum())
                w.write_table(t.filter(pa.array(m)))
    finally:
        src.close()
    import gc
    import time
    gc.collect()  # 释放 pyarrow 底层文件句柄
    for attempt in range(5):
        try:
            target.unlink(missing_ok=True)
            tmp.rename(target)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(3)
    logger.info("longs 过滤: 保留 %d，剔除 %d", keep, drop)



def formal_tier_map(
    vocab_path: Path,
    grade_path: Path,
    a_concept_path: Path,
) -> np.ndarray:
    """skill_code → 最终 A/B/C confidence tier（仅冻结文件输入）。"""
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    n_codes = max(vocab.values()) + 1
    tier_map = np.full(n_codes, "", dtype=object)

    a = pd.read_csv(a_concept_path, encoding="utf-8-sig")
    if "skill_id" not in a.columns or "confidence_tier" not in a.columns:
        raise RuntimeError("A 级冻结 concept 文件缺 skill_id/confidence_tier")
    for sid, tier in zip(a.skill_id.astype(str), a.confidence_tier):
        if sid in vocab and not pd.isna(tier):
            tier_map[int(vocab[sid])] = str(tier)

    _, _, tier_by_sid = load_formal_legacy_spec(grade_path)
    for sid, tier in tier_by_sid.items():
        if sid in vocab:
            tier_map[int(vocab[sid])] = tier

    used = np.array(sorted(vocab.values()), dtype=np.int64)
    missing = [int(code) for code in used if not str(tier_map[int(code)]).strip()]
    if missing:
        raise RuntimeError(
            f"正式词表有 {len(missing)} 个 skill_code 缺 confidence_tier，"
            f"示例={missing[:10]}"
        )
    return tier_map


def enrich_job_skill_long(
    target: Path,
    vocab_path: Path,
    tier_map: np.ndarray,
    *,
    dictionary_version: str,
) -> None:
    """给 §11 job_skill_long 补正式 skill_id/tier/dictionary_version。"""
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    n_codes = max(vocab.values()) + 1
    code_to_sid = np.empty(n_codes, dtype=object)
    code_to_sid[:] = None
    for sid, code in vocab.items():
        code_to_sid[int(code)] = str(sid)
    if any(x is None for x in code_to_sid):
        raise RuntimeError("skill_vocab code 非连续，无法稳定解码 job_skill_long")

    src = pq.ParquetFile(target)
    existing = set(src.schema_arrow.names)
    extra = {"skill_id", "confidence_tier", "dictionary_version"}
    if extra <= existing:
        src.close()
        return
    if existing & extra:
        src.close()
        raise RuntimeError(
            f"job_skill_long 仅含部分正式字段: {sorted(existing & extra)}"
        )

    tmp = target.with_name("_job_skill_long_enriched.parquet")
    writer = None
    try:
        for rb in src.iter_batches(batch_size=1_000_000):
            t = pa.Table.from_batches([rb])
            sc = t["skill_code"].to_numpy()
            if (sc < 0).any() or (sc >= n_codes).any():
                raise RuntimeError("job_skill_long 含 vocab 外 skill_code")
            sids = code_to_sid[sc]
            tiers = tier_map[sc]
            if any(not str(x).strip() for x in tiers):
                raise RuntimeError("job_skill_long 存在空 confidence_tier")
            t = t.append_column("skill_id", pa.array(sids, type=pa.string()))
            t = t.append_column(
                "confidence_tier", pa.array(tiers, type=pa.string())
            )
            t = t.append_column(
                "dictionary_version",
                pa.array([dictionary_version] * len(t), type=pa.string()),
            )
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
            writer.write_table(t)
    finally:
        src.close()
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("job_skill_long 为空，拒绝正式发布")
    tmp.replace(target)


def main() -> None:
    ap = argparse.ArgumentParser(description="v2h 规则 b 一致版重算")
    ap.add_argument("--run-id", default="20260910_v2h")
    args = ap.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_v2h.log")
    t0 = datetime.now()
    rel2 = paths.output_dir / "release" / "panel_v2b"   # 扫描合并产物（输入）
    rel3 = paths.output_dir / "release" / "panel_v2h"   # 本代际发布（输出）
    rel3.mkdir(parents=True, exist_ok=True)
    a_concept = (
        paths.output_dir / "dictionary"
        / "skill_concept_bilingual_a_frozen_v1.1.csv"
    )
    a_alias = (
        paths.output_dir / "dictionary"
        / "skill_alias_active_bilingual_a_frozen_v1.1.csv"
    )

    # 1) 输入件从 v2b 移入 v2h（flag/long/firm 为规则 b 扫描产物）
    scan_inputs = ("job_anchor_flag.parquet", "job_skill_long.parquet",
                   "job_firm.parquet")
    for f in scan_inputs:
        shutil.copy2(rel2 / f, rel3 / f)

    # 2) 最终语义分级与正式 job_skill_long
    gcsv = paths.output_dir / "dictionary" / "skill_legacy_graded_BCD_v2.csv"
    grade = pd.read_csv(gcsv, encoding="utf-8-sig")
    pass2b = paths.output_dir / "panel_v2" / "pass2b"
    vocab_path = pass2b / "skill_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    # 新 formal scan 理论上已经不包含 D；保留这一步作防御性过滤。
    d_ids = set(grade[grade.final_grade == "D"].skill_id.astype(str))
    codes = np.array(sorted(vocab[s] for s in d_ids if s in vocab), np.int32)
    logger.info("formal scan 后残余 D 级码: %d", len(codes))
    filter_longs(rel3, codes)
    tiers = formal_tier_map(vocab_path, gcsv, a_concept)
    enrich_job_skill_long(
        rel3 / "job_skill_long.parquet",
        vocab_path,
        tiers,
        dictionary_version=LEX_VERSION,
    )

    # 3) counts（新 flag）→ relevance
    counts = decode_skill_ids(compute_counts(rel3), vocab_path)
    counts.to_parquet(rel3 / "skill_ai_counts.parquet", index=False)
    logger.info("skill_ai_counts: %d 行", len(counts))
    rel_df = decode_skill_ids(
        compute_relevance(counts, tiers), vocab_path
    )
    rel_df["dictionary_version"] = LEX_VERSION
    rel_df.to_parquet(rel3 / "skill_ai_relevance.parquet", index=False)
    logger.info("skill_ai_relevance: %d 行", len(rel_df))

    # 4) scoring + quality
    scoring.run(rel3)
    quality.run(rel3)

    # 5) 装配：严格按 §18 从冻结 A 文件 + 最终语义分级重建词典。
    for source in (a_concept, a_alias, gcsv):
        if not source.exists():
            raise RuntimeError(f"正式发布输入缺失: {source}")
    export_final_dictionaries(
        rel3,
        gcsv,
        dictionary_version=LEX_VERSION,
        a_concept_path=a_concept,
        a_alias_path=a_alias,
    )
    export_legacy_disposition(rel3)
    pd.DataFrame(anchor_dictionary_rows()).to_csv(
        rel3 / "ai_anchor_dictionary_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )
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
        ("job_skill_long.parquet", "job_id+skill_id",
         "panel_v2/scan.py(证据)+v2h.py(正式词典解码)"),
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
    manifest_inputs += [
        a_concept,
        a_alias,
        gcsv,
        vocab_path,
        pass2b / "formal_legacy_spec.json",
        pass2b / "skill_vocab.stamp.json",
    ]
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
