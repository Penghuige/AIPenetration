"""v2d 面板重建：治理后词表（legacy §8 通道化 + taut 得分排除）的重打分发布。

关键等价性论证（免重扫依据）：本次词表变更**只删不增**——Aho-Corasick 键集
收缩不改变幸存键的命中位置，故"过滤 job_skill_long"与"用新词表重扫 pass2"
逐对相等；counts/relevance 是逐技能统计，其他技能的排除不影响幸存技能的
c/n——直接按行过滤复用（过滤后 pooled=Σannual 等不变量仍成立）。
scoring/quality 必须全量重算（岗位得分=幸存技能均值，变了）。

产物：output/release/panel_v2d/（14 件同构 + 治理件），run_id 默认 v2d。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.v2d --run-id 20260909_v2d
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from .export_release import _meta
from .lexicon_v2d import LEXICON_VERSION
from . import quality, scoring

logger = logging.getLogger("ai_penetration.panel_v2.v2d")

EXCLUDED_DISPOSITIONS = ["removed_taut", "removed_lowfreq",
                         "score_excluded_taut_atier"]


def excluded_codes(dic: Path) -> np.ndarray:
    """治理表 → 被排除 skill_id → pass2 vocab 数字码。"""
    taut = pd.read_csv(dic / "v2d_taut_excluded_skill_ids.csv",
                       encoding="utf-8-sig")
    keep = pd.read_csv(dic / "skill_legacy_activated_v1.2.csv",
                       encoding="utf-8-sig")
    keep_ids, taut_ids = set(keep.skill_id), set(taut.skill_id)
    n_taut_lg = sum(s.startswith("legacy:") for s in taut_ids)
    vocab = json.loads((get_project_paths().output_dir / "panel_v2" / "pass2"
                        / "skill_vocab.json").read_text(encoding="utf-8"))
    # 低频清单不单独落表：legacy ∧ ¬keep ∧ ¬taut ≡ removed_lowfreq（完备推导）
    ids = set(taut_ids)
    n_low = 0
    for sid in vocab:
        if (sid.startswith("legacy:") and sid not in keep_ids
                and sid not in taut_ids):
            ids.add(sid)
            n_low += 1
    codes = np.array(sorted(vocab[s] for s in ids if s in vocab), np.int32)
    assert len(codes) == len(ids) - sum(
        1 for s in ids if s not in vocab), "存在 vocab 外排除 id"
    logger.info("v2d 排除: legacy taut %d + A级 taut %d + 低频 %d → 技能码 %d",
                n_taut_lg, len(taut_ids) - n_taut_lg, n_low, len(codes))
    return codes


def build_inputs(rel: Path, rel2: Path, codes: np.ndarray) -> None:
    """v2d 目录输入件：过滤 longs/counts/relevance，复制 flag/firm。"""
    rel2.mkdir(parents=True, exist_ok=True)
    bad = set(codes.tolist())
    # longs 流式过滤（2 亿行，B1 内存纪律）
    src = pq.ParquetFile(rel / "job_skill_long.parquet")
    keep_rows = drop_rows = 0
    with pq.ParquetWriter(rel2 / "job_skill_long.parquet", src.schema_arrow,
                          compression="zstd") as w:
        for rb in src.iter_batches(batch_size=4_000_000,
                                   columns=["job_id", "year", "skill_code"]):
            t = pa.Table.from_batches([rb])
            sc = t["skill_code"].to_numpy()
            m = ~np.isin(sc, codes)
            keep_rows += int(m.sum())
            drop_rows += int((~m).sum())
            w.write_table(t.filter(pa.array(m)))
    logger.info("longs 过滤: 保留 %d，剔除 %d（%.1f%%）", keep_rows, drop_rows,
                drop_rows / (keep_rows + drop_rows) * 100)
    # counts/relevance 行过滤（493k，pandas 足够）
    for name, key in (("skill_ai_counts", None), ("skill_ai_relevance", None)):
        df = pd.read_parquet(rel / f"{name}.parquet")
        df = df[~df.skill_code.isin(bad)]
        df.to_parquet(rel2 / f"{name}.parquet", index=False)
        logger.info("%s 过滤后 %d 行", name, len(df))
    (rel2 / "job_anchor_flag.parquet").write_bytes(
        (rel / "job_anchor_flag.parquet").read_bytes())
    (rel2 / "job_firm.parquet").write_bytes(
        (rel / "job_firm.parquet").read_bytes())


def finalize(rel: Path, rel2: Path, codes: np.ndarray, run_id: str) -> None:
    """复制字典/锚点/治理件并逐件 metadata（11 字段，v1.2 词表版本）。"""
    for f in ("skill_concept_v1.parquet", "skill_alias_v1.parquet",
              "ai_anchor_dictionary_v1.csv", "skill_legacy_v1.parquet",
              "skill_candidate_d_v1.parquet"):
        (rel2 / f).write_bytes((rel / f).read_bytes())
    dic = get_project_paths().output_dir / "dictionary"
    for f in ("skill_legacy_activated_v1.2.csv", "v2d_taut_excluded_skill_ids.csv"):
        import shutil
        shutil.copy2(dic / f, rel2 / f)
    n_pairs = pq.ParquetFile(rel2 / "job_skill_long.parquet").metadata.num_rows
    specs = [
        ("skill_concept_v1.parquet", "skill_id", "ai_dict.skill_concepts"),
        ("skill_alias_v1.parquet", "alias_id", "ai_dict.skill_aliases(active)"),
        ("skill_candidate_d_v1.parquet", "term", "pass2/skill_vocab.json"),
        ("skill_legacy_v1.parquet", "term", "panel_v2/lexicon.py(union构建处置)"),
        ("skill_legacy_activated_v1.2.csv", "skill_id",
         "panel_v2/lexicon_v2d.py(§8通道化)"),
        ("v2d_taut_excluded_skill_ids.csv", "skill_id",
         "panel_v2/lexicon_v2d.py"),
        ("ai_anchor_dictionary_v1.csv", "anchor_version+keyword",
         "panel_v2/anchors.py"),
        ("job_anchor_flag.parquet", "job_id", "panel_v2/scan.py(v2a 复用)"),
        ("job_skill_long.parquet", "job_id+skill_code",
         f"panel_v2/v2d.py(过滤自 v2a，{n_pairs:,}对)"),
        ("job_firm.parquet", "job_id", "panel_v2/scan.py(v2a 复用)"),
        ("skill_ai_counts.parquet", "skill+ver+win+year", "panel_v2/v2d.py(行过滤)"),
        ("skill_ai_relevance.parquet", "skill+ver+win+year", "panel_v2/v2d.py(行过滤)"),
        ("job_ai_classification.parquet", "job_id", "panel_v2/scoring.py(v2d)"),
        ("job_ai_score_loo.parquet", "job_id", "panel_v2/scoring.py(v2d)"),
        ("quality_control_report.md", "-", "panel_v2/quality.py(v2d)"),
    ]
    made = 0
    for name, pk, src in specs:
        p = rel2 / name
        if not p.exists():
            logger.warning("缺文件跳过: %s", name)
            continue
        _meta(p, run_id, pk, src, anchor_version="main,cn_paper,babina",
              dictionary_version=LEXICON_VERSION)
        made += 1
    single = rel2 / "job_ai_score.parquet"
    if not single.exists():  # 流式合并（内存纪律，同 export_release 教训）
        files = sorted((rel2 / "job_ai_score").glob("*.parquet"))
        schema = pq.read_schema(files[0])
        with pq.ParquetWriter(single, schema, compression="zstd") as w:
            for f in files:
                for rb in pq.ParquetFile(f).iter_batches(batch_size=2_000_000):
                    w.write_table(pa.Table.from_batches([rb]))
    _meta(single, run_id, "job+ver+win+stype", "panel_v2/scoring.py(v2d)",
          anchor_version="main,cn_paper,babina",
          dictionary_version=LEXICON_VERSION)
    print(f"v2d 发布完成: {made + 1} 件 + metadata @ {rel2}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v2d 治理后重打分发布")
    parser.add_argument("--run-id", default="20260909_v2d")
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_v2d.log")
    rel = paths.output_dir / "release" / "panel_v2"
    rel2 = paths.output_dir / "release" / "panel_v2d"
    if (rel2 / "quality_control_report.md").exists():
        raise SystemExit("panel_v2d 已生成（删目录重跑或改 run-id）")
    t0 = datetime.now()
    codes = excluded_codes(paths.output_dir / "dictionary")
    build_inputs(rel, rel2, codes)
    scoring.run(rel2)
    quality.run(rel2)
    finalize(rel, rel2, codes, args.run_id)
    print(f"v2d 全链完成，用时 {(datetime.now()-t0).total_seconds()/60:.1f} 分钟")


if __name__ == "__main__":
    main()
