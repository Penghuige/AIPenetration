"""v2e：按指南原预案复原的主口径代际（2026-09-09，用户指示纠偏）。

与原预案的差距修复（docs/10 复盘结论）：
1. **主指标复原 >0.05**——指南 §2.4"主AI岗位标识：得分严格大于0.05"、
   §16"主阈值为0.05；0.10和0.15用于与Babina方法及阈值敏感性比较"。
   测量对象按指南定义="AI 相关岗位"；"净核心职责 AI 岗位占比"是我方追加的
   解释性问题（金标准），作补充分析层保留，不再反过来改主口径。
2. **撤销 v2d 的 A 级 36 键得分排除**——§10.3"分级只表来源稳定性，
   不代表不同权重；A/B/C 均参与"；"机器学习"等键 ω≡1 是指针设计本义
   （§1.10 允许记录性测量误差），仅作披露属性登记。
3. **legacy 层归位 §10 A/B/C/D 通道**：B=df≥100、C 三分支、D=其余不进正式
   匹配（频率用 zh_alias_freq 的 (platform,规范化文本) 去重语料口径，§10.3.2
   的 COUNT(DISTINCT text_hash) 之保守同族；文本面差异在 QC 披露）。
   v2d 的"taut 剔除/频数门"动作降级为敏感性世代保留。

产物 release/panel_v2e/（run_id 20260909_v2e，词表版本
bilingual_a_frozen_v1.1+legacy_grade_v1）。等价性依据同 v2d（只删不增）。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.v2e --run-id 20260909_v2e
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from .export_release import _meta
from .lexicon_v2d import grade_legacy_frame
from .v2d import build_inputs

logger = logging.getLogger("ai_penetration.panel_v2.v2e")

LEX_VERSION = "bilingual_a_frozen_v1.1+legacy_grade_v1"


def grade_and_exclusions(rel: Path, out_dir: Path) -> tuple[pd.DataFrame, "object"]:
    """分级并给出排除码集（仅 legacy D 级）。"""
    vocab = json.loads((out_dir / "panel_v2" / "pass2" / "skill_vocab.json")
                       .read_text(encoding="utf-8"))
    code2sid = {v: k for k, v in vocab.items()}
    df_csv = out_dir / "dictionary" / "legacy_df_freq_v1.csv"
    dfreq = pd.read_csv(df_csv, encoding="utf-8-sig")
    legacy_keys = dict(zip(dfreq.match_key, dfreq.skill_id))
    df_map = dict(zip(dfreq.match_key, dfreq.df_unique_text))
    counts = pq.read_table(rel / "skill_ai_counts.parquet").to_pandas()
    main = counts[counts.anchor_version == "main"]
    pooled = main[main.window_type == "pooled"]
    cooc = {code2sid[c]: (n and cc / n) for c, n, cc in
            zip(pooled.skill_code, pooled.n_skill, pooled.n_ai_cooccur)
            if c in code2sid}
    ann = main[main.window_type == "annual"]
    fy = {}
    for c, y in zip(ann.skill_code, ann.year):
        if c in code2sid:
            s = code2sid[c]
            fy[s] = min(fy.get(s, 9999), int(y))
    frame = grade_legacy_frame(legacy_keys, df_map, cooc, fy)
    n = frame.grade.value_counts()
    logger.info("v2e 分级: B %d / C %d / D %d（legacy %d 键）",
                int(n.get("B", 0)), int(n.get("C", 0)), int(n.get("D", 0)),
                len(frame))
    return frame, frame


def main() -> None:
    parser = argparse.ArgumentParser(description="v2e 原预案复原代际")
    parser.add_argument("--run-id", default="20260909_v2e")
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_v2e.log")
    rel = paths.output_dir / "release" / "panel_v2"
    rel3 = paths.output_dir / "release" / "panel_v2e"
    if (rel3 / "quality_control_report.md").exists():
        raise SystemExit("panel_v2e 已存在（删目录或改 run-id）")
    t0 = datetime.now()
    import numpy as np
    frame, _ = grade_and_exclusions(rel, paths.output_dir)
    vocab = json.loads((paths.output_dir / "panel_v2" / "pass2"
                        / "skill_vocab.json").read_text(encoding="utf-8"))
    d_ids = set(frame[frame.grade == "D"].skill_id)
    codes = np.array(sorted(vocab[s] for s in d_ids if s in vocab), np.int32)
    print(f"D 级排除 {len(codes)} 键；A 级全参与；B/C 共 "
          f"{int((frame.grade != 'D').sum())} 键正式参与")
    build_inputs(rel, rel3, codes)
    from . import quality, scoring
    scoring.run(rel3)
    quality.run(rel3)
    # 分级治理件
    import shutil
    gcsv = paths.output_dir / "dictionary" / "skill_legacy_graded_BCD_v1.csv"
    frame.to_csv(gcsv, index=False, encoding="utf-8-sig")
    for f in ("skill_concept_v1.parquet", "skill_alias_v1.parquet",
              "ai_anchor_dictionary_v1.csv", "skill_legacy_v1.parquet",
              "skill_candidate_d_v1.parquet"):
        (rel3 / f).write_bytes((rel / f).read_bytes())
    shutil.copy2(gcsv, rel3 / "skill_legacy_graded_BCD_v1.csv")
    single = rel3 / "job_ai_score.parquet"
    import pyarrow as pa
    if not single.exists():
        files = sorted((rel3 / "job_ai_score").glob("*.parquet"))
        with pq.ParquetWriter(single, pq.read_schema(files[0]),
                              compression="zstd") as w:
            for f in files:
                for rb in pq.ParquetFile(f).iter_batches(batch_size=2_000_000):
                    w.write_table(pa.Table.from_batches([rb]))
    specs = [
        ("skill_concept_v1.parquet", "skill_id", "ai_dict.skill_concepts"),
        ("skill_alias_v1.parquet", "alias_id", "ai_dict.skill_aliases(active)"),
        ("skill_candidate_d_v1.parquet", "term", "pass2/skill_vocab.json"),
        ("skill_legacy_v1.parquet", "term", "panel_v2/lexicon.py(union构建处置)"),
        ("skill_legacy_graded_BCD_v1.csv", "skill_id",
         "panel_v2/lexicon_v2d.py(§10.3.1 A/B/C/D 分级)"),
        ("ai_anchor_dictionary_v1.csv", "anchor_version+keyword",
         "panel_v2/anchors.py"),
        ("job_anchor_flag.parquet", "job_id", "panel_v2/scan.py(v2a 复用)"),
        ("job_skill_long.parquet", "job_id+skill_code", "panel_v2/v2e.py(D级行过滤)"),
        ("job_firm.parquet", "job_id", "panel_v2/scan.py(v2a 复用)"),
        ("skill_ai_counts.parquet", "skill+ver+win+year", "panel_v2/v2e.py(行过滤)"),
        ("skill_ai_relevance.parquet", "skill+ver+win+year", "panel_v2/v2e.py(行过滤)"),
        ("job_ai_score.parquet", "job+ver+win+stype", "panel_v2/scoring.py(v2e)"),
        ("job_ai_classification.parquet", "job_id", "panel_v2/scoring.py(v2e)"),
        ("job_ai_score_loo.parquet", "job_id", "panel_v2/scoring.py(§14.3 v2e)"),
        ("quality_control_report.md", "-", "panel_v2/quality.py(v2e)"),
    ]
    made = 0
    for name, pk, src in specs:
        p = rel3 / name
        if not p.exists():
            logger.warning("缺文件跳过: %s", name)
            continue
        _meta(p, args.run_id, pk, src, anchor_version="main,cn_paper,babina",
              dictionary_version=LEX_VERSION)
        made += 1
    print(f"v2e 发布: {made} 件 @ {rel3}，用时 "
          f"{(datetime.now() - t0).total_seconds() / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
