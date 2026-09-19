"""零技能子样本曲线（接收方追加口径，2026-09-15）。

动机：年度 AI 率与语料可匹配度（零技能率）同呼吸（年度相关 -0.82）——把
"有技能子样本"上的口径作为并列曲线输出，可分离机械分量（详见交接 §3.2
可匹配度注记）。本口径为**追加稳健性**，不改动指南主口径（>0.05 全体岗位）。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.subsample_curve [--rel panel_v2h]
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from .anchors import ANCHOR_RULES_VERSION

logger = logging.getLogger("ai_penetration.panel_v2.subsample")


def curve_frame(cls: pd.DataFrame, score: pd.DataFrame,
                y_min: int = 2016) -> pd.DataFrame:
    """年度曲线（全体 vs 有技能子样本）。

    Args:
        cls: job_ai_classification 帧（需 job_id/year/zero_skill_override/
            aijob_main_annual_raw_005/aijob_main_annual_raw_015）。
        score: main_annual_raw 岗位得分帧（job_id/ai_score）。
        y_min: 起始年（2014/2015 为装饰性小样本，默认剔除）。

    Returns:
        DataFrame[year, n, zero_skill_rate, r005_all, r005_matched,
        hi030_all, hi030_matched]。
    """
    required_cls = {
        "job_id", "year", "zero_skill_override", "aijob_main_annual_raw_005"
    }
    required_score = {"job_id", "ai_score"}
    missing_cls = required_cls - set(cls.columns)
    missing_score = required_score - set(score.columns)
    if missing_cls or missing_score:
        raise ValueError(
            f"子样本曲线输入字段缺失: cls={sorted(missing_cls)} "
            f"score={sorted(missing_score)}"
        )
    if cls.job_id.duplicated().any() or score.job_id.duplicated().any():
        raise ValueError("子样本曲线要求 classification/score 的 job_id 均唯一")

    df = cls.merge(
        score[["job_id", "ai_score"]],
        on="job_id",
        how="left",
        validate="one_to_one",
    )
    if df.ai_score.isna().ne(df.zero_skill_override.astype(bool)).any():
        bad = int(
            df.ai_score.isna().ne(df.zero_skill_override.astype(bool)).sum()
        )
        raise ValueError(
            f"score 与 zero_skill_override 不一致（{bad} 岗位）；拒绝静默删行/补零"
        )
    df["hi030"] = (df.ai_score > 0.30).astype(float)
    rows = []
    for y, sub in df[df.year >= y_min].groupby("year"):
        ok = sub[sub.zero_skill_override == 0]
        rows.append({
            "year": int(y), "n": int(len(sub)),
            "zero_skill_rate": round(float(sub.zero_skill_override.mean()), 5),
            "r005_all": round(float(sub["aijob_main_annual_raw_005"].mean()), 5),
            "r005_matched": round(float(ok["aijob_main_annual_raw_005"].mean()), 5),
            "hi030_all": round(float(sub.hi030.mean()), 5),
            "hi030_matched": round(float(ok.hi030.mean()), 5)})
    return pd.DataFrame(rows)


def run(rel_name: str) -> Path:
    paths = get_project_paths()
    rel = paths.output_dir / "release" / rel_name
    cls = pq.read_table(rel / "job_ai_classification.parquet",
                        columns=["job_id", "year", "zero_skill_override",
                                 "aijob_main_annual_raw_005",
                                 "aijob_main_annual_raw_015"]).to_pandas()
    score = pq.read_table(rel / "job_ai_score" / "main_annual_raw.parquet",
                          columns=["job_id", "ai_score"]).to_pandas()
    frame = curve_frame(cls, score)
    out = paths.report_dir / (
        f"zero_skill_subsample_curve_{datetime.now():%Y%m%d}.csv")
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("子样本曲线落盘 %s（锚点规则 %s，release=%s）",
                out, ANCHOR_RULES_VERSION, rel_name)
    print(frame.to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="零技能子样本曲线（追加口径）")
    ap.add_argument("--rel", default="panel_v2h", help="release 子目录名")
    args = ap.parse_args()
    setup_logging(get_project_paths().log_dir / "panel_v2_subsample.log")
    run(args.rel)


if __name__ == "__main__":
    main()
