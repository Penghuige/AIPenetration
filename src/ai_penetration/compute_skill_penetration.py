"""基于技能 AI 相关度的渗透率汇总。"""
from __future__ import annotations

import logging

import pandas as pd

from .skill_cooccurrence import compute_job_ai_relevance, compute_skill_ai_scores

logger = logging.getLogger("ai_penetration.skill_pen")


def compute_skill_ai_scores_from_df(
    df: pd.DataFrame,
    min_count: int = 5,
) -> dict[str, float]:
    """从岗位技能数据计算技能 AI 共现度。

    Args:
        df: load_job_skill_data 输出，含 is_ai_seed 与 skills。
        min_count: 技能最低出现次数。

    Returns:
        skill -> ai_score 映射。
    """
    ai_df = df[df["is_ai_seed"]]
    n_ai = len(ai_df)
    n_all = len(df)
    counts_ai: dict[str, int] = {}
    counts_all: dict[str, int] = {}
    for skills in df["skills"]:
        for s in skills:
            counts_all[s] = counts_all.get(s, 0) + 1
    for skills in ai_df["skills"]:
        for s in skills:
            counts_ai[s] = counts_ai.get(s, 0) + 1
    return compute_skill_ai_scores(counts_ai, counts_all, n_ai, n_all, min_count)


def compute_job_relevance_series(
    df: pd.DataFrame,
    ai_scores: dict[str, float],
) -> pd.Series:
    """计算每岗位 AI 相关度。"""
    return df["skills"].apply(
        lambda sk: compute_job_ai_relevance(list(sk), ai_scores)
    )
