"""技能 AI 共现度与岗位 AI 相关度计算模块。

数据驱动：ai_score(skill) = P(skill|AI岗位) / P(skill|全部岗位)。
岗位 AI 相关度 = 该岗位所有技能 ai_score 的平均值。

默认对 ai_score 做 sigmoid 归一化（ratio/(1+ratio)），将原始比率压缩到
[0,1]，避免单个离群技能（如深度学习 301）主导岗位相关度。
"""
from __future__ import annotations

import logging

logger = logging.getLogger("ai_penetration.cooccurrence")


def _normalize_score(ratio: float) -> float:
    """Sigmoid 归一化：ratio/(1+ratio)，将 [0,∞) 映射到 [0,1)。

    Args:
        ratio: 原始共现比率。

    Returns:
        [0,1) 区间的归一化分数。
    """
    if ratio <= 0:
        return 0.0
    return ratio / (1.0 + ratio)


def compute_skill_ai_scores(
    skill_counts_ai: dict[str, int],
    skill_counts_all: dict[str, int],
    n_ai_jobs: int,
    n_all_jobs: int,
    min_count: int = 5,
    min_ai_count: int = 3,
    alpha: float = 0.5,
    normalize: bool = True,
) -> dict[str, float]:
    """计算每个技能的 AI 共现度。

    共现度 = P(skill|AI岗位) / P(skill|全部岗位)，用拉普拉斯平滑抑制
    小样本噪声。全部岗位或 AI 岗位中出现次数过低的技能得分被压缩。

    Args:
        skill_counts_ai: 技能在 AI 岗位中的出现次数。
        skill_counts_all: 技能在全部岗位中的出现次数。
        n_ai_jobs: AI 岗位总数。
        n_all_jobs: 全部岗位总数。
        min_count: 技能在全部岗位的最低出现次数（低于则压缩分数）。
        min_ai_count: 技能在 AI 岗位的最低出现次数（低于则压缩分数）。
        alpha: 拉普拉斯平滑参数。
        normalize: 是否做 sigmoid 归一化（默认 True）。

    Returns:
        skill -> ai_score 映射。normalize=True 时分数在 [0,1)；
        normalize=False 时原始比率，>1 表示偏向 AI 岗位。
    """
    if n_ai_jobs <= 0 or n_all_jobs <= 0:
        return {}
    scores: dict[str, float] = {}
    for skill in skill_counts_all:
        cnt_all = skill_counts_all.get(skill, 0)
        cnt_ai = skill_counts_ai.get(skill, 0)
        if cnt_all == 0:
            continue
        # 拉普拉斯平滑
        p_all = (cnt_all + alpha) / (n_all_jobs + 2 * alpha)
        p_ai = (cnt_ai + alpha) / (n_ai_jobs + 2 * alpha)
        raw = p_ai / p_all if p_all else 0.0
        if cnt_all < min_count or cnt_ai < min_ai_count:
            # 稀疏技能：原始得分不超过 1.0，避免低频噪声放大
            raw = min(1.0, raw)
        scores[skill] = _normalize_score(raw) if normalize else raw
    logger.info("技能共现度计算完成: %d 个技能", len(scores))
    return scores


def compute_job_ai_relevance(
    skills: list[str],
    ai_scores: dict[str, float],
) -> float:
    """计算岗位 AI 相关度（技能平均 ai_score）。

    Args:
        skills: 岗位抽取出的技能列表。
        ai_scores: skill -> ai_score 映射。

    Returns:
        岗位 AI 相关度；无技能或全部无分值时返回 0.0。
    """
    if not skills:
        return 0.0
    scored = [ai_scores[s] for s in skills if s in ai_scores]
    if not scored:
        return 0.0
    return sum(scored) / len(scored)
