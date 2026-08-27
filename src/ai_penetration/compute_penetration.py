"""AI 职业渗透率计算模块。

用时间断层法识别 AI 新职业（2022 前零出现 + 2022 后出现次数达标），
并计算 AI 新职业占当年总职业数的比例（职业数占比），按职业大类分组。
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("ai_penetration.compute")

AI_BOUNDARY_YEAR = 2022


def identify_ai_new_occupations(
    position_stats: pd.DataFrame,
    position_occ_map: dict[str, dict],
    min_post_ai_count: int = 5,
    boundary_year: int = AI_BOUNDARY_YEAR,
) -> set[str]:
    """识别 AI 新职业。

    判定规则：某职业名在 boundary_year 前零出现，且在 boundary_year 及之后
    累计出现次数 ≥ min_post_ai_count。

    Args:
        position_stats: position/year/quarter/count 统计表。
        position_occ_map: position -> {occupation_name, occupation_category} 映射。
        min_post_ai_count: AI 后最低出现次数阈值。
        boundary_year: AI 前后分界年。

    Returns:
        AI 新职业名（occupation_name）集合。
    """
    # position -> occupation_name
    pos_to_occ = {
        pos: item["occupation_name"]
        for pos, item in position_occ_map.items()
    }
    df = position_stats.copy()
    df["occupation"] = df["position"].map(pos_to_occ).fillna(df["position"])

    pre = df[df["year"] < boundary_year]
    post = df[df["year"] >= boundary_year]

    pre_occ = set(pre["occupation"].unique())
    post_counts = post.groupby("occupation")["count"].sum()

    ai_new = set()
    for occ, cnt in post_counts.items():
        if occ not in pre_occ and cnt >= min_post_ai_count:
            ai_new.add(occ)
    logger.info("识别出 AI 新职业 %d 个", len(ai_new))
    return ai_new


def _occupation_mapping(position_occ_map: dict[str, dict]) -> tuple[dict[str, str], dict[str, str]]:
    """构建 position → occupation 与 occupation → category 映射。"""
    pos_to_occ: dict[str, str] = {}
    occ_to_cat: dict[str, str] = {}
    for pos, item in position_occ_map.items():
        occ = item["occupation_name"] or pos
        cat = item["occupation_category"] or "其他"
        pos_to_occ[pos] = occ
        occ_to_cat[occ] = cat
    return pos_to_occ, occ_to_cat


def compute_penetration(
    position_stats: pd.DataFrame,
    position_occ_map: dict[str, dict],
    ai_new: set[str],
    boundary_year: int = AI_BOUNDARY_YEAR,
) -> pd.DataFrame:
    """按年计算渗透率。

    分母：当年出现过的总职业数；分子：当年属于 AI 新职业的职业数。
    按 (year, category) 分组。

    Args:
        position_stats: position/year/quarter/count 统计表。
        position_occ_map: position -> {occupation_name, occupation_category}。
        ai_new: AI 新职业名集合（identify_ai_new_occupations 产出）。
        boundary_year: AI 前后分界年（用于标注但渗透率全年度计算）。

    Returns:
        DataFrame，列 year / category / total_occupations / ai_occupations / penetration。
    """
    pos_to_occ, occ_to_cat = _occupation_mapping(position_occ_map)
    df = position_stats.copy()
    df["occupation"] = df["position"].map(pos_to_occ).fillna(df["position"])
    df["category"] = df["occupation"].map(occ_to_cat).fillna("其他")
    df["is_ai"] = df["occupation"].isin(ai_new)

    rows = []
    for (year, cat), group in df.groupby(["year", "category"]):
        total = group["occupation"].nunique()
        ai_cnt = group[group["is_ai"]]["occupation"].nunique()
        rows.append({
            "year": year,
            "category": cat,
            "total_occupations": total,
            "ai_occupations": ai_cnt,
            "penetration": ai_cnt / total if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["year", "category"]).reset_index(drop=True)


def compute_quarterly_penetration(
    position_stats: pd.DataFrame,
    position_occ_map: dict[str, dict],
    ai_new: set[str],
    boundary_year: int = AI_BOUNDARY_YEAR,
) -> pd.DataFrame:
    """按季度计算渗透率（2022 后，捕捉生成式 AI 爆发）。

    Args:
        同 compute_penetration。

    Returns:
        DataFrame，列 year / quarter / category / total_occupations / ai_occupations / penetration。
    """
    pos_to_occ, occ_to_cat = _occupation_mapping(position_occ_map)
    df = position_stats.copy()
    df = df[df["year"] >= boundary_year]
    df["occupation"] = df["position"].map(pos_to_occ).fillna(df["position"])
    df["category"] = df["occupation"].map(occ_to_cat).fillna("其他")
    df["is_ai"] = df["occupation"].isin(ai_new)

    rows = []
    for (year, quarter, cat), group in df.groupby(["year", "quarter", "category"]):
        total = group["occupation"].nunique()
        ai_cnt = group[group["is_ai"]]["occupation"].nunique()
        rows.append({
            "year": year,
            "quarter": quarter,
            "category": cat,
            "total_occupations": total,
            "ai_occupations": ai_cnt,
            "penetration": ai_cnt / total if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["year", "quarter", "category"]).reset_index(drop=True)
