"""eps 岗位描述加载与技能抽取流水线。

从广东省分片表加载岗位描述，用技能词典做简单子串匹配抽取技能，
标注种子 AI 岗位（岗位名命中 AI 关键词），供共现度计算使用。
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from sqlalchemy import text

from .keyword_penetration import match_ai
from .load_guangdong import GD_SHARDS
from .skill_dictionary import load_skill_names

logger = logging.getLogger("ai_penetration.skill_data")


# AI 技能词表模块级缓存（避免每岗位重复加载文件）
_AI_SKILL_TERMS_CACHE: list[str] | None = None


def _extract_ai_skills(
    description: str,
    ai_skill_terms: list[str] | None = None,
) -> list[str]:
    """从岗位描述中匹配 AI 技术技能词。

    用 dicts/ai_skill_terms.txt 的 AI 技能词典做词边界匹配。
    短英文词（如 R/C）用词边界，中文词子串匹配。
    词表只在首次调用时加载并缓存。

    Args:
        description: 岗位描述文本。
        ai_skill_terms: AI 技能词表；None 时使用模块级缓存。

    Returns:
        命中的 AI 技能词列表。
    """
    global _AI_SKILL_TERMS_CACHE
    if ai_skill_terms is None:
        if _AI_SKILL_TERMS_CACHE is None:
            from .skill_dictionary import load_ai_skill_terms

            _AI_SKILL_TERMS_CACHE = load_ai_skill_terms()
        ai_skill_terms = _AI_SKILL_TERMS_CACHE
    desc = description or ""
    matched = []
    for term in ai_skill_terms:
        if not term:
            continue
        if _needs_boundary(term):
            if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", desc):
                matched.append(term)
        elif term in desc:
            matched.append(term)
    return matched


def _extract_skills(description: str, skill_names: list[str]) -> list[str]:
    """从岗位描述中匹配技能词。

    纯英文技能名（如 R/C/Go）用词边界匹配，避免误匹配到其他词的子串
    （如 "R" 误匹配 "Spring"）。含中文或长度 ≥5 的英文技能名用子串匹配。

    Args:
        description: 岗位描述文本。
        skill_names: 技能词典词表。

    Returns:
        命中的技能名列表（按词典顺序去重）。
    """
    desc = description or ""
    matched = []
    for name in skill_names:
        if not name:
            continue
        if _needs_boundary(name):
            if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", desc):
                matched.append(name)
        elif name in desc:
            matched.append(name)
    return matched


def _needs_boundary(name: str) -> bool:
    """判断技能名是否需要词边界匹配。

    纯英文且长度 ≤4（如 R/C/Go/AI）时，子串匹配误报风险高，需词边界。

    Args:
        name: 技能名。

    Returns:
        True 表示需词边界匹配。
    """
    if not name:
        return False
    if re.search(r"[一-鿿]", name):
        return False  # 含中文，子串匹配即可
    return len(name) <= 4


def load_job_skill_data(
    engine,
    cities: list[str] | None = None,
    min_freq: int = 10,
    max_jobs: int = 200000,
) -> pd.DataFrame:
    """加载岗位描述、抽取技能、标注种子 AI 岗位。

    Args:
        engine: eps 数据库 engine。
        cities: 限定城市；None 表示全部 21 市。
        min_freq: 技能频次阈值，预留给频次过滤；当前实现按描述非空加载。
        max_jobs: 抽样岗位数上限（控制技能抽取规模）。

    Returns:
        DataFrame，列 job_id / position / is_ai_seed / skills（list）。
    """
    shards = {c: s for c, s in GD_SHARDS.items() if not cities or c in cities}
    skill_names = load_skill_names()
    if not skill_names:
        logger.warning("技能词典为空，跳过技能抽取")
        return pd.DataFrame(columns=["job_id", "position", "is_ai_seed", "skills"])

    rows = []
    remaining = max_jobs if max_jobs > 0 else 0
    for city, shard in shards.items():
        if max_jobs > 0 and remaining <= 0:
            logger.info("已达全局岗位数上限 %d，提前结束", max_jobs)
            break
        sql = f"""
            SELECT position, job_description,
                   substr(publish_time, 1, 4) AS yr
            FROM public.{shard}
            WHERE job_description IS NOT NULL AND job_description != ''
              AND position IS NOT NULL AND position != ''
              AND publish_time ~ '^[0-9]{{4}}-'
        """
        if max_jobs > 0:
            sql += f" LIMIT {int(remaining)}"
        with engine.connect() as conn:
            result = conn.execute(text(sql)).mappings().all()
        for r in result:
            position = str(r["position"] or "")
            desc = str(r["job_description"] or "")
            year = int(r["yr"]) if r["yr"] else 0
            rows.append({
                "job_id": f"{city}_{len(rows)}",
                "position": position,
                "description": desc,
                "year": year,
                "is_ai_seed": match_ai(position),
                "skills": _extract_skills(desc, skill_names),
            })
        if max_jobs > 0:
            remaining -= len(result)
        logger.info("%s 已加载 %d 条岗位（累计 %d）", city, len(result), len(rows))
    df = pd.DataFrame(rows)
    logger.info("技能数据加载完成: %d 条, 种子 AI %d 条",
                len(df), int(df["is_ai_seed"].sum()) if not df.empty else 0)
    return df
