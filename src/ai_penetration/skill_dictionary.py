"""技能词典与 AI 关键词加载模块。"""
from __future__ import annotations

import logging
from pathlib import Path

from config.paths import get_project_paths

logger = logging.getLogger("ai_penetration.skill_dict")


def load_ai_keyword_set() -> set[str]:
    """从 dicts/ai_occupation_keywords.txt 加载 AI 关键词集。

    Returns:
        AI 关键词集合。
    """
    path = get_project_paths().project_root / "dicts" / "ai_occupation_keywords.txt"
    if not path.exists():
        logger.warning("AI 关键词词典不存在: %s", path)
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def load_skill_names() -> list[str]:
    """加载通用技能名（静态词典）。

    原依赖 Employ26 库 dict.hard_skills/soft_skills，分离后改为
    dicts/skill_names_general.txt 静态文件（2026-09-02 自该库导出 1994 项）。

    Returns:
        技能名列表；文件不存在时返回空列表。
    """
    path = get_project_paths().project_root / "dicts" / "skill_names_general.txt"
    if not path.exists():
        logger.warning("通用技能词典不存在: %s", path)
        return []
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    logger.info("技能词典加载: %d 个技能", len(names))
    return names


def _ai_skill_path() -> Path:
    """返回 AI 技能词典路径。"""
    return get_project_paths().project_root / "dicts" / "ai_skill_terms.txt"


def _parse_weighted_terms(path: Path) -> list[tuple[str, float]]:
    """解析「词|权重」格式的词表。

    Args:
        path: 词表文件路径。

    Returns:
        (词, 权重) 列表；未标注权重时默认 3.0（特定 AI 技术词，单独出现即 AI）。
    """
    if not path.exists():
        logger.warning("词表不存在: %s", path)
        return []
    terms: list[tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        term = parts[0].strip()
        weight = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 3.0
        terms.append((term, weight))
    return terms


def load_ai_skill_terms() -> list[str]:
    """从 dicts/ai_skill_terms.txt 加载 AI 技能词典（仅词名）。

    Returns:
        AI 技术技能词列表。
    """
    terms = [t for t, _ in _parse_weighted_terms(_ai_skill_path())]
    logger.info("AI 技能词典加载: %d 个词", len(terms))
    return terms


def load_ai_skill_weights() -> dict[str, float]:
    """加载 AI 技能词权重 {词: 权重}。

    Returns:
        技能词 → 权重 的映射；强词 2.0，弱词 1.0。
    """
    return dict(_parse_weighted_terms(_ai_skill_path()))
