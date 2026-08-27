"""加权 AI 判定：岗位名关键词 + 描述技能词各带权重，累加超阈值才算 AI。

权重分三层（见 dicts/ai_skill_terms.txt / ai_occupation_keywords.txt）：
- 3 = 特定 AI 技术词（岗位名关键词默认、描述中如 PyTorch/SLAM/轨迹预测/RAG…，单独出现即真 AI）
- 2 = 通用强词（深度学习/大模型/人脸识别/数字人…，公司简介/模板爱用，需与其他词佐证）
- 1 = 弱词（AI设计/内容安全/特征工程/岗位名中的 "AI"…，歧义大，需共同佐证）

判定规则：``岗位名得分 + 描述得分 >= AI_SCORE_THRESHOLD（3）`` 才算 AI 渗透。

- 一个特定技术词（3）→ 命中；通用强词+任一佐证（2+≥1）→ 命中；弱词需多个
- 误报语境排除：如「AI设计」后紧跟「软件」（=Adobe Illustrator）时不计数
- 口语同形词处理：「深度学习/强化学习」仅在 AI 技术语境计分，
  与方法 B（skill_ai_anchor）口径一致（见提交 0eb2677 的修复说明）

使用示例::

    from src.ai_penetration.ai_scoring import is_ai_job, match_keywords_scored

    is_ai_job("AI司机", "负责接送...")  # False（AI=1 分，不足以判定）
    is_ai_job("图像算法工程师", "负责深度学习模型与PyTorch部署")  # True（2+3=5）
    is_ai_job("平面设计师", "熟练PS、AI设计软件")  # False（AI设计被排除）
"""
from __future__ import annotations

import re

from .keyword_penetration import (
    _STRIP_COMPANY_BRACKET_RE,
    _keyword_matches,
    AI_KEYWORDS,
    AI_KEYWORD_WEIGHTS,
)
from .skill_data import _needs_boundary
from .skill_dictionary import load_ai_skill_weights

# 判定阈值：岗位名得分 + 描述得分 >= 阈值
# 3 = 需一个特定技术词（权重3）或 通用强词+佐证（2+≥1）或 标题强词（3）
AI_SCORE_THRESHOLD = 3.0

# 误报语境排除：词 → 排除正则（命中该正则则该词不计分）
_EXCLUDED_CONTEXTS: dict[str, re.Pattern] = {
    # "AI设计软件" = Adobe Illustrator（平面设计岗常见），非 AI
    "AI设计": re.compile(r"AI设计(?=软件)"),
}

# 口语同形异义词 AI 语境要求：词 → 语境正则。
# 这些词在招聘文案中常作口语用法（如"深入学习能力"），只有当正文出现
# 其 AI 技术语境（后接技术后缀/非口语标记）时才计分，否则剔除。
# 模式与 skill_ai_anchor._AMBIGUOUS_AI_TERMS 保持一致，
# 保证方法 A 与方法 B/融合口径判定一致。
_AI_CONTEXT_REQUIRED: dict[str, re.Pattern] = {
    "深度学习": re.compile(r"深度学习(?!能力|精神|意识|态度|习惯|思考|领悟|钻研|学习|力)"),
    "强化学习": re.compile(r"强化学习(?!能力|学习)"),
}

_AI_SKILL_WEIGHTS_CACHE: dict[str, float] | None = None


def _ai_skill_weights() -> dict[str, float]:
    """加载 AI 技能词权重（模块级缓存）。"""
    global _AI_SKILL_WEIGHTS_CACHE
    if _AI_SKILL_WEIGHTS_CACHE is None:
        _AI_SKILL_WEIGHTS_CACHE = load_ai_skill_weights()
    return _AI_SKILL_WEIGHTS_CACHE


def match_keywords_scored(position: str) -> tuple[list[tuple[str, float]], float]:
    """岗位名命中 AI 关键词并累加权重。

    Args:
        position: 岗位名。

    Returns:
        (命中的 (关键词, 权重) 列表, 总权重分)。
    """
    text = _STRIP_COMPANY_BRACKET_RE.sub("", position or "")
    upper = text.upper()
    hits: list[tuple[str, float]] = []
    total = 0.0
    for kw in AI_KEYWORDS:
        weight = AI_KEYWORD_WEIGHTS.get(kw, 3.0)
        if _keyword_matches(kw, upper):
            hits.append((kw, weight))
            total += weight
    return hits, total


def match_skills_scored(description: str) -> tuple[list[tuple[str, float]], float]:
    """描述命中 AI 技能词并累加权重（含误报语境排除）。

    Args:
        description: 岗位描述。

    Returns:
        (命中的 (技能词, 权重) 列表, 总权重分)。
    """
    desc = description or ""
    hits: list[tuple[str, float]] = []
    total = 0.0
    for term, weight in _ai_skill_weights().items():
        excluded = _EXCLUDED_CONTEXTS.get(term)
        if excluded is not None and excluded.search(desc):
            continue
        # 口语同形异义词：仅当出现 AI 技术语境时才计分
        context_required = _AI_CONTEXT_REQUIRED.get(term)
        if context_required is not None and not context_required.search(desc):
            continue
        if _needs_boundary(term):
            matched = re.search(
                rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", desc
            )
        else:
            matched = term in desc
        if matched:
            hits.append((term, weight))
            total += weight
    return hits, total


def is_ai_job(position: str, description: str) -> bool:
    """加权判定岗位是否为 AI 岗位。

    岗位名得分 + 描述得分 >= AI_SCORE_THRESHOLD 时判定为 AI。

    Args:
        position: 岗位名。
        description: 岗位描述。

    Returns:
        True 表示该岗位为 AI 相关。
    """
    _, kw_score = match_keywords_scored(position)
    _, sk_score = match_skills_scored(description)
    return (kw_score + sk_score) >= AI_SCORE_THRESHOLD


def build_prefilter_regex(terms: list[str]) -> re.Pattern:
    """构建快筛合并正则（忽略大小写，作为评分匹配的超集）。

    用于流式处理先快速排除绝大多数非 AI 岗位，命中后再做详细计分，
    避免对全量数据逐词匹配。

    Args:
        terms: 词表（关键词或技能词）。

    Returns:
        编译后的合并正则。
    """
    return re.compile(
        "|".join(re.escape(t) for t in terms),
        re.IGNORECASE,
    )
