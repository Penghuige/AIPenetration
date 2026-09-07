"""三套 AI 锚点字典与确定性匹配（指南 §12 逐字落地）。

锚点集合（§12.1–12.3）：
- ``main``      六组：AI / ML / NLP / CV(计算机视觉+图像识别) / LLM(大语言模型系) /
                TRANS(Transformer+模型/架构限定，禁裸词与"变换器模型"，禁裸"大模型")
- ``cn_paper``  四组：AI / ML / NLP / 图像识别（无"计算机视觉"；禁独立 IR）
- ``babina``    四组：AI / ML / NLP / 计算机视觉（无"图像识别"）

匹配规则（§12.6）：
1. 文本先 NFKC 规范化 + 小写（全角英文归一）；中文关键词按连续子串匹配；
2. 英文短语大小写不敏感，允许一个或多个空格/连字符（machine-learning == machine learning）；
3. 缩写（AI/ML/NLP/LLM/LLMs）用 ASCII 字母数字边界，不在其他单词内部命中；
4. 不使用独立 CV、IR 锚点；不使用裸词 Transformer/Transformers/大模型/变换器模型；
5. 每个岗位保存命中组与命中词明细（不能只存 0/1，§12.6.7）；
6. 同组多次命中只置位一次（§12.6.8）；
7. 岗位名称不进入任何锚点版本（§12.6.1）——本模块只对描述文本调用。

产出：ai_anchor_dictionary_v1 记录（§12.4 字段）与岗位锚点标记
（job_anchor_flag 的三列 + main 命中明细两列，§12.5）。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# (组标签, 关键词, 语言, 匹配规则类型)
# 规则类型：zh=子串；en=短语(空格/连字符灵活)；abbr=ASCII边界缩写；
#          trans_zh=Transformer+中文限定；trans_en=Transformer+英文限定词
_TERM_TABLE: list[tuple[str, str, str]] = [
    ("AI", "人工智能", "zh"),
    ("AI", "artificial intelligence", "en"),
    ("AI", "AI", "abbr"),
    ("ML", "机器学习", "zh"),
    ("ML", "machine learning", "en"),
    ("ML", "ML", "abbr"),
    ("NLP", "自然语言处理", "zh"),
    ("NLP", "natural language processing", "en"),
    ("NLP", "NLP", "abbr"),
    ("CVISION", "计算机视觉", "zh"),
    ("CVISION", "computer vision", "en"),
    ("CIMAGE", "图像识别", "zh"),
    ("CIMAGE", "image recognition", "en"),
    ("LLM", "大语言模型", "zh"),
    ("LLM", "大型语言模型", "zh"),
    ("LLM", "large language model", "en"),
    ("LLM", "LLM", "abbr_llms"),
    ("TRANS", "Transformer模型", "trans_zh"),
    ("TRANS", "Transformer架构", "trans_zh"),
    ("TRANS", "transformer model", "trans_en"),
    ("TRANS", "transformer architecture", "trans_en"),
]

# 各锚点版本包含的组（§12.1–12.3；cn_paper/babina 把 CV 组拆开）
ANCHOR_VERSIONS: dict[str, tuple[str, ...]] = {
    "main": ("AI", "ML", "NLP", "CVISION", "CIMAGE", "LLM", "TRANS"),
    "cn_paper": ("AI", "ML", "NLP", "CIMAGE"),
    "babina": ("AI", "ML", "NLP", "CVISION"),
}

# 歧义标记（§12.4 ambiguity_flag）：缩写词存在语境歧义，命中明细保留供审计
_AMBIGUOUS_RULES = {"abbr", "abbr_llms"}


def _phrase_pattern(term: str) -> str:
    """英文短语 → 空格/连字符灵活的正则片段。"""
    return r"[\s\-]+".join(re.escape(w) for w in term.split())


def _compile_term(term: str, rule: str) -> re.Pattern:
    """按规则类型编译单个关键词正则（作用于 NFKC+lower 后文本）。"""
    low = unicodedata.normalize("NFKC", term).lower()
    if rule == "zh":
        return re.compile(re.escape(low))
    if rule == "en":
        return re.compile(rf"(?<![a-z0-9]){_phrase_pattern(low)}(?![a-z0-9])")
    if rule == "abbr":
        return re.compile(rf"(?<![a-z0-9]){re.escape(low)}(?![a-z0-9])")
    if rule == "abbr_llms":
        return re.compile(rf"(?<![a-z0-9]){re.escape(low)}s?(?![a-z0-9])")
    if rule == "trans_zh":  # transformer 与 模型/架构 组合，允许间隔
        tail = low.split("transformer", 1)[1]
        return re.compile(rf"(?<![a-z0-9])transformer[\s\-]*{re.escape(tail)}")
    if rule == "trans_en":
        return re.compile(rf"(?<![a-z0-9]){_phrase_pattern(low)}")
    raise ValueError(f"未知匹配规则: {rule}")


@dataclass(frozen=True)
class AnchorHit:
    """单个岗位在单个锚点版本下的命中结果（§12.5 明细要求）。"""

    flag: int
    groups: tuple[str, ...]
    terms: tuple[str, ...]


# 预编译：version -> [(组, 原词, 规则, compiled)]
_COMPILED: dict[str, list[tuple[str, str, str, re.Pattern]]] = {
    ver: [(grp, term, rule, _compile_term(term, rule))
          for grp, term, rule in _TERM_TABLE if grp in set(groups)]
    for ver, groups in ANCHOR_VERSIONS.items()
}

_MATCH_TEXT_RE = re.compile(r"[\s]+")


def normalize_desc(text: str) -> str:
    """锚点匹配用规范化：NFKC + 小写 + 连续空白折叠为单空格。

    指南英文短语允许"一个或多个空格或连字符"，折叠空白使
    "自然语言处理"等子串与多空格排版等价。
    """
    norm = unicodedata.normalize("NFKC", text or "").lower()
    return _MATCH_TEXT_RE.sub(" ", norm)


def match_anchors(version: str, norm_text: str) -> AnchorHit:
    """对已规范化的描述文本执行某锚点版本的确定性匹配。

    Args:
        version: main / cn_paper / babina。
        norm_text: normalize_desc() 输出（仅描述文本；岗位名不得传入）。

    Returns:
        AnchorHit：flag(0/1) + 命中组 + 命中词（明细，§12.6.7）。
    """
    groups_hit: list[str] = []
    terms_hit: list[str] = []
    for grp, term, _rule, pat in _COMPILED[version]:
        if pat.search(norm_text):
            if grp not in groups_hit:
                groups_hit.append(grp)
            terms_hit.append(term)
    return AnchorHit(
        flag=1 if groups_hit else 0,
        groups=tuple(groups_hit),
        terms=tuple(terms_hit),
    )


def match_all_versions(desc: str) -> dict[str, AnchorHit]:
    """三套锚点一次匹配（供扫描 worker 调用）。

    Args:
        desc: 岗位描述原文（内部做规范化）。

    Returns:
        {version: AnchorHit}。
    """
    norm = normalize_desc(desc)
    return {ver: match_anchors(ver, norm) for ver in ANCHOR_VERSIONS}


def anchor_dictionary_rows() -> list[dict]:
    """导出 §12.4 ai_anchor_dictionary_v1 记录。

    Returns:
        每行 {anchor_version, anchor_group, keyword, keyword_normalized,
        language, matching_rule, ambiguity_flag}。
    """
    rows = []
    for ver, groups in ANCHOR_VERSIONS.items():
        gset = set(groups)
        for grp, term, rule in _TERM_TABLE:
            if grp not in gset:
                continue
            rows.append({
                "anchor_version": ver,
                "anchor_group": grp,
                "keyword": term,
                "keyword_normalized": unicodedata.normalize("NFKC", term).lower(),
                "language": "en" if re.search(r"[a-z]", term) else "zh",
                "matching_rule": rule,
                "ambiguity_flag": 1 if rule in _AMBIGUOUS_RULES else 0,
            })
    return rows
