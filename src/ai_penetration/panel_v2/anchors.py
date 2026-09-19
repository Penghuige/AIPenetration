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
（job_anchor_flag：三列 0/1 + main 命中组 bitmask）。
锚点规则版本 ``ANCHOR_RULES_VERSION``（扫描日志与 QC 报告落戳）：
- ``20260908_a``：发布面板（v2ac）所用规则——LLM 全拼复数不可命中、
  TRANS 英文无尾界、语言字段按"含小写字母即 en"误判全大写缩写。
- ``20260909_b``（当前）：修订 LLM 复数（en_pl 规则）、TRANS 尾界+复数、
  词典语言字段按规则类型派生。影响上界 ≤0.018% 岗位（LLM|TRANS 全组命中
  4,214），发布数字不回算，自全国 392 城重跑起生效；跨版本对比须核对戳记。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# 规则版本戳（20260909_b：LLM 复数 en_pl、TRANS 尾界、语言字段派生修正）
ANCHOR_RULES_VERSION = "20260909_b"

# (组标签, 关键词, 匹配规则类型)
# 规则类型：zh=子串；en=短语(空格/连字符灵活,ASCII 边界)；
#          en_pl=短语+可选复数 s（§12.1 "large language models"）；
#          abbr=ASCII边界缩写；abbr_llms=缩写+可选 s；
#          trans_zh=Transformer+中文限定；trans_en=Transformer+英文限定词(尾界+可选复数)
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
    ("LLM", "large language model", "en_pl"),
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

# 语言字段按规则类型派生（20260909_b：旧式"含小写字母即en"把全大写缩写标成
# zh、把 Transformer模型 这类中英混排规则标成 en，共 6/43 行错）
_RULE_LANG = {"zh": "zh", "trans_zh": "zh", "en": "en", "en_pl": "en",
              "abbr": "en", "abbr_llms": "en", "trans_en": "en"}


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
    if rule == "en_pl":  # 20260909_b：短语+可选复数（large language models）
        return re.compile(rf"(?<![a-z0-9]){_phrase_pattern(low)}s?(?![a-z0-9])")
    if rule == "abbr":
        return re.compile(rf"(?<![a-z0-9]){re.escape(low)}(?![a-z0-9])")
    if rule == "abbr_llms":
        return re.compile(rf"(?<![a-z0-9]){re.escape(low)}s?(?![a-z0-9])")
    if rule == "trans_zh":  # transformer 与 模型/架构 组合，允许间隔
        tail = low.split("transformer", 1)[1]
        return re.compile(rf"(?<![a-z0-9])transformer[\s\-]*{re.escape(tail)}")
    if rule == "trans_en":  # 20260909_b：补尾界防 modeling 误命中，容复数
        return re.compile(rf"(?<![a-z0-9]){_phrase_pattern(low)}s?(?![a-z0-9])")
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
    for grp, _term, _rule, pat in _COMPILED[version]:
        matches = list(pat.finditer(norm_text))
        if matches:
            if grp not in groups_hit:
                groups_hit.append(grp)
            for match in matches:
                surface = match.group(0)
                if surface not in terms_hit:
                    terms_hit.append(surface)
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
                "language": _RULE_LANG[rule],
                "matching_rule": rule,
                "ambiguity_flag": 1 if rule in _AMBIGUOUS_RULES else 0,
            })
    return rows
