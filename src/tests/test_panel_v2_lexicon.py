"""union 词表与三态文本清洗的离线测试。"""
from __future__ import annotations

from src.ai_penetration.panel_v2.anchors import normalize_desc
from src.ai_penetration.panel_v2.lexicon import LEGACY_PREFIX, build_union_lexicon
from src.ai_penetration.text_clean import (
    clean_description,
    match_from_raw,
    normalize_position,
    text_hash,
)

_ALIAS_DATA = [
    ("计算机视觉", "uuid-cv"),
    ("机器学习", "uuid-ml"),
    ("深度学习", "uuid-dl"),          # 同形词
    ("PyTorch", "uuid-pytorch"),
    ("大语言模型", "uuid-llm"),
]


def _lex():
    return build_union_lexicon(
        legacy_terms=["tensorflow", "halcon", "PyTorch", "目标检测"],
        aliases=_ALIAS_DATA,
    )


def test_extract_unions_both_layers():
    lex = _lex()
    got = lex.extract(normalize_desc("负责机器学习、PyTorch 与 tensorflow 训练"))
    assert {"uuid-ml", "uuid-pytorch", LEGACY_PREFIX + "tensorflow"} <= got
    # A 级已覆盖 PyTorch，legacy 同名不重复制键
    assert LEGACY_PREFIX + "pytorch" not in got
    assert "PyTorch" in lex.overlap_terms


def test_ascii_boundary_on_legacy_and_atier():
    lex = _lex()
    # "tensorflow" 嵌在长单词内部不得命中
    assert lex.extract(normalize_desc("mytensorflowx 平台")) == set()
    assert lex.extract(normalize_desc("精通tensorflow。")) != set()


def test_homograph_requires_ai_context():
    lex = _lex()
    hits = lex.extract(normalize_desc("具有深度学习能力，认真负责"))
    assert "uuid-dl" not in hits           # 口语"深入学习"剔除
    hits2 = lex.extract(normalize_desc("负责深度学习模型训练"))
    assert "uuid-dl" in hits2              # AI 语境保留


def test_legacy_namespace_isolated_from_uuids():
    lex = _lex()
    hits = lex.extract(normalize_desc("使用halcon做视觉"))
    assert LEGACY_PREFIX + "halcon" in hits


# ---------- text_clean ----------

def test_clean_keeps_skill_characters():
    raw = "精通C++/C#.NET，R与Go；负责LLM（大模型）微调！！！！"
    out = clean_description(raw)
    for token in ("C++", "C#", ".NET", "LLM", "（大模型）"):
        assert token in out, token
    assert "！！" not in out  # 重复标点压缩


def test_clean_template_lines_removed():
    raw = "职位介绍\n\n岗位：算法\n\n----------\n任职要求：本科"
    out = clean_description(raw)
    assert "----------" not in out
    assert "算法" in out and "任职要求" in out  # 结构性标题行可删，正文保留


def test_match_hash_stable_across_layout():
    a = match_from_raw("负责  机器学习\n\t开发")
    b = match_from_raw("负责机器学习开发")
    assert text_hash(normalize_desc(a)) == text_hash(normalize_desc(b))
    assert text_hash(normalize_desc("机器学习")) != text_hash(normalize_desc("深度学习"))


def test_normalize_position_strips_noise():
    # 同企业编号噪声归并（去重目的），但字母内容保留（保守规则）
    assert normalize_position("软件工程师（HS0287）") == normalize_position("软件工程师（HS0277）")
    assert normalize_position("AI 算法工程师") == "ai算法工程师"
