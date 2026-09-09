"""三套锚点字典的指南 §12.6 测试：正例 + 全部规定反例。"""
from __future__ import annotations

from src.ai_penetration.panel_v2.anchors import (
    anchor_dictionary_rows,
    match_all_versions,
    match_anchors,
    normalize_desc,
)


def _flag(text: str, version: str = "main") -> int:
    return match_all_versions(text)[version].flag


# ---------- 正例 ----------

def test_positive_phrases_and_case_variants():
    assert _flag("熟悉 Machine-Learning 与 natural  language  processing") == 1
    assert _flag("负责自然语言处理NLP项目") == 1
    hit = match_anchors("main", normalize_desc("部署 LLMs 与 AI 平台"))
    assert set(hit.groups) == {"LLM", "AI"}
    assert hit.flag == 1
    assert "LLM" in hit.terms


def test_positive_transformer_with_qualifier():
    assert _flag("基于Transformer架构的视觉大模型") == 1  # TRANS 组
    assert _flag("transformer models 推理优化") == 1


def test_positive_fullwidth_and_punctuation():
    assert _flag("全角ＬＬＭ适配") == 1
    assert _flag("使用ML、AI。等") == 1


def test_hit_details_recorded_not_only_flag():
    hit = match_anchors("main", normalize_desc("计算机视觉与图像识别研发"))
    assert hit.flag == 1
    assert set(hit.groups) == {"CVISION", "CIMAGE"}
    assert "计算机视觉" in hit.terms


# ---------- 反例（§12.6 明文要求覆盖） ----------

def test_negative_bare_transformer():
    assert _flag("负责Transformer的推理优化") == 0
    assert _flag("Transformer技术栈") == 0


def test_negative_bare_large_model_and_alt_name():
    assert _flag("有大模型应用经验") == 0
    assert _flag("变换器模型研究") == 0


def test_negative_abbr_inside_words():
    assert _flag("AIML平台") == 0
    assert _flag("SNLP与GLLM统计") == 0


def test_negative_standalone_cv_ir():
    assert _flag("熟悉CV与IR规范") == 0


def test_negative_transformer_middle_word():
    # 短语允许空格/连字符但 transformer-based architecture 不是指南认可组合
    assert _flag("transformer-based pipeline") == 0


# ---------- 三套版本差异 ----------

def test_version_split_image_vs_vision():
    assert (_flag("图像识别", "main"), _flag("图像识别", "cn_paper"),
            _flag("图像识别", "babina")) == (1, 1, 0)
    assert (_flag("计算机视觉", "main"), _flag("计算机视觉", "cn_paper"),
            _flag("计算机视觉", "babina")) == (1, 0, 1)


# ---------- 20260909_b 修订回归（审计 D4/D5/语言字段） ----------

def test_llm_full_phrase_plural_hits():
    # 指南 §12.1 明列 "large language models"，旧规则复数被尾界阻断
    assert _flag("熟悉large language models相关技术") == 1
    assert _flag("large language model微调") == 1  # 单数不回退


def test_transformer_trailing_boundary():
    # 旧 trans_en 无尾界："transformer modeling"（非"模型"）误命中
    assert _flag("transformer modeling经验") == 0
    assert _flag("transformer modelling工作") == 0
    assert _flag("transformer models部署") == 1     # 真复数仍命中


def test_dictionary_language_derived_from_rule():
    rows = anchor_dictionary_rows()
    for r in rows:
        if r["keyword"] in ("AI", "ML", "NLP", "LLM"):
            assert r["language"] == "en", r  # 全大写缩写曾误标 zh
        if r["matching_rule"] == "trans_zh":
            assert r["language"] == "zh", r  # Transformer模型 曾误标 en
    assert all(r["language"] in ("zh", "en") for r in rows)


def test_dictionary_rows_shape():
    rows = anchor_dictionary_rows()
    by_ver = {}
    for r in rows:
        by_ver.setdefault(r["anchor_version"], []).append(r)
    assert len(by_ver["main"]) == 21
    assert len(by_ver["cn_paper"]) == 11  # AI/ML/NLP 各3 + CIMAGE 2
    assert len(by_ver["babina"]) == 11    # AI/ML/NLP 各3 + CVISION 2
    # 缩写行必须带 ambiguity_flag
    assert all(r["ambiguity_flag"] == 1 for r in rows
               if r["matching_rule"] in ("abbr", "abbr_llms"))
