"""v2d/v2e 治理纯函数离线测试（注入词表，不连库）。"""
from __future__ import annotations

import pandas as pd

from src.ai_penetration.panel_v2.lexicon import LEGACY_PREFIX, build_union_lexicon
from src.ai_penetration.panel_v2.lexicon_llm import apply_semantic_review_to_grade
from src.ai_penetration.panel_v2.lexicon_v2d import (
    governance_frame,
    grade_legacy_frame,
    tautological_keys,
)

_ALIASES = [
    ("机器学习", "uuid-ml"),        # 与 ML 锚点同形 → A级 taut
    ("数据分析", "uuid-da"),
    ("PyTorch", "uuid-torch"),      # 非锚点
]
_LEGACY = ["ai技术", "pytorch", "小词", "普通技能", "视觉算法"]


def _lex():
    return build_union_lexicon(legacy_terms=_LEGACY, aliases=_ALIASES)


def test_tautological_keys_enumeration():
    taut = tautological_keys(_lex())
    assert "机器学习" in taut and "ML" in taut["机器学习"]
    assert "ai技术" in taut and "AI" in taut["ai技术"]   # 键内含锚点同形串
    assert "pytorch" not in taut
    assert "视觉算法" not in taut                        # 非锚点词不误伤


def test_governance_frame_dispositions():
    lex = _lex()
    taut = tautological_keys(lex)
    pooled = {LEGACY_PREFIX + "小词": 5, LEGACY_PREFIX + "普通技能": 999,
              LEGACY_PREFIX + "视觉算法": 250, "uuid-da": 100}
    frame = governance_frame(lex, taut, pooled)
    d = dict(zip(frame.term, frame.disposition))
    assert d["机器学习"] == "score_excluded_taut_atier"   # 词典不动，得分排除
    assert d["ai技术"] == "removed_taut"                  # taut 先于频数
    assert d["小词"] == "removed_lowfreq"
    assert d["普通技能"] == "activated_keep"
    assert d["视觉算法"] == "activated_keep"
    # 全键覆盖 + 确定性
    assert len(frame) == lex.n_concepts
    pd.testing.assert_frame_equal(frame, governance_frame(lex, taut, pooled))
    assert (frame.loc[frame.disposition == "removed_lowfreq",
                      "pooled_freq"] < 100).all()


def test_grade_legacy_frame_guide_rules():
    """§10.3.1 A/B/C/D 自动分级三分支（v2e 复原通道）。"""
    keys = {"机器学习基础": LEGACY_PREFIX + "a", "pytorch": LEGACY_PREFIX + "b",
            "midjourney": LEGACY_PREFIX + "c", "小工具x": LEGACY_PREFIX + "d",
            "ai": LEGACY_PREFIX + "e"}
    dfreq = {"机器学习基础": 500, "pytorch": 50, "midjourney": 7,
             "小工具x": 50, "ai": 30000}
    cooc = {keys[k]: v for k, v in
            zip(keys, (0.9, 0.8, 0.55, 0.05, 1.0))}
    fy = {keys[k]: v for k, v in zip(keys, (2016, 2016, 2023, 2015, 2014))}
    g = grade_legacy_frame(keys, dfreq, cooc, fy)
    d = dict(zip(g.term, g.grade))
    assert d["机器学习基础"] == "B"          # df≥100
    assert d["pytorch"] == "C"              # 10≤df<100
    assert d["midjourney"] == "C"           # df≥5 且 cooc≥0.5 且新出现
    assert d["小工具x"] == "C"               # 10≤df<100
    assert d["ai"] == "B"                   # 原预案：频数即 B，taut 仅披露不行动
    assert int(g.loc[g.term == "ai", "tautological"].iloc[0]) == 1
    assert int(g.loc[g.term == "机器学习基础", "tautological"].iloc[0]) == 1
    # 确定性
    pd.testing.assert_frame_equal(g, grade_legacy_frame(keys, dfreq, cooc, fy))



def test_semantic_review_closes_grade_and_mapping_contract():
    grade = pd.DataFrame([
        {
            "term": "oldalias", "skill_id": "legacy:oldalias",
            "df_freq": 120, "cand_cooc": 0.1, "first_year": 2015,
            "tautological": 0, "grade": "B",
        },
        {
            "term": "newmodel", "skill_id": "legacy:newmodel",
            "df_freq": 7, "cand_cooc": 0.1, "first_year": 2023,
            "tautological": 0, "grade": "C",
        },
        {
            "term": "proxyonly", "skill_id": "legacy:proxyonly",
            "df_freq": 7, "cand_cooc": 0.1, "first_year": 2023,
            "tautological": 0, "grade": "C",
        },
        {
            "term": "ambig", "skill_id": "legacy:ambig",
            "df_freq": 100, "cand_cooc": 0.6, "first_year": 2020,
            "tautological": 0, "grade": "B",
        },
    ])
    t1 = {term: "skill" for term in grade.term}
    t2 = {
        "oldalias": {"s": True, "c": "tool", "n": False, "r": 0, "a": False},
        "newmodel": {"s": True, "c": "model", "n": True, "r": -1, "a": False},
        "proxyonly": {"s": True, "c": "model", "n": False, "r": -1, "a": False},
        "ambig": {"s": True, "c": "other", "n": False, "r": -2, "a": True},
    }
    items = {
        "oldalias": {"cand_skill_ids": ["uuid-existing"]},
        "newmodel": {"cand_skill_ids": []},
        "proxyonly": {"cand_skill_ids": []},
        "ambig": {"cand_skill_ids": ["uuid-a", "uuid-b"]},
    }
    out = apply_semantic_review_to_grade(
        grade, t1_cat=t1, t2_by_term=t2, t2_items_by_term=items
    ).set_index("term")

    assert out.loc["oldalias", "final_grade"] == "A"
    assert out.loc["oldalias", "formal_skill_id"] == "uuid-existing"
    assert out.loc["newmodel", "final_grade"] == "C"
    assert out.loc["proxyonly", "final_grade"] == "D"
    assert "c_new_tech_not_confirmed" in out.loc["proxyonly", "demote_reason"]
    assert out.loc["ambig", "final_grade"] == "D"
    assert "t2_ambiguous_mapping" in out.loc["ambig", "demote_reason"]
