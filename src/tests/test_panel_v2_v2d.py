"""v2d 治理纯函数离线测试（注入词表，不连库）。"""
from __future__ import annotations

import pandas as pd

from src.ai_penetration.panel_v2.lexicon import LEGACY_PREFIX, build_union_lexicon
from src.ai_penetration.panel_v2.lexicon_v2d import governance_frame, tautological_keys

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
