"""原始交接要求的纯函数回归测试。"""
from __future__ import annotations

import pandas as pd

from src.ai_penetration.panel_v2.governance import (
    governed_skill_map,
    materialize_formal_dictionary,
    stable_bc_skill_id,
)


def _base_frames():
    concepts = pd.DataFrame([
        {
            "skill_id": "uuid-a",
            "canonical_zh": "机器学习",
            "canonical_en": "machine learning",
            "skill_type": "method_algorithm",
            "skill_category": "A",
            "confidence_tier": "A",
            "translation_status": "translated",
            "dictionary_version": "base",
        }
    ])
    aliases = pd.DataFrame([
        {
            "alias_id": "alias-a",
            "skill_id": "uuid-a",
            "alias": "机器学习",
            "alias_normalized": "机器学习",
            "language": "zh",
            "is_active": "1",
            "activation_reason": "base",
        }
    ])
    return concepts, aliases


def test_stable_bc_skill_id_is_deterministic_and_year_bound():
    a = stable_bc_skill_id("PyTorch插件", 2021)
    b = stable_bc_skill_id("pytorch插件", 2021)
    c = stable_bc_skill_id("PyTorch插件", 2022)
    assert a == b
    assert a != c
    assert len(a) == 36


def test_governed_skill_map_excludes_d_and_keeps_mapped_a():
    grades = pd.DataFrame([
        {"term": "ML", "final_grade": "A", "final_skill_id": "uuid-a"},
        {"term": "新框架", "final_grade": "B", "final_skill_id": "uuid-b"},
        {"term": "噪声词", "final_grade": "D", "final_skill_id": ""},
    ])
    got = governed_skill_map(grades)
    assert got == {"ml": "uuid-a", "新框架": "uuid-b"}


def test_materialized_release_dictionary_contains_formal_abc_only():
    concepts, aliases = _base_frames()
    sid_b = stable_bc_skill_id("新框架", 2021)
    sid_c = stable_bc_skill_id("新方法", 2022)
    grades = pd.DataFrame([
        {
            "term": "ML",
            "final_grade": "A",
            "final_skill_id": "uuid-a",
            "t2_cat": "method",
        },
        {
            "term": "新框架",
            "final_grade": "B",
            "final_skill_id": sid_b,
            "t2_cat": "framework",
        },
        {
            "term": "新方法",
            "final_grade": "C",
            "final_skill_id": sid_c,
            "t2_cat": "method",
        },
        {
            "term": "噪声词",
            "final_grade": "D",
            "final_skill_id": "",
            "t2_cat": "other",
        },
    ])
    out_c, out_a, out_d = materialize_formal_dictionary(
        concepts, aliases, grades, "dict-v-test"
    )

    assert set(out_c.skill_id.astype(str)) == {"uuid-a", sid_b, sid_c}
    assert set(out_d.term.astype(str)) == {"噪声词"}
    assert set(out_a.skill_id.astype(str)) <= set(out_c.skill_id.astype(str))
    assert "ML" in set(out_a.alias.astype(str))
    assert "新框架" in set(out_a.alias.astype(str))
    assert "新方法" in set(out_a.alias.astype(str))
    assert "噪声词" not in set(out_a.alias.astype(str))
