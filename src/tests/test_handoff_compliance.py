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



def test_saturation_uses_two_consecutive_incremental_rounds_only():
    from src.ai_penetration.panel_v2.discovery_formal import saturation_pass

    metrics = pd.DataFrame([
        {"round": 0, "new_standard_concepts": 1, "coverage_gain_pp": 99.0},
        {"round": 1, "new_standard_concepts": 4, "coverage_gain_pp": 99.0},
    ])
    assert not saturation_pass(metrics)

    metrics = pd.concat([
        metrics,
        pd.DataFrame([{
            "round": 2, "new_standard_concepts": 3,
            "coverage_gain_pp": 99.0,
        }]),
    ], ignore_index=True)
    assert saturation_pass(metrics)


def test_saturation_rejects_nonconsecutive_incremental_rounds():
    from src.ai_penetration.panel_v2.discovery_formal import saturation_pass

    metrics = pd.DataFrame([
        {"round": 1, "new_standard_concepts": 2, "coverage_gain_pp": 0.0},
        {"round": 3, "new_standard_concepts": 1, "coverage_gain_pp": 0.0},
    ])
    assert not saturation_pass(metrics)


def test_discovery_text_group_id_is_platform_and_text_bound():
    from src.ai_penetration.panel_v2.discovery_frame import _discovery_id

    a = _discovery_id("platform-a", 123)
    assert a == _discovery_id("platform-a", 123)
    assert a != _discovery_id("platform-b", 123)
    assert a != _discovery_id("platform-a", 124)
    assert len(a) == 64


def test_translation_missing_ids_can_only_backfill_with_order_evidence():
    from src.ai_penetration.translation_completion import _resolve_batch_ids

    source = [
        {"source_skill_id": "s1", "canonical_en": "Python"},
        {"source_skill_id": "s2", "canonical_en": "PyTorch"},
    ]
    result = [
        {"canonical_en": "Python", "canonical_zh": "Python"},
        {"canonical_en": "PyTorch", "canonical_zh": "PyTorch"},
    ]
    ids, backfilled, ratio = _resolve_batch_ids(result, source, 1)
    assert ids == ["s1", "s2"]
    assert backfilled is True
    assert ratio == 1.0


def test_translation_backfill_rejects_wrong_order():
    import pytest
    from src.ai_penetration.translation_completion import _resolve_batch_ids

    source = [
        {"source_skill_id": "s1", "canonical_en": "Python"},
        {"source_skill_id": "s2", "canonical_en": "PyTorch"},
    ]
    result = [
        {"canonical_en": "PyTorch"},
        {"canonical_en": "Python"},
    ]
    with pytest.raises(ValueError, match="98%"):
        _resolve_batch_ids(result, source, 1)


def test_benchmark_effective_span_accepts_deterministic_fallback():
    from src.ai_penetration.panel_v2.model_benchmark import _validate

    text = "熟悉 Python 数据分析"
    parsed = {
        "job_id": "j1",
        "skills": [{
            "surface": "Python",
            "canonical_suggestion": "Python",
            "skill_type": "programming_language",
            "evidence": "熟悉 Python 数据分析",
            "start": 0,
            "end": 1,
            "existing_skill_id": None,
        }],
    }
    got = _validate(
        parsed, "j1", text, {"programming_language"}
    )
    valid, direct, effective, total, hallucinated, evidence_ok, *_ = got
    assert valid is True
    assert direct == 0
    assert effective == 1
    assert total == 1
    assert hallucinated == 0
    assert evidence_ok == 1
