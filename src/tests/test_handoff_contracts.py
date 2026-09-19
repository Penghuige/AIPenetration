"""原始交接 §10/§18 的正式词典发布契约测试。"""
from __future__ import annotations

import pandas as pd

from src.ai_penetration.panel_v2.export_release import (
    build_final_dictionary_frames,
)
from src.ai_penetration.panel_v2.lexicon import load_formal_legacy_spec


def _a_concepts() -> pd.DataFrame:
    return pd.DataFrame([{
        "skill_id": "uuid-a",
        "canonical_zh": "机器学习",
        "canonical_en": "machine learning",
        "skill_type": "method_algorithm",
        "skill_category": "AI",
        "definition": "",
        "source": "ESCO",
        "source_version": "v1",
        "source_id": "a",
        "valid_from": "",
        "valid_to": "",
        "confidence_tier": "A",
        "dictionary_version": "a-frozen",
    }])


def _a_aliases() -> pd.DataFrame:
    return pd.DataFrame([{
        "alias_id": "alias-a",
        "skill_id": "uuid-a",
        "alias": "机器学习",
        "alias_normalized": "机器学习",
        "language": "zh",
        "source": "ESCO",
        "matching_rule": "substring",
        "ambiguity_flag": 0,
        "confidence_tier": "A",
        "dictionary_version": "a-frozen",
        "is_active": "1",
        "primary_skill_id": "",
        "boundary_rule": "",
        "case_sensitive": "0",
        "activation_reason": "source",
    }])


def _grade() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "term": "机器学习方法", "skill_id": "legacy:mapped",
            "formal_skill_id": "uuid-a", "final_grade": "A",
            "t2_type": "method", "t2_ambig": False, "first_year": 2016,
            "t2_relation": 0,
        },
        {
            "term": "新框架B", "skill_id": "legacy:新框架b",
            "formal_skill_id": "legacy:新框架b", "final_grade": "B",
            "t2_type": "framework", "t2_ambig": False, "first_year": 2020,
            "t2_relation": -1,
        },
        {
            "term": "新模型C", "skill_id": "legacy:新模型c",
            "formal_skill_id": "legacy:新模型c", "final_grade": "C",
            "t2_type": "model", "t2_ambig": False, "first_year": 2023,
            "t2_relation": -1,
        },
        {
            "term": "非技能D", "skill_id": "legacy:非技能d",
            "formal_skill_id": "legacy:非技能d", "final_grade": "D",
            "t2_type": "other", "t2_ambig": True, "first_year": 2020,
            "t2_relation": -3,
        },
    ])


def test_final_dictionary_separates_formal_and_candidate():
    concepts, aliases, d = build_final_dictionary_frames(
        _a_concepts(), _a_aliases(), _grade(),
        dictionary_version="combined-test",
    )
    ids = set(concepts.skill_id.astype(str))
    assert ids == {"uuid-a", "legacy:新框架b", "legacy:新模型c"}
    assert "legacy:mapped" not in ids

    alias_map = dict(zip(aliases.alias, aliases.skill_id))
    assert alias_map["机器学习方法"] == "uuid-a"
    assert alias_map["新框架B"] == "legacy:新框架b"
    assert alias_map["新模型C"] == "legacy:新模型c"
    assert "非技能D" not in alias_map

    assert list(d.term) == ["非技能D"]
    assert set(d.final_grade) == {"D"}


def test_formal_legacy_spec_excludes_d_and_preserves_a_mapping(tmp_path):
    path = tmp_path / "grade.csv"
    _grade().to_csv(path, index=False, encoding="utf-8-sig")

    terms, sid_by_key, tier_by_sid = load_formal_legacy_spec(path)

    assert "非技能D" not in terms
    assert sid_by_key["机器学习方法"] == "uuid-a"
    assert sid_by_key["新框架b"] == "legacy:新框架b"
    assert sid_by_key["新模型c"] == "legacy:新模型c"
    assert tier_by_sid == {
        "legacy:新框架b": "B",
        "legacy:新模型c": "C",
    }
