"""原始交接 §7/§10 的正式词典治理辅助。

本模块把 legacy 候选的临时字符串空间转换成可发布的概念空间：
- MATCH_EXISTING → 复用已有 A 级 skill_id；
- B/C 新概念 → 稳定 UUIDv5；
- D → 仅保留候选，不进入正式匹配。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid

import pandas as pd

_PROJECT_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/Penghuige/AIPenetration/skill-concepts",
)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")

_T2_TYPE_MAP = {
    "method": "method_algorithm",
    "tool": "software_tool",
    "software": "software_tool",
    "model": "method_algorithm",
    "framework": "platform_framework_library",
    "data": "data_database",
    "language": "programming_language",
    "other": "other_skill",
}


def normalize_term(term: str) -> str:
    """与正式 matcher 一致的概念名称规范化键。"""
    return unicodedata.normalize("NFKC", str(term)).lower().strip()


def stable_bc_skill_id(term: str, first_year: int) -> str:
    """按规范名 + 首次发现年份生成稳定 UUIDv5（指南 §7.4.2）。"""
    key = normalize_term(term)
    year = int(first_year)
    if not key or year < 1900 or year > 2100:
        raise ValueError(
            f"B/C 新概念缺少有效规范名/首次发现年份: term={term!r}, year={first_year!r}"
        )
    return str(uuid.uuid5(_PROJECT_NAMESPACE, f"{key}|{year}"))


def governed_skill_map(grades: pd.DataFrame) -> dict[str, str]:
    """返回正式可匹配 term → final_skill_id；D 级明确排除。"""
    required = {"term", "final_grade", "final_skill_id"}
    missing = required - set(grades.columns)
    if missing:
        raise ValueError("治理表缺列: " + ", ".join(sorted(missing)))
    formal = grades[grades.final_grade.isin(["A", "B", "C"])].copy()
    empty_id = formal.final_skill_id.isna() | (
        formal.final_skill_id.astype(str).str.len() == 0
    )
    if empty_id.any():
        raise ValueError("A/B/C 治理行存在空 final_skill_id")
    out: dict[str, str] = {}
    for term, sid in zip(formal.term.astype(str), formal.final_skill_id.astype(str)):
        key = normalize_term(term)
        if key in out and out[key] != sid:
            raise ValueError(f"正式治理词同形多概念: {key!r} -> {out[key]!r}/{sid!r}")
        out[key] = sid
    return out


def _alias_id(skill_id: str, term: str) -> str:
    digest = hashlib.sha256(
        (str(skill_id) + "\\0" + normalize_term(term)).encode("utf-8")
    ).hexdigest()[:24]
    return "gov-" + digest


def materialize_formal_dictionary(
    base_concepts: pd.DataFrame,
    base_aliases: pd.DataFrame,
    grades: pd.DataFrame,
    dictionary_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成 §18 的 A/B/C 正式概念、正式别名与 D 级候选三件。"""
    required = {"term", "final_grade", "final_skill_id"}
    missing = required - set(grades.columns)
    if missing:
        raise ValueError("治理表缺列: " + ", ".join(sorted(missing)))

    concepts = base_concepts.copy()
    aliases = base_aliases.copy()
    if concepts.skill_id.astype(str).duplicated().any():
        raise ValueError("A级概念 skill_id 不唯一")

    formal = grades[grades.final_grade.isin(["A", "B", "C"])].copy()
    d_candidates = grades[grades.final_grade == "D"].copy()

    concept_rows: list[dict] = []
    for _, row in formal[formal.final_grade.isin(["B", "C"])].iterrows():
        sid = str(row.final_skill_id)
        term = str(row.term)
        rec = {col: pd.NA for col in concepts.columns}
        rec["skill_id"] = sid
        if "canonical_zh" in rec:
            rec["canonical_zh"] = term if _CJK_RE.search(term) else pd.NA
        if "canonical_en" in rec:
            rec["canonical_en"] = term if not _CJK_RE.search(term) else pd.NA
        if "skill_type" in rec:
            rec["skill_type"] = _T2_TYPE_MAP.get(
                str(row.get("t2_cat", "")), "other_skill"
            )
        if "skill_category" in rec:
            rec["skill_category"] = "china_recruitment_governed"
        if "confidence_tier" in rec:
            rec["confidence_tier"] = str(row.final_grade)
        if "translation_status" in rec:
            rec["translation_status"] = "not_applicable_governed"
        if "dictionary_version" in rec:
            rec["dictionary_version"] = dictionary_version
        concept_rows.append(rec)

    if concept_rows:
        concepts = pd.concat([concepts, pd.DataFrame(concept_rows)], ignore_index=True)
    if concepts.skill_id.astype(str).duplicated().any():
        dup = concepts.loc[
            concepts.skill_id.astype(str).duplicated(), "skill_id"
        ].head().tolist()
        raise ValueError(f"物化后 skill_id 重复: {dup}")

    existing_alias: dict[str, str] = {}
    for alias, sid in zip(aliases.alias.astype(str), aliases.skill_id.astype(str)):
        key = normalize_term(alias)
        if key in existing_alias and existing_alias[key] != sid:
            raise ValueError(
                f"正式基础别名同形多概念未消歧: {key!r} -> "
                f"{existing_alias[key]!r}/{sid!r}"
            )
        existing_alias[key] = sid

    alias_rows: list[dict] = []
    for _, row in formal.iterrows():
        term = str(row.term)
        sid = str(row.final_skill_id)
        key = normalize_term(term)
        if key in existing_alias:
            if existing_alias[key] != sid:
                raise ValueError(
                    f"治理别名与正式 A 级概念冲突: {key!r} -> "
                    f"{existing_alias[key]!r}/{sid!r}"
                )
            continue
        rec = {col: pd.NA for col in aliases.columns}
        rec["alias_id"] = _alias_id(sid, term)
        rec["skill_id"] = sid
        rec["alias"] = term
        if "alias_normalized" in rec:
            rec["alias_normalized"] = key
        if "language" in rec:
            rec["language"] = "zh" if _CJK_RE.search(term) else "en"
        if "alias_type" in rec:
            rec["alias_type"] = "governed_surface"
        if "source" in rec:
            rec["source"] = "china_recruitment_governance"
        is_ascii = all(ord(ch) < 128 for ch in key)
        if "matching_rule" in rec:
            rec["matching_rule"] = (
                "ascii_alnum_boundary" if is_ascii else "substring_longest"
            )
        if "ambiguity_flag" in rec:
            rec["ambiguity_flag"] = "1" if bool(row.get("t2_ambig", False)) else "0"
        if "confidence_tier" in rec:
            rec["confidence_tier"] = str(row.final_grade)
        if "dictionary_version" in rec:
            rec["dictionary_version"] = dictionary_version
        if "is_active" in rec:
            rec["is_active"] = "1"
        if "primary_skill_id" in rec:
            rec["primary_skill_id"] = sid
        if "boundary_rule" in rec:
            rec["boundary_rule"] = (
                "ascii_alnum" if is_ascii else "none"
            )
        if "case_sensitive" in rec:
            rec["case_sensitive"] = "0"
        if "activation_reason" in rec:
            rec["activation_reason"] = (
                "mapped_existing_by_t2"
                if str(row.final_grade) == "A"
                else f"governed_grade_{row.final_grade}"
            )
        if "translation_status" in rec:
            rec["translation_status"] = "not_applicable_governed"
        alias_rows.append(rec)
        existing_alias[key] = sid

    if alias_rows:
        aliases = pd.concat([aliases, pd.DataFrame(alias_rows)], ignore_index=True)
    if aliases.alias_id.astype(str).duplicated().any():
        raise ValueError("物化后 alias_id 重复")

    if "dictionary_version" in concepts.columns:
        concepts["dictionary_version"] = dictionary_version
    if "dictionary_version" in aliases.columns:
        aliases["dictionary_version"] = dictionary_version

    concept_ids = set(concepts.skill_id.astype(str))
    dangling = set(aliases.skill_id.astype(str)) - concept_ids
    if dangling:
        raise ValueError(f"正式别名存在无概念引用: {sorted(dangling)[:5]}")

    return concepts, aliases, d_candidates.reset_index(drop=True)
