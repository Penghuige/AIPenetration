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



def _text(value, default: str = "") -> str:
    """CSV/DataFrame 空值安全转文本，避免 NaN 被写成字面量 'nan'。"""
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


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


def materialize_source_skill_records(
    base_concepts: pd.DataFrame,
    dictionary_version: str,
) -> pd.DataFrame:
    """从冻结 A 级概念中的显式来源字段物化 §7.4.1 来源映射表。

    只使用原文件已经存在的来源标识，不猜测缺失 release/version。
    """
    required = {
        "skill_id", "canonical_zh", "canonical_en", "skill_type",
        "source_primary", "source_version_primary", "source_id_primary",
        "esco_uri", "onet_element_ids",
    }
    missing = required - set(base_concepts.columns)
    if missing:
        raise ValueError("冻结概念表缺来源字段: " + ", ".join(sorted(missing)))

    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add(
        sid: str,
        source_name: str,
        source_version: str,
        source_skill_id: str,
        label: str,
        description: str,
        category: str,
        mapping_type: str,
        mapping_evidence: str,
    ) -> None:
        source_name = _text(source_name)
        source_skill_id = _text(source_skill_id)
        if not source_name or not source_skill_id:
            return
        key = (source_name, source_skill_id, sid)
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "source_name": source_name,
            "source_version": _text(source_version, "UNKNOWN_LOCAL_SNAPSHOT"),
            "source_skill_id": source_skill_id,
            "source_label": label,
            "source_description": description,
            "source_category": category,
            "internal_skill_id": sid,
            "mapping_type": mapping_type,
            "mapping_evidence": mapping_evidence,
            "dictionary_version": dictionary_version,
        })

    for _, row in base_concepts.iterrows():
        sid = str(row.skill_id)
        label = _text(row.get("canonical_en")) or _text(row.get("canonical_zh"))
        desc = _text(row.get("definition_en")) or _text(row.get("description_en"))
        cat = _text(row.get("skill_type"))
        add(
            sid,
            _text(row.get("source_primary")),
            _text(row.get("source_version_primary"), "UNKNOWN_LOCAL_SNAPSHOT"),
            _text(row.get("source_id_primary")),
            label, desc, cat,
            "primary_source_record",
            "frozen_concept.source_primary/source_id_primary",
        )
        primary_name = _text(row.get("source_primary")).upper()
        primary_version = _text(
            row.get("source_version_primary"), "UNKNOWN_LOCAL_SNAPSHOT"
        )
        add(
            sid, "ESCO",
            primary_version if primary_name == "ESCO" else "UNKNOWN_LOCAL_SNAPSHOT",
            _text(row.get("esco_uri")), label, desc, cat,
            "source_crosswalk",
            "frozen_concept.esco_uri",
        )
        add(
            sid, "O*NET",
            primary_version
            if primary_name in {"O*NET", "ONET"} else "UNKNOWN_LOCAL_SNAPSHOT",
            _text(row.get("onet_element_ids")), label, desc, cat,
            "source_crosswalk",
            "frozen_concept.onet_element_ids (preserved verbatim)",
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("无法从冻结概念表物化任何 source_skill_record")
    if out[["source_name", "source_skill_id", "internal_skill_id"]].duplicated().any():
        raise ValueError("source_skill_record 主键重复")
    return out.reset_index(drop=True)


def materialize_dictionary_changelog(
    grades: pd.DataFrame,
    dictionary_version: str,
) -> pd.DataFrame:
    """物化 §10.4 词典变更账本；不修改任何分级结果。"""
    required = {"term", "final_grade", "final_skill_id"}
    missing = required - set(grades.columns)
    if missing:
        raise ValueError("治理表缺列: " + ", ".join(sorted(missing)))
    rows = []
    for row in grades.itertuples(index=False):
        data = row._asdict()
        grade = str(data.get("final_grade", ""))
        rows.append({
            "term": str(data.get("term", "")),
            "final_skill_id": _text(data.get("final_skill_id")),
            "final_grade": grade,
            "dictionary_status": (
                "formal" if grade in {"A", "B", "C"} else "candidate_d"
            ),
            "mapping_action": _text(data.get("mapping_action")),
            "source": _text(data.get("source"), "legacy_governance_v3"),
            "df_unique_description": data.get(
                "df_freq", data.get("df_unique_description", pd.NA)
            ),
            "candidate_anchor_cooc": data.get(
                "cand_cooc", data.get("candidate_anchor_cooc", pd.NA)
            ),
            "first_year": data.get("first_year", pd.NA),
            "demote_reason": _text(data.get("demote_reason")),
            "dictionary_version": dictionary_version,
        })
    out = pd.DataFrame(rows)
    if out.term.astype(str).map(normalize_term).duplicated().any():
        raise ValueError("词典 changelog term 不唯一")
    return out.sort_values(
        ["dictionary_status", "final_grade", "term"], kind="stable"
    ).reset_index(drop=True)


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
    if formal.final_skill_id.isna().any():
        raise ValueError("A/B/C 存在空 final_skill_id")
    grade_conflict = (
        formal.groupby(formal.final_skill_id.astype(str)).final_grade.nunique()
    )
    if (grade_conflict > 1).any():
        bad = grade_conflict[grade_conflict > 1].index.tolist()[:5]
        raise ValueError(f"同一 final_skill_id 出现多个 confidence tier: {bad}")

    controlled_types = set(_T2_TYPE_MAP.values()) | {
        "hardware_equipment", "domain_knowledge", "business_management",
        "general_work_skill", "soft_skill", "other_skill",
    }
    existing_concept_ids = set(concepts.skill_id.astype(str))
    concept_rows: list[dict] = []
    bc = formal[formal.final_grade.isin(["B", "C"])].copy()
    sort_cols = ["final_skill_id", "term"]
    if "df_freq" in bc.columns:
        bc["_df_sort"] = pd.to_numeric(bc["df_freq"], errors="coerce").fillna(-1)
        bc = bc.sort_values(
            ["final_skill_id", "_df_sort", "term"],
            ascending=[True, False, True],
            kind="stable",
        )
    else:
        bc = bc.sort_values(sort_cols, kind="stable")
    for _, row in bc.iterrows():
        sid = str(row.final_skill_id)
        if sid in existing_concept_ids:
            continue
        term = str(row.term)
        rec = {col: pd.NA for col in concepts.columns}
        rec["skill_id"] = sid
        if "canonical_zh" in rec:
            rec["canonical_zh"] = term if _CJK_RE.search(term) else pd.NA
        if "canonical_en" in rec:
            rec["canonical_en"] = term if not _CJK_RE.search(term) else pd.NA
        raw_type = _text(row.get("t2_cat"))
        if "skill_type" in rec:
            rec["skill_type"] = (
                raw_type if raw_type in controlled_types
                else _T2_TYPE_MAP.get(raw_type, "other_skill")
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
        existing_concept_ids.add(sid)

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
            amb = _text(row.get("t2_ambig")).lower() in {"1", "true", "yes"}
            rec["ambiguity_flag"] = "1" if amb else "0"
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
            action = _text(row.get("mapping_action"))
            rec["activation_reason"] = (
                "mapped_existing_by_review"
                if action.startswith("MATCH_EXISTING")
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
