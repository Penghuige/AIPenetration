"""panel_v2 M4-b：§18 发布物组装（13 件 + legacy 处置表 + 同名 metadata.json）。

把 counts/relevance/scoring 产物与词典、锚点表汇总到 release/panel_v2/，
逐文件生成 §18.1 metadata（指南清单 11 字段原文照单，"至少包含"语义；
2026-09-09 审计修正：本仓旧述"12 字段"系笔误，config/docs 已同步 11）。
--force-release 哨兵检查覆盖四件核心数据件（classification/score/flag/QC；
审计 D10：旧版仅查 QC 一件，哨兵缺失时词典件会被静默替换）。

legacy 处置表（skill_legacy_v1，2026-09-09 增补件）：§18 原十三件只含 A 级
概念/别名，不含自建 6,872 词在 union 构建中的完整去向——合成独立概念
6,328、同名并入 A 级 367、自建表内部归一重复 172、短词丢弃 5（"539 重叠"
=367+172 两 disposition 之和，性质不同不可混称）。本件逐词记录 disposition，
与 build_union_lexicon 计数强制对账，使发布包对 legacy 空间自含可审。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.export_release --run-id 20260907_v2a
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

from config.paths import get_project_paths

from ..common import eps_conn_params, setup_logging
from ..skill_ai_anchor import load_merged_skills
from .anchors import anchor_dictionary_rows
from .lexicon import LEGACY_PREFIX, _resolve_active_alias_rows, build_union_lexicon

logger = logging.getLogger("ai_penetration.panel_v2.export")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash() -> str:
    root = get_project_paths().config_dir
    h = hashlib.sha256()
    for f in sorted(root.glob("*.yaml")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _meta(p: Path, run_id: str, primary_key: str, source_files: str,
          anchor_version: str = "-",
          dictionary_version: str = "bilingual_a_frozen_v1.1+legacy_union_v1") -> None:
    import pyarrow.parquet as _pq
    if p.suffix == ".parquet":
        schema = str(_pq.read_schema(p))
        n = _pq.ParquetFile(p).metadata.num_rows
    else:
        text = p.read_text(encoding="utf-8-sig")
        schema = "csv"
        n = max(text.count("\n") - 1, 0)
    meta = {
        "file_name": p.name, "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": n, "column_schema": schema, "primary_key": primary_key,
        "source_files": source_files,
        "dictionary_version": dictionary_version,
        "anchor_version": anchor_version,
        "config_hash": _config_hash(), "sha256": _sha(p),
    }
    p.with_suffix(p.suffix + ".metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")



_T2_TYPE_TO_GUIDE = {
    "method": "method_algorithm",
    "tool": "software_tool",
    "software": "software_tool",
    "model": "method_algorithm",
    "framework": "platform_framework_library",
    "data": "data_database",
    "language": "programming_language",
    "other": "other_skill",
}


def _legacy_alias_id(skill_id: str, term: str) -> str:
    """为 legacy/映射别名生成稳定 alias_id。"""
    raw = f"{skill_id}\0{term}".encode("utf-8")
    return "legacy_alias:" + hashlib.sha256(raw).hexdigest()[:24]


def _alias_language(term: str) -> str:
    has_ascii = any("a" <= ch.lower() <= "z" for ch in term)
    has_non_ascii = any(ord(ch) > 127 for ch in term)
    if has_ascii and has_non_ascii:
        return "mixed"
    return "en" if has_ascii else "zh"


def build_final_dictionary_frames(
    a_concepts: pd.DataFrame,
    a_aliases: pd.DataFrame,
    grade: pd.DataFrame,
    *,
    dictionary_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按指南 §10/§18 组装 A+B+C 正式词典与 D 候选。

    A 级来自冻结外部词典；最终 grade=A 的 legacy 表面只作为既有 A 概念别名；
    grade=B/C 建立正式概念；grade=D 仅进入候选表。
    """
    required_grade = {
        "term", "skill_id", "final_grade", "formal_skill_id",
        "t2_type", "t2_ambig",
    }
    missing = required_grade - set(grade.columns)
    if missing:
        raise RuntimeError(
            "最终分级缺少 §10 发布字段: " + ", ".join(sorted(missing))
        )
    if not set(grade.final_grade.dropna().astype(str)).issubset({"A", "B", "C", "D"}):
        raise RuntimeError("final_grade 存在 A/B/C/D 之外取值")

    # A aliases 必须与正式 matcher 使用同一 primary_skill_id 解析规则。
    alias_rows = [
        (str(r.alias), str(r.skill_id),
         None if pd.isna(r.primary_skill_id) else str(r.primary_skill_id))
        for _, r in a_aliases.iterrows()
    ]
    resolved = dict(_resolve_active_alias_rows(alias_rows))
    kept_a = a_aliases[
        a_aliases.apply(
            lambda r: resolved.get(str(r.alias)) == str(r.skill_id), axis=1
        )
    ].copy()
    kept_a = kept_a.sort_values("alias_id", kind="stable").drop_duplicates(
        ["alias", "skill_id"], keep="first"
    )
    # 与 matcher 一样，NFKC/lower 后不能指向多个概念。
    import unicodedata
    norm_sid: dict[str, str] = {}
    for _, row in kept_a.iterrows():
        key = unicodedata.normalize("NFKC", str(row.alias)).lower()
        sid = str(row.skill_id)
        if key in norm_sid and norm_sid[key] != sid:
            raise RuntimeError(
                f"A 级规范化别名键 {key!r} 仍映射多个概念"
            )
        norm_sid[key] = sid

    formal = grade[grade.final_grade.isin(["A", "B", "C"])].copy()
    d_frame = grade[grade.final_grade == "D"].copy().reset_index(drop=True)

    concept_cols = [
        "skill_id", "canonical_zh", "canonical_en", "skill_type",
        "skill_category", "definition", "source", "source_version",
        "source_id", "valid_from", "valid_to", "confidence_tier",
        "dictionary_version",
    ]
    concepts = a_concepts.copy()
    for col in concept_cols:
        if col not in concepts.columns:
            concepts[col] = pd.NA
    concepts = concepts[concept_cols]

    extra_concepts: list[dict] = []
    extra_aliases: list[dict] = []
    a_ids = set(concepts.skill_id.astype(str))
    for _, row in formal.iterrows():
        term = str(row.term)
        sid = str(row.formal_skill_id)
        tier = str(row.final_grade)
        if not sid or sid == "nan":
            raise RuntimeError(f"正式词条 {term!r} formal_skill_id 为空")

        if tier in {"B", "C"}:
            if sid in a_ids:
                raise RuntimeError(
                    f"{tier} 级词条 {term!r} 错误复用 A 级 skill_id {sid}"
                )
            extra_concepts.append({
                "skill_id": sid,
                "canonical_zh": term if _alias_language(term) != "en" else "",
                "canonical_en": term if _alias_language(term) == "en" else "",
                "skill_type": _T2_TYPE_TO_GUIDE.get(
                    str(row.t2_type), "other_skill"
                ),
                "skill_category": "china_job_legacy",
                "definition": "",
                "source": "china_job_legacy",
                "source_version": "legacy_grade_v2",
                "source_id": term,
                "valid_from": (
                    int(row.first_year)
                    if "first_year" in row and not pd.isna(row.first_year)
                    else pd.NA
                ),
                "valid_to": pd.NA,
                "confidence_tier": tier,
                "dictionary_version": dictionary_version,
            })
        elif sid not in a_ids:
            raise RuntimeError(
                f"grade=A 的词条 {term!r} 未映射到现有 A 概念: {sid}"
            )

        key = unicodedata.normalize("NFKC", term).lower()
        if key in norm_sid and norm_sid[key] != sid:
            raise RuntimeError(
                f"正式别名 {term!r} 与已有 A 别名概念冲突"
            )
        norm_sid[key] = sid
        extra_aliases.append({
            "alias_id": _legacy_alias_id(sid, term),
            "skill_id": sid,
            "alias": term,
            "alias_normalized": key,
            "language": _alias_language(term),
            "source": "china_job_legacy",
            "matching_rule": (
                "ascii_alnum_boundary"
                if all(ord(ch) < 128 for ch in term)
                else "substring"
            ),
            "ambiguity_flag": int(bool(row.t2_ambig)),
            "confidence_tier": (
                tier if tier in {"B", "C"}
                else concepts.set_index("skill_id")
                    .get("confidence_tier", pd.Series(dtype=object))
                    .get(sid, "A")
            ),
            "dictionary_version": dictionary_version,
            "is_active": "1",
            "primary_skill_id": sid,
            "boundary_rule": (
                "ascii_alnum" if all(ord(ch) < 128 for ch in term) else ""
            ),
            "case_sensitive": "0",
            "activation_reason": f"final_grade_{tier}",
        })

    if extra_concepts:
        concepts = pd.concat(
            [concepts, pd.DataFrame(extra_concepts)[concept_cols]],
            ignore_index=True,
        )
    if concepts.skill_id.astype(str).duplicated().any():
        raise RuntimeError("skill_concept_v1 出现重复 skill_id")

    alias_cols = [
        "alias_id", "skill_id", "alias", "alias_normalized", "language",
        "source", "matching_rule", "ambiguity_flag", "confidence_tier",
        "dictionary_version", "is_active", "primary_skill_id",
        "boundary_rule", "case_sensitive", "activation_reason",
    ]
    for col in alias_cols:
        if col not in kept_a.columns:
            kept_a[col] = pd.NA
    aliases = kept_a[alias_cols]
    if extra_aliases:
        aliases = pd.concat(
            [aliases, pd.DataFrame(extra_aliases)[alias_cols]],
            ignore_index=True,
        )
    if aliases.alias_id.astype(str).duplicated().any():
        raise RuntimeError("skill_alias_v1 出现重复 alias_id")

    return (
        concepts.sort_values("skill_id", kind="stable").reset_index(drop=True),
        aliases.sort_values(["alias", "skill_id"], kind="stable").reset_index(drop=True),
        d_frame.sort_values("term", kind="stable").reset_index(drop=True),
    )


def export_final_dictionaries(
    rel: Path,
    grade_path: Path,
    *,
    dictionary_version: str,
    a_concept_path: Path | None = None,
    a_alias_path: Path | None = None,
) -> None:
    """从冻结 A 级文件 + 最终语义分级组装指南 §18 的正式三件。

    正式发布优先使用冻结 CSV，使 A 级输入也能被 run manifest 做文件哈希绑定；
    DB 查询仅保留给旧入口兼容。
    """
    if a_concept_path is not None and a_alias_path is not None:
        raw_concepts = pd.read_csv(a_concept_path, encoding="utf-8-sig")
        concepts = pd.DataFrame({
            "skill_id": raw_concepts["skill_id"],
            "canonical_zh": raw_concepts.get("canonical_zh", ""),
            "canonical_en": raw_concepts.get("canonical_en", ""),
            "skill_type": raw_concepts.get("skill_type", ""),
            "skill_category": raw_concepts.get("skill_category", ""),
            "definition": raw_concepts.get(
                "definition_en",
                raw_concepts.get("description_en", ""),
            ),
            "source": raw_concepts.get("source_primary", ""),
            "source_version": raw_concepts.get("source_version_primary", ""),
            "source_id": raw_concepts.get("source_id_primary", ""),
            "valid_from": raw_concepts.get("valid_from", pd.NA),
            "valid_to": raw_concepts.get("valid_to", pd.NA),
            "confidence_tier": raw_concepts.get("confidence_tier", "A"),
            "dictionary_version": raw_concepts.get(
                "dictionary_version", "bilingual_a_frozen_v1.1"
            ),
        })
        aliases = pd.read_csv(a_alias_path, encoding="utf-8-sig")
        aliases = aliases[aliases["is_active"].astype(str) == "1"].copy()
    else:
        conn = psycopg2.connect(**eps_conn_params())
        try:
            concepts = pd.read_sql(
                """SELECT skill_id, canonical_zh, canonical_en, skill_type,
                          skill_category,
                          coalesce(definition_en, description_en, '') AS definition,
                          source_primary AS source,
                          source_version_primary AS source_version,
                          source_id_primary AS source_id,
                          valid_from, valid_to, confidence_tier, dictionary_version
                   FROM ai_dict.skill_concepts""",
                conn,
            )
            aliases = pd.read_sql(
                """SELECT alias_id, skill_id, alias, alias_normalized, language,
                          source, matching_rule, ambiguity_flag, confidence_tier,
                          dictionary_version, is_active, primary_skill_id,
                          boundary_rule, case_sensitive, activation_reason
                   FROM ai_dict.skill_aliases
                   WHERE is_active='1'
                   ORDER BY alias, skill_id""",
                conn,
            )
        finally:
            conn.close()

    grade = pd.read_csv(grade_path, encoding="utf-8-sig")
    concepts, aliases, d_frame = build_final_dictionary_frames(
        concepts, aliases, grade, dictionary_version=dictionary_version
    )
    concepts.to_parquet(rel / "skill_concept_v1.parquet", index=False)
    aliases.to_parquet(rel / "skill_alias_v1.parquet", index=False)
    d_frame.to_parquet(rel / "skill_candidate_d_v1.parquet", index=False)
    logger.info(
        "§18 正式词典: concepts=%d aliases=%d D=%d",
        len(concepts), len(aliases), len(d_frame),
    )


def export_dictionaries(rel: Path) -> None:
    """词典三件 + 锚点表（§18）。"""
    conn = psycopg2.connect(**eps_conn_params())
    try:
        concepts = pd.read_sql(
            "SELECT skill_id, canonical_zh, canonical_en, skill_type, "
            "skill_category, confidence_tier, translation_status, dictionary_version "
            "FROM ai_dict.skill_concepts", conn)
        aliases = pd.read_sql(
            "SELECT alias_id, skill_id, alias, alias_normalized, language, "
            "is_active, activation_reason FROM ai_dict.skill_aliases "
            "WHERE is_active='1'", conn)
    finally:
        conn.close()
    concepts.to_parquet(rel / "skill_concept_v1.parquet", index=False)
    aliases.to_parquet(rel / "skill_alias_v1.parquet", index=False)
    # D 级候选：union 词表 legacy 空间（A 级未收录、由自建表贡献的词）
    vocab_path = get_project_paths().output_dir / "panel_v2" / "pass2" / "skill_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    legacy = sorted(s[len("legacy:"):] for s in vocab
                    if s.startswith("legacy:"))
    pd.DataFrame({"term": legacy}).assign(
        grade_candidate="D_legacy_union_contrib"
    ).to_parquet(rel / "skill_candidate_d_v1.parquet", index=False)
    rows = anchor_dictionary_rows()
    pd.DataFrame(rows).to_csv(rel / "ai_anchor_dictionary_v1.csv",
                              index=False, encoding="utf-8-sig")


def legacy_disposition_frame(legacy_terms: list[str], lex) -> pd.DataFrame:
    """自建词表逐词处置表（纯函数，与 build_union_lexicon 构建序一致）。

    Args:
        legacy_terms: 自建合并技能词典词表（与 union 构建同序输入）。
        lex: 已构建的 UnionLexicon（用其 keys_map/homograph 判归属）。

    Returns:
        DataFrame[term, match_key, skill_id, disposition, homograph_guard]，
        disposition ∈ {legacy_concept（合成独立概念）, covered_by_atier
        （A 级已有同形键，并入该概念）, duplicate_term（自建表内部归一重复）,
        skipped_short（归一键 <2 字符）}；按 (disposition, match_key, term)
        排序保证跨运行确定性。
    """
    import unicodedata

    rows = []
    created: set[str] = set()
    for term in legacy_terms:
        key = unicodedata.normalize("NFKC", str(term)).lower()
        if len(key) < 2:
            disp, sid = "skipped_short", ""
        else:
            sid = lex.keys_map.get(key, "")
            if not sid.startswith(LEGACY_PREFIX):
                disp = "covered_by_atier"
            elif key in created:
                disp = "duplicate_term"
            else:
                created.add(key)
                disp = "legacy_concept"
        rows.append({
            "term": str(term), "match_key": key, "skill_id": sid,
            "disposition": disp,
            "homograph_guard": int(bool(sid) and sid in lex.homograph),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["disposition", "match_key", "term"],
                           kind="stable").reset_index(drop=True)


def export_legacy_disposition(rel: Path) -> pd.DataFrame:
    """用生产同源路径重建 union 词表并落 skill_legacy_v1.parquet。

    断言强制对账：legacy_concept 计数 == lex.n_legacy，
    covered+duplicate == len(lex.overlap_terms)，全量 == len(terms)。
    """
    terms = load_merged_skills(include_llm=True)
    lex = build_union_lexicon(include_legacy=True, legacy_terms=terms)
    frame = legacy_disposition_frame(terms, lex)
    n = frame.disposition.value_counts()
    assert int(n.get("legacy_concept", 0)) == lex.n_legacy, "legacy 计数失配"
    assert int(n.get("covered_by_atier", 0)) + int(n.get("duplicate_term", 0)) \
        == len(lex.overlap_terms), "重叠计数失配"
    assert len(frame) == len(terms), "词数失配"
    frame.to_parquet(rel / "skill_legacy_v1.parquet", index=False)
    logger.info("legacy 处置表: %d 词 → 合成概念 %d / 并入A级 %d / 内部重复 %d / 短词 %d",
                len(frame), int(n.get("legacy_concept", 0)),
                int(n.get("covered_by_atier", 0)),
                int(n.get("duplicate_term", 0)), int(n.get("skipped_short", 0)))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="panel_v2 §18 发布组装")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--force-release", action="store_true")
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_export.log")
    rel = paths.output_dir / "release" / "panel_v2"
    rel.mkdir(parents=True, exist_ok=True)
    sentinels = ("quality_control_report.md", "job_ai_classification.parquet",
                 "job_ai_score.parquet", "job_anchor_flag.parquet")
    if not args.force_release and any((rel / s).exists() for s in sentinels):
        raise SystemExit("发布目录已有本次结果（--force-release 覆盖，或改 run_id）")

    export_dictionaries(rel)
    export_legacy_disposition(rel)
    # job_ai_score 以 18 单元 dataset 产出（B1 内存纪律），发布前流式合并
    score_dir = rel / "job_ai_score"
    score_single = rel / "job_ai_score.parquet"
    # 幂等判断用真实规模（上次崩溃可能留残缺文件；4.5 亿行不可能小于 100MB）
    if score_single.exists() and score_single.stat().st_size < 100 * 1024 * 1024:
        score_single.unlink()
    if score_dir.is_dir() and not score_single.exists():
        import pyarrow.parquet as _pq
        import pyarrow as _pa
        files = sorted(score_dir.glob("*.parquet"))
        schema = _pq.read_schema(files[0])
        with _pq.ParquetWriter(score_single, schema,
                               compression="zstd") as w:
            for f in files:
                for rb in _pq.ParquetFile(f).iter_batches(batch_size=2_000_000):
                    w.write_table(_pa.Table.from_batches([rb]))
        logger.info("job_ai_score 合并 %d 单元 -> 单文件", len(files))
    specs = [
        ("skill_concept_v1.parquet", "skill_id", "ai_dict.skill_concepts"),
        ("skill_alias_v1.parquet", "alias_id", "ai_dict.skill_aliases(active)"),
        ("skill_candidate_d_v1.parquet", "term", "pass2/skill_vocab.json"),
        ("skill_legacy_v1.parquet", "term", "panel_v2/lexicon.py(union构建处置)"),
        ("ai_anchor_dictionary_v1.csv", "anchor_version+keyword", "panel_v2/anchors.py"),
        ("job_anchor_flag.parquet", "job_id", "panel_v2/scan.py"),
        ("job_skill_long.parquet", "job_id+skill_code", "panel_v2/scan.py"),
        ("job_firm.parquet", "job_id", "panel_v2/scan.py"),
        ("skill_ai_counts.parquet", "skill+ver+win+year", "panel_v2/counts.py"),
        ("skill_ai_relevance.parquet", "skill+ver+win+year", "panel_v2/relevance.py"),
        ("job_ai_score.parquet", "job+ver+win+stype", "panel_v2/scoring.py"),
        ("job_ai_classification.parquet", "job_id", "panel_v2/scoring.py"),
        ("job_ai_score_loo.parquet", "job_id", "panel_v2/scoring.py(§14.3)"),
        ("quality_control_report.md", "-", "panel_v2/quality.py"),
    ]
    made = 0
    for name, pk, src in specs:
        p = rel / name
        if not p.exists():
            logger.warning("缺文件跳过: %s", name)
            continue
        _meta(p, args.run_id, pk, src,
              anchor_version="main,cn_paper,babina")
        made += 1
    print(f"发布组装完成: {made} 件 + metadata @ {rel}")


if __name__ == "__main__":
    main()
