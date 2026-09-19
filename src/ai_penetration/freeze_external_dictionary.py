"""导出正式外部 A 级双语词典与 QC 报告（交接冻结前置·步骤4）。

在别名激活（zh_alias_activation --apply）完成后运行：从 ai_dict 导出
冻结版概念表与激活别名表，生成质量检查报告，把发布状态从
complete_candidate_not_frozen 推进为 A 级正式冻结版。

版本命名遵守指南 §4.1：不用 final/new/latest，语义版本 v1.1 + 运行编号。

产出（output/dictionary/）：
- skill_concept_bilingual_a_frozen_v1.1.csv     全概念表（37 列）
- skill_alias_bilingual_a_active_frozen_v1.1.csv 激活别名表（17 列）

QC 报告（output/reports/）：行数、SHA-256、激活分布、不变量检查结果。

使用示例::

    python -m src.ai_penetration.freeze_external_dictionary
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import psycopg2

from config.paths import get_project_paths

from .common import eps_conn_params, setup_logging

logger = logging.getLogger("ai_penetration.freeze_dict")

VERSION = "bilingual_a_frozen_v1.1"


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_query_to_csv(cur_dir: str, query: str, out_path: Path,
                       columns: list[str]) -> int:
    """执行查询并以 CSV（含表头）落盘，返回数据行数。"""
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
    finally:
        conn.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows)


def check_invariants() -> tuple[list[str], list[str], list[str]]:
    """执行不变量自检（指南 §11.1.3 / 执行记录规则 5-8 的相关约束）。

    歧义/needs_review 不变量限 zh/mixed 候选（en 来源别名按 §7.6.4 豁免
    中文频数限制；source_english 状态为交接包既有属性）。

    Returns:
        (通过项列表, 失败项列表, 豁免登记项列表)。
    """
    conn = psycopg2.connect(**eps_conn_params())
    ok: list[str] = []
    bad: list[str] = []
    notes: list[str] = []
    try:
        cur = conn.cursor()
        # 0) 上一任交接的 170 批中文化必须先真正完成，才能称为 frozen。
        cur.execute("""
            SELECT count(*) FROM ai_dict.skill_concepts
            WHERE translation_status IS NULL
               OR translation_status = ''
               OR translation_status = 'pending_codex_zh'
        """)
        n_pending = int(cur.fetchone()[0])
        (ok if n_pending == 0 else bad).append(
            f"正式概念无 pending_codex_zh/空翻译状态: {n_pending} 违例"
        )
        cur.execute("""
            SELECT count(DISTINCT dictionary_version),
                   count(*) FILTER (WHERE skill_id IS NULL OR skill_id = '')
            FROM ai_dict.skill_concepts
        """)
        n_versions, empty_skill_id = cur.fetchone()
        (ok if int(n_versions) == 1 else bad).append(
            f"概念表 dictionary_version 单一: {int(n_versions)} 个版本"
        )
        (ok if int(empty_skill_id) == 0 else bad).append(
            f"概念 skill_id 全部非空: {int(empty_skill_id)} 违例"
        )

        # 1) 歧义别名不得为激活态（限 zh/mixed；en 来源别名按指南 §7.6.4 豁免
        #    中文频数限制，其歧义由来源侧 activation_reason=unique_source_label 管理）
        cur.execute("""
            SELECT count(*) FROM ai_dict.skill_aliases s
            JOIN ai_dict.alias_ambiguity a ON a.alias = s.alias
            WHERE s.is_active='1' AND a.n_skills > 1
              AND s.language IN ('zh','mixed')
        """)
        n = cur.fetchone()[0]
        (ok if n == 0 else bad).append(f"歧义 zh/mixed 别名未激活: {n} 违例")
        cur.execute("""
            SELECT count(*) FROM ai_dict.skill_aliases s
            JOIN ai_dict.alias_ambiguity a ON a.alias = s.alias
            WHERE s.is_active='1' AND a.n_skills > 1 AND s.language = 'en'
        """)
        n_en = cur.fetchone()[0]
        notes.append(f"en 来源别名与 zh/mixed 歧义同名共存 {n_en} 行（§7.6.4 豁免，登记）")
        # 2) 过短别名不得激活（长度<=2 的 zh/mixed）
        cur.execute("""
            SELECT count(*) FROM ai_dict.skill_aliases
            WHERE is_active='1' AND language IN ('zh','mixed')
              AND length(trim(alias)) <= 2
        """)
        n = cur.fetchone()[0]
        (ok if n == 0 else bad).append(f"过短 zh/mixed 别名未激活: {n} 违例")
        # 3) 不变量限 Codex 中文候选：ambiguity_flag=1 的 zh/mixed 行必须
        #    needs_review；en 行 translation_status=source_english 为交接包既有属性
        cur.execute("""
            SELECT count(*) FROM ai_dict.skill_aliases
            WHERE ambiguity_flag='1' AND translation_status != 'needs_review'
              AND language IN ('zh','mixed')
        """)
        n = cur.fetchone()[0]
        (ok if n == 0 else bad).append(f"Codex 中文候选 ambiguity⇔needs_review 一致: {n} 违例")
        cur.execute("""
            SELECT count(*) FROM ai_dict.skill_aliases
            WHERE ambiguity_flag='1' AND translation_status = 'source_english'
        """)
        notes.append(f"en 行 source_english 歧义标记 {cur.fetchone()[0]} 行（交接包既有属性，登记）")
        # 4) 激活别名数不少于交接候选版既有激活数（只增不减）
        cur.execute("SELECT count(*) FROM ai_dict.skill_aliases WHERE is_active='1'")
        total_active = cur.fetchone()[0]
        (ok if total_active >= 107467 else bad).append(
            f"激活总数 {total_active} >= 候选版 107467")
    finally:
        conn.close()
    return ok, bad, notes


def stats_by_language() -> list[tuple]:
    """激活状态的 language × is_active 统计。"""
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT language, is_active, count(*)
            FROM ai_dict.skill_aliases GROUP BY 1,2 ORDER BY 1,2
        """)
        return cur.fetchall()
    finally:
        conn.close()


def _latest_activation_manifest(report_dir: Path) -> dict | None:
    """读取最近的激活 manifest（阈值依据入报告）。"""
    manifests = sorted(report_dir.glob("alias_activation_manifest_*.json"))
    return json.loads(manifests[-1].read_text(encoding="utf-8")) if manifests else None


CONCEPT_COLUMNS = [
    "skill_id", "canonical_zh", "canonical_en", "skill_type", "skill_category",
    "definition_en", "description_en", "source_primary", "source_version_primary",
    "source_id_primary", "source_names", "source_record_count", "source_skill_types",
    "esco_uri", "esco_skill_type", "esco_reuse_level", "onet_element_ids",
    "onet_categories", "onet_hot_technology", "onet_in_demand", "is_digital",
    "is_green", "is_transversal", "is_research_skill", "is_language_skill",
    "confidence_tier", "dictionary_version", "translation_status", "valid_from",
    "valid_to", "is_active_concept", "source_count", "translation_strategy",
    "translation_confidence", "translation_ambiguity_flag", "translation_notes",
    "translation_source",
]
ALIAS_COLUMNS = [
    "alias_id", "skill_id", "alias", "alias_normalized", "language", "alias_type",
    "source", "matching_rule", "ambiguity_flag", "confidence_tier",
    "dictionary_version", "is_active", "primary_skill_id", "boundary_rule",
    "case_sensitive", "activation_reason", "translation_status",
]


def export_and_report(stamp: str) -> dict[str, str]:
    """导出两表 + 生成 QC 报告。

    Returns:
        {产物路径: sha256}。
    """
    paths = get_project_paths()
    out_dir = paths.output_dir / "dictionary"
    concept_csv = out_dir / f"skill_concept_{VERSION}.csv"
    alias_csv = out_dir / f"skill_alias_active_{VERSION}.csv"

    concept_select = ", ".join(
        f"'{VERSION}' AS dictionary_version" if col == "dictionary_version" else col
        for col in CONCEPT_COLUMNS
    )
    alias_select = ", ".join(
        f"'{VERSION}' AS dictionary_version" if col == "dictionary_version" else col
        for col in ALIAS_COLUMNS
    )
    n_concept = _copy_query_to_csv(
        "ai_dict",
        f"SELECT {concept_select} FROM ai_dict.skill_concepts ORDER BY skill_id",
        concept_csv, CONCEPT_COLUMNS,
    )
    n_alias = _copy_query_to_csv(
        "ai_dict",
        f"""SELECT {alias_select} FROM ai_dict.skill_aliases
            WHERE is_active='1' ORDER BY alias_id""",
        alias_csv, ALIAS_COLUMNS,
    )
    sha = {str(concept_csv): _sha256_file(concept_csv), str(alias_csv): _sha256_file(alias_csv)}
    ok, bad, notes = check_invariants()
    lang_stats = stats_by_language()
    manifest = _latest_activation_manifest(paths.report_dir)

    lines = [
        f"# A 级双语词典冻结 QC 报告（{VERSION}）",
        "",
        f"- 运行时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 发布状态: `complete_candidate_not_frozen` → `a_level_frozen_{VERSION}`",
        "- 语料依据: eps 广深 2014–2024，激活频数口径 COUNT(DISTINCT 规范化text_hash)",
        "",
        "## 导出",
        "",
        "| 文件 | 行数 | SHA-256 |",
        "|---|---|---|",
        f"| {concept_csv.name} | {n_concept} | `{sha[str(concept_csv)][:16]}…` |",
        f"| {alias_csv.name} | {n_alias} | `{sha[str(alias_csv)][:16]}…` |",
        "",
        "## 激活统计（language × is_active）",
        "",
        "| language | is_active | 行数 |",
        "|---|---|---|",
    ]
    for lang, active, cnt in lang_stats:
        lines.append(f"| {lang} | {active} | {cnt} |")
    if manifest:
        lines += [
            "",
            "## 激活阈值依据（步骤3 manifest）",
            "",
            f"- 阈值: freq_total >= {manifest['min_freq']} 且不在 alias_ambiguity",
            f"- 候选别名: {manifest['candidates']}；skill_aliases 更新行: {manifest['rows_updated']}",
            f"- reason 标记: `{manifest['reason']}`；明细: `{manifest['candidates_csv']}`",
            "- 候选集最高频别名见该 CSV（freq 降序）",
        ]
    lines += ["", "## 不变量检查", ""]
    for item in ok:
        lines.append(f"- ✅ {item}")
    for item in bad:
        lines.append(f"- ❌ {item}")
    if notes:
        lines += ["", "## 豁免登记（交接既有属性，非本次变更引入）", ""]
        for item in notes:
            lines.append(f"- ℹ️ {item}")
    report_path = paths.report_dir / f"freeze_qc_{VERSION}_{stamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("QC 报告: %s", report_path)
    if bad:
        raise SystemExit(2)
    return sha


def main() -> None:
    """冻结导出入口。"""
    parser = argparse.ArgumentParser(description="导出 A 级冻结词典与 QC 报告")
    args = parser.parse_args()
    del args
    paths = get_project_paths()
    setup_logging(paths.log_dir / "freeze_external_dictionary.log")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sha = export_and_report(stamp)
    print("导出完成:")
    for path, digest in sha.items():
        print(f"  {path}\n    sha256={digest}")


if __name__ == "__main__":
    main()
