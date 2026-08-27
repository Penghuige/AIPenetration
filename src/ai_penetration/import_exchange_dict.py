"""导入交接包双语技能词典到 eps 数据库（ai_dict schema）。

来源：D:\\PythonProjects\\exchange\\02_完整双语候选词典
- skill_concept_bilingual_complete_candidate_v1.0.csv → ai_dict.skill_concepts
- skill_alias_bilingual_complete_candidate_v1.0.csv   → ai_dict.skill_aliases

使用 COPY 导入（15 万行级别），导入后校验行数与 QC 报告一致性并建索引。

使用示例::

    python -m src.ai_penetration.import_exchange_dict --exchange-dir D:/PythonProjects/exchange
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import psycopg2


from .common import eps_conn_params

logger = logging.getLogger("ai_penetration.import_dict")

SCHEMA = "ai_dict"
CONCEPT_TABLE = f"{SCHEMA}.skill_concepts"
ALIAS_TABLE = f"{SCHEMA}.skill_aliases"

CONCEPT_CSV = "02_完整双语候选词典/skill_concept_bilingual_complete_candidate_v1.0.csv"
ALIAS_CSV = "02_完整双语候选词典/skill_alias_bilingual_complete_candidate_v1.0.csv"

EXPECTED_CONCEPTS = 22683
EXPECTED_ALIASES = 153804


def _connect():
    """连接 eps 库。"""
    params = eps_conn_params()
    return psycopg2.connect(**params)


def _csv_columns(path: Path) -> list[str]:
    """读取 CSV 表头列名。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def _copy_csv(cur, table: str, path: Path) -> int:
    """将 CSV 整体 COPY 入表（列按表头顺序对齐），返回行数。"""
    cols = _csv_columns(path)
    col_list = ", ".join('"%s"' % c for c in cols)
    with path.open(encoding="utf-8-sig", newline="") as f:
        # 去掉表头行后按 CSV 格式 COPY
        cur.copy_expert(
            f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT csv, HEADER false)",
            _SkipHeader(f),
        )
    cur.execute(f"SELECT count(*) FROM {table}")
    return cur.fetchone()[0]


class _SkipHeader:
    """包装文件对象，跳过首行（表头）供 COPY 使用。"""

    def __init__(self, f):
        self._f = f
        self._header_skipped = False

    def read(self, size: int) -> str:
        if not self._header_skipped:
            self._f.readline()
            self._header_skipped = True
        return self._f.read(size)

    def readline(self) -> str:
        if not self._header_skipped:
            self._f.readline()
            self._header_skipped = True
        return self._f.readline()

    def __iter__(self):
        return iter(self._f)

    @property
    def closed(self) -> bool:
        return self._f.closed


def main() -> None:
    """导入交接包词典入口。"""
    parser = argparse.ArgumentParser(description="导入交接包技能词典")
    parser.add_argument("--exchange-dir", type=str,
                        default=r"D:\PythonProjects\exchange")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    root = Path(args.exchange_dir)
    concept_csv = root / CONCEPT_CSV
    alias_csv = root / ALIAS_CSV
    if not concept_csv.exists() or not alias_csv.exists():
        logger.error("词典文件不存在: %s / %s", concept_csv, alias_csv)
        return

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SET LOCAL work_mem = '1GB'")

        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

        # 概念表（列定义与 CSV 一致，全部 text 兼容）
        cur.execute(f"DROP TABLE IF EXISTS {CONCEPT_TABLE}")
        cur.execute(f"""
            CREATE TABLE {CONCEPT_TABLE} (
                skill_id text PRIMARY KEY,
                canonical_zh text,
                canonical_en text,
                skill_type text,
                skill_category text,
                definition_en text,
                description_en text,
                source_primary text,
                source_version_primary text,
                source_id_primary text,
                source_names text,
                source_record_count text,
                source_skill_types text,
                esco_uri text,
                esco_skill_type text,
                esco_reuse_level text,
                onet_element_ids text,
                onet_categories text,
                onet_hot_technology text,
                onet_in_demand text,
                is_digital text,
                is_green text,
                is_transversal text,
                is_research_skill text,
                is_language_skill text,
                confidence_tier text,
                dictionary_version text,
                translation_status text,
                valid_from text,
                valid_to text,
                is_active_concept text,
                source_count text,
                translation_strategy text,
                translation_confidence text,
                translation_ambiguity_flag text,
                translation_notes text,
                translation_source text
            )
        """)
        n_concept = _copy_csv(cur, CONCEPT_TABLE, concept_csv)
        logger.info("概念表导入: %d 行 (期望 %d)", n_concept, EXPECTED_CONCEPTS)

        # 别名表
        cur.execute(f"DROP TABLE IF EXISTS {ALIAS_TABLE}")
        cur.execute(f"""
            CREATE TABLE {ALIAS_TABLE} (
                alias_id text PRIMARY KEY,
                skill_id text REFERENCES {CONCEPT_TABLE}(skill_id),
                alias text NOT NULL,
                alias_normalized text,
                language text,
                alias_type text,
                source text,
                matching_rule text,
                ambiguity_flag text,
                confidence_tier text,
                dictionary_version text,
                is_active text,
                primary_skill_id text,
                boundary_rule text,
                case_sensitive text,
                activation_reason text,
                translation_status text
            )
        """)
        n_alias = _copy_csv(cur, ALIAS_TABLE, alias_csv)
        logger.info("别名表导入: %d 行 (期望 %d)", n_alias, EXPECTED_ALIASES)

        # 匹配用索引
        cur.execute(
            f"CREATE INDEX idx_skill_aliases_norm ON {ALIAS_TABLE} (alias_normalized)")
        cur.execute(
            f"CREATE INDEX idx_skill_aliases_skill ON {ALIAS_TABLE} (skill_id)")
        cur.execute(
            f"CREATE INDEX idx_skill_aliases_active ON {ALIAS_TABLE} (is_active)")
        conn.commit()
        logger.info("索引创建完成")

        # 校验
        cur.execute(f"SELECT count(*), count(DISTINCT skill_id) FROM {CONCEPT_TABLE}")
        c, d = cur.fetchone()
        cur.execute(f"""
            SELECT count(*), count(*) FILTER (WHERE is_active='1')
            FROM {ALIAS_TABLE}
        """)
        a, act = cur.fetchone()
        ok = c == d == EXPECTED_CONCEPTS and a == EXPECTED_ALIASES and act == 107467
        logger.info("校验: 概念 %d/%d唯一, 别名 %d (激活 %d), 一致=%s",
                    c, d, a, act, ok)
        if not ok:
            logger.warning("与 QC 报告不一致，请人工核查")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
