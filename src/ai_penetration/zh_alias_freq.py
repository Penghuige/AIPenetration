"""中文/混合别名岗位描述频数计算（交接包冻结前置步骤）。

对 ai_dict.skill_aliases 中未激活的 zh/mixed 别名，在 eps 广深语料中计算
「命中的不同岗位描述数」（distinct jobs），并输出歧义检查表。这是交接说明
规定的中文化别名冻结前置步骤（频数 + 歧义）。

方法：Aho-Corasick 自动机一次扫描全文，岗位内去重后按别名累计 distinct 描述数。
为效率逐年份处理并存断点。

输出：
- ai_dict.zh_alias_freq(alias_id, alias, freq, year)  各年 distinct 频数
- ai_dict.alias_ambiguity(...)                        一对多/过短歧义标记

使用示例::

    python -m src.ai_penetration.zh_alias_freq --years 2024 --cities "广州市,深圳市"
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import ahocorasick
import psycopg2

from config.paths import get_project_paths

from .common import eps_connect, eps_conn_params, setup_logging
from .load_guangdong import GD_SHARDS

logger = logging.getLogger("ai_penetration.zh_alias_freq")


def load_zh_mixed_aliases() -> dict[str, tuple[str, str]]:
    """从 PG 加载未激活的 zh/mixed 别名 {alias: (alias_id, language)}。

    同名别名取首个 alias_id（一对多映射另行歧义标记）。

    Returns:
        alias -> (alias_id, language)。
    """
    conn = eps_connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT alias, min(alias_id), min(language)
            FROM ai_dict.skill_aliases
            WHERE is_active='0' AND language IN ('zh','mixed')
              AND length(trim(alias)) >= 2
            GROUP BY alias
        """)
        rows = cur.fetchall()
        logger.info("加载 zh/mixed 未激活别名(去重): %d 条", len(rows))
        return {r[0]: (r[1], r[2]) for r in rows}
    finally:
        conn.close()


def build_automaton(alias_map: dict[str, tuple[str, str]]):
    """构建 Aho-Corasick 自动机。

    扫描前语料会转小写，因此自动机的键统一为 alias.lower()；
    存储值保留原始别名，保证命中后能映射回正确条目。

    Args:
        alias_map: alias -> (alias_id, language) 映射。

    Note:
        仅大小写不同的别名会冲突，后写入者覆盖先写入者；
        同形异名的 alias_id 一对多歧义由 alias_ambiguity 表另行标记。
    """
    automaton = ahocorasick.Automaton()
    for alias, (aid, _lang) in alias_map.items():
        key = alias.lower()
        automaton.add_word(key, (aid, alias))
    automaton.make_automaton()
    return automaton


def count_freq_year(year: int, cities: list[str], automaton, batch_size: int) -> dict[str, int]:
    """扫描某年语料，统计每别名的 distinct 岗位描述数。

    Args:
        year: 年份。
        cities: 城市 shard 列表对应的市名。
        automaton: 构建好的自动机。
        batch_size: fetchmany 批大小。

    Returns:
        alias_id -> distinct job 数。
    """
    freq: dict[str, int] = defaultdict(int)
    conn = eps_connect()
    total_jobs = 0
    try:
        cur = conn.cursor()
        cur.execute("SET LOCAL work_mem = '1GB'")
        for city in cities:
            shard = GD_SHARDS[city]
            sql = f"""
                SELECT job_description FROM public.{shard}
                WHERE substr(publish_time, 1, 4) = %s
                  AND job_description IS NOT NULL AND job_description != ''
                  AND position IS NOT NULL AND position != ''
            """
            cur.execute(sql, (str(year),))
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                for (desc,) in batch:
                    text = str(desc).lower()
                    # iter_long 返回 (结束位置, (alias_id, alias))，取存储值的别名 ID
                    hits = {info[1][0] for info in automaton.iter_long(text)}
                    for aid in hits:
                        freq[aid] += 1
                    total_jobs += 1
                logger.info("%s %d 已处理 %d 条", city, year, total_jobs)
    finally:
        conn.close()
    logger.info("%d 年扫描完成: %d 岗位, %d 别名命中过", year, total_jobs, len(freq))
    return dict(freq)


def write_freq_table(conn_params: dict, freq_by_year: dict[int, dict[str, int]]) -> None:
    """写入 ai_dict.zh_alias_freq 表。"""
    conn = psycopg2.connect(**conn_params)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS ai_dict.zh_alias_freq")
        cur.execute("""
            CREATE TABLE ai_dict.zh_alias_freq (
                alias_id text PRIMARY KEY,
                freq_total bigint,
                freq_by_year jsonb
            )
        """)
        # 合并各年
        merged: dict[str, dict] = {}
        for year, fmap in freq_by_year.items():
            for aid, n in fmap.items():
                m = merged.setdefault(aid, {"total": 0, "years": {}})
                m["total"] += n
                m["years"][str(year)] = n
        rows = [(aid, v["total"], json.dumps(v["years"])) for aid, v in merged.items()]
        from psycopg2.extras import execute_values
        execute_values(cur,
            "INSERT INTO ai_dict.zh_alias_freq (alias_id, freq_total, freq_by_year) VALUES %s",
            rows, page_size=5000)
        conn.commit()
        logger.info("zh_alias_freq 写入: %d 条", len(rows))
    finally:
        conn.close()


def write_ambiguity_table() -> None:
    """生成歧义检查表：一对多映射、过短词、跨技能同名。"""
    conn = psycopg2.connect(**_cp())
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS ai_dict.alias_ambiguity")
        cur.execute("""
            CREATE TABLE ai_dict.alias_ambiguity AS
            SELECT s.alias, s.alias_normalized, count(DISTINCT s.skill_id) AS n_skills,
                   bool_or(length(trim(s.alias)) <= 2) AS too_short
            FROM ai_dict.skill_aliases s
            WHERE s.is_active='0' AND s.language IN ('zh','mixed')
            GROUP BY s.alias, s.alias_normalized
            HAVING count(DISTINCT s.skill_id) > 1 OR bool_or(length(trim(s.alias)) <= 2)
        """)
        cur.execute("SELECT count(*) FROM ai_dict.alias_ambiguity")
        n = cur.fetchone()[0]
        conn.commit()
        logger.info("alias_ambiguity 歧义条目: %d", n)
    finally:
        conn.close()


def _cp():
    """连接 eps（内部工具函数）。"""
    return eps_conn_params()


def main() -> None:
    """中文别名频数计算入口。"""
    parser = argparse.ArgumentParser(description="中文/混合别名频数计算")
    parser.add_argument("--years", type=str, default="2024")
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    parser.add_argument("--batch-size", type=int, default=100000)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "zh_alias_freq.log")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    alias_map = load_zh_mixed_aliases()
    automaton = build_automaton(alias_map)
    logger.info("自动机构建完成")

    freq_by_year: dict[int, dict[str, int]] = {}
    for year in years:
        freq_by_year[year] = count_freq_year(year, cities, automaton, args.batch_size)

    write_freq_table(_cp(), freq_by_year)
    write_ambiguity_table()

    # 频数分布概览
    print("\n完成。各年命中别名数:", {y: len(f) for y, f in freq_by_year.items()})


if __name__ == "__main__":
    main()
