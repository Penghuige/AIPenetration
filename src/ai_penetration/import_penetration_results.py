"""将 AI 渗透率分析结果 CSV 导入结果数据库（public schema）。

结果库与 eps 源库物理隔离：目标库由 config/database.yaml 的
``database.results_db``（默认 ``ai_pen_results``）或环境变量
``AIPEN_RESULTS_DBNAME`` 决定，本脚本不写 eps 源库。
若配置把结果库解析回源库名，除非显式传 ``--allow-source-db``，否则拒绝执行。

导入三张结果表（幂等：TRUNCATE 后重写，可重复执行）：

- ``public.ai_penetration_fused_cities``：21 市（广东）× 11 年融合面板
- ``public.ai_penetration_fused_industry``：广深 2024 城市 × 行业三口径
  （GB/T 大类；导入时剔除 total < 500 的小样本组合）
- ``public.ai_penetration_national``：全国 392 城 × 年份融合面板
  （来源 fused_national.py，含 A/B/C 三口径与分解）

数据源文件从 ``report_dir`` 按模式取最新（结果文件由分析脚本产出，本脚本只读）。

使用示例::

    python -m src.ai_penetration.import_penetration_results            # 导入到结果库
    python -m src.ai_penetration.import_penetration_results --create-db  # 结果库不存在时先创建
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
from sqlalchemy import create_engine, text

from config.paths import get_project_paths

logger = logging.getLogger("ai_penetration.import_results")

# (表名, report_dir 文件模式, 额外处理) 规格
SPECS: list[tuple[str, str, str]] = [
    ("ai_penetration_fused_cities", "ai_penetration_fused_cities_*.csv", ""),
    ("ai_penetration_fused_industry", "ai_penetration_fused_industry_*.csv", "industry"),
    ("ai_penetration_national", "ai_penetration_national_*.csv", ""),
]

DDL_COMMON = """
CREATE TABLE IF NOT EXISTS public.{table} (
    city           VARCHAR(64)  NOT NULL,
    year           INT          NOT NULL,
    total          BIGINT       NOT NULL,
    a_jobs         BIGINT       NOT NULL,
    b_jobs         BIGINT       NOT NULL,
    fused_jobs     BIGINT       NOT NULL,
    {extra}
    a_rate         DOUBLE PRECISION NOT NULL,
    b_rate         DOUBLE PRECISION NOT NULL,
    fused_rate     DOUBLE PRECISION NOT NULL,
    PRIMARY KEY ({pk})
)
"""


def _ddl(table: str) -> str:
    """按表名生成 DDL（fused_cities/national 带分解列，industry 带行业列）。"""
    if table.endswith("_industry"):
        return DDL_COMMON.format(
            table=table,
            extra="industry       VARCHAR(64)  NOT NULL,",
            pk="city, year, industry",
        )
    return DDL_COMMON.format(
        table=table,
        extra=(
            "ab_both_jobs   BIGINT       NOT NULL,\n"
            "    a_only_jobs    BIGINT       NOT NULL,\n"
            "    b_only_jobs    BIGINT       NOT NULL,"
        ),
        pk="city, year",
    )


def _latest(paths, pattern: str):
    """在 report_dir 找匹配模式的最新 CSV；缺失返回 None。"""
    files = sorted(paths.glob(pattern))
    return files[-1] if files else None


def check_target_db(
    results_params: dict,
    source_params: dict,
    allow_source_db: bool,
) -> None:
    """校验结果库目标不会落到 eps 源库（纯函数，离线可测）。

    Args:
        results_params: 结果库连接参数。
        source_params: eps 源库连接参数。
        allow_source_db: 显式豁免开关。

    Raises:
        RuntimeError: 结果库与源库同名（host+port+dbname 一致）且未显式豁免。
    """
    if allow_source_db:
        return
    same = (
        str(results_params.get("host")) == str(source_params.get("host"))
        and int(results_params.get("port", 5432)) == int(source_params.get("port", 5432))
        and results_params.get("dbname") == source_params.get("dbname")
    )
    if same:
        raise RuntimeError(
            "结果库解析为源库 "
            f"{source_params.get('dbname')}，本脚本拒绝向 eps 源库写入结果表。\n"
            "请设置 config/database.yaml 的 database.results_db 或环境变量 "
            "AIPEN_RESULTS_DBNAME 指向独立结果库；确需写源库时显式加 --allow-source-db。"
        )


def ensure_results_db(create: bool) -> None:
    """确认结果库存在；不存在时按 create 决定创建或报错指引。

    通过维护库（postgres）查询 pg_database 判断，不触碰 eps 数据。

    Args:
        create: True 时自动 CREATE DATABASE（结果库为全新库，不影响任何既有数据）。

    Raises:
        RuntimeError: 结果库不存在且未允许创建时，抛出手工创建指引。
    """
    paths = get_project_paths()
    rp = paths.results_connection_params
    maint = create_engine(
        paths._url_for({**rp, "dbname": "postgres"}, None),
        isolation_level="AUTOCOMMIT",
    )
    with maint.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": rp["dbname"]}
        ).scalar()
    maint.dispose()
    if exists:
        return
    if not create:
        raise RuntimeError(
            f"结果库 {rp['dbname']} 不存在。先创建（PSQL: CREATE DATABASE "
            f"{rp['dbname']};）或加 --create-db 让本脚本代建。"
        )
    admin = create_engine(
        paths._url_for({**rp, "dbname": "postgres"}, None),
        isolation_level="AUTOCOMMIT",  # CREATE DATABASE 不能在事务块内执行
    )
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{rp["dbname"]}"'))
    admin.dispose()
    logger.info("已创建结果库: %s", rp["dbname"])


def import_tables(allow_source_db: bool = False, create_db: bool = False) -> dict[str, int]:
    """导入三张结果表到结果库（TRUNCATE 后重写，幂等）。

    Args:
        allow_source_db: 显式豁免源库保护（不推荐）。
        create_db: 结果库不存在时自动创建。

    Returns:
        {表名: 导入行数}。

    Raises:
        RuntimeError: 目标解析回源库且未豁免，或结果库缺失且未允许创建。
        FileNotFoundError: 任一数据源 CSV 在 report_dir 缺失。
    """
    paths = get_project_paths()
    check_target_db(
        paths.results_connection_params, paths.pg_connection_params, allow_source_db
    )
    ensure_results_db(create_db)
    engine = create_engine(paths.results_pg_sqlalchemy_url())
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        for table, pattern, mode in SPECS:
            src = _latest(paths.report_dir, pattern)
            if src is None:
                raise FileNotFoundError(
                    f"缺少数据源：report_dir 下无 {pattern}（先运行对应分析脚本）"
                )
            df = pd.read_csv(src)
            if mode == "industry":
                if "year" not in df.columns:
                    df["year"] = 2024
                before = len(df)
                df = df[df["total"] >= 500].reset_index(drop=True)
                logger.info("%s 剔除 total<500: %d -> %d 行", table, before, len(df))
            conn.execute(text(_ddl(table)))
            conn.execute(text(f"TRUNCATE public.{table}"))
            df.to_sql(table, conn, if_exists="append", index=False,
                      method="multi", chunksize=500)
            counts[table] = len(df)
            logger.info("已导入 %s.public.%s: %d 行（源 %s）",
                        paths.results_dbname, table, len(df), src.name)
    engine.dispose()
    return counts


def verify_anchor(allow_source_db: bool = False) -> None:
    """导入后校验锚点行（广州 2024 恰一条），确认结果库数据完整。

    Raises:
        AssertionError: 锚点行缺失或重复。
    """
    paths = get_project_paths()
    check_target_db(
        paths.results_connection_params, paths.pg_connection_params, allow_source_db
    )
    engine = create_engine(paths.results_pg_sqlalchemy_url())
    with engine.connect() as conn:
        gz = conn.execute(text(
            "SELECT count(*), max(total), max(fused_jobs) "
            "FROM public.ai_penetration_fused_cities "
            "WHERE city='广州市' AND year=2024"
        )).fetchone()
    engine.dispose()
    assert gz[0] == 1, "广州 2024 锚点行缺失或重复"
    logger.info("校验锚点 广州2024: total=%s fused=%s", gz[1], gz[2])


def main() -> None:
    """入口：导入并校验。"""
    parser = argparse.ArgumentParser(description="导入渗透率结果表（独立结果库）")
    parser.add_argument("--allow-source-db", action="store_true",
                        help="显式允许结果表写入源库（不推荐）")
    parser.add_argument("--create-db", action="store_true",
                        help="结果库不存在时自动创建")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    paths = get_project_paths()
    logger.info("目标结果库: %s:%s/%s（源库 %s 只读）",
                paths.results_connection_params["host"],
                paths.results_connection_params["port"],
                paths.results_dbname, paths.pg_dbname)
    try:
        counts = import_tables(
            allow_source_db=args.allow_source_db, create_db=args.create_db
        )
    except (RuntimeError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        sys.exit(2)
    verify_anchor(allow_source_db=args.allow_source_db)
    print("导入完成:", counts)


if __name__ == "__main__":
    main()
