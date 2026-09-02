"""将 AI 渗透率分析结果 CSV 导入本项目配置数据库（public schema）。

目标库由 config/database.yaml（或 AIPEN_PG_* 环境变量）决定，代码不硬编码库名。
导入三张结果表（幂等：TRUNCATE 后重写，可重复执行）：

- ``public.ai_penetration_fused_cities``：21 市（广东）× 11 年融合面板
- ``public.ai_penetration_fused_industry``：广深 2024 城市 × 行业三口径
  （GB/T 大类；导入时剔除 total < 500 的小样本组合）
- ``public.ai_penetration_national``：全国 392 城 × 年份融合面板
  （来源 fused_national.py，含 A/B/C 三口径与分解）

数据源文件从 ``report_dir`` 按模式取最新（结果文件由分析脚本产出，本脚本只读）。

使用示例::

    python -m src.ai_penetration.import_penetration_results
"""
from __future__ import annotations

import logging

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


def import_tables() -> dict[str, int]:
    """导入三张结果表（TRUNCATE 后重写，幂等）。

    Returns:
        {表名: 导入行数}。

    Raises:
        FileNotFoundError: 任一数据源 CSV 在 report_dir 缺失。
    """
    paths = get_project_paths()
    engine = create_engine(paths.pg_sqlalchemy_url())
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
            logger.info("已导入 public.%s: %d 行（源 %s）", table, len(df), src.name)
    return counts


def main() -> None:
    """入口：导入并校验。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    paths = get_project_paths()
    counts = import_tables()
    engine = create_engine(paths.pg_sqlalchemy_url())
    with engine.connect() as conn:
        n = conn.execute(text(
            "SELECT count(*) FROM public.ai_penetration_fused_cities "
            "WHERE city='广州市' AND year=2024"
        )).scalar()
        gz = conn.execute(text(
            "SELECT total, fused_jobs FROM public.ai_penetration_fused_cities "
            "WHERE city='广州市' AND year=2024"
        )).fetchone()
    assert n == 1, "广州 2024 锚点行缺失"
    logger.info("校验锚点 广州2024: total=%s fused=%s", gz[0], gz[1])
    print("导入完成:", counts)


if __name__ == "__main__":
    main()
