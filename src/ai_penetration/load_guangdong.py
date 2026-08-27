"""广东省招聘数据加载模块。

从 eps 数据库加载广东省 21 个地级市分片表的岗位数据，
聚合为 position × (year, quarter) 的出现次数统计表。
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config.paths import get_project_paths

logger = logging.getLogger("ai_penetration.load")

# 广东省 21 个地级市 → eps 分片表名（探索自 eps 数据库，见 output/eps_city_shard_map.json）
GD_SHARDS: dict[str, str] = {
    "广州市": "job_p0387",
    "深圳市": "job_p0389",
    "东莞市": "job_p0374",
    "佛山市": "job_p0372",
    "中山市": "job_p0361",
    "珠海市": "job_p0354",
    "惠州市": "job_p0360",
    "江门市": "job_p0339",
    "肇庆市": "job_p0246",
    "汕头市": "job_p0341",
    "潮州市": "job_p0268",
    "揭阳市": "job_p0285",
    "汕尾市": "job_p0122",
    "湛江市": "job_p0276",
    "茂名市": "job_p0270",
    "阳江市": "job_p0209",
    "云浮市": "job_p0160",
    "韶关市": "job_p0232",
    "清远市": "job_p0306",
    "梅州市": "job_p0215",
    "河源市": "job_p0222",
}


def get_eps_engine() -> Engine:
    """创建连接 eps 数据库的 SQLAlchemy engine。

    eps 是本项目的唯一数据源，连接参数统一来自 config（database.yaml /
    环境变量），URL 由 paths.pg_sqlalchemy_url() 生成并对凭据做 URL 编码。

    Returns:
        指向 eps 数据库的 SQLAlchemy Engine。
    """
    return create_engine(get_project_paths().pg_sqlalchemy_url())


def _parse_year_quarter(publish_time: str) -> tuple[int, int] | None:
    """解析 publish_time 为 (year, quarter)。

    Args:
        publish_time: 形如 'YYYY-MM-DD' 的发布日期字符串。

    Returns:
        (year, quarter)；无法解析时返回 None。
    """
    try:
        dt = datetime.strptime(str(publish_time).strip()[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    return dt.year, (dt.month - 1) // 3 + 1


def _build_position_stats(rows: list[tuple]) -> pd.DataFrame:
    """从 (position, publish_time) 元组列表聚合出现次数统计。

    Args:
        rows: 每项为 (position, publish_time) 的元组列表。

    Returns:
        DataFrame，列为 position / year / quarter / count，
        按 position 排序，空 position 与无法解析的日期被忽略。
    """
    records = []
    for position, publish_time in rows:
        pos = str(position or "").strip()
        if not pos:
            continue
        yq = _parse_year_quarter(publish_time)
        if yq is None:
            continue
        records.append({"position": pos, "year": yq[0], "quarter": yq[1]})
    if not records:
        return pd.DataFrame(columns=["position", "year", "quarter", "count"])
    df = pd.DataFrame(records)
    stats = (
        df.groupby(["position", "year", "quarter"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return stats.sort_values("position").reset_index(drop=True)


def load_gd_position_stats(
    engine: Engine,
    limit_per_shard: int = 0,
    cities: list[str] | None = None,
) -> pd.DataFrame:
    """加载广东省岗位数据并聚合为统计表。

    逐表查询 position 与 publish_time，合并后聚合。

    Args:
        engine: eps 数据库 engine。
        limit_per_shard: 每张分片表最多取多少行；0 表示不限制。
        cities: 只加载指定城市；为空时加载全部 GD_SHARDS。

    Returns:
        position × (year, quarter) 出现次数统计表。

    Raises:
        KeyError: cities 中包含不在 GD_SHARDS 内的城市名。
    """
    shards = {c: GD_SHARDS[c] for c in cities} if cities else GD_SHARDS
    all_rows: list[tuple] = []
    for city, shard in shards.items():
        sql = f"SELECT position, publish_time FROM public.{shard}"
        if limit_per_shard > 0:
            sql += f" LIMIT {int(limit_per_shard)}"
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [(str(r[0] or ""), str(r[1] or "")) for r in result]
        all_rows.extend(rows)
        logger.info("已加载 %s（%s）: %d 行", city, shard, len(rows))
    return _build_position_stats(all_rows)
