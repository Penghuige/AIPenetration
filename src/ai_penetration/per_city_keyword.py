"""广东各市 AI 渗透率关键词分析（逐市单独计算）。

对广东省 21 个地级市分别聚合高频岗位并用 AI 关键词匹配，
计算每市每年的 AI 渗透率，输出到单个 CSV 与报告。

逐市单独聚合可避免一次全量 GROUP BY 对 PostgreSQL 的内存压力
（大市单表聚合已验证稳定，如广州/深圳）。

使用示例::

    python -m src.ai_penetration.per_city_keyword --min-freq 10

输出：
- output/reports/ai_penetration_per_city.csv — 各市 × 年 的渗透率矩阵
- output/reports/ai_penetration_per_city_<ts>.md — 报告
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.paths import get_project_paths

from .common import setup_logging
from .keyword_penetration import compute_keyword_penetration
from .load_guangdong import GD_SHARDS, get_eps_engine, load_gd_position_stats

logger = logging.getLogger("ai_penetration.per_city")


def run_per_city(engine, min_freq: int = 10) -> pd.DataFrame:
    """逐市聚合并计算 AI 关键词渗透率。

    Args:
        engine: eps 数据库 engine。
        min_freq: position 全程最低出现次数阈值（客户端过滤 count >= min_freq）。

    Returns:
        DataFrame，列 city / year / total_positions / ai_positions /
        position_rate / total_jobs / ai_jobs / job_rate。
    """
    rows: list[dict] = []
    for city, shard in GD_SHARDS.items():
        logger.info("处理 %s（%s）...", city, shard)
        try:
            stats = load_gd_position_stats(engine, cities=[city])
            if not stats.empty and min_freq > 0:
                stats = stats[stats["count"] >= min_freq].copy()
            if stats.empty:
                logger.warning("%s 过滤频次 >= %d 后无数据，跳过", city, min_freq)
                continue
            pen = compute_keyword_penetration(stats)
            for _, r in pen.iterrows():
                rows.append({
                    "city": city,
                    "year": int(r["year"]),
                    "total_positions": int(r["total_positions"]),
                    "ai_positions": int(r["ai_positions"]),
                    "position_rate": round(r["position_rate"], 6),
                    "total_jobs": int(r["total_jobs"]),
                    "ai_jobs": int(r["ai_jobs"]),
                    "job_rate": round(r["job_rate"], 6),
                })
            logger.info(
                "%s 完成: %d 个岗位, 命中 %d",
                city, stats["count"].sum(),
                pen["ai_jobs"].sum() if not pen.empty else 0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s 失败: %s", city, exc)
    return pd.DataFrame(rows)


def write_report(
    df: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """生成各市渗透率报告 MD 与 CSV。

    Args:
        df: run_per_city 结果。
        output_dir: 输出目录。
        timestamp: 时间戳。

    Returns:
        报告 MD 路径。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ai_penetration_per_city.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 广东各市 AI 渗透率（关键词匹配）",
        "",
        f"- 分析时间：{timestamp}",
        "- 方法：逐市聚合高频岗位（频次≥10），岗位名命中 AI 关键词",
        "",
        "## 各市 2024 年 AI 渗透率",
        "",
        "| 城市 | 总岗位数 | AI岗位数 | 岗位渗透率 | 总职业数 | AI职业数 | 职业渗透率 |",
        "|---|---|---|---|---|---|---|",
    ]
    df24 = df[df["year"] == 2024].sort_values("job_rate", ascending=False)
    for _, r in df24.iterrows():
        lines.append(
            f"| {r['city']} | {int(r['total_positions'])} | {int(r['ai_positions'])} | "
            f"{r['position_rate']:.4f} | {int(r['total_jobs'])} | {int(r['ai_jobs'])} | "
            f"{r['job_rate']:.4f} |"
        )

    lines.extend([
        "",
        "## 各市各年 AI 岗位渗透率趋势",
        "",
        "| 城市 | 年份 | AI岗位数 | 岗位渗透率 | 职业渗透率 |",
        "|---|---|---|---|---|",
    ])
    for _, r in df.sort_values(["city", "year"]).iterrows():
        lines.append(
            f"| {r['city']} | {int(r['year'])} | {int(r['ai_positions'])} | "
            f"{r['position_rate']:.4f} | {r['job_rate']:.4f} |"
        )

    lines.extend([
        "",
        "## 文件",
        "",
        f"- 详细数据：`{csv_path.name}`",
    ])
    report_path = output_dir / f"ai_penetration_per_city_{timestamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已写入: %s", report_path)
    logger.info("CSV 已写入: %s", csv_path)
    return report_path


def main() -> None:
    """逐市 AI 渗透率分析入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="广东各市 AI 渗透率（逐市）")
    parser.add_argument("--min-freq", type=int, default=10)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_per_city.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    engine = get_eps_engine()
    df = run_per_city(engine, min_freq=args.min_freq)
    if df.empty:
        logger.error("无结果，请检查 PG 连接")
        return
    report_path = write_report(df, paths.output_dir / "reports", timestamp)
    logger.info("报告: %s", report_path)


if __name__ == "__main__":
    main()
