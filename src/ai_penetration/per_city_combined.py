"""广东各市 AI 综合渗透率（关键词 + AI 技能词典，逐市分析）。

对广东省 21 个地级市分别加载岗位描述，用关键词法 + AI 技能词典法
计算每市的 AI 岗位渗透率（按年），输出各市对比。

使用示例::

    python -m src.ai_penetration.per_city_combined --per-city-max-jobs 50000
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

import pandas as pd

from config.paths import get_project_paths

from .common import setup_logging
from .combined_penetration import compute_combined_penetration
from .load_guangdong import GD_SHARDS, get_eps_engine
from .skill_data import load_job_skill_data

logger = logging.getLogger("ai_penetration.per_city_combined")




def main() -> None:
    """逐市综合渗透率入口。"""
    parser = argparse.ArgumentParser(description="广东各市 AI 综合渗透率（逐市）")
    parser.add_argument("--per-city-max-jobs", type=int, default=50000,
                        help="每市加载岗位描述上限（小市不足则全量）")
    parser.add_argument("--cities", type=str, default="",
                        help="限定城市，逗号分隔；空则全部 21 市")
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_per_city_combined.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    shards = {c: s for c, s in GD_SHARDS.items() if not cities or c in cities}

    engine = get_eps_engine()
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    for city in shards:
        logger.info("处理 %s ...", city)
        df = load_job_skill_data(
            engine, cities=[city], max_jobs=args.per_city_max_jobs
        )
        if df.empty:
            logger.warning("%s 无数据", city)
            continue
        pen = compute_combined_penetration(df)
        # 该市 2024 年渗透率（无 2024 则取最新年）
        recent = pen[pen["year"] >= 2022].sort_values("year")
        if not recent.empty:
            latest = recent.iloc[-1]
            summary_rows.append({
                "city": city,
                "sample": int(df[df["year"] > 0].shape[0]),
                "latest_year": int(latest["year"]),
                "keyword_rate": round(latest["keyword_rate"], 4),
                "aiskill_rate": round(latest["aiskill_rate"], 4),
                "combined_rate": round(latest["combined_rate"], 4),
                "combined_ai": int(latest["combined_ai"]),
            })
        for _, r in pen.iterrows():
            detail_rows.append({"city": city, **r.to_dict()})
        logger.info("%s 完成: 最新综合率=%.4f", city, summary_rows[-1]["combined_rate"] if summary_rows else 0)

    summary = pd.DataFrame(summary_rows).sort_values("combined_rate", ascending=False)
    detail = pd.DataFrame(detail_rows)

    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ai_penetration_per_city_combined.csv"
    detail.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report_path = out_dir / f"ai_penetration_per_city_combined_{timestamp}.md"
    lines = [
        "# 广东各市 AI 综合渗透率（关键词 + AI 技能词典）",
        "",
        f"- 时间：{timestamp}",
        f"- 每市加载上限：{args.per_city_max_jobs} 条",
        "",
        "## 各市最新年综合渗透率（按综合率排序）",
        "",
        "| 城市 | 样本数 | 年份 | 关键词率 | AI技能率 | 综合率 | 综合AI岗位 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['city']} | {int(r['sample'])} | {int(r['latest_year'])} | "
            f"{r['keyword_rate']:.4f} | {r['aiskill_rate']:.4f} | "
            f"{r['combined_rate']:.4f} | {int(r['combined_ai'])} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已写入: %s", report_path)
    logger.info("明细 CSV: %s", csv_path)
    print("\n" + summary[["city", "sample", "latest_year", "keyword_rate",
                         "aiskill_rate", "combined_rate", "combined_ai"]].to_string(index=False))


if __name__ == "__main__":
    main()
