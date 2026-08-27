"""岗位 AI 渗透率（方法一关键词 + 方法三 AI 技能词典）。

岗位 AI 判定 = 岗位名命中 AI 关键词 **或** 描述命中 AI 技能词典。
按年统计 AI 岗位渗透率，并输出岗位 AI 相关度（AI 技能命中数）。

使用示例::

    python -m src.ai_penetration.combined_penetration --cities "广州市,深圳市" --max-jobs 100000
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

import pandas as pd

from config.paths import get_project_paths

from .common import setup_logging
from .ai_scoring import AI_SCORE_THRESHOLD, match_keywords_scored, match_skills_scored
from .load_guangdong import get_eps_engine
from .skill_data import load_job_skill_data

logger = logging.getLogger("ai_penetration.combined")




def compute_combined_penetration(df: pd.DataFrame) -> pd.DataFrame:
    """按年计算综合渗透率。

    岗位 AI 判定 = 关键词命中（is_ai_seed）或 描述命中 AI 技能。

    Args:
        df: load_job_skill_data 输出（含 year / is_ai_seed / description）。

    Returns:
        DataFrame，列 year / total / keyword_ai / aiskill_ai / combined_ai /
        keyword_rate / aiskill_rate / combined_rate。
    """
    df = df[df["year"] > 0].copy()
    kw_scores = df["position"].apply(lambda p: match_keywords_scored(p)[1])
    sk_matches = df["description"].apply(lambda d: match_skills_scored(d))
    df["kw_score"] = kw_scores
    df["sk_score"] = sk_matches.apply(lambda x: x[1])
    df["ai_skill_count"] = sk_matches.apply(lambda x: len(x[0]))
    df["ai_skill_hit"] = df["sk_score"] >= AI_SCORE_THRESHOLD
    df["keyword_ai"] = df["kw_score"] >= AI_SCORE_THRESHOLD
    df["combined_ai"] = (df["kw_score"] + df["sk_score"]) >= AI_SCORE_THRESHOLD
    logger.info(
        "岗位 AI 相关度（AI 技能命中数）分布: mean=%.3f, 命中≥1占比=%.1f%%, 命中≥2占比=%.1f%%",
        df["ai_skill_count"].mean(),
        (df["ai_skill_count"] >= 1).mean() * 100,
        (df["ai_skill_count"] >= 2).mean() * 100,
    )

    rows = []
    for year, group in df.groupby("year"):
        n = len(group)
        kw = int(group["keyword_ai"].sum())
        sk = int(group["ai_skill_hit"].sum())
        both = int(group["combined_ai"].sum())
        rows.append({
            "year": int(year),
            "total": n,
            "keyword_ai": kw,
            "aiskill_ai": sk,
            "combined_ai": both,
            "keyword_rate": kw / n if n else 0.0,
            "aiskill_rate": sk / n if n else 0.0,
            "combined_rate": both / n if n else 0.0,
        })
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    logger.info(
        "综合渗透率完成: 2024 关键词=%.4f AI技能=%.4f 综合=%.4f",
        out[out.year == 2024]["keyword_rate"].iloc[0] if not out[out.year == 2024].empty else 0,
        out[out.year == 2024]["aiskill_rate"].iloc[0] if not out[out.year == 2024].empty else 0,
        out[out.year == 2024]["combined_rate"].iloc[0] if not out[out.year == 2024].empty else 0,
    )
    return out


def main() -> None:
    """综合渗透率入口。"""
    parser = argparse.ArgumentParser(description="AI 渗透率（关键词 + AI 技能词典）")
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    parser.add_argument("--max-jobs", type=int, default=100000)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_combined.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    engine = get_eps_engine()
    df = load_job_skill_data(engine, cities=cities, max_jobs=args.max_jobs)
    logger.info("加载岗位 %d 条", len(df))

    pen = compute_combined_penetration(df)
    print(pen[["year", "total", "keyword_ai", "aiskill_ai", "combined_ai",
               "keyword_rate", "aiskill_rate", "combined_rate"]].to_string(index=False))

    # 输出报告
    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"ai_penetration_combined_{timestamp}.md"
    lines = [
        "# AI 渗透率（关键词 + AI 技能词典）",
        "",
        f"- 时间：{timestamp}",
        f"- 数据：{len(df)} 条岗位（{'、'.join(cities)}）",
        "",
        "## 按年渗透率",
        "",
        "| 年份 | 总岗位 | 关键词AI | AI技能AI | 综合AI | 关键词率 | AI技能率 | 综合率 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in pen.iterrows():
        lines.append(
            f"| {int(r['year'])} | {int(r['total'])} | {int(r['keyword_ai'])} | "
            f"{int(r['aiskill_ai'])} | {int(r['combined_ai'])} | "
            f"{r['keyword_rate']:.4f} | {r['aiskill_rate']:.4f} | {r['combined_rate']:.4f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已写入: %s", report_path)


if __name__ == "__main__":
    main()
