"""锚点共现法行业渗透率（方法 B 的行业维度交叉验证）。

复用已计算的 ωsAI（skill_ai_anchor 锚点共现），流式扫描岗位描述，
经 LATERAL join 关联 ent 企业表取行业大类，岗位 ωjAI >= 阈值判为 AI，
按 城市 × 行业 聚合渗透率，对比方法 A（加权判定）的行业排名。

使用示例::

    python -m src.ai_penetration.anchor_industry --year 2024 --threshold 0.10
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .common import DEFAULT_OMEGA_SNAPSHOT, eps_conn_params, setup_logging
from .anchor_penetration import DEFAULT_OMEGA_THRESHOLD
from .industry_classification import classify_industry
from .load_guangdong import GD_SHARDS
from .skill_ai_anchor import build_skill_regex, extract_skills_fast, load_merged_skills

logger = logging.getLogger("ai_penetration.anchor_industry")




def compute_industry_penetration(
    cities: list[str],
    year: int,
    omega_scores: dict[str, float],
    regex,
    threshold: float,
) -> pd.DataFrame:
    """流式扫描岗位，按锚点相关度 + 行业聚合。

    Args:
        cities: 城市列表。
        year: 年份。
        omega_scores: skill -> ωsAI。
        regex: 技能合并正则。
        threshold: 岗位 AI 判定阈值。

    Returns:
        DataFrame，列 city / industry / total / ai_jobs / ai_rate。
    """
    params = eps_conn_params()
    stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"total": 0, "ai_jobs": 0}
    )
    conn = psycopg2.connect(**params)
    try:
        cur = conn.cursor()
        for city in cities:
            shard = GD_SHARDS[city]
            ent_shard = shard.replace("job_", "ent_", 1)
            sql = f"""
                SELECT j.job_description, e.industry_code
                FROM public.{shard} j
                LEFT JOIN LATERAL (
                    SELECT industry_code FROM public.{ent_shard}
                    WHERE recruit_id = j.recruit_id
                    LIMIT 1
                ) e ON true
                WHERE substr(j.publish_time, 1, 4) = %s
                  AND j.job_description IS NOT NULL AND j.job_description != ''
                  AND j.position IS NOT NULL AND j.position != ''
            """
            cur.execute(sql, (str(year),))
            total = 0
            ai_total = 0
            for (desc, ind_code) in cur:
                desc = str(desc or "")
                industry = classify_industry(ind_code)
                key = (city, industry)
                stats[key]["total"] += 1
                total += 1
                skills = extract_skills_fast(desc, regex)
                scored = [omega_scores[s] for s in skills if s in omega_scores]
                omega_j = sum(scored) / len(scored) if scored else 0.0
                if omega_j >= threshold:
                    stats[key]["ai_jobs"] += 1
                    ai_total += 1
                if total % 1000000 == 0:
                    logger.info("%s %d 已处理 %d 条, AI %d",
                                city, year, total, ai_total)
            logger.info("%s %d 年完成: %d 条, AI %d", city, year, total, ai_total)
    finally:
        conn.close()
    rows = [
        {"city": c, "industry": ind, **s,
         "ai_rate": s["ai_jobs"] / s["total"] if s["total"] else 0.0}
        for (c, ind), s in stats.items()
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """锚点法行业渗透率入口。"""
    parser = argparse.ArgumentParser(description="锚点法行业渗透率")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--threshold", type=float, default=DEFAULT_OMEGA_THRESHOLD)
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    parser.add_argument("--omega-file", type=str,
                        default=DEFAULT_OMEGA_SNAPSHOT)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_anchor_industry.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    # 复用已算好的 ωsAI
    omega_path = Path(args.omega_file)
    if not omega_path.exists():
        omega_path = paths.project_root / args.omega_file
    omega_scores = json.loads(omega_path.read_text(encoding="utf-8"))
    logger.info("ωsAI 加载: %d 个技能", len(omega_scores))

    skills = load_merged_skills(include_llm=True)
    regex = build_skill_regex(skills)

    df = compute_industry_penetration(cities, args.year, omega_scores, regex, args.threshold)

    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ai_penetration_anchor_industry_{timestamp}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("结果已写入: %s", out_path)

    # 终端预览：跨城市行业 Top
    g = df.groupby("industry").agg(
        total=("total", "sum"), ai_jobs=("ai_jobs", "sum")
    ).reset_index()
    g["rate"] = g["ai_jobs"] / g["total"] * 100
    g = g.sort_values("rate", ascending=False)
    print("\n=== 锚点法 %d 年 行业 AI 渗透率 Top20 ===" % args.year)
    print(g.head(20).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
