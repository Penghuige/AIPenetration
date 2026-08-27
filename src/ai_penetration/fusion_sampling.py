"""融合判定渗透率（按年抽样估算，方法 A 或 方法 B-过滤）。

两法已单独全量跑过，但融合（并集）需每岗位重叠信息，聚合结果不含。
本脚本按年抽样（每城市年份最多 5 万条），逐岗位计算
`A 判 AI 或 B(ωjAI≥0.15 且 max≥0.5) 判 AI`，得到该年融合渗透率的抽样估计。

使用示例::

    python -m src.ai_penetration.fusion_sampling --cities "广州市,深圳市" --sample-size 50000
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .common import DEFAULT_OMEGA_SNAPSHOT, eps_conn_params, setup_logging
from .load_guangdong import GD_SHARDS
from .skill_ai_anchor import is_ai_fused, load_merged_skills, build_skill_regex

logger = logging.getLogger("ai_penetration.fusion_sampling")




def main() -> None:
    """按年抽样融合渗透率入口。"""
    parser = argparse.ArgumentParser(description="融合判定渗透率（按年抽样）")
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--year-start", type=int, default=2014)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--omega-file", type=str,
                        default=DEFAULT_OMEGA_SNAPSHOT)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_fusion.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    years = list(range(args.year_start, args.year_end + 1))

    # ωsAI
    omega_path = Path(args.omega_file)
    if not omega_path.exists():
        omega_path = paths.project_root / args.omega_file
    omega_scores = json.loads(omega_path.read_text(encoding="utf-8"))
    logger.info("ωsAI 加载: %d 个技能", len(omega_scores))

    skills = load_merged_skills(include_llm=True)
    regex = build_skill_regex(skills)
    logger.info("技能词典 %d 项", len(skills))

    params = eps_conn_params()
    rows: list[dict] = []
    conn = psycopg2.connect(**params)
    try:
        cur = conn.cursor()
        for city in cities:
            shard = GD_SHARDS[city]
            for year in years:
                sql = f"""
                    SELECT position, job_description FROM public.{shard}
                    WHERE substr(publish_time, 1, 4) = %s
                      AND job_description IS NOT NULL AND job_description != ''
                      AND position IS NOT NULL AND position != ''
                    ORDER BY random() LIMIT %s
                """
                cur.execute(sql, (str(year), int(args.sample_size)))
                fetched = cur.fetchall()
                n = len(fetched)
                a_ai = 0
                fused_ai = 0
                for pos, desc in fetched:
                    pos = str(pos or "")
                    desc = str(desc or "")
                    a, b, f = is_ai_fused(pos, desc, omega_scores, regex)
                    if a:
                        a_ai += 1
                    if f:
                        fused_ai += 1
                rows.append({
                    "city": city, "year": year,
                    "sample": n, "a_ai": a_ai, "fused_ai": fused_ai,
                    "a_rate": a_ai / n if n else 0.0,
                    "fused_rate": fused_ai / n if n else 0.0,
                })
                logger.info("%s %d 年: 抽样 %d, A率=%.4f, 融合率=%.4f",
                            city, year, n, rows[-1]["a_rate"], rows[-1]["fused_rate"])
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ai_penetration_fusion_{timestamp}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("结果已写入: %s", out_path)

    # 终端展示：融合率 vs A率
    print("\n=== 融合渗透率（按年抽样）vs 方法A ===")
    for city in cities:
        print("--- %s ---" % city)
        sub = df[df["city"] == city][["year", "a_rate", "fused_rate"]]
        sub.columns = ["年份", "方法A率%", "融合率%"]
        print((sub * [1, 100, 100]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
