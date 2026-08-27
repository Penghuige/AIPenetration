"""AI 岗位判定抽样复核：检查加权判定法的多选/漏选。

从指定城市分片表随机抽样指定年份的岗位（position + job_description），
用加权评分法（ai_scoring：关键词/技能词带权重，得分 >= 阈值）判定，
按五桶输出样本明细供人工复核：

- BOTH：关键词与技能各自单独达标
- KW：仅关键词达标
- SK：仅技能达标
- WEAK：单项都不达标但合计达标（弱词相互佐证，重点复核歧义词）
- NON：均未达标

样本明细写入 output/ai_penetration/sample_review_{city}_{year}.txt。

使用示例::

    python -m src.ai_penetration.sample_check --city 广州市 --years 2024,2019 --n 20000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from config.paths import get_project_paths

from .ai_scoring import AI_SCORE_THRESHOLD, match_keywords_scored, match_skills_scored
from .load_guangdong import GD_SHARDS, get_eps_engine

_DESC_SNIPPET_LEN = 80
_NON_CAP = 400


def sample_review(
    city: str,
    years: list[int],
    n: int,
) -> Path:
    """抽样并输出复核明细。

    Args:
        city: 城市名（GD_SHARDS 键）。
        years: 要抽样的年份列表。
        n: 每年抽样条数。

    Returns:
        复核明细文件路径。
    """
    shard = GD_SHARDS[city]
    engine = get_eps_engine()

    out_dir = get_project_paths().output_dir / "ai_penetration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        # ORDER BY random() 做随机抽样（过滤脏数据后）
        sql = f"""
            SELECT position, job_description
            FROM public.{shard}
            WHERE substr(publish_time, 1, 4) = :yr
              AND job_description IS NOT NULL AND job_description != ''
              AND position IS NOT NULL AND position != ''
            ORDER BY random()
            LIMIT :n
        """
        rows = engine.connect().execute(
            __import__("sqlalchemy").text(sql),
            {"yr": str(year), "n": int(n)},
        ).fetchall()
        print(f"[{city} {year}] 抽样 {len(rows)} 条")

        buckets = {"KW": [], "SK": [], "BOTH": [], "WEAK": [], "NON": []}
        for position, desc in rows:
            position = str(position or "")
            desc = str(desc or "")
            kw_hits, kw_score = match_keywords_scored(position)
            sk_hits, sk_score = match_skills_scored(desc)
            combined = kw_score + sk_score
            if combined < AI_SCORE_THRESHOLD:
                buckets["NON"].append((position, desc, kw_hits, sk_hits, kw_score, sk_score))
            elif kw_score >= AI_SCORE_THRESHOLD and sk_score >= AI_SCORE_THRESHOLD:
                buckets["BOTH"].append((position, desc, kw_hits, sk_hits, kw_score, sk_score))
            elif kw_score >= AI_SCORE_THRESHOLD:
                buckets["KW"].append((position, desc, kw_hits, sk_hits, kw_score, sk_score))
            elif sk_score >= AI_SCORE_THRESHOLD:
                buckets["SK"].append((position, desc, kw_hits, sk_hits, kw_score, sk_score))
            else:
                buckets["WEAK"].append((position, desc, kw_hits, sk_hits, kw_score, sk_score))

        ai_buckets = ("BOTH", "KW", "SK", "WEAK")
        ai_total = sum(len(buckets[b]) for b in ai_buckets)
        print(f"  AI 判定（both|kw|sk|weak|none）: "
              f"{len(buckets['BOTH'])} | {len(buckets['KW'])} | {len(buckets['SK'])} | "
              f"{len(buckets['WEAK'])} | {len(buckets['NON'])}")

        out_path = out_dir / f"sample_review_{city}_{year}.txt"
        lines = [
            f"# {city} {year} AI 判定抽样复核（n={len(rows)}）",
            f"- AI 岗位合计: {ai_total} ({ai_total / len(rows) * 100:.2f}%)",
            f"- 关键词率: {(len(buckets['BOTH']) + len(buckets['KW'])) / len(rows) * 100:.2f}%",
            f"- AI技能率: {(len(buckets['BOTH']) + len(buckets['SK'])) / len(rows) * 100:.2f}%",
            "",
        ]
        for bucket in ai_buckets:
            lines.append(f"===== {bucket}（{len(buckets[bucket])}）=====")
            for position, desc, kw_hits, sk_hits, kw_score, sk_score in buckets[bucket]:
                kw_str = "、".join(f"{t}({w:.0f})" for t, w in kw_hits) or "-"
                sk_str = "、".join(f"{t}({w:.0f})" for t, w in sk_hits) or "-"
                snippet = desc[: _DESC_SNIPPET_LEN].replace("\n", " ")
                lines.append(
                    f"[kw:{kw_str}] [sk:{sk_str}] "
                    f"kw={kw_score:.0f} sk={sk_score:.0f} | {position} | {snippet}"
                )
            lines.append("")
        lines.append(f"===== NON（共 {len(buckets['NON'])}，随机展示 {_NON_CAP}）=====")
        for position, desc, _, _, _, _ in random.sample(
            buckets["NON"], min(len(buckets["NON"]), _NON_CAP)
        ):
            snippet = desc[: _DESC_SNIPPET_LEN].replace("\n", " ")
            lines.append(f"{position} | {snippet}")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  复核明细: {out_path}")
    return out_dir


def main() -> None:
    """抽样复核入口。"""
    parser = argparse.ArgumentParser(description="AI 岗位判定抽样复核")
    parser.add_argument("--city", type=str, default="广州市",
                        help="城市名（GD_SHARDS 键）")
    parser.add_argument("--years", type=str, default="2024,2019",
                        help="抽样年份，逗号分隔")
    parser.add_argument("--n", type=int, default=20000,
                        help="每年抽样条数")
    args = parser.parse_args()
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    sample_review(args.city, years, args.n)


if __name__ == "__main__":
    main()
