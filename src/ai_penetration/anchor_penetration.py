"""锚点共现法 AI 渗透率（方法四：LLM 挖掘 + 技能-技能共现）。

两阶段：
1. **计算技能 AI 相关度 ωsAI**：从岗位描述样本中，用技能-技能共现（与 AI 锚点
   技能的共现比例）计算每项技能的人工智能相关度（见 skill_ai_anchor）。
2. **计算岗位 AI 相关度 ωjAI 与渗透率**：流式扫描岗位，抽取技能，岗位 AI 相关度
   = 该岗位全部技能 ωsAI 的平均值；ωjAI ≥ 阈值判为 AI 岗位，按年/城市聚合渗透率。

区别于方法一（关键词）、方法三（三层加权）：本方法完全数据驱动，不依赖手工词典
权重与岗位名，技能词表由 LLM 从数据挖掘（dicts/ai_skill_terms_llm.txt）。

使用示例::

    python -m src.ai_penetration.anchor_penetration --omega-sample 150000 --threshold 0.3
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .common import eps_conn_params, setup_logging
from .load_guangdong import GD_SHARDS
from .skill_ai_anchor import (
    build_skill_regex,
    compute_anchor_ai_scores,
    extract_skills_fast,
    load_merged_skills,
)

logger = logging.getLogger("ai_penetration.anchor")

# 默认岗位 AI 判定阈值（ωjAI）
DEFAULT_OMEGA_THRESHOLD = 0.3


def sample_jobs_skills(
    shard: str,
    year: int,
    n: int,
    regex: re.Pattern,
    batch_size: int = 100000,
) -> list[set[str]]:
    """抽样指定年份岗位的技能集（用于计算 ωsAI）。

    Args:
        shard: 分片表名。
        year: 年份。
        n: 抽样条数。
        regex: 技能合并正则。
        batch_size: 分批行数。

    Returns:
        每岗位技能集合列表。
    """
    jobs_skills: list[set[str]] = []
    conn = psycopg2.connect(**eps_conn_params())
    try:
        # 命名游标 = 服务端游标：抽样行数虽有限制，描述可达数 KB，
        # 逐批取数避免整结果集一次性缓冲进 Python 内存
        cur = conn.cursor(f"anchor_sample_{shard}_{year}")
        cur.itersize = 10000
        sql = f"""
            SELECT job_description FROM public.{shard}
            WHERE substr(publish_time, 1, 4) = %s
              AND job_description IS NOT NULL AND job_description != ''
              AND position IS NOT NULL AND position != ''
            ORDER BY random() LIMIT %s
        """
        cur.execute(sql, (str(year), int(n)))
        while True:
            batch = cur.fetchmany(10000)
            if not batch:
                break
            for (desc,) in batch:
                jobs_skills.append(extract_skills_fast(str(desc), regex))
        cur.close()
    finally:
        conn.close()
    logger.info("%s %d 年抽样 %d 条技能集", shard, year, len(jobs_skills))
    return jobs_skills


def _checkpoint_path() -> Path:
    """返回锚点渗透率断点文件路径。"""
    return get_project_paths().output_dir / "ai_penetration" / "anchor_checkpoint.json"


def _load_cp() -> dict[str, dict]:
    """加载锚点渗透率断点。"""
    path = _checkpoint_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("锚点断点: 已完成 %d 个城市年份", len(data))
        return data
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def _save_cp(done: dict[str, dict]) -> None:
    """保存锚点渗透率断点。"""
    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")


def compute_penetration_by_anchor(
    cities: list[str],
    years: list[int],
    omega_scores: dict[str, float],
    regex: re.Pattern,
    threshold: float,
) -> pd.DataFrame:
    """流式扫描岗位，按锚点相关度计算逐年渗透率（逐年断点可续跑）。

    Args:
        cities: 城市列表。
        years: 年份列表。
        omega_scores: skill -> ωsAI 映射。
        regex: 技能合并正则。
        threshold: 岗位 AI 判定阈值（ωjAI >= 阈值）。

    Returns:
        DataFrame，列 city / year / total / ai_jobs / ai_rate。
    """
    done = _load_cp()  # {"city|year": {"total","ai_jobs"}}
    rows: list[dict] = []
    conn = psycopg2.connect(**eps_conn_params())
    try:
        for city in cities:
            shard = GD_SHARDS[city]
            for year in years:
                cp_key = f"{city}|{year}"
                if cp_key in done:
                    logger.info("%s %d 年已有断点，跳过", city, year)
                    cp_total = int(done[cp_key].get("total", 0))
                    cp_ai = int(done[cp_key].get("ai_jobs", 0))
                    rows.append({
                        "city": city,
                        "year": year,
                        "total": cp_total,
                        "ai_jobs": cp_ai,
                        # 断点只存 total/ai_jobs，恢复行重算比率，
                        # 避免下游 pivot/report 出现 NaN
                        "ai_rate": cp_ai / cp_total if cp_total else 0.0,
                    })
                    continue
                sql = f"""
                    SELECT job_description FROM public.{shard}
                    WHERE substr(publish_time, 1, 4) = %s
                      AND job_description IS NOT NULL AND job_description != ''
                      AND position IS NOT NULL AND position != ''
                """
                # 命名游标 = 服务端游标，逐批取数避免整结果集缓冲进内存
                cur = conn.cursor(f"anchor_scan_{shard}_{year}")
                cur.itersize = 10000
                cur.execute(sql, (str(year),))
                total = 0
                ai = 0
                batch: list[str] = []
                while True:
                    fetched = cur.fetchmany(100000)
                    if not fetched:
                        break
                    for (desc,) in fetched:
                        total += 1
                        batch.append(str(desc))
                        if len(batch) >= 100000:
                            ai += _count_ai_batch(batch, regex, omega_scores, threshold)
                            batch = []
                cur.close()
                if batch:
                    ai += _count_ai_batch(batch, regex, omega_scores, threshold)
                done[cp_key] = {"total": total, "ai_jobs": ai}
                _save_cp(done)
                rows.append({
                    "city": city, "year": year,
                    "total": total, "ai_jobs": ai,
                    "ai_rate": ai / total if total else 0.0,
                })
                logger.info("%s %d 年: %d 条, AI %d (%.4f)",
                            city, year, total, ai, ai / total if total else 0)
    finally:
        conn.close()
    return pd.DataFrame(rows)


def _count_ai_batch(
    descs: list[str],
    regex: re.Pattern,
    omega_scores: dict[str, float],
    threshold: float,
) -> int:
    """统计一批描述中 AI 岗位数（ωjAI >= 阈值）。"""
    ai = 0
    for desc in descs:
        skills = extract_skills_fast(desc, regex)
        scored = [omega_scores[s] for s in skills if s in omega_scores]
        omega_j = sum(scored) / len(scored) if scored else 0.0
        if omega_j >= threshold:
            ai += 1
    return ai


def main() -> None:
    """锚点共现法渗透率入口。"""
    parser = argparse.ArgumentParser(description="锚点共现法 AI 渗透率")
    parser.add_argument("--omega-sample", type=int, default=150000,
                        help="计算 ωsAI 的抽样条数")
    parser.add_argument("--threshold", type=float, default=DEFAULT_OMEGA_THRESHOLD,
                        help="岗位 AI 判定阈值（ωjAI >= 阈值）")
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    parser.add_argument("--year-start", type=int, default=2014)
    parser.add_argument("--year-end", type=int, default=2024)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_anchor.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    years = list(range(args.year_start, args.year_end + 1))

    skills = load_merged_skills(include_llm=True)
    regex = build_skill_regex(skills)
    logger.info("技能词典 %d 项", len(skills))

    # 阶段一：抽样计算 ωsAI
    all_jobs_skills: list[set[str]] = []
    for city in cities:
        all_jobs_skills += sample_jobs_skills(
            GD_SHARDS[city], max(years), args.omega_sample, regex
        )
    omega_df = compute_anchor_ai_scores(all_jobs_skills)
    omega_scores = dict(zip(omega_df["skill"], omega_df["omega_ai"]))
    logger.info("ωsAI 计算完成: %d 个技能", len(omega_scores))

    # 阶段二：流式计算渗透率
    pen = compute_penetration_by_anchor(
        cities, years, omega_scores, regex, args.threshold
    )
    print("\n=== 锚点共现法渗透率（阈值 ωjAI>=%.2f）===" % args.threshold)
    print(pen.pivot_table(index="year", columns="city", values="ai_rate") * 100)

    # 输出
    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pen.to_csv(out_dir / f"ai_penetration_anchor_{timestamp}.csv",
               index=False, encoding="utf-8-sig")
    (out_dir / f"omega_ai_scores_{timestamp}.json").write_text(
        json.dumps(dict(zip(omega_df["skill"], omega_df["omega_ai"])),
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    logger.info("结果已写入: %s", out_dir / f"ai_penetration_anchor_{timestamp}.csv")

    # 清理断点
    cp = _checkpoint_path()
    if cp.exists():
        cp.unlink()
        logger.info("已清理断点: %s", cp)


if __name__ == "__main__":
    main()
