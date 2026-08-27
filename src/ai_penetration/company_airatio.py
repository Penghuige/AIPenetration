"""公司级 AI 投入强度（AIRatio）——对齐论文核心产出。

AIRatio = 公司 AI 岗位数 / 公司总岗位数（公司-年度层面）。

数据链：job（岗位）→ ent（recruit_id 关联）→ company（company_id / company_name）。
AI 判定先用方法 A（is_ai_job，岗位名+描述加权，快速），可扩展为融合口径。

为控制噪声，只输出发布数 >= min_jobs 的公司。

使用示例::

    python -m src.ai_penetration.company_airatio --year 2024 --min-jobs 10
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
from src.ai_penetration.ai_scoring import is_ai_job

from .common import (
    DEFAULT_OMEGA_SNAPSHOT,
    eps_conn_params,
    resolve_artifact_path,
    setup_logging,
)
from .load_guangdong import GD_SHARDS

logger = logging.getLogger("ai_penetration.company_airatio")


def _cp_path(method: str) -> Path:
    """断点文件路径（按 AI 判定方法区分，避免混用两种口径）。"""
    return (
        get_project_paths().output_dir
        / "ai_penetration"
        / f"company_airatio_cp_{method}.json"
    )


def _load_cp(method: str) -> tuple[dict, set]:
    """加载指定方法的断点（stats + 已完成 city|year）。

    Args:
        method: 当前运行的 AI 判定方法（'a' 或 'fused'）。断点内记录的
            方法与当前不一致时视为不兼容，返回空断点并告警。

    Returns:
        (stats 字典, 已完成 city|year 集合)。
    """
    path = _cp_path(method)
    if not path.exists():
        return {}, set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cp_method = str(data.get("method", ""))
        if cp_method != method:
            logger.warning(
                "断点方法 %s 与当前 --method %s 不一致，忽略该断点重新计算",
                cp_method or "(未记录)", method,
            )
            return {}, set()
        stats = {tuple(k.split("|")): v for k, v in data["stats"].items()}
        stats = {(int(y), c): v for (y, c), v in stats.items()}
        completed = set(data.get("completed", []))
        logger.info("断点加载: %d 公司年份, 已完成 %d 个城市年份",
                    len(stats), len(completed))
        return stats, completed
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("断点文件损坏，忽略: %s", path)
        return {}, set()


def _save_cp(stats: dict, completed: set, method: str) -> None:
    """保存断点（含判定方法标记）。"""
    path = _cp_path(method)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "method": method,
        "stats": {"%d|%s" % (y, c): v for (y, c), v in stats.items()},
        "completed": sorted(completed),
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """公司 AIRatio 计算入口。"""
    parser = argparse.ArgumentParser(description="公司级 AI 投入强度")
    parser.add_argument("--year-start", type=int, default=2024)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--years", type=str, default="",
                        help="指定年份，逗号分隔（覆盖 year-start/end）")
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    parser.add_argument("--min-jobs", type=int, default=10,
                        help="公司最少发布数（低于则不输出，控制噪声）")
    parser.add_argument("--method", type=str, default="a", choices=["a", "fused"],
                        help="AI 判定方法：a=方法A加权, fused=融合(A或B-过滤)")
    parser.add_argument("--omega-file", type=str, default=DEFAULT_OMEGA_SNAPSHOT)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.log_dir / "ai_penetration_company.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    # AI 判定函数（签名统一为 (pos, desc) -> bool）：方法 A 或融合口径
    if args.method == "fused":
        from .skill_ai_anchor import (
            build_skill_regex,
            is_ai_fused,
            load_merged_skills,
        )

        omega_path = resolve_artifact_path(args.omega_file, artifact="ωsAI 分数快照")
        omega_scores = json.loads(omega_path.read_text(encoding="utf-8"))
        fused_regex = build_skill_regex(load_merged_skills(include_llm=True))
        logger.info("融合模式: ωsAI %d 技能", len(omega_scores))

        def ai_judge(pos: str, desc: str) -> bool:
            """融合口径判定：方法 A 判 AI 或 B（ωjAI≥0.15 且 maxω≥0.5）。"""
            return is_ai_fused(pos, desc, omega_scores, fused_regex)[2]
    else:
        def ai_judge(pos: str, desc: str) -> bool:
            """方法 A 加权判定（岗位名 + 描述评分达到阈值）。"""
            return is_ai_job(pos, desc)

    # {(year, company_id): {"name": ..., "total": n, "ai": n}}
    loaded_stats, completed = _load_cp(args.method)
    stats: dict[tuple[int, str], dict] = defaultdict(
        lambda: {"name": "", "total": 0, "ai": 0}
    )
    stats.update(loaded_stats)
    conn = psycopg2.connect(**eps_conn_params())
    try:
        years = [int(y) for y in args.years.split(",") if y.strip()]
        if not years:
            years = list(range(args.year_start, args.year_end + 1))
        for year in years:
            for city in cities:
                cp_key = f"{city}|{year}"
                if cp_key in completed:
                    logger.info("%s %d 已有断点，跳过", city, year)
                    continue
                shard = GD_SHARDS[city]
                ent_shard = shard.replace("job_", "ent_", 1)
                sql = f"""
                    SELECT j.position, j.job_description, e.company_id, e.company_name
                    FROM public.{shard} j
                    LEFT JOIN LATERAL (
                        SELECT company_id, company_name FROM public.{ent_shard}
                        WHERE recruit_id = j.recruit_id
                        LIMIT 1
                    ) e ON true
                    WHERE substr(j.publish_time, 1, 4) = %s
                      AND j.job_description IS NOT NULL AND j.job_description != ''
                      AND j.position IS NOT NULL AND j.position != ''
                """
                # 会话级优化：大 join 的哈希内存（同 penetration_detail 流式实现）
                with conn.cursor() as setup_cur:
                    setup_cur.execute("SET LOCAL work_mem = '1GB'")
                # 命名游标 = 服务端游标，逐批取数，避免整结果集缓冲进 Python 内存
                cur = conn.cursor(f"airatio_{shard}_{year}")
                cur.execute(sql, (str(year),))
                batch_size = 100000
                processed = 0
                while True:
                    rows_batch = cur.fetchmany(batch_size)
                    if not rows_batch:
                        break
                    for pos, desc, cid, cname in rows_batch:
                        cid = str(cid or "")
                        if not cid:
                            continue
                        s = stats[(int(year), cid)]
                        if not s["name"]:
                            s["name"] = str(cname or "")
                        s["total"] += 1
                        if ai_judge(str(pos or ""), str(desc or "")):
                            s["ai"] += 1
                        processed += 1
                    if processed % 1000000 < batch_size:
                        logger.info("%s %d 已处理 %d 条", city, year, processed)
                cur.close()
                logger.info("%s %d 完成: %d 条", city, year, processed)
                completed.add(cp_key)
                _save_cp(stats, completed, args.method)
    finally:
        conn.close()

    # 过滤 + 计算 AIRatio
    rows = []
    for (year, cid), s in stats.items():
        if s["total"] < args.min_jobs:
            continue
        rows.append({
            "year": year, "company_id": cid, "company_name": s["name"],
            "total_jobs": s["total"], "ai_jobs": s["ai"],
            "airatio": s["ai"] / s["total"] if s["total"] else 0.0,
        })
    df = pd.DataFrame(
        rows,
        columns=["year", "company_id", "company_name", "total_jobs", "ai_jobs", "airatio"],
    )
    if not df.empty:
        df = df.sort_values(["year", "airatio"], ascending=[True, False])
    logger.info("公司数(>=%d发布): %d", args.min_jobs, len(df))

    out_dir = paths.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"company_airatio_{timestamp}.csv"
    # 报告写成功之后才清理断点：写盘失败时断点保留，可续跑
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("结果已写入: %s", out_path)
    cp = _cp_path(args.method)
    if cp.exists():
        cp.unlink()
        logger.info("已清理断点: %s", cp)

    print("\n=== 公司 AIRatio 年度汇总（发布>=%d）===" % args.min_jobs)
    summary = df.groupby("year").agg(
        公司数=("company_id", "count"),
        AIRatio均值=("airatio", "mean"),
        有AI公司=("ai_jobs", lambda x: (x > 0).sum()),
        AI公司占比=("ai_jobs", lambda x: (x > 0).mean() * 100),
        AIRatio_ge5pct=("airatio", lambda x: (x >= 0.05).sum()),
    ).reset_index()
    summary["AIRatio均值%"] = summary["AIRatio均值"] * 100
    summary["AI公司占比%"] = summary["AI公司占比"]
    print(summary[["year", "公司数", "AIRatio均值%", "有AI公司", "AI公司占比%", "AIRatio_ge5pct"]]
          .round(2).to_string(index=False))
    if args.year_start == args.year_end:
        print("\nTop 20:")
        print(df[["company_name", "total_jobs", "ai_jobs", "airatio"]]
              .head(20).to_string(index=False))


if __name__ == "__main__":
    main()
