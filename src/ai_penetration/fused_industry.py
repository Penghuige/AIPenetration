"""融合判定行业渗透率（方法 A / B / C 三口径，多进程并行）。

复用 skill_ai_anchor.is_ai_fused（A/B/C 三判定），流式扫描岗位并经
LATERAL join 关联 ent 企业表取行业大类，一次扫描同时输出三口径的
城市 × 行业 聚合。

加速设计（面向多核机器）：
- 主进程按城市顺序持有 server-side cursor 读批（IO 密集）
- 每批提交给 ProcessPoolExecutor 并行评分（CPU 密集的正则提取 + 判定）
- worker 经 initializer 加载 ωsAI 与技能正则一次，避免重复构建

使用示例::

    python -m src.ai_penetration.fused_industry --year 2024 --workers 24
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from config.paths import get_project_paths

from .common import DEFAULT_OMEGA_SNAPSHOT, eps_conn_params, setup_logging

from .industry_classification import classify_industry
from .load_guangdong import GD_SHARDS

logger = logging.getLogger("ai_penetration.fused_industry")

# worker 全局状态（initializer 一次性加载）
_WORKER: dict = {}




def _init_worker(omega_path: str) -> None:
    """worker 初始化：加载 ωsAI、构建技能正则（每 worker 一次）。"""
    from .skill_ai_anchor import build_skill_regex, load_merged_skills

    omega = json.loads(Path(omega_path).read_text(encoding="utf-8"))
    _WORKER["omega"] = omega
    _WORKER["regex"] = build_skill_regex(load_merged_skills(include_llm=True))


def _score_batch(rows: list[tuple]) -> dict:
    """评分一批 (position, desc, industry_code)，返回 {industry: 计数}。

    Args:
        rows: 数据库批次行。

    Returns:
        {industry: {"total","a","b","fused"}} 局部聚合。
    """
    from .skill_ai_anchor import is_ai_fused

    out: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "a": 0, "b": 0, "fused": 0}
    )
    for position, desc, ind_code in rows:
        industry = classify_industry(ind_code)
        s = out[industry]
        s["total"] += 1
        a_ai, b_ai, f_ai = is_ai_fused(
            str(position or ""), str(desc or ""), _WORKER["omega"], _WORKER["regex"]
        )
        if a_ai:
            s["a"] += 1
        if b_ai:
            s["b"] += 1
        if f_ai:
            s["fused"] += 1
    return dict(out)


def _scan_city_shard(
    city: str,
    shard: str,
    year: int,
    params: dict,
    pool: ProcessPoolExecutor,
    scan_workers: int,
    scan_id: int,
    lock: threading.Lock,
    stats_merge: dict,
) -> None:
    """单分片多线程 ctid 切片扫描：每线程独立连接扫一段物理块区间。

    Args:
        city: 城市名。
        shard: 岗位分片表名。
        year: 年份。
        params: PG 连接参数。
        pool: 共享评分进程池。
        scan_workers: 扫描切片数。
        scan_id: 当前城市序号（日志用）。
        lock: 聚合锁。
        stats_merge: 共享聚合字典（就地更新）。
    """
    import psycopg2.extras

    # 取总块数（relpages 可能为 stale 0，用关系大小精确计算）
    conn0 = psycopg2.connect(**params)
    try:
        cur0 = conn0.cursor()
        cur0.execute(
            "SELECT pg_relation_size(%s) / current_setting('block_size')::int",
            (f"public.{shard}",),
        )
        total_blocks = max(1, int(cur0.fetchone()[0]))
    finally:
        conn0.close()

    step = (total_blocks + scan_workers - 1) // scan_workers
    ranges = [
        (i * step, min((i + 1) * step, total_blocks))
        for i in range(scan_workers)
        if i * step < total_blocks
    ]
    ent_shard = shard.replace("job_", "ent_", 1)

    def scan_range(seg: int, lo: int, hi: int) -> None:
        """扫描一个 ctid 块区间。"""
        p = dict(params)
        conn = psycopg2.connect(**p)
        try:
            cur = conn.cursor(f"fused_ind_{seg}")
            cur.itersize = 0
            sql = f"""
                SELECT j.position, j.job_description, e.industry_code
                FROM public.{shard} j
                LEFT JOIN LATERAL (
                    SELECT industry_code FROM public.{ent_shard}
                    WHERE recruit_id = j.recruit_id
                    LIMIT 1
                ) e ON true
                WHERE j.ctid >= '({lo},0)'::tid AND j.ctid < '({hi},0)'::tid
                  AND substr(j.publish_time, 1, 4) = %s
                  AND j.job_description IS NOT NULL AND j.job_description != ''
                  AND j.position IS NOT NULL AND j.position != ''
            """
            cur.execute(sql, (str(year),))
            seg_total = 0
            while True:
                batch = cur.fetchmany(50000)
                if not batch:
                    break
                partial = pool.submit(_score_batch, batch).result()
                with lock:
                    for industry, s in partial.items():
                        acc = stats_merge[(city, industry)]
                        for k in ("total", "a", "b", "fused"):
                            acc[k] += s[k]
                seg_total += len(batch)
            logger.info("[%d] %s 段%d 块[%d,%d) 完成：%d 条",
                        scan_id, city, seg, lo, hi, seg_total)
        finally:
            conn.close()

    threads = [
        threading.Thread(target=scan_range, args=(i, lo, hi), daemon=True)
        for i, (lo, hi) in enumerate(ranges)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def compute_fused_industry(
    cities: list[str],
    year: int,
    omega_path: str,
    workers: int,
    scan_workers: int = 8,
) -> pd.DataFrame:
    """多连接 ctid 并行扫描 + 多进程评分，输出 A/B/C 三口径聚合。

    Args:
        cities: 城市列表。
        year: 年份。
        omega_path: ωsAI JSON 路径。
        workers: 评分进程池大小。
        scan_workers: 每城市扫描切片数（并行 PG 连接数）。

    Returns:
        DataFrame，列 city / industry / total / a_jobs / b_jobs / fused_jobs 及率。
    """
    params = eps_conn_params()
    stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"total": 0, "a": 0, "b": 0, "fused": 0}
    )
    lock = threading.Lock()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(omega_path,)) as pool:
        for idx, city in enumerate(cities):
            shard = GD_SHARDS[city]
            logger.info("%s 开始并行扫描（%d 切片）...", city, scan_workers)
            _scan_city_shard(city, shard, year, params, pool,
                             scan_workers, idx, lock, stats)
            logger.info("%s 完成", city)
    rows = []
    for (c, ind), s in stats.items():
        n = s["total"]
        rows.append({
            "city": c, "industry": ind, "total": n,
            "a_jobs": s["a"], "b_jobs": s["b"], "fused_jobs": s["fused"],
            "a_rate": s["a"] / n if n else 0.0,
            "b_rate": s["b"] / n if n else 0.0,
            "fused_rate": s["fused"] / n if n else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """融合判定行业渗透率入口。"""
    parser = argparse.ArgumentParser(description="融合判定行业渗透率（A/B/C 三口径，多进程）")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    parser.add_argument("--workers", type=int, default=24,
                        help="评分进程池大小")
    parser.add_argument("--scan-workers", type=int, default=8,
                        help="每城市并行扫描连接数（ctid 切片）")
    parser.add_argument("--omega-file", type=str,
                        default=DEFAULT_OMEGA_SNAPSHOT)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_fused_industry.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    df = compute_fused_industry(cities, args.year, args.omega_file,
                                args.workers, args.scan_workers)

    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ai_penetration_fused_industry_{timestamp}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("结果已写入: %s", out_path)

    # 终端预览：跨城市行业三口径 Top15（样本>=500，按融合率）
    g = df.groupby("industry")[["total", "a_jobs", "b_jobs", "fused_jobs"]].sum().reset_index()
    g["a%"] = g["a_jobs"] / g["total"] * 100
    g["b%"] = g["b_jobs"] / g["total"] * 100
    g["fused%"] = g["fused_jobs"] / g["total"] * 100
    g = g[g["total"] >= 500].sort_values("fused%", ascending=False)
    print("\n=== %d 行业三口径 Top15（样本>=500，按融合率） ===" % args.year)
    print(g[["industry", "total", "a%", "b%", "fused%"]].head(15).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
