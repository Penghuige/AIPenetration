"""融合判定 21 市全量渗透率（A/B/C 三口径，多进程并行，城市级断点）。

对广东省 21 市分片表做一次全表扫描（ctid 切片并行），逐岗位融合判定
（A 加权 / B 锚点过滤 / C 融合），按 城市 × 年份 聚合三口径渗透率。

设计：
- 不按年过滤、不 join ent：单遍扫描聚合全部年份（口径与 substr(publish_time,1,4) 一致）
- 每市 8 连接 ctid 切片扫描 + 24 进程评分（fused_industry 同款架构）
- 城市级断点：每市完成后写 JSON，中断续跑跳过已完成城市
- 按表大小升序处理（小市先出结果）

使用示例::

    python -m src.ai_penetration.fused_cities --workers 24 --scan-workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .common import DEFAULT_OMEGA_SNAPSHOT, eps_conn_params, resolve_artifact_path, setup_logging

from .load_guangdong import GD_SHARDS

logger = logging.getLogger("ai_penetration.fused_cities")

_WORKER: dict = {}
_YEAR_RE = re.compile(r"^\s*(\d{4})")




def _checkpoint_path() -> Path:
    """返回城市级断点文件路径。"""
    return get_project_paths().output_dir / "ai_penetration" / "fused_cities_checkpoint.json"


def _load_checkpoint() -> dict[str, dict[str, dict]]:
    """加载断点 {city: {year: stats}}；损坏返回空。"""
    path = _checkpoint_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("断点损坏，忽略: %s", exc)
        return {}


def _save_checkpoint(data: dict[str, dict[str, dict]]) -> None:
    """保存断点。"""
    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _init_worker(omega_path: str) -> None:
    """worker 初始化：加载 ωsAI 与技能正则（每 worker 一次）。"""
    from .skill_ai_anchor import build_skill_regex, load_merged_skills

    _WORKER["omega"] = json.loads(
        resolve_artifact_path(omega_path, artifact="ωsAI").read_text(encoding="utf-8"))
    _WORKER["regex"] = build_skill_regex(load_merged_skills(include_llm=True))


def _score_batch(rows: list[tuple]) -> dict[int, dict]:
    """评分一批 (position, desc, publish_time)，按年聚合三口径计数。

    年份口径：publish_time 前 4 位数字（与 substr(publish_time,1,4) 一致），
    无法解析的行丢弃。

    Args:
        rows: 数据库批次行。

    Returns:
        {year: {"total","a","b","fused"}}。
    """
    from .skill_ai_anchor import is_ai_fused

    out: dict[int, dict] = defaultdict(
        lambda: {"total": 0, "a": 0, "b": 0, "fused": 0}
    )
    for position, desc, pub in rows:
        m = _YEAR_RE.match(str(pub or ""))
        if not m:
            continue
        year = int(m.group(1))
        s = out[year]
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


def _scan_city(
    city: str,
    shard: str,
    params: dict,
    pool: ProcessPoolExecutor,
    scan_workers: int,
    lock: threading.Lock,
    years_out: dict[int, dict],
) -> None:
    """单市 ctid 多连接并行扫描，评分结果按年合并到 years_out。"""
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

    def scan_range(seg: int, lo: int, hi: int) -> None:
        p = dict(params)
        conn = psycopg2.connect(**p)
        try:
            cur = conn.cursor(f"fused_city_{city}_{seg}")
            sql = f"""
                SELECT j.position, j.job_description, j.publish_time
                FROM public.{shard} j
                WHERE j.ctid >= '({lo},0)'::tid AND j.ctid < '({hi},0)'::tid
                  AND j.job_description IS NOT NULL AND j.job_description != ''
                  AND j.position IS NOT NULL AND j.position != ''
            """
            cur.execute(sql)
            seg_total = 0
            # 流水线：每扫描线程保持 pipeline 批在途，吃满评分进程池
            pipeline = 4
            pending: list = []
            while True:
                batch = cur.fetchmany(50000)
                if not batch:
                    break
                pending.append(pool.submit(_score_batch, batch))
                if len(pending) >= pipeline:
                    partial = pending.pop(0).result()
                    with lock:
                        for year, s in partial.items():
                            acc = years_out.setdefault(
                                year, {"total": 0, "a": 0, "b": 0, "fused": 0}
                            )
                            for k in ("total", "a", "b", "fused"):
                                acc[k] += s[k]
                seg_total += len(batch)
            for f in pending:
                partial = f.result()
                with lock:
                    for year, s in partial.items():
                        acc = years_out.setdefault(
                            year, {"total": 0, "a": 0, "b": 0, "fused": 0}
                        )
                        for k in ("total", "a", "b", "fused"):
                            acc[k] += s[k]
            logger.info("%s 段%d 块[%d,%d) 完成：%d 条", city, seg, lo, hi, seg_total)
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


def main() -> None:
    """21 市融合渗透率入口。"""
    parser = argparse.ArgumentParser(description="融合判定 21 市全量渗透率")
    parser.add_argument("--cities", type=str, default="",
                        help="限定城市逗号分隔；空=全部 21 市（按表大小升序）")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--omega-file", type=str,
                        default=DEFAULT_OMEGA_SNAPSHOT)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_fused_cities.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = eps_conn_params()

    # 城市列表：指定则用之；否则全部 21 市按表大小升序
    if args.cities:
        cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    else:
        conn = psycopg2.connect(**params)
        try:
            cur = conn.cursor()
            sizes = []
            for city, shard in GD_SHARDS.items():
                cur.execute(
                    "SELECT pg_relation_size(%s)", (f"public.{shard}",)
                )
                sizes.append((cur.fetchone()[0], city))
            cities = [c for _, c in sorted(sizes)]
        finally:
            conn.close()
    logger.info("处理顺序: %s", cities)

    done = _load_checkpoint()  # {city: {year_str: stats}}
    lock = threading.Lock()
    results: dict[str, dict[int, dict]] = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.omega_file,)) as pool:
        for city in cities:
            if city in done:
                logger.info("%s 已有断点，跳过", city)
                continue
            shard = GD_SHARDS[city]
            logger.info("=== %s 开始 ===", city)
            years_out: dict[int, dict] = {}
            _scan_city(city, shard, params, pool, args.scan_workers, lock, years_out)
            done[city] = {str(y): s for y, s in years_out.items()}
            _save_checkpoint(done)
            logger.info("=== %s 完成：%d 年，断点已存 ===", city, len(years_out))

    # 汇总输出（含 A/B 拆分分解：both = a+b-fused）
    rows = []
    for city, years in done.items():
        for ys, s in years.items():
            n = s["total"]
            ab_both = s["a"] + s["b"] - s["fused"]
            rows.append({
                "city": city, "year": int(ys), "total": n,
                "a_jobs": s["a"], "b_jobs": s["b"], "fused_jobs": s["fused"],
                "ab_both_jobs": ab_both,
                "a_only_jobs": s["a"] - ab_both,
                "b_only_jobs": s["b"] - ab_both,
                "a_rate": s["a"] / n if n else 0.0,
                "b_rate": s["b"] / n if n else 0.0,
                "fused_rate": s["fused"] / n if n else 0.0,
            })
    df = pd.DataFrame(rows).sort_values(["city", "year"])
    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ai_penetration_fused_cities_{timestamp}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("结果已写入: %s", out_path)

    cp = _checkpoint_path()
    if cp.exists():
        cp.unlink()
        logger.info("断点已清理")

    print("\n=== 21 市 2024 融合率 ===")
    y24 = df[df["year"] == 2024].sort_values("fused_rate", ascending=False)
    print(y24[["city", "total", "a_rate", "b_rate", "fused_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
