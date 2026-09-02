"""融合判定全国城市渗透率（自动枚举分片，跳过已跑城市，支持时限基准）。

fused_cities.py 的全国版：
- 分片枚举：从 pg_class 找全部 job_p% 表，每表取 city 字段作为城市名（不再用
  GD_SHARDS 硬编码）；空表跳过并记录。
- 广东复用：启动时预载 21 市已有全量面板 CSV 进断点，不重跑。
- 基准模式：--bench 对单表按时限测吞吐（不写结果），供跑前选型 workers/scan-workers。
- 城市级断点：中断续跑；--max-minutes 单城时限（可选保险）。

使用示例::

    # 生产
    python -m src.ai_penetration.fused_national --workers 64 --scan-workers 8
    # 基准（runner 自动调用）
    python -m src.ai_penetration.fused_national --bench job_p0021 --workers 64 --scan-workers 8 --minutes 6
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .common import DEFAULT_OMEGA_SNAPSHOT, eps_conn_params, resolve_artifact_path, setup_logging

logger = logging.getLogger("ai_penetration.fused_national")

_WORKER: dict = {}
_YEAR_RE = re.compile(r"^\s*(\d{4})")

# 广东 21 市既有全量面板（复用不重跑）
GD_PANEL_GLOB = "ai_penetration_fused_cities_*.csv"  # report_dir 下最新广东面板（预载跳过 21 市）


def _guard_single_instance() -> None:
    """单实例保护：已有更早启动的 fused_national 进程在跑则本进程退出。

    防多 runner 重生实例在 03:00 同时拉起多个全国面板互相踩踏断点。
    """
    import os
    import time

    try:
        import psutil
    except ImportError:
        return
    me = os.getpid()
    my_start = psutil.Process(me).create_time()
    for p in psutil.process_iter():
        try:
            if p.pid == me:
                continue
            if "python" not in (p.name() or "").lower():
                continue
            cmd = " ".join(p.cmdline() or [])
            if "fused_national" in cmd and "bench" not in cmd:
                if p.create_time() < my_start:
                    logger.warning("检测到更早启动的 fused_national PID=%s，本实例退出"
                                   "（单实例保护）", p.pid)
                    print("NATIONAL_SKIPPED: another instance already running")
                    sys.exit(0)
        except (psutil.Error, OSError):
            continue




def _checkpoint_path() -> Path:
    """返回全国断点文件路径。"""
    return get_project_paths().output_dir / "ai_penetration" / "fused_national_checkpoint.json"


def _load_checkpoint() -> dict:
    """加载断点 {city: {year: stats}}。"""
    path = _checkpoint_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("断点损坏，忽略: %s", exc)
        return {}


def _save_checkpoint(done: dict) -> None:
    """保存断点。"""
    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")


def _preload_gd() -> dict:
    """从广东 21 市全量面板预载 done（跳过重跑）。

    Returns:
        {city: {year_str: {total,a,b,fused}}}。
    """
    report_dir = get_project_paths().report_dir
    candidates = sorted(report_dir.glob(GD_PANEL_GLOB))
    if not candidates:
        logger.warning("广东面板 CSV 不存在于 %s，将重跑 21 市", report_dir)
        return {}
    path = candidates[-1]
    logger.info("广东面板预载文件: %s", path.name)
    df = pd.read_csv(path)
    done: dict = defaultdict(dict)
    for r in df.itertuples():
        done[r.city][str(r.year)] = {
            "total": int(r.total), "a": int(r.a_jobs),
            "b": int(r.b_jobs), "fused": int(r.fused_jobs),
        }
    logger.info("预载广东面板: %d 市 %d 城-年", df["city"].nunique(), len(df))
    return {k: dict(v) for k, v in done.items()}


def enumerate_national_tables(params: dict) -> list[tuple[str, str, float]]:
    """枚举全国 job 分片表并按大小降序。

    Args:
        params: eps PG 连接参数。

    Returns:
        [(shard, city, size_bytes)]，空表/无 city 表跳过并告警。
    """
    conn = psycopg2.connect(**params)
    out: list[tuple[str, str, float]] = []
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT c.relname, pg_relation_size(c.oid)
            FROM pg_class c JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' AND c.relname LIKE 'job_p%' AND c.relkind = 'r'
            ORDER BY pg_relation_size(c.oid) DESC
        """)
        tables = cur.fetchall()
        for shard, size in tables:
            try:
                cur.execute(f"SELECT city FROM public.{shard} LIMIT 1")
                row = cur.fetchone()
            except psycopg2.Error:
                conn.rollback()
                row = None
            if not row or not str(row[0] or "").strip():
                logger.warning("跳过无城市名/空表: %s", shard)
                continue
            out.append((shard, str(row[0]).strip(), float(size)))
    finally:
        conn.close()
    logger.info("全国分片枚举: 可用 %d / 总 %d", len(out), len(tables))
    return out


def _init_worker(omega_path: str) -> None:
    """worker 初始化：加载 ωsAI 与技能正则。"""
    from .skill_ai_anchor import build_skill_regex, load_merged_skills

    _WORKER["omega"] = json.loads(
        resolve_artifact_path(omega_path, artifact="ωsAI").read_text(encoding="utf-8"))
    _WORKER["regex"] = build_skill_regex(load_merged_skills(include_llm=True))


def _score_batch(rows: list[tuple]) -> dict:
    """评分一批并返回 {year: {total,a,b,fused}} 局部聚合。"""
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


def _run_city(
    city: str,
    shard: str,
    params: dict,
    pool: ProcessPoolExecutor,
    scan_workers: int,
    lock: threading.Lock,
    years_out: dict,
    deadline: float | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """ctid 多连接并行扫描单城，评分聚合进 years_out。

    Args:
        city: 城市名。
        shard: 分片表名。
        params: PG 连接参数。
        pool: 评分进程池。
        scan_workers: ctid 切片数。
        lock: 聚合锁。
        years_out: 输出聚合（就地更新）。
        deadline: Unix 时间戳硬限（超时中止本城，基准用）。
        stop_event: 中止信号。

    Returns:
        处理行数（deadline 中止时为部分行数）。
    """
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
        for i in range(scan_workers) if i * step < total_blocks
    ]
    counter = {"rows": 0}
    c_lock = threading.Lock()

    def scan_range(seg: int, lo: int, hi: int) -> None:
        conn = psycopg2.connect(**params)
        try:
            # SET LOCAL 必须走匿名 cursor：命名游标会把首条语句包装成
            # DECLARE ... FOR <stmt>，SET 语法不兼容（全国移植时的 bug）
            setup_cur = conn.cursor()
            setup_cur.execute("SET LOCAL work_mem = '256MB'")
            setup_cur.close()
            cur = conn.cursor(f"nat_{shard}_{seg}")
            sql = f"""
                SELECT position, job_description, publish_time
                FROM public.{shard}
                WHERE ctid >= '({lo},0)'::tid AND ctid < '({hi},0)'::tid
                  AND job_description IS NOT NULL AND job_description != ''
                  AND position IS NOT NULL AND position != ''
            """
            cur.execute(sql)
            pending: list = []
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if deadline is not None and time.time() > deadline:
                    if stop_event is not None:
                        stop_event.set()
                    break
                batch = cur.fetchmany(50000)
                if not batch:
                    break
                with c_lock:
                    counter["rows"] += len(batch)  # 取到即计数（小表不足4批时也计）
                pending.append(pool.submit(_score_batch, batch))
                if len(pending) >= 4:
                    partial = pending.pop(0).result()
                    with lock:
                        for year, s in partial.items():
                            acc = years_out.setdefault(
                                year, {"total": 0, "a": 0, "b": 0, "fused": 0}
                            )
                            for k in ("total", "a", "b", "fused"):
                                acc[k] += s[k]
            for f in pending:
                partial = f.result()
                with lock:
                    for year, s in partial.items():
                        acc = years_out.setdefault(
                            year, {"total": 0, "a": 0, "b": 0, "fused": 0}
                        )
                        for k in ("total", "a", "b", "fused"):
                            acc[k] += s[k]
        except psycopg2.Error as exc:
            logger.error("%s 段%d 扫描异常: %s", city, seg, str(exc)[:150])
            if stop_event is not None:
                stop_event.set()
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
    return counter["rows"]


def _run_city_client(
    city: str,
    shard: str,
    params: dict,
    pool: ProcessPoolExecutor,
    lock: threading.Lock,
    years_out: dict,
) -> int:
    """小表快速通道：客户端游标直读（免服务端 tuplestore 物化）。

    1.5GB 以下的表整表流式拉到客户端内存（结果约 2-3GB 上限），
    分批提交评分。省去命名游标"写临时文件再读回"的 HDD 双倍开销。

    Args:
        city: 城市名。
        shard: 分片表名。
        params: PG 连接参数。
        pool: 评分进程池。
        lock: 聚合锁。
        years_out: 聚合输出（就地更新）。

    Returns:
        读取行数。
    """
    conn = psycopg2.connect(**params)
    total = 0
    try:
        cur = conn.cursor()
        cur.execute("SET LOCAL work_mem = '256MB'")
        # 普通 SELECT（非 DECLARE）可触发 Parallel Seq Scan：多 worker 并行
        # 读表与 TOAST 抽取，是小表逐城耗时的真正解药
        cur.execute("SET LOCAL max_parallel_workers_per_gather = 4")
        cur.itersize = 50000
        sql = f"""
            SELECT position, job_description, publish_time
            FROM public.{shard}
            WHERE job_description IS NOT NULL AND job_description != ''
              AND position IS NOT NULL AND position != ''
        """
        cur.execute(sql)
        pending: list = []
        # itersize 生效需逐批迭代结果集
        for batch in _iter_batches(cur, 50000):
            total += len(batch)
            pending.append(pool.submit(_score_batch, batch))
            if len(pending) >= 3:
                partial = pending.pop(0).result()
                with lock:
                    for year, s in partial.items():
                        acc = years_out.setdefault(
                            year, {"total": 0, "a": 0, "b": 0, "fused": 0}
                        )
                        for k in ("total", "a", "b", "fused"):
                            acc[k] += s[k]
        for f in pending:
            partial = f.result()
            with lock:
                for year, s in partial.items():
                    acc = years_out.setdefault(
                        year, {"total": 0, "a": 0, "b": 0, "fused": 0}
                    )
                    for k in ("total", "a", "b", "fused"):
                        acc[k] += s[k]
    except psycopg2.Error as exc:
        logger.error("%s 客户端扫描异常: %s", city, str(exc)[:150])
    finally:
        conn.close()
    return total


def _iter_batches(cur, size: int):
    """按批迭代游标结果（itersize 生效，避免整表进内存）。"""
    while True:
        batch = cur.fetchmany(size)
        if not batch:
            return
        yield batch


def bench(shard: str, workers: int, scan_workers: int, minutes: float,
          omega_file: str) -> float:
    """对单表跑时限基准，返回吞吐（行/秒）。

    Args:
        shard: 分片表名。
        workers: 评分进程数。
        scan_workers: ctid 切片数。
        minutes: 时限。
        omega_file: ωsAI 路径。

    Returns:
        处理行数 / 用时秒。
    """
    params = eps_conn_params()
    # 取一个 city 名仅用于日志
    stop_event = threading.Event()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(omega_file,)) as pool:
        rows = _run_city(
            f"bench:{shard}", shard, params, pool, scan_workers,
            threading.Lock(), {}, deadline=t0 + minutes * 60, stop_event=stop_event,
        )
    el = time.time() - t0
    rate = rows / el if el else 0.0
    logger.info("BENCH %s w=%d s=%d -> %d rows / %.0fs = %.0f rows/s",
                shard, workers, scan_workers, rows, el, rate)
    return rate


def main() -> None:
    """全国城市渗透率入口（生产 / 基准两模式）。"""
    parser = argparse.ArgumentParser(description="融合判定全国城市渗透率")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--city-concurrency", type=int, default=2,
                        help="同时处理的城市数（小表并行直读解锁 PG 并行扫描后，2 城并发最优）")
    parser.add_argument("--omega-file", type=str,
                        default=DEFAULT_OMEGA_SNAPSHOT)
    parser.add_argument("--bench", type=str, default="",
                        help="基准模式：给定分片表名，跑 --minutes 测吞吐后退出")
    parser.add_argument("--minutes", type=float, default=6.0)
    parser.add_argument("--resume-soft", action="store_true",
                        help="生产模式：遇 PG 连接失败暂停 10 分钟重试一次")
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_fused_national.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = eps_conn_params()

    if args.bench:
        rate = bench(args.bench, args.workers, args.scan_workers, args.minutes,
                     args.omega_file)
        print(f"BENCH_RESULT rows_per_sec={rate:.0f} shard={args.bench} "
              f"workers={args.workers} scan={args.scan_workers}")
        return

    tables = enumerate_national_tables(params)
    done = _load_checkpoint() or _preload_gd()
    _save_checkpoint(done)
    lock = threading.Lock()
    remaining = [(s, c) for s, c, _sz in tables if c not in done]
    logger.info("总 %d 城，已完成 %d，待跑 %d",
                len(tables), len(tables) - len(remaining), len(remaining))

    done_lock = threading.Lock()

    def process_one(shard: str, city: str, size_gb: float) -> None:
        """单城处理：小表客户端直读（并行扫描），中大表命名游标 ctid 切片。

        Args:
            shard: 分片表名。
            city: 城市名。
            size_gb: 表大小（字节）。
        """
        client_mode = size_gb < 2e9
        scan_w = 4 if size_gb < 8e9 else args.scan_workers
        logger.info("=== %s (%s, %.1fGB, %s) 开始 ===",
                    city, shard, size_gb / 1e9,
                    "并行直读" if client_mode else f"游标{scan_w}路")
        years_out: dict = {}
        t0 = time.time()
        attempts = 0
        while True:
            attempts += 1
            years_out.clear()
            if client_mode:
                rows = _run_city_client(city, shard, params, pool, lock, years_out)
            else:
                rows = _run_city(city, shard, params, pool, scan_w,
                                 lock, years_out)
            if rows > 0:
                break
            if not args.resume_soft or attempts >= 3:
                logger.error("%s 连续 %d 轮 0 行，跳过（需人工核查该表）", city, attempts)
                return
            logger.warning("%s 本轮 0 行（第 %d/3 次），10 分钟后重试", city, attempts)
            time.sleep(600)
        with done_lock:
            done[city] = {str(y): s for y, s in years_out.items()}
            _save_checkpoint(done)
        logger.info("=== %s 完成：%d 年 %.1f 分钟 ===",
                    city, len(years_out), (time.time() - t0) / 60)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.omega_file,)) as pool:
        todo = [(s, c, sz) for s, c, sz in tables if c not in done]
        # 城市并发：小城（<2GB）2 城同时（各 4 并行 worker 扫）；
        # 为控制磁盘流数与内存，统一并发 2
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.city_concurrency) as ctp:
            list(ctp.map(lambda t: process_one(*t), todo))

    # 汇总输出
    rows = []
    for city, years in done.items():
        for ys, s in years.items():
            n = s["total"]
            ab_both = s["a"] + s["b"] - s["fused"]
            rows.append({
                "city": city, "year": int(ys), "total": n,
                "a_jobs": s["a"], "b_jobs": s["b"], "fused_jobs": s["fused"],
                "ab_both_jobs": ab_both, "a_only_jobs": s["a"] - ab_both,
                "b_only_jobs": s["b"] - ab_both,
                "a_rate": s["a"] / n if n else 0.0,
                "b_rate": s["b"] / n if n else 0.0,
                "fused_rate": s["fused"] / n if n else 0.0,
            })
    df = pd.DataFrame(rows).sort_values(["city", "year"])
    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ai_penetration_national_{timestamp}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("全国面板已写入: %s (%d 城 %d 行)", out_path,
                df["city"].nunique(), len(df))
    cp = _checkpoint_path()
    if cp.exists():
        cp.unlink()


if __name__ == "__main__":
    main()
