"""全国 392 城 城市×行业 三口径面板（年度/半年度/季度三粒度，含行业分解）。

fused_industry.py（广深 2024）的全国版，三处关键升级：
1. 单次扫描同时聚合 年度/半年度/季度 三种时间粒度
2. 行业关联弃用逐行 LATERAL（全国 1.3 亿行下约 90 小时不可行），
   改为每城两遍 numpy：先流式读 ent 建 recruit_id→industry_code
   排序数组（searchsorted 批量映射），job 扫描时主进程映射、
   worker 仅做并行 A/B/C 评分（worker 无需共享大映射）
3. 输出即含分解列：ab_both / a_only / b_only（行业层分解）

工程约定与 fused_national 一致：城市级断点（每城一个 cells JSON）、
小表（<2GB）并行直读、大表 ctid 切片、描述截断在评分内生效、
单实例保护。输出时剔除 total < MIN_TOTAL_THRESHOLD 的组合。

使用示例::

    python -m src.ai_penetration.fused_industry_national --workers 24
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .common import (
    DEFAULT_OMEGA_SNAPSHOT,
    eps_conn_params,
    resolve_artifact_path,
    setup_logging,
)
from .fused_national import _guard_single_instance, enumerate_national_tables
from .industry_classification import classify_industry
from .skill_ai_anchor import is_ai_fused

logger = logging.getLogger("ai_penetration.fused_industry_national")

_WORKER: dict = {}
_YEAR_MO_RE = re.compile(r"^\s*(\d{4})-(\d{2})")
MIN_TOTAL_THRESHOLD = 500
_CELLS_DIRNAME = "fused_industry_national_cells"


def _init_worker(omega_path: str) -> None:
    """worker 初始化：加载 ωsAI 与技能正则（每 worker 一次）。

    Args:
        omega_path: ωsAI JSON 路径（可为相对 report_dir 的模式名）。
    """
    from .skill_ai_anchor import build_skill_regex, load_merged_skills

    _WORKER["omega"] = json.loads(
        resolve_artifact_path(omega_path, artifact="ωsAI").read_text(encoding="utf-8"))
    _WORKER["regex"] = build_skill_regex(load_merged_skills(include_llm=True))


def _score_aligned(pairs: list[tuple]) -> list[tuple]:
    """对一批 (position, description) 返回对齐的 (a, b, fused) 判定三元组。

    Args:
        pairs: 每元素为 (position, job_description)。

    Returns:
        与输入对齐的三元组列表。
    """
    return [
        is_ai_fused(str(p or ""), str(d or ""), _WORKER["omega"], _WORKER["regex"])
        for p, d in pairs
    ]


def _build_ent_map(city: str, ent_shard: str, params: dict):
    """流式读 ent 表建 recruit_id→industry 的排序数组映射。

    重复 recruit_id 经稳定排序后 searchsorted 命中首条，
    与 LATERAL LIMIT 1 语义一致。

    Args:
        city: 城市名（日志用）。
        ent_shard: ent 分片表名。
        params: eps 连接参数。

    Returns:
        (ids_sorted, inds_sorted) numpy 'S' 数组，表空/失败返回 None。
    """
    t0 = time.time()
    ids, inds = [], []
    conn = psycopg2.connect(**params)
    try:
        cur = conn.cursor(f"entmap_{ent_shard}")
        cur.execute(f"SELECT recruit_id, industry_code FROM public.{ent_shard}")
        while True:
            batch = cur.fetchmany(200000)
            if not batch:
                break
            for rid, ind in batch:
                ids.append((rid or "").strip())
                inds.append((ind or "").strip())
    except psycopg2.Error as exc:
        logger.error("%s ent 映射读取失败: %s", city, str(exc)[:150])
        return None
    finally:
        conn.close()
    if not ids:
        return None
    ids_arr = np.array(ids, dtype="S32")
    inds_arr = np.array(inds, dtype="S16")
    del ids, inds
    order = np.argsort(ids_arr, kind="stable")
    logger.info("%s ent 映射就绪: %d 行，%.0f 秒", city, len(order), time.time() - t0)
    return ids_arr[order], inds_arr[order]


def _lookup_industry(rids: list, ent_map) -> list:
    """numpy 排序数组批量查行业码并归类到 GB/T 大类。

    Args:
        rids: recruit_id 字符串列表。
        ent_map: (ids_sorted, inds_sorted) 或 None。

    Returns:
        与 rids 对齐的行业大类标签列表（未命中/空为"未知"）。
    """
    if ent_map is None:
        return ["未知"] * len(rids)
    ids_arr, inds_arr = ent_map
    q = np.array([r.strip() for r in rids], dtype="S32")
    idx = np.searchsorted(ids_arr, q)
    idx_c = np.clip(idx, 0, len(ids_arr) - 1)
    hit = ids_arr[idx_c] == q
    codes = np.where(hit, inds_arr[idx_c], b"")
    return [classify_industry(c.decode("ascii", "ignore")) for c in codes]


def _period_keys(pub) -> list:
    """由 publish_time 生成三粒度期间键。

    Args:
        pub: 发布时间字符串（'YYYY-MM...' 起头）。

    Returns:
        [("年度", y, 0, 0)]，月份可解析时追加半年度/季度键；
        年份不可解析时返回空列表。
    """
    m = _YEAR_MO_RE.match(str(pub or ""))
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return [
                ("年度", y, 0, 0),
                ("半年度", y, 1 if mo <= 6 else 2, 0),
                ("季度", y, 0, (mo - 1) // 3 + 1),
            ]
        return [("年度", y, 0, 0)]
    m2 = re.match(r"^\s*(\d{4})", str(pub or ""))
    return [("年度", int(m2.group(1)), 0, 0)] if m2 else []


def _run_city_industry(
    city: str,
    shard: str,
    size_bytes: float,
    params: dict,
    pool: ProcessPoolExecutor,
    scan_workers: int,
) -> dict | None:
    """单城两遍：ent 映射 + job 扫描，聚合三粒度 cells。

    Args:
        city: 城市名。
        shard: job 分片表名。
        size_bytes: job 表大小（<2GB 走并行直读，否则 ctid 切片）。
        params: eps 连接参数。
        pool: 评分进程池。
        scan_workers: 大表切片数。

    Returns:
        cells 字典 {f"{gran}|{year}|{half}|{q}|{industry}": {total,a,b,fused}}；
        失败返回 None。
    """
    ent_shard = shard.replace("job_", "ent_", 1)
    ent_map = _build_ent_map(city, ent_shard, params)
    cells: dict = defaultdict(lambda: {"total": 0, "a": 0, "b": 0, "fused": 0})
    counter = {"rows": 0}
    c_lock = threading.Lock()

    def consume(batch_rows: list, scores: list) -> None:
        rids = [str(r[3] or "") for r in batch_rows]
        inds = _lookup_industry(rids, ent_map)
        for row, (a_ai, b_ai, f_ai), industry in zip(batch_rows, scores, inds):
            for gran, y, h, q in _period_keys(row[2]):
                s = cells[f"{gran}|{y}|{h}|{q}|{industry}"]
                s["total"] += 1
                s["a"] += a_ai
                s["b"] += b_ai
                s["fused"] += f_ai
        with c_lock:
            counter["rows"] += len(batch_rows)

    def scan_range(lo: int, hi: int) -> None:
        conn = psycopg2.connect(**params)
        try:
            setup_cur = conn.cursor()
            setup_cur.execute("SET LOCAL work_mem = '256MB'")
            if hi < 0:
                setup_cur.execute("SET LOCAL max_parallel_workers_per_gather = 4")
            setup_cur.close()
            if hi >= 0:
                cur = conn.cursor(f"fin_{shard}_{lo}")
                where = f"ctid >= '({lo},0)'::tid AND ctid < '({hi},0)'::tid AND "
            else:
                cur = conn.cursor()
                cur.itersize = 50000
                where = ""
            cur.execute(f"""
                SELECT position, job_description, publish_time, recruit_id
                FROM public.{shard}
                WHERE {where}job_description IS NOT NULL AND job_description != ''
                  AND position IS NOT NULL AND position != ''
            """)
            pending = []
            while True:
                batch = cur.fetchmany(50000)
                if not batch:
                    break
                pairs = [(r[0], r[1]) for r in batch]
                pending.append((batch, pool.submit(_score_aligned, pairs)))
                if len(pending) >= 3:
                    b, fut = pending.pop(0)
                    consume(b, fut.result())
            for b, fut in pending:
                consume(b, fut.result())
        except psycopg2.Error as exc:
            logger.error("%s 扫描异常: %s", city, str(exc)[:150])
        finally:
            conn.close()

    if size_bytes < 2e9:
        scan_range(-1, -1)  # 并行直读单遍
    else:
        conn0 = psycopg2.connect(**params)
        try:
            c0 = conn0.cursor()
            c0.execute(
                "SELECT pg_relation_size(%s) / current_setting('block_size')::int",
                (f"public.{shard}",),
            )
            blocks = max(1, int(c0.fetchone()[0]))
        finally:
            conn0.close()
        step = (blocks + scan_workers - 1) // scan_workers
        threads = [
            threading.Thread(target=scan_range, args=(i * step, min((i + 1) * step, blocks)),
                             daemon=True)
            for i in range(scan_workers) if i * step < blocks
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    if counter["rows"] == 0:
        logger.error("%s 扫描 0 行", city)
        return None
    del ent_map
    return dict(cells)


def _cells_dir() -> Path:
    """城市 cells 断点目录。"""
    return get_project_paths().output_dir / _CELLS_DIRNAME


def build_panels(timestamp: str) -> dict[str, int]:
    """汇总各城 cells 断点，按三粒度过滤小样本并输出 CSV。

    Args:
        timestamp: 输出文件名时间戳。

    Returns:
        {粒度: 输出行数}。
    """
    paths = get_project_paths()
    agg: dict[str, list] = {"年度": [], "半年度": [], "季度": []}
    for fp in sorted(_cells_dir().glob("*.json")):
        city = fp.stem
        cells = json.loads(fp.read_text(encoding="utf-8"))
        for key, s in cells.items():
            gran, year, half, q, industry = key.split("|")
            n = s["total"]
            if n < MIN_TOTAL_THRESHOLD:
                continue
            ab = s["a"] + s["b"] - s["fused"]
            period = ("全年" if gran == "年度"
                      else "上半年" if half == "1" and gran == "半年度"
                      else "下半年" if gran == "半年度"
                      else f"Q{int(q)}")
            agg[gran].append({
                "city": city, "industry": industry, "year": int(year), "期间": period,
                "total": n, "a_jobs": s["a"], "b_jobs": s["b"], "fused_jobs": s["fused"],
                "ab_both_jobs": ab, "a_only_jobs": s["a"] - ab, "b_only_jobs": s["b"] - ab,
                "a_rate": s["a"] / n, "b_rate": s["b"] / n, "fused_rate": s["fused"] / n,
            })
    counts = {}
    for gran, rows in agg.items():
        suffix = {"年度": "year", "半年度": "half", "季度": "quarter"}[gran]
        if not rows:
            counts[gran] = 0
            continue
        df = pd.DataFrame(rows).sort_values(["city", "year", "industry"])
        path = paths.report_dir / f"ai_penetration_industry_national_{suffix}_{timestamp}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        counts[gran] = len(df)
        logger.info("%s 面板已写入: %s（%d 行）", gran, path.name, len(df))
    return counts


def main() -> None:
    """全国城市×行业三粒度面板入口。"""
    _guard_single_instance()
    parser = argparse.ArgumentParser(description="全国城市×行业三口径三粒度面板")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--city-concurrency", type=int, default=2)
    parser.add_argument("--omega-file", type=str, default=DEFAULT_OMEGA_SNAPSHOT)
    parser.add_argument("--cities", type=str, default="",
                        help="限定城市，逗号分隔；空=全部 392 城")
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "fused_industry_national.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = eps_conn_params()

    tables = enumerate_national_tables(params)
    if args.cities:
        want = {c.strip() for c in args.cities.split(",") if c.strip()}
        tables = [t for t in tables if t[1] in want]
    cdir = _cells_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    done_cities = {p.stem for p in cdir.glob("*.json")}
    todo = [t for t in tables if t[1] not in done_cities]
    logger.info("总 %d 城，已完成 %d，待跑 %d",
                len(tables), len(tables) - len(todo), len(todo))

    def process(shard: str, city: str, size_bytes: float) -> None:
        t0 = time.time()
        logger.info("=== %s (%s, %.1fGB) 开始 ===", city, shard, size_bytes / 1e9)
        for attempt in range(1, 4):
            cells = _run_city_industry(city, shard, size_bytes, params, pool,
                                       args.scan_workers)
            if cells is not None:
                (cdir / f"{city}.json").write_text(
                    json.dumps(cells, ensure_ascii=False), encoding="utf-8")
                logger.info("=== %s 完成：%d cells，%.0f 分钟 ===",
                            city, len(cells), (time.time() - t0) / 60)
                return
            logger.warning("%s 第 %d/3 轮失败，10 分钟后重试", city, attempt)
            time.sleep(600)
        logger.error("%s 三轮失败，跳过", city)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.omega_file,)) as pool:
        with ThreadPoolExecutor(max_workers=args.city_concurrency) as ctp:
            list(ctp.map(lambda t: process(*t), todo))

    if not args.cities:
        counts = build_panels(timestamp)
        print("三粒度面板输出:", counts)


if __name__ == "__main__":
    main()
