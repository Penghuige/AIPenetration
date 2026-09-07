"""panel_v2 M3-a：Pass2 识别扫描（§12.5 job_anchor_flag + job_skill_long）。

输入 M2 产物 job_master_gzsz（canonical 岗位主表，结果库）；eps 只读；
产物写 output/panel_v2/pass2/ 与 release 合并件。要点（评审确认版）：

- canonical 过滤：master 导出为按 key=blake2b-63(recruit_id) 稳定排序的 npz
  数组，worker `np.searchsorted` 判命中（mmap 语义，OS 页缓存跨进程共享；
  Windows 无 fork，避免每 worker 复制大 dict）。导出时断言 key 全局无碰撞。
- 一次规范化共用：match_from_raw 产出 match 文本，锚点与 union 词表都吃它。
- 锚点三套一次匹配：flag 三列 + main 命中组 bitmask（§12.6.7 组/词明细，
  词级明细在 dup 审计需要时由 groups_main_bits + 重放还原，v2a 存组级）。
- 技能长表 (job_id, year, skill_code)：skill_code 用全局确定性词表
  （自动机键全集排序编号，主/子进程独立重建，一致性由构造保证）。
- 留一输入表 (job_id, year, company_code)（§14.3）。
- 守恒断言：Σworker canonical 命中 == job_master 行数，不符即停。

使用示例::

    python -X utf8 -m src.ai_penetration.panel_v2.scan --bench   # 单片吞吐
    python -X utf8 -m src.ai_penetration.panel_v2.scan           # 全量 pass2
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg2

from config.paths import get_project_paths

from ..common import eps_conn_params, setup_logging
from ..text_clean import match_from_raw
from .anchors import match_all_versions
from .dedup import SHARDS, _blocks, _h63

logger = logging.getLogger("ai_penetration.panel_v2.scan")

GROUP_BITS = {"AI": 1, "ML": 2, "NLP": 4, "CVISION": 8,
              "CIMAGE": 16, "LLM": 32, "TRANS": 64}

_WORKER: dict = {}


def _results_conn():
    import psycopg2 as _p
    rp = get_project_paths().results_connection_params
    return _p.connect(**rp)


def export_master(out_dir: Path) -> None:
    """job_master → 排序 npz + company 词表（幂等：文件存在即跳过）。"""
    npz = out_dir / "master_keys.npz"
    if npz.exists():
        return
    conn = _results_conn()
    cur = conn.cursor()
    cur.execute("SELECT rid, job_id, city, year, company_id FROM public.job_master_gzsz")
    rows = cur.fetchall()
    conn.close()
    n = len(rows)
    logger.info("job_master 载入: %d 行", n)
    key = np.empty(n, np.int64)
    job_id = np.empty(n, np.int64)
    year = np.empty(n, np.int32)
    company = np.empty(n, np.int32)
    comps: dict[str, int] = {}
    for i, (rid, jid, _c, y, comp) in enumerate(rows):
        key[i] = _h63(str(rid))
        job_id[i] = int(jid)
        year[i] = int(y)
        company[i] = comps.setdefault(str(comp), len(comps))
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    assert not np.any(k_sorted[1:] == k_sorted[:-1]), "canonical key 哈希碰撞"
    # 注意：压缩 npz 与 mmap_mode="r" 不兼容，必须用未压缩 savez
    np.savez(npz, key=k_sorted, job_id=job_id[order], year=year[order],
             company=company[order])
    (out_dir / "company_vocab.json").write_text(
        json.dumps(comps), encoding="utf-8")
    logger.info("master npz 导出完成: %d 条 / company %d", n, len(comps))


def _init_worker(npz_path: str) -> None:
    """worker 初始化：mmap master + 重建 union 词表与确定性编号。

    skill_code 编号 = sorted(keys_map 值全集) 的下标，跨进程一致性由
    构造确定性保证（同一别名表 + 同一 legacy 词典），词表落盘供解码。
    """
    _WORKER["m"] = np.load(npz_path, mmap_mode="r")
    from ..skill_ai_anchor import load_merged_skills
    from .lexicon import _load_atier_aliases, build_union_lexicon
    aliases = _load_atier_aliases()
    lex = build_union_lexicon(legacy_terms=load_merged_skills(include_llm=True),
                              aliases=aliases)
    all_sids = sorted(set(lex.keys_map.values()))
    _WORKER["sid_to_code"] = {s: i for i, s in enumerate(all_sids)}
    _WORKER["lex"] = lex
    vocab_path = Path(npz_path).parent / "skill_vocab.json"
    if not vocab_path.exists():
        vocab_path.write_text(
            json.dumps(_WORKER["sid_to_code"], ensure_ascii=False),
            encoding="utf-8")


def scan_slice(shard: str, city_id: int, lo: int, hi: int, out_dir: str) -> dict:
    """一个 ctid 切片的 pass2 识别，输出 3 个 parquet 分片；meta 断点。"""
    out = Path(out_dir)
    task = f"{shard}_{lo}"
    done = out / f"{task}.done.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    m = _WORKER["m"]
    lex = _WORKER["lex"]
    sid2code = _WORKER["sid_to_code"]
    keys = m["key"]
    conn = psycopg2.connect(**eps_conn_params())
    n_rows = n_canon = 0
    flag_rows: list[list[int]] = []
    long_rows: list[list[int]] = []
    firm_rows: list[list[int]] = []
    try:
        with conn.cursor() as setup:
            setup.execute("SET LOCAL work_mem = '256MB'")
            setup.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cur = conn.cursor(f"pass2_{task}")
        cur.itersize = 50000
        cur.execute(
            f"SELECT recruit_id, job_description FROM public.{shard} "
            "WHERE ctid >= '(%s,0)'::tid AND ctid < '(%s,0)'::tid "
            "  AND job_description IS NOT NULL AND job_description != '' "
            "  AND position IS NOT NULL AND position != '' "
            "  AND recruit_id IS NOT NULL",
            (int(lo), int(hi)))
        while True:
            batch = cur.fetchmany(50000)
            if not batch:
                break
            for rid, desc in batch:
                n_rows += 1
                k = _h63(str(rid))
                idx = int(np.searchsorted(keys, k))
                if idx >= keys.size or int(keys[idx]) != k:
                    continue
                n_canon += 1
                job_id = int(m["job_id"][idx])
                yr = int(m["year"][idx])
                match_txt = match_from_raw(str(desc))
                hits = match_all_versions(match_txt)
                bits = 0
                for g in hits["main"].groups:
                    bits |= GROUP_BITS[g]
                codes = sorted(sid2code[s] for s in lex.extract(match_txt))
                flag_rows.append([job_id, yr, hits["main"].flag,
                                  hits["cn_paper"].flag, hits["babina"].flag, bits])
                long_rows.extend([job_id, yr, c] for c in codes)
                firm_rows.append([job_id, yr, int(m["company"][idx])])
        cur.close()
    finally:
        conn.close()
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({
        "job_id": pa.array([r[0] for r in flag_rows], pa.int64()),
        "year": pa.array([r[1] for r in flag_rows], pa.int32()),
        "anchor_main": pa.array([r[2] for r in flag_rows], pa.int8()),
        "anchor_cn_paper": pa.array([r[3] for r in flag_rows], pa.int8()),
        "anchor_babina": pa.array([r[4] for r in flag_rows], pa.int8()),
        "groups_main_bits": pa.array([r[5] for r in flag_rows], pa.int16()),
    }), out / f"{task}.flags.parquet")
    pq.write_table(pa.table({
        "job_id": pa.array([r[0] for r in long_rows], pa.int64()),
        "year": pa.array([r[1] for r in long_rows], pa.int32()),
        "skill_code": pa.array([r[2] for r in long_rows], pa.int32()),
    }), out / f"{task}.long.parquet")
    pq.write_table(pa.table({
        "job_id": pa.array([r[0] for r in firm_rows], pa.int64()),
        "year": pa.array([r[1] for r in firm_rows], pa.int32()),
        "company_code": pa.array([r[2] for r in firm_rows], pa.int32()),
    }), out / f"{task}.firm.parquet")
    stat = {"task": task, "rows": n_rows, "canonical": n_canon,
            "skill_pairs": len(long_rows)}
    done.write_text(json.dumps(stat), encoding="utf-8")
    logger.info("pass2 切片完成 %s: canonical=%d pairs=%d", task, n_canon,
                len(long_rows))
    return stat


def _plan(slices: int) -> list[tuple]:
    conn = psycopg2.connect(**eps_conn_params())
    cur = conn.cursor()
    tasks = []
    for _city, shard, city_id in SHARDS:
        total = _blocks(cur, shard)
        step = total // slices + 1
        for i in range(slices):
            lo = i * step
            hi = 4294967295 if i == slices - 1 else min(total, (i + 1) * step)
            if lo < hi:
                tasks.append((shard, city_id, lo, hi))
    conn.close()
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="panel_v2 pass2 识别扫描")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--slices", type=int, default=4)
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_scan.log")
    out_dir = paths.output_dir / "panel_v2" / "pass2"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    export_master(out_dir)
    npz = str(out_dir / "master_keys.npz")
    tasks = _plan(args.slices)
    if args.bench:
        tasks = tasks[:1]

    # Windows spawn：initializer 传 npz 路径；ProcessPoolExecutor 需顶层函数
    from concurrent.futures import ProcessPoolExecutor
    stats = []
    with ProcessPoolExecutor(
            max_workers=min(args.workers, 8),
            initializer=_init_worker, initargs=(npz,)) as pool:
        futs = [pool.submit(scan_slice, s, c, lo, hi, str(out_dir))
                for s, c, lo, hi in tasks]
        for f in futs:
            stats.append(f.result())

    # 守恒核验（§9 纪律：跑完先守恒，再谈产物）
    conn = _results_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM public.job_master_gzsz")
    n_master = cur.fetchone()[0]
    conn.close()
    canon = sum(s["canonical"] for s in stats)
    if canon != n_master:
        raise SystemExit(f"守恒失败: pass2 canonical {canon} != master {n_master}"
                         f"（差 {n_master - canon}，检查切片覆盖/日期过滤）")
    # 合并 release 输入件
    import pyarrow as pa
    import pyarrow.parquet as pq
    rel = paths.output_dir / "release" / "panel_v2"
    rel.mkdir(parents=True, exist_ok=True)
    for name, suffix in (("job_anchor_flag", "flags"), ("job_skill_long", "long"),
                         ("job_firm", "firm")):
        files = sorted(out_dir.glob(f"*.{suffix}.parquet"))
        tables = [pq.read_table(f) for f in files]
        pq.write_table(pa.concat_tables(tables), rel / f"{name}.parquet")
        logger.info("合并 %s.parquet: %d 分片", name, len(tables))
    dur = (datetime.now() - t0).total_seconds() / 3600
    print(f"pass2 完成: canonical={canon:,} pairs={sum(s['skill_pairs'] for s in stats):,}"
          f" 用时 {dur:.2f}h，守恒核验通过 ✓")


if __name__ == "__main__":
    main()
