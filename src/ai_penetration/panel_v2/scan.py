"""panel_v2 M3-a：Pass2 识别扫描（§12.5 job_anchor_flag + job_skill_long）。

输入 M2 产物 job_master_gzsz（结果库）；eps 只读；产物写
output/panel_v2/pass2/。审计修复版（B1/B3/M9）：

- master 导出为**裸 .npy + np.load(mmap_mode="r")**（npz 成员不真 mmap，
  实证每 worker 会私载全量——已修正），按 key=blake2b-63(recruit_id) 排序。
- skill 词表由**主进程单点构建**并原子落盘（含 ORDER BY 的别名查询消除
  36 个实证碰撞键的跨 worker 分歧），worker 读文件 + 一致性断言。
- 产物**按批落盘为 parquet dataset 分区**（flags/long/firm 三目录），
  不在内存累积全量行。
- 守恒断言：Σcanonical == job_master 行数。

使用示例::

    python -X utf8 -m src.ai_penetration.panel_v2.scan --bench
    python -X utf8 -m src.ai_penetration.panel_v2.scan
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
FLUSH_ROWS = 500_000  # 每子批落盘行数（B1：禁止全切片累积）

_WORKER: dict = {}


def _results_conn():
    import psycopg2 as _p
    rp = get_project_paths().results_connection_params
    return _p.connect(**rp)


def export_master(out_dir: Path) -> None:
    """job_master → 4 个排序裸 .npy（mmap 真共享）。原子写（M9）。"""
    npy_dir = out_dir / "master_npy"
    if all((npy_dir / f"{n}.npy").exists()
           for n in ("key", "job_id", "year", "company")):
        return
    conn = _results_conn()
    cur = conn.cursor()
    cur.execute("SELECT job_id_raw, job_id, city, year, company_id "
                "FROM public.job_master_gzsz")
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
    npy_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_master_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, arr in (("key", k_sorted), ("job_id", job_id[order]),
                      ("year", year[order]), ("company", company[order])):
        p = tmp / f"{name}.npy"
        np.save(p, arr)
        p.rename(npy_dir / f"{name}.npy")   # 原子发布
    (tmp / "done").write_text("ok")
    (out_dir / "company_vocab.json").write_text(
        json.dumps(comps), encoding="utf-8")
    logger.info("master npy 导出: %d 条 / company %d", n, len(comps))


def build_skill_vocab(out_dir: Path) -> None:
    """主进程单点构建 union 词表并原子落盘（B3：ORDER BY 确定性）。"""
    path = out_dir / "skill_vocab.json"
    if path.exists():
        return
    from ..skill_ai_anchor import load_merged_skills
    from .lexicon import _load_atier_aliases, build_union_lexicon
    aliases = _load_atier_aliases()   # 查询已 ORDER BY，first-wins 确定
    lex = build_union_lexicon(legacy_terms=load_merged_skills(include_llm=True),
                              aliases=aliases)
    sids = sorted(set(lex.keys_map.values()))
    tmp = out_dir / "_vocab.json.tmp"
    tmp.write_text(json.dumps({s: i for i, s in enumerate(sids)},
                              ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)
    logger.info("skill_vocab 落盘: %d skill", len(sids))


def _init_worker(npy_dir: str) -> None:
    """worker：mmap master + 从落盘词表反查一致性（B3）。"""
    _WORKER["m"] = {n: np.load(Path(npy_dir) / f"{n}.npy", mmap_mode="r")
                    for n in ("key", "job_id", "year", "company")}
    from ..skill_ai_anchor import load_merged_skills
    from .lexicon import _load_atier_aliases, build_union_lexicon
    aliases = _load_atier_aliases()
    lex = build_union_lexicon(legacy_terms=load_merged_skills(include_llm=True),
                              aliases=aliases)
    vocab_path = Path(npy_dir).parent / "skill_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    built = sorted(set(lex.keys_map.values()))
    assert len(built) == len(vocab) and all(
        vocab[s] == i for i, s in enumerate(built)), \
        "worker 词表与落盘 vocab 不一致（B3 防御断言）"
    _WORKER["sid_to_code"] = vocab
    _WORKER["lex"] = lex


def _flush_parts(out: Path, task: str, part: int,
                 flags_buf: list, long_buf: list, firm_buf: list) -> None:
    """子批 buffer 即时写 parquet 分区（B1）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if flags_buf:
        cols = list(zip(*flags_buf))
        pq.write_table(pa.table({
            "job_id": pa.array(cols[0], pa.int64()),
            "year": pa.array(cols[1], pa.int32()),
            "anchor_main": pa.array(cols[2], pa.int8()),
            "anchor_cn_paper": pa.array(cols[3], pa.int8()),
            "anchor_babina": pa.array(cols[4], pa.int8()),
            "groups_main_bits": pa.array(cols[5], pa.int16()),
        }), out / "parts_flags" / f"{task}_{part:04d}.parquet")
    if long_buf:
        cols = list(zip(*long_buf))
        pq.write_table(pa.table({
            "job_id": pa.array(cols[0], pa.int64()),
            "year": pa.array(cols[1], pa.int32()),
            "skill_code": pa.array(cols[2], pa.int32()),
        }), out / "parts_long" / f"{task}_{part:04d}.parquet")
    if firm_buf:
        cols = list(zip(*firm_buf))
        pq.write_table(pa.table({
            "job_id": pa.array(cols[0], pa.int64()),
            "year": pa.array(cols[1], pa.int32()),
            "company_code": pa.array(cols[2], pa.int32()),
        }), out / "parts_firm" / f"{task}_{part:04d}.parquet")


def scan_slice(shard: str, city_id: int, lo: int, hi: int, out_dir: str) -> dict:
    """一个 ctid 切片的 pass2：buffer 满子批即落盘，meta 为完成标记。"""
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
    n_rows = n_canon = part = 0
    flags_buf: list = []
    long_buf: list = []
    firm_buf: list = []
    n_pairs = 0
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
        for d in ("parts_flags", "parts_long", "parts_firm"):
            (out / d).mkdir(parents=True, exist_ok=True)
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
                flags_buf.append((job_id, yr, hits["main"].flag,
                                  hits["cn_paper"].flag, hits["babina"].flag,
                                  bits))
                long_buf.extend((job_id, yr, c) for c in codes)
                firm_buf.append((job_id, yr, int(m["company"][idx])))
                n_pairs += len(codes)
                if len(flags_buf) >= FLUSH_ROWS:
                    _flush_parts(out, task, part, flags_buf, long_buf, firm_buf)
                    flags_buf, long_buf, firm_buf = [], [], []
                    part += 1
            if len(flags_buf) >= FLUSH_ROWS:  # fetchmany 边界也检查
                _flush_parts(out, task, part, flags_buf, long_buf, firm_buf)
                flags_buf, long_buf, firm_buf = [], [], []
                part += 1
        cur.close()
        _flush_parts(out, task, part, flags_buf, long_buf, firm_buf)
    finally:
        conn.close()
    stat = {"task": task, "rows": n_rows, "canonical": n_canon,
            "skill_pairs": n_pairs}
    tmp = out / f".{task}.done.tmp"
    tmp.write_text(json.dumps(stat), encoding="utf-8")
    tmp.rename(done)
    logger.info("pass2 切片完成 %s: canonical=%d pairs=%d", task, n_canon, n_pairs)
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


def merge_parts(out_dir: Path, rel_dir: Path) -> None:
    """子批分区合并为 §18 单文件（流式写，避免全表内存）。"""
    import pyarrow.parquet as pq
    rel_dir.mkdir(parents=True, exist_ok=True)
    for name, sub in (("job_anchor_flag", "parts_flags"),
                      ("job_skill_long", "parts_long"),
                      ("job_firm", "parts_firm")):
        files = sorted((out_dir / sub).glob("*.parquet"))
        assert files, f"{sub} 无分片"
        schema = pq.read_schema(files[0])
        with pq.ParquetWriter(rel_dir / f"{name}.parquet", schema) as w:
            for f in files:
                w.write_table(pq.read_table(f))
        logger.info("合并 %s: %d 分片", name, len(files))


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
    build_skill_vocab(out_dir)
    npy_dir = str(out_dir / "master_npy")
    tasks = _plan(args.slices)
    if args.bench:
        tasks = tasks[:1]

    from concurrent.futures import ProcessPoolExecutor
    stats = []
    with ProcessPoolExecutor(
            max_workers=min(args.workers, 8),
            initializer=_init_worker, initargs=(npy_dir,)) as pool:
        futs = [pool.submit(scan_slice, s, c, lo, hi, str(out_dir))
                for s, c, lo, hi in tasks]
        for f in futs:
            stats.append(f.result())

    conn = _results_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM public.job_master_gzsz")
    n_master = cur.fetchone()[0]
    conn.close()
    canon = sum(s["canonical"] for s in stats)
    if canon != n_master:
        raise SystemExit(f"守恒失败: pass2 canonical {canon} != master {n_master}")
    merge_parts(out_dir, paths.output_dir / "release" / "panel_v2")
    dur = (datetime.now() - t0).total_seconds() / 3600
    print(f"pass2 完成: canonical={canon:,} pairs={sum(s['skill_pairs'] for s in stats):,}"
          f" 用时 {dur:.2f}h，守恒核验通过 ✓")


if __name__ == "__main__":
    main()
