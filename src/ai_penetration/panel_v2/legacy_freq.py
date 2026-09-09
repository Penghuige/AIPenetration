"""v2e：legacy 词 df_unique_description 频数扫描（指南 §10.3.2 分级通道补课）。

背景（docs/10 F1 + 原预案核对）：自建词当年未过 §10 B/C/D 分级；分级须用
频率语料的 `COUNT(DISTINCT text_hash)` 口径（zh_alias_freq 对 A 级用的
platform×text_hash 键，比指南文本口径更保守——沿用同键并披露）。本模块只扫
legacy 层 6,328 键，匹配语义对齐生产 union 词表（ASCII 词边界守卫，A 级管线
无守卫不适用英文词）。eps 只读；结果写 ai_dict.legacy_term_freq（词典治理
合法 schema）+ 本地 CSV 治理件。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.legacy_freq --workers 8 --slices 4
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from hashlib import blake2b
from pathlib import Path

import numpy as np
import psycopg2

from config.paths import get_project_paths

from ..common import eps_conn_params, setup_logging
from ..zh_alias_freq import aggregate_counts, platform_id
from .anchors import normalize_desc
from .dedup import SHARDS, _blocks
from .lexicon import LEGACY_PREFIX, _is_ascii_alnum, build_union_lexicon

logger = logging.getLogger("ai_penetration.panel_v2.legacy_freq")

_W: dict = {}


def _legacy_terms() -> dict[str, str]:
    """生产同源 union 构建，取 legacy 键 → skill_id。"""
    from ..skill_ai_anchor import load_merged_skills
    from .lexicon import _load_atier_aliases
    lex = build_union_lexicon(legacy_terms=load_merged_skills(include_llm=True),
                              aliases=_load_atier_aliases())
    return {k: sid for k, sid in lex.keys_map.items()
            if str(sid).startswith(LEGACY_PREFIX)}


def _init_worker(terms: dict[str, str]) -> None:
    import ahocorasick
    auto = ahocorasick.Automaton()
    order = sorted(terms)
    idx = {k: i for i, k in enumerate(order)}
    for k in order:
        auto.add_word(k, k)
    auto.make_automaton()
    _W["auto"] = auto
    _W["idx"] = idx
    _W["n"] = len(order)
    _W["ascii"] = {k for k in order if all(ord(c) < 128 for c in k)}


def scan_slice(table: str, b_start: int, b_end: int, tmp_dir: str) -> dict:
    """一个 ctid 切片：文本规范化→(platform,hash) 去重→带边界 Aho→(aid,key)。

    实现纪律（2026-09-10 重跑版：首版把全部命中对攒进 Python list，
    千万级对 × 8 worker 造成整机换页停滞且进度不可见）：
    ①定长环形缓冲（100 万对 ≈ 12MB/worker），满则溢出落 part 文件；
    ②心跳文件 <task>.hb.json（rows/phase），外部可观测；
    ③worker 本地日志文件（spawn 子进程 logging 未初始化，不依赖父配置）；
    ④连接超时 15s，失败重试 2 次后硬失败（防无提示悬挂）。
    """
    auto, idx, n = _W["auto"], _W["idx"], _W["n"]
    ascii_keys = _W["ascii"]
    task = f"lg_{table}_{b_start}_{b_end}"
    kf = Path(tmp_dir) / f"{task}.keys.bin"
    af = Path(tmp_dir) / f"{task}.aids.bin"
    hb = Path(tmp_dir) / f"{task}.hb.json"
    wl = (Path(tmp_dir) / f"{task}.wlog.txt").open("a", encoding="utf-8")

    if kf.exists() and af.exists():
        wl.close()
        return json.loads((Path(tmp_dir) / f"{task}.json").read_text(encoding="utf-8"))

    def wlog(msg: str) -> None:
        wl.write(f"{datetime.now():%H:%M:%S} {msg}\n")
        wl.flush()
    FLUSH = 1_000_000
    buf_k = np.empty(FLUSH, np.uint64)
    buf_a = np.empty(FLUSH, np.uint32)
    pos = seq = pairs = 0

    def spill() -> None:
        """溢出前桶内 (aid,key) 排序去重——模板相邻重复在桶级即消，
        全局正确性仍由 aggregate_counts 的最终去重保证（跨桶重复无害）。
        基准外推原始对量 ~6 亿，不去重将压垮聚合阶段内存。"""
        nonlocal pos, seq, pairs
        if not pos:
            return
        kk, aa = buf_k[:pos], buf_a[:pos]
        o = np.lexsort((kk, aa))
        ks, as_ = kk[o], aa[o]
        keep = np.ones(pos, dtype=bool)
        keep[1:] = (ks[1:] != ks[:-1]) | (as_[1:] != as_[:-1])
        ks[keep].tofile(Path(tmp_dir) / f"{task}.k{seq:03d}.part")
        as_[keep].tofile(Path(tmp_dir) / f"{task}.a{seq:03d}.part")
        pairs += int(keep.sum())
        seq += 1
        pos = 0

    def connect():
        last = None
        for _ in range(3):
            try:
                return psycopg2.connect(**eps_conn_params(), connect_timeout=15)
            except Exception as e:  # noqa 连接期异常原样重试
                last = e
                wlog(f"connect fail: {e}")
        raise RuntimeError(f"eps 连接失败（3 次）: {last}")

    rows = 0
    # 无 per-slice 文本去重集（首版 local_seen 千万级 Python int ≈ 700MB/worker，
    # 8 路并发即换页停滞）；(aid,text) 全局去重由 aggregate_counts 完成，
    # 本处只做有界溢出落盘——内存 O(FLUSH)。
    try:
        conn = connect()
        wlog(f"connected, scan {b_start}-{b_end}")
        hb.write_text(json.dumps({"phase": "scan", "rows": 0}), encoding="utf-8")
        with conn.cursor() as st:
            st.execute("SET LOCAL work_mem = '256MB'")
            st.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cur = conn.cursor(f"lgfreq_{task[:48]}")
        cur.itersize = 50000
        # 语料谓词与 A 级频率管线一致（词典语料面，非主样本准入面）
        cur.execute(
            f"SELECT platform, job_description FROM public.{table} "
            "WHERE ctid >= '(%s,0)'::tid AND ctid < '(%s,0)'::tid "
            "AND job_description IS NOT NULL AND job_description != '' "
            "AND position IS NOT NULL AND position != ''",
            (int(b_start), int(b_end)))
        while True:
            batch = cur.fetchmany(50000)
            if not batch:
                break
            for platform, desc in batch:
                rows += 1
                # 文本面=生产匹配语义（NFKC+lower+空白折叠），多词英文键可配；
                # 去重键结构同 A 级 (platform, 规范化文本)（口径差异在 QC 披露）
                norm = normalize_desc(str(desc))
                h48 = int.from_bytes(blake2b(norm.encode(), digest_size=6).digest(),
                                     "big")
                key = (platform_id(str(platform)) << 48) | h48
                for end, k in auto.iter(norm):
                    if k in ascii_keys:
                        s0 = end - len(k) + 1
                        bef = norm[s0 - 1] if s0 > 0 else ""
                        aft = norm[end + 1] if end + 1 < len(norm) else ""
                        if _is_ascii_alnum(bef) or _is_ascii_alnum(aft):
                            continue
                    buf_k[pos] = key
                    buf_a[pos] = idx[k]
                    pos += 1
                    if pos == FLUSH:
                        spill()
            hb.write_text(json.dumps({"phase": "scan", "rows": rows,
                                      "pairs": pairs + pos}), encoding="utf-8")
            if rows % 2_000_000 == 0:
                wlog(f"rows={rows:,} pairs={pairs + pos:,}")
        cur.close()
        conn.close()
        spill()
        wlog(f"fetch done rows={rows:,} pairs={pairs:,}; merging")
        hb.write_text(json.dumps({"phase": "merge", "rows": rows,
                                  "pairs": pairs}), encoding="utf-8")
        parts = sorted(Path(tmp_dir).glob(f"{task}.k*.part"))
        if pairs:
            ka = np.concatenate([np.fromfile(p, dtype=np.uint64) for p in parts])
            aa = np.concatenate([np.fromfile(p, dtype=np.uint32)
                                 for p in Path(tmp_dir).glob(f"{task}.a*.part")])
            ka.tofile(kf)
            aa.tofile(af)
            n_uniq_local = int(len(np.unique(ka)))
            del ka, aa
        else:
            np.empty(0, np.uint64).tofile(kf)
            np.empty(0, np.uint32).tofile(af)
            n_uniq_local = 0
        for p in Path(tmp_dir).glob(f"{task}.*.part"):
            p.unlink()
    finally:
        wl.close()
    meta = {"task": task, "rows": rows, "pairs": int(pairs),
            "local_new": n_uniq_local,
            "keys_file": str(kf), "aids_file": str(af)}
    (Path(tmp_dir) / f"{task}.json").write_text(json.dumps(meta), encoding="utf-8")
    hb.unlink(missing_ok=True)
    logger.info("legacy 切片 %s 完成: rows=%d pairs=%d", task, rows, pairs)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="legacy df_unique_description 频数")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--slices", type=int, default=4)
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_legacy_freq.log")
    assert args.workers <= 8, "HDD 纪律"
    terms = _legacy_terms()
    logger.info("legacy 键 %d，规划切片", len(terms))
    conn = psycopg2.connect(**eps_conn_params())
    cur = conn.cursor()
    tasks = []
    for _, table, _ in SHARDS:
        total = _blocks(cur, table)
        step = total // args.slices + 1
        for i in range(args.slices):
            lo = i * step
            hi = 4294967295 if i == args.slices - 1 else min(total, (i + 1) * step)
            if lo < hi:
                tasks.append((table, lo, hi))
    conn.close()
    tmp = paths.output_dir / "panel_v2" / "legacy_freq_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    metas = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(terms,)) as pool:
        for f in [pool.submit(scan_slice, t, lo, hi, str(tmp))
                  for t, lo, hi in tasks]:
            metas.append(f.result())
    freq = aggregate_counts(metas, tmp, n_alias=len(terms))
    order = sorted(terms)
    rows = [(k, terms[k], int(freq[i])) for i, k in enumerate(order)]
    out_csv = paths.output_dir / "dictionary" / "legacy_df_freq_v1.csv"
    import pandas as pd
    pd.DataFrame(rows, columns=["match_key", "skill_id", "df_unique_text"]
                 ).to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info("legacy 频数落盘 %s（df>0: %d，df>=100: %d）", out_csv,
                int((freq > 0).sum()), int((freq >= 100).sum()))
    # 治理结果表（eps ai_dict 唯一合法写入目标；不动 A 级 zh_alias_freq）
    conn = psycopg2.connect(**eps_conn_params())
    try:
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS ai_dict.legacy_term_freq")
        c.execute("""CREATE TABLE ai_dict.legacy_term_freq (
            match_key text PRIMARY KEY, skill_id text, df_unique_text bigint)""")
        from psycopg2.extras import execute_values
        execute_values(c, "INSERT INTO ai_dict.legacy_term_freq VALUES %s",
                       rows, page_size=5000)
        conn.commit()
    finally:
        conn.close()
    print(f"legacy 频数完成: {len(rows)} 键，df≥100 {int((freq >= 100).sum())}，"
          f"df∈[10,100) {int(((freq >= 10) & (freq < 100)).sum())}，"
          f"df∈[5,10) {int(((freq >= 5) & (freq < 10)).sum())}，"
          f"df<5 {int((freq < 5).sum())}")


if __name__ == "__main__":
    main()
