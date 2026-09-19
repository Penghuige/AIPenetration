"""panel_v2 M3-a：Pass2 识别扫描（§12.5 job_anchor_flag + job_skill_long）。

输入 M2 产物 job_master_gzsz（结果库）；eps 只读；产物写
output/panel_v2/pass2/。审计修复版（B1/B3/M9）：

- master 导出为**裸 .npy + np.load(mmap_mode="r")**（npz 成员不真 mmap，
  实证每 worker 会私载全量——已修正），按 key=blake2b-63(recruit_id) 排序；
  目录含 .stamp.json 版号戳（MASTER_VERSION+n），复用前校验。
- 命中键最终定稿 **rid-only**（master rid 实证全局唯一）：同一 rid 的多份
  raw 拷贝全部映射到同一 job_id，跨切片重复由 worker hit_seen 片内去重 +
  merge 端 PG DISTINCT ON 仲裁；keep-first 的"多拷贝识别等价"假设由
  merge_parts 冲突计数**实测披露**（flag/firm 逐 job 多值计数），全国重跑
  预检须复核该计数（审计 D4）。
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
from ..text_clean import match_from_raw, text_hash
from .anchors import ANCHOR_RULES_VERSION, match_all_versions
from .dedup import ADMISSION_WHERE, MASTER_VERSION, SHARDS, _blocks, _h63

logger = logging.getLogger("ai_penetration.panel_v2.scan")

GROUP_BITS = {"AI": 1, "ML": 2, "NLP": 4, "CVISION": 8,
              "CIMAGE": 16, "LLM": 32, "TRANS": 64}
FLUSH_ROWS = 500_000  # 每子批落盘行数（B1：禁止全切片累积）
SCAN_PIPELINE_VERSION = "handoff_v3_longest_span_primary_20260919"

_WORKER: dict = {}


def _results_conn():
    import psycopg2 as _p
    rp = get_project_paths().results_connection_params
    return _p.connect(**rp)




def export_master(out_dir: Path) -> None:
    """job_master → 4 个排序裸 .npy（mmap 真共享）。原子写（M9）。

    过滤键 = h63(job_id_raw)（**rid-only 定稿**，2026-09-08 三迭代结论：
    复合键 (plat,city,rid) 因 pmap 采样对长尾平台编码覆盖缺口实证漏命中
    9,867 例已弃用；master rid 全局唯一由下方碰撞断言保证）。
    复用需过版号戳校验（审计 D2：三件存在≠同代）。
    """
    npy_dir = out_dir / "master_npy"
    stamp = npy_dir / ".stamp.json"
    names = ("key", "job_id", "year", "company", "city", "thash")
    if all((npy_dir / f"{n}.npy").exists() for n in names) and stamp.exists():
        s = json.loads(stamp.read_text(encoding="utf-8"))
        if (s.get("version") == MASTER_VERSION
                and s.get("scan_pipeline_version") == SCAN_PIPELINE_VERSION):
            logger.info("master npy 复用（%s，n=%d）", s["version"], s["n"])
            return
        logger.warning("master npy 版号不符（%s != %s），重建",
                       s.get("version"), MASTER_VERSION)
    conn = _results_conn()
    cur = conn.cursor()
    cur.execute("SELECT job_id_raw, plat, job_id, city, year, company_id, thash "
                "FROM public.job_master_gzsz")
    rows = cur.fetchall()
    conn.close()
    n = len(rows)
    logger.info("job_master 载入: %d 行", n)
    key = np.empty(n, np.int64)
    job_id = np.empty(n, np.int64)
    year = np.empty(n, np.int32)
    company = np.empty(n, np.int32)
    city = np.empty(n, np.int16)
    thash = np.empty(n, np.int64)
    comps: dict[str, int] = {}
    # key = h63(rid)：master 的 rid 全局唯一（实证 0 重复组），跨切片/跨平台
    # 的重复命中由 worker hit_seen + PG 端 DISTINCT ON 仲裁（rid 复合平台编码
    # 会因映射覆盖缺口产生漏命中，2026-09-07 实证 9867 例，已弃用）
    for i, (rid, _plat, jid, city_id, y, comp, hash_value) in enumerate(rows):
        key[i] = _h63(str(rid))
        job_id[i] = int(jid)
        year[i] = int(y)
        company[i] = comps.setdefault(str(comp), len(comps))
        city[i] = int(city_id)
        thash[i] = int(hash_value)
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    assert not np.any(k_sorted[1:] == k_sorted[:-1]), "canonical key 哈希碰撞"
    npy_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_master_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, arr in (("key", k_sorted), ("job_id", job_id[order]),
                      ("year", year[order]), ("company", company[order]),
                      ("city", city[order]), ("thash", thash[order])):
        p = tmp / f"{name}.npy"
        np.save(p, arr)
        p.rename(npy_dir / f"{name}.npy")   # 原子发布
    stamp.write_text(json.dumps({
        "version": MASTER_VERSION, "scan_pipeline_version": SCAN_PIPELINE_VERSION,
        "n": n}), encoding="utf-8")
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
    """worker：mmap master + 词表一致性断言（B3）。"""
    _WORKER["m"] = {n: np.load(Path(npy_dir) / f"{n}.npy", mmap_mode="r")
                    for n in ("key", "job_id", "year", "company", "city", "thash")}
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
        stat = json.loads(done.read_text(encoding="utf-8"))
        # 三校验（审计 D2：--slices 变化时同名任务范围不同，旧"完成"不可信）
        if (stat.get("version") == MASTER_VERSION and stat.get("hi") == hi
                and stat.get("rules") == ANCHOR_RULES_VERSION):
            return stat
        logger.warning("切片 %s 完成戳不符（%s/%s/%s），清分片重扫", task,
                       stat.get("version"), stat.get("hi"), stat.get("rules"))
        for d in ("parts_flags", "parts_long", "parts_firm"):
            for stale in (out / d).glob(f"{task}_*.parquet"):
                stale.unlink()  # 防旧分片混入 merge（重扫部分失败留残）
        done.unlink()
    m = _WORKER["m"]
    lex = _WORKER["lex"]
    sid2code = _WORKER["sid_to_code"]
    keys = m["key"]
    conn = psycopg2.connect(**eps_conn_params())
    n_rows = n_canon = part = 0
    dup_hits = 0
    hit_seen: set[int] = set()  # raw 同 (plat,city,rid) 真重复行只识别一次（§6.2.1.1）
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
            + ADMISSION_WHERE,  # 单源谓词（审计 D3：禁再手抄）
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
                if k in hit_seen:
                    dup_hits += 1
                    continue
                hit_seen.add(k)
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
            "skill_pairs": n_pairs, "dup_hits": dup_hits,
            "version": MASTER_VERSION, "lo": int(lo), "hi": int(hi),
            "rules": ANCHOR_RULES_VERSION}
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
    """parts 经结果库 PG 去重（DISTINCT ON）后流式转 parquet。

    2026-09-07 教训：2 亿行 pandas concat+drop_duplicates 内存洪峰疑似
    压垮同机 PG（UNLOGGED 表被重启清空）——大表 dedup 是 PG 的本职，
    Python 端只做流式 CSV→parquet 转换。跨切片重复命中（rid 在 raw 有
    真重复行）由 DISTINCT ON keep-first 仲裁；"同 canonical 识别等价"
    是假设而非前提（审计 D4），此处逐 job 实测冲突数并披露（flag/firm
    多值即冲突；long 表 keep-first 天然吸收集合差异，不可从此路测量）。
    """
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    rel_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        ("job_anchor_flag", "parts_flags",
         "job_id int8, year int, anchor_main smallint, anchor_cn_paper smallint,"
         " anchor_babina smallint, groups_main_bits int",
         "job_id"),
        ("job_skill_long", "parts_long",
         "job_id int8, year int, skill_code int",
         "job_id, skill_code"),
        ("job_firm", "parts_firm", "job_id int8, year int, company_code int",
         "job_id"),
    )
    conn = _results_conn()
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET maintenance_work_mem = '2GB'")
    cur.execute("SET work_mem = '1GB'")
    for name, sub, cols, dedup_key in specs:
        stg = f"_{name}_stg"
        fin = f"_{name}_fin"
        cur.execute(f"DROP TABLE IF EXISTS public.{stg} CASCADE")
        cur.execute(f"DROP TABLE IF EXISTS public.{fin} CASCADE")
        cur.execute(f"CREATE TABLE public.{stg} ({cols})")
        files = sorted((out_dir / sub).glob("*.parquet"))
        assert files, f"{sub} 无分片"
        for f in files:
            bio = _table_to_csv_buf(f)
            cur.copy_expert(
                f"COPY public.{stg} FROM STDIN WITH (FORMAT csv)", bio)
        conn.commit()
        # keep-first 等价假设实测（审计 D4）：同一 job_id 的多拷贝行若识别
        # 结果不同，即存在"扫的描述≠定群描述"的真实分歧面
        conflict = 0
        if name == "job_anchor_flag":
            cur.execute(
                f"SELECT count(*) FROM (SELECT job_id FROM public.{stg} "
                "GROUP BY job_id HAVING count(DISTINCT (year, anchor_main,"
                " anchor_cn_paper, anchor_babina, groups_main_bits)) > 1) x")
            conflict = cur.fetchone()[0]
        elif name == "job_firm":
            cur.execute(
                f"SELECT count(*) FROM (SELECT job_id FROM public.{stg} "
                f"GROUP BY job_id HAVING count(DISTINCT company_code) > 1) x")
            conflict = cur.fetchone()[0]
        if conflict:
            logger.warning("keep-first 冲突披露 %s: %d 个 job 多拷贝识别不等价"
                           "（受影响行由 ctid 序 keep-first 裁决，跨运行可翻转）",
                           name, conflict)
        cur.execute(f"CREATE TABLE public.{fin} AS SELECT DISTINCT ON ({dedup_key}) *"
                    f" FROM public.{stg} ORDER BY {dedup_key}, job_id")
        conn.commit()
        cur.execute(f"SELECT count(*) FROM public.{stg}")
        before = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM public.{fin}")
        after = cur.fetchone()[0]
        # PG COPY 出 CSV → pyarrow 流式转 parquet（内存 O(row group)）
        csv_tmp = out_dir / f"{name}.csv"
        with csv_tmp.open("w", encoding="utf-8", newline="") as fh:
            cur.copy_expert(
                f"COPY public.{fin} TO STDOUT WITH (FORMAT csv, HEADER true)",
                fh)
        tbl = pcsv.read_csv(
            csv_tmp,
            read_options=pcsv.ReadOptions(block_size=1 << 24),
            convert_options=pcsv.ConvertOptions(
                column_types={c: t for c, t in zip(
                    [x.split()[0] for x in cols.split(", ")],
                    _arrow_types(name))}))
        pq.write_table(tbl, rel_dir / f"{name}.parquet", compression="zstd")
        csv_tmp.unlink(missing_ok=True)
        cur.execute(f"DROP TABLE public.{stg}")
        cur.execute(f"DROP TABLE public.{fin}")
        conn.commit()
        logger.info("PG 去重合并 %s: %d -> %d 行", name, before, after)
    conn.close()


def _arrow_types(name: str) -> list:
    import pyarrow as pa
    if name == "job_anchor_flag":
        return [pa.int64(), pa.int32(), pa.int16(), pa.int16(), pa.int16(),
                pa.int16()]
    if name == "job_skill_long":
        return [pa.int64(), pa.int32(), pa.int32()]
    return [pa.int64(), pa.int32(), pa.int32()]


def _table_to_csv_buf(path):
    """parquet 分片转 CSV BytesIO（COPY 输入，无表头）。"""
    import io
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    tbl = pq.read_table(path)
    buf = io.BytesIO()
    pcsv.write_csv(tbl, buf, write_options=pcsv.WriteOptions(
        include_header=False))
    buf.seek(0)
    return buf


def main() -> None:
    parser = argparse.ArgumentParser(description="panel_v2 pass2 识别扫描")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--slices", type=int, default=4)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--out-tag", default="",
                        help="输出目录标签：pass2<tag> 与 release/panel_v2<tag>"
                             "（v2h 重扫用，避免覆盖 v2ac 基底）")
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / f"panel_v2_scan{args.out_tag}.log")
    out_dir = paths.output_dir / "panel_v2" / f"pass2{args.out_tag}"
    rel_out = paths.output_dir / "release" / f"panel_v2{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("锚点规则版本: %s | 主样本版号: %s",
                ANCHOR_RULES_VERSION, MASTER_VERSION)
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
    dup_hits = sum(s.get("dup_hits", 0) for s in stats)
    merge_parts(out_dir, rel_out)
    # 守恒终判：合并去重后的 flag 行数必须等于 master（Σcanonical 允许多计
    # 跨切片重复命中，由 merge keep-first 归一）
    import pyarrow.parquet as pq
    n_flag = pq.ParquetFile(
        rel_out / "job_anchor_flag.parquet").metadata.num_rows
    if n_flag != n_master:
        raise SystemExit(
            f"守恒失败: 合并后 flag {n_flag} != master {n_master}"
            f"（Σcanonical={canon} 本地去重 {dup_hits}）")
    dur = (datetime.now() - t0).total_seconds() / 3600
    print(f"pass2 完成: canonical={canon:,}(dup hits {dup_hits:,}) "
          f"pairs={sum(s['skill_pairs'] for s in stats):,} "
          f"flag={n_flag:,} 用时 {dur:.2f}h，守恒核验通过 ✓")


if __name__ == "__main__":
    main()
