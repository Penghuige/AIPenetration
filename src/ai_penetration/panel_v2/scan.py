"""panel_v2 M3-a：Pass2 识别扫描（§12.5 job_anchor_flag + job_skill_long）。

输入 M2 产物 job_master_gzsz（结果库）；eps 只读；产物写
output/panel_v2/pass2_handoff_scan/。交接合规版：

- master 导出为**裸 .npy + np.load(mmap_mode="r")**；按 §3.2 稳定 job_id 的
  63-bit 计算代理排序；
  目录含 .stamp.json 版号戳（MASTER_VERSION+n），复用前校验。
- raw 行通过 `SHA256(platform|job_id_raw)` 的同一 63-bit 代理回到 master；
  再用 canonical city + text_hash 双重确认，只扫描去重阶段选中的那条描述。
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
from ..text_clean import clean_description, text_hash, to_match
from .anchors import ANCHOR_RULES_VERSION, match_all_versions
from .dedup import (ADMISSION_WHERE, MASTER_VERSION, SHARDS, _blocks,
                    _stable_job_id, _stable_job_id_sha256)

logger = logging.getLogger("ai_penetration.panel_v2.scan")

GROUP_BITS = {"AI": 1, "ML": 2, "NLP": 4, "CVISION": 8,
              "CIMAGE": 16, "LLM": 32, "TRANS": 64}
FLUSH_ROWS = 500_000  # 每子批落盘行数（B1：禁止全切片累积）
SCAN_PIPELINE_VERSION = "handoff_v4_full_evidence_20260920"

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
    cur.execute("SELECT job_id, city, year, company_id, thash "
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
    # key 直接使用 §3.2 稳定 job_id = SHA256(platform|raw_id) 的 63-bit 代理。
    # pass2 对原始行用同一公式重算，避免任何 rid-only/平台编码旁路。
    for i, (jid, city_id, y, comp, hash_value) in enumerate(rows):
        key[i] = int(jid)
        job_id[i] = int(jid)
        year[i] = int(y)
        company[i] = comps.setdefault(str(comp), len(comps))
        city[i] = int(city_id)
        thash[i] = int(hash_value)
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    if np.any(k_sorted[1:] == k_sorted[:-1]):
        raise RuntimeError("canonical 稳定 job_id 碰撞")
    npy_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_master_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, arr in (("key", k_sorted), ("job_id", job_id[order]),
                      ("year", year[order]), ("company", company[order]),
                      ("city", city[order]), ("thash", thash[order])):
        p = tmp / f"{name}.npy"
        np.save(p, arr)
        p.replace(npy_dir / f"{name}.npy")   # 原子发布
    stamp.write_text(json.dumps({
        "version": MASTER_VERSION, "scan_pipeline_version": SCAN_PIPELINE_VERSION,
        "n": n}), encoding="utf-8")
    (out_dir / "company_vocab.json").write_text(
        json.dumps(comps), encoding="utf-8")
    logger.info("master npy 导出: %d 条 / company %d", n, len(comps))


def _load_governed_legacy_map() -> tuple[dict[str, str], set[str], str]:
    """读取 §10 治理后的正式 A/B/C map、歧义键与内容哈希。"""
    import hashlib
    import pandas as pd
    from .governance import governed_skill_map

    path = (
        get_project_paths().output_dir
        / "dictionary"
        / "skill_governed_ABCD_v4.csv"
    )
    if not path.exists():
        raise RuntimeError(
            "缺少 handoff-compliant 治理表 skill_governed_ABCD_v4.csv；"
            "请先完成 formal discovery review"
        )
    grades = pd.read_csv(path, encoding="utf-8-sig")
    mapping = governed_skill_map(grades)
    formal = grades[grades.final_grade.isin(["A", "B", "C"])].copy()
    ambiguous = {
        str(term).lower() for term, flag in zip(formal.term, formal.t2_ambig)
        if str(flag).strip().lower() in {"1", "true", "yes"}
    }
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return mapping, ambiguous, digest


def build_skill_vocab(out_dir: Path) -> None:
    """用最终 A/B/C 概念映射构建 skill_code 词表并绑定治理表哈希。"""
    path = out_dir / "skill_vocab.json"
    stamp = out_dir / "skill_vocab.stamp.json"
    governed, ambiguous, governance_hash = _load_governed_legacy_map()
    if path.exists() and stamp.exists():
        meta = json.loads(stamp.read_text(encoding="utf-8"))
        if (
            meta.get("scan_pipeline_version") == SCAN_PIPELINE_VERSION
            and meta.get("governance_sha256") == governance_hash
        ):
            return
    from .lexicon import _load_atier_alias_records, build_union_lexicon
    aliases = _load_atier_alias_records()
    lex = build_union_lexicon(
        legacy_terms=sorted(governed),
        aliases=aliases,
        legacy_id_map=governed,
        legacy_ambiguous_keys=ambiguous,
    )
    sids = sorted(set(lex.keys_map.values()))
    tmp = out_dir / "_vocab.json.tmp"
    tmp.write_text(
        json.dumps({s: i for i, s in enumerate(sids)}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    stamp.write_text(
        json.dumps({
            "scan_pipeline_version": SCAN_PIPELINE_VERSION,
            "governance_sha256": governance_hash,
            "n_skills": len(sids),
        }),
        encoding="utf-8",
    )
    logger.info("skill_vocab 落盘: %d skill（governance=%s）", len(sids),
                governance_hash[:12])


def _init_worker(npy_dir: str) -> None:
    """worker：mmap canonical master + 同一治理版本正式词表。"""
    _WORKER["m"] = {
        n: np.load(Path(npy_dir) / f"{n}.npy", mmap_mode="r")
        for n in ("key", "job_id", "year", "company", "city", "thash")
    }
    from .lexicon import _load_atier_alias_records, build_union_lexicon
    governed, ambiguous, governance_hash = _load_governed_legacy_map()
    aliases = _load_atier_alias_records()
    lex = build_union_lexicon(
        legacy_terms=sorted(governed),
        aliases=aliases,
        legacy_id_map=governed,
        legacy_ambiguous_keys=ambiguous,
    )
    vocab_path = Path(npy_dir).parent / "skill_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    built = sorted(set(lex.keys_map.values()))
    if not (
        len(built) == len(vocab)
        and all(vocab[s] == i for i, s in enumerate(built))
    ):
        raise RuntimeError("worker 词表与落盘 vocab 不一致")
    _WORKER["sid_to_code"] = vocab
    _WORKER["lex"] = lex
    _WORKER["governance_hash"] = governance_hash


def _flush_parts(out: Path, task: str, part: int,
                 flags_buf: list, long_buf: list, firm_buf: list,
                 text_buf: list) -> None:
    """子批 buffer 即时写 zstd parquet 分区（指南 §4.1）。"""
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
            "matched_anchor_groups_main": pa.array(cols[6], pa.string()),
            "matched_anchor_terms_main": pa.array(cols[7], pa.string()),
        }), out / "parts_flags" / f"{task}_{part:04d}.parquet",
                       compression="zstd")
    if long_buf:
        cols = list(zip(*long_buf))
        pq.write_table(pa.table({
            "job_id": pa.array(cols[0], pa.int64()),
            "year": pa.array(cols[1], pa.int32()),
            "skill_code": pa.array(cols[2], pa.int32()),
            "skill_id": pa.array(cols[3], pa.string()),
            "surface_form": pa.array(cols[4], pa.string()),
            "match_start": pa.array(cols[5], pa.int32()),
            "match_end": pa.array(cols[6], pa.int32()),
            "mention_count": pa.array(cols[7], pa.int32()),
            "match_method": pa.array(cols[8], pa.string()),
            "ambiguity_flag": pa.array(cols[9], pa.int8()),
            "span_verified": pa.array(cols[10], pa.int8()),
            "covered_candidate_count": pa.array(cols[11], pa.int32()),
            "covered_candidates": pa.array(cols[12], pa.string()),
        }), out / "parts_long" / f"{task}_{part:04d}.parquet",
                       compression="zstd")
    if text_buf:
        cols = list(zip(*text_buf))
        pq.write_table(pa.table({
            "job_id": pa.array(cols[0], pa.int64()),
            "job_id_sha256": pa.array(cols[1], pa.string()),
            "job_id_raw": pa.array(cols[2], pa.string()),
            "source_platform": pa.array(cols[3], pa.string()),
            "year": pa.array(cols[4], pa.int32()),
            "job_description_raw": pa.array(cols[5], pa.string()),
            "job_description_clean": pa.array(cols[6], pa.string()),
            "job_description_match": pa.array(cols[7], pa.string()),
            "text_hash": pa.array(cols[8], pa.int64()),
        }), out / "parts_text" / f"{task}_{part:04d}.parquet",
                       compression="zstd")
    if firm_buf:
        cols = list(zip(*firm_buf))
        pq.write_table(pa.table({
            "job_id": pa.array(cols[0], pa.int64()),
            "year": pa.array(cols[1], pa.int32()),
            "company_code": pa.array(cols[2], pa.int32()),
        }), out / "parts_firm" / f"{task}_{part:04d}.parquet",
                       compression="zstd")

def scan_slice(shard: str, city_id: int, lo: int, hi: int, out_dir: str) -> dict:
    """一个 ctid 切片的 pass2：buffer 满子批即落盘，meta 为完成标记。"""
    out = Path(out_dir)
    task = f"{shard}_{lo}"
    done = out / f"{task}.done.json"
    if done.exists():
        stat = json.loads(done.read_text(encoding="utf-8"))
        # 三校验（审计 D2：--slices 变化时同名任务范围不同，旧"完成"不可信）
        if (stat.get("version") == MASTER_VERSION and stat.get("hi") == hi
                and stat.get("rules") == ANCHOR_RULES_VERSION
                and stat.get("scan_pipeline_version") == SCAN_PIPELINE_VERSION
                and stat.get("governance_sha256") == _WORKER.get("governance_hash")):
            return stat
        logger.warning("切片 %s 完成戳不符（%s/%s/%s），清分片重扫", task,
                       stat.get("version"), stat.get("hi"), stat.get("rules"))
        for d in ("parts_flags", "parts_long", "parts_firm", "parts_text"):
            for stale in (out / d).glob(f"{task}_*.parquet"):
                stale.unlink()  # 防旧分片混入 merge（重扫部分失败留残）
        done.unlink()
    m = _WORKER["m"]
    lex = _WORKER["lex"]
    sid2code = _WORKER["sid_to_code"]
    keys = m["key"]
    conn = psycopg2.connect(**eps_conn_params())
    n_rows = n_canon = part = 0
    dup_hits = noncanonical_copies = 0
    hit_seen: set[int] = set()  # raw 同 (plat,city,rid) 真重复行只识别一次（§6.2.1.1）
    flags_buf: list = []
    long_buf: list = []
    firm_buf: list = []
    text_buf: list = []
    n_pairs = 0
    try:
        with conn.cursor() as setup:
            setup.execute("SET LOCAL work_mem = '256MB'")
            setup.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cur = conn.cursor(f"pass2_{task}")
        cur.itersize = 50000
        cur.execute(
            f"SELECT recruit_id, platform, job_description FROM public.{shard} "
            "WHERE ctid >= '(%s,0)'::tid AND ctid < '(%s,0)'::tid "
            + ADMISSION_WHERE,  # 单源谓词（审计 D3：禁再手抄）
            (int(lo), int(hi)))
        for d in ("parts_flags", "parts_long", "parts_firm", "parts_text"):
            (out / d).mkdir(parents=True, exist_ok=True)
        while True:
            batch = cur.fetchmany(50000)
            if not batch:
                break
            for rid, platform, desc in batch:
                n_rows += 1
                k = _stable_job_id(str(platform or ""), str(rid))
                idx = int(np.searchsorted(keys, k))
                if idx >= keys.size or int(keys[idx]) != k:
                    continue
                # 必须扫描 dedup 选中的 canonical 文本，而不是同 recruit_id 的
                # 任意 raw 拷贝。否则 ctid 顺序变化即可改变技能/锚点结果。
                if int(m["city"][idx]) != int(city_id):
                    noncanonical_copies += 1
                    continue
                raw_txt = str(desc)
                clean_txt = clean_description(raw_txt)
                match_txt = to_match(clean_txt)
                if text_hash(match_txt) != int(m["thash"][idx]):
                    noncanonical_copies += 1
                    continue
                if k in hit_seen:
                    dup_hits += 1
                    continue
                hit_seen.add(k)
                n_canon += 1
                job_id = int(m["job_id"][idx])
                yr = int(m["year"][idx])
                hits = match_all_versions(match_txt)
                bits = 0
                for g in hits["main"].groups:
                    bits |= GROUP_BITS[g]
                matches = lex.extract_matches(match_txt)
                flags_buf.append((
                    job_id, yr, hits["main"].flag, hits["cn_paper"].flag,
                    hits["babina"].flag, bits,
                    "|".join(hits["main"].groups),
                    "|".join(hits["main"].terms),
                ))
                for match in matches:
                    code = sid2code[match.skill_id]
                    long_buf.append((
                        job_id, yr, code, match.skill_id, match.surface_form,
                        match.start, match.end, match.mention_count,
                        match.match_method, match.ambiguity_flag,
                        int(match_txt[match.start:match.end] == match.surface_form),
                        match.covered_candidate_count,
                        match.covered_candidates,
                    ))
                firm_buf.append((job_id, yr, int(m["company"][idx])))
                text_buf.append((
                    job_id,
                    _stable_job_id_sha256(str(platform or ""), str(rid)),
                    str(rid), str(platform or ""), yr, raw_txt, clean_txt,
                    match_txt, int(m["thash"][idx]),
                ))
                n_pairs += len(matches)
                if len(flags_buf) >= FLUSH_ROWS:
                    _flush_parts(out, task, part, flags_buf, long_buf, firm_buf, text_buf)
                    flags_buf, long_buf, firm_buf, text_buf = [], [], [], []
                    part += 1
            if len(flags_buf) >= FLUSH_ROWS:  # fetchmany 边界也检查
                _flush_parts(out, task, part, flags_buf, long_buf, firm_buf, text_buf)
                flags_buf, long_buf, firm_buf, text_buf = [], [], [], []
                part += 1
        cur.close()
        _flush_parts(out, task, part, flags_buf, long_buf, firm_buf, text_buf)
    finally:
        conn.close()
    stat = {"task": task, "rows": n_rows, "canonical": n_canon,
            "skill_pairs": n_pairs, "dup_hits": dup_hits,
            "version": MASTER_VERSION, "lo": int(lo), "hi": int(hi),
            "rules": ANCHOR_RULES_VERSION,
            "scan_pipeline_version": SCAN_PIPELINE_VERSION,
            "governance_sha256": _WORKER.get("governance_hash"),
            "noncanonical_copies": noncanonical_copies}
    tmp = out / f".{task}.done.tmp"
    tmp.write_text(json.dumps(stat), encoding="utf-8")
    tmp.replace(done)
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
         " anchor_babina smallint, groups_main_bits int,"
         " matched_anchor_groups_main text, matched_anchor_terms_main text",
         "job_id"),
        ("job_skill_long", "parts_long",
         "job_id int8, year int, skill_code int, skill_id text, surface_form text,"
         " match_start int, match_end int, mention_count int, match_method text,"
         " ambiguity_flag smallint, span_verified smallint,"
         " covered_candidate_count int, covered_candidates text",
         "job_id, skill_code"),
        ("job_firm", "parts_firm", "job_id int8, year int, company_code int",
         "job_id"),
        ("job_text_clean", "parts_text",
         "job_id int8, job_id_sha256 text, job_id_raw text, source_platform text,"
         " year int, job_description_raw text, job_description_clean text,"
         " job_description_match text, text_hash int8",
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
        if not files:
            raise RuntimeError(f"{sub} 无分片")
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
        elif name == "job_text_clean":
            cur.execute(
                f"SELECT count(*) FROM (SELECT job_id FROM public.{stg} "
                f"GROUP BY job_id HAVING count(DISTINCT "
                f"(job_id_sha256, job_id_raw, source_platform, year, text_hash,"
                f" job_description_match)) > 1) x")
            conflict = cur.fetchone()[0]
        if conflict:
            raise RuntimeError(
                f"canonical 扫描出现不等价重复 {name}: {conflict} 个 job；"
                "拒绝用 ctid keep-first 仲裁")
        order_by = (f"{dedup_key}, match_start, match_end, surface_form"
                    if name == "job_skill_long" else f"{dedup_key}, job_id")
        cur.execute(f"CREATE TABLE public.{fin} AS SELECT DISTINCT ON ({dedup_key}) *"
                    f" FROM public.{stg} ORDER BY {order_by}")
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
                pa.int16(), pa.string(), pa.string()]
    if name == "job_skill_long":
        return [pa.int64(), pa.int32(), pa.int32(), pa.string(), pa.string(),
                pa.int32(), pa.int32(), pa.int32(), pa.string(), pa.int16(),
                pa.int16(), pa.int32(), pa.string()]
    if name == "job_text_clean":
        return [pa.int64(), pa.string(), pa.string(), pa.string(), pa.int32(),
                pa.string(), pa.string(), pa.string(), pa.int64()]
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
    parser.add_argument(
        "--out-tag", default="_handoff_scan",
        help="输出目录标签；默认隔离到 pass2_handoff_scan / panel_v2_handoff_scan，"
             "避免覆盖历史 panel_v2",
    )
    args = parser.parse_args()
    if args.workers > 8:
        raise SystemExit("--workers 不得超过 8（大表 IO 纪律）")
    paths = get_project_paths()
    setup_logging(paths.log_dir / f"panel_v2_scan{args.out_tag}.log")
    out_dir = paths.output_dir / "panel_v2" / f"pass2{args.out_tag}"
    rel_out = paths.output_dir / "release" / f"panel_v2{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_out.mkdir(parents=True, exist_ok=True)
    scan_manifest_path = rel_out / "scan_manifest.json"
    # 任何新 scan 一开始先撤销旧“完整”凭证；只有全链成功后才重建。
    scan_manifest_path.unlink(missing_ok=True)
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
    n_text = pq.ParquetFile(
        rel_out / "job_text_clean.parquet").metadata.num_rows
    if n_flag != n_master or n_text != n_master:
        raise SystemExit(
            f"守恒失败: flag={n_flag}, text={n_text}, master={n_master}"
            f"（Σcanonical={canon} 本地去重 {dup_hits}）")
    from .reproducibility import sha256_file
    governed_path = (
        paths.output_dir / "dictionary" / "skill_governed_ABCD_v4.csv"
    )
    scan_files = [
        rel_out / "job_anchor_flag.parquet",
        rel_out / "job_skill_long.parquet",
        rel_out / "job_firm.parquet",
        rel_out / "job_text_clean.parquet",
    ]
    scan_manifest = {
        "status": "formal_pass",
        "master_version": MASTER_VERSION,
        "scan_pipeline_version": SCAN_PIPELINE_VERSION,
        "anchor_rules_version": ANCHOR_RULES_VERSION,
        "governance_sha256": sha256_file(governed_path),
        "n_master": int(n_master),
        "n_flag": int(n_flag),
        "n_text": int(n_text),
        "files": {
            p.name: sha256_file(p) for p in scan_files
        },
    }
    scan_manifest_path.write_text(
        json.dumps(scan_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dur = (datetime.now() - t0).total_seconds() / 3600
    print(f"pass2 完成: canonical={canon:,}(dup hits {dup_hits:,}) "
          f"pairs={sum(s['skill_pairs'] for s in stats):,} "
          f"flag={n_flag:,} text={n_text:,} 用时 {dur:.2f}h，守恒核验通过 ✓")


if __name__ == "__main__":
    main()
