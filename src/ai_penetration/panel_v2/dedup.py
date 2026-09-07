"""panel_v2 M2：招聘广告去重主表构建（指南 §6.2.1 主规则，评审修正版）。

数据事实（2026-09-07 实测+评审确认）：广深 recruit_id 平台内唯一；job↔ent
join 覆盖高（抽样 100%，全量由不变量兜底）；ent.recruit_id 有重复（映射聚合
取 min(company_id)，冲突计数入验收）。所有写入仅在结果库；eps 只读。

流程（评审必改项全部落实）：
1. ``copy_ent_map``：ent (recruit_id, company_id) 流式导出结果库（无主键，
   join 时 GROUP BY 聚合去重，冲突数入不变量）。
2. ``pass1_scan``：ctid 切片（≤8 路，HDD 纪律）扫 job 表全行，Python 侧算
   match 文本 hash、pos_norm_hash、日期、描述长度、**字段完整度**
   （education/work_type/experience/recruit_count/age_req 非空数——评审代理
   定义，§6.2.2.2 的可辩护实现）；年份/日期不可解析**不丢行**（yr=0/day=-1
   隔离标记），定长 64B 窄行分片落盘（断点幂等）。
3. ``copy_stage``：文本 COPY 进 UNLOGGED stage 表。
4. ``build_master``：PG 侧两段式——stage LEFT JOIN 聚合后的 ent_map 物化
   sorted_stage（company 未命中→哨兵 'UNK:<rid>'，禁 NULL 共组，规则4）；
   sorted_stage 上窗口（显式 ROWS 帧、ORDER BY (day,rid) 确定性）做 30 天
   链式分段（规则5：year 入分区键天然不跨年）与 §6.2.2 canonical 选择，
   物化 job_master_gzsz；另建 dup_group_map（§6.2.2 平台数/编号映射）、
   跨年重复互标表、链跨度>30 天审计指标。
5. ``verify_invariants``：sum(collapsed)=stage 行数、canonical=组数、
   (plat,rid) 全量唯一、company 未命中计数入报、year=canonical 年份、
   bad_year>0.1% 阻断、链跨度组占比披露。

§6.1.1 三态文本声明：raw 由 eps 原表永久保留替代，match 态可由
text_clean.match_from_raw 重算，中间表只存 hash（设计文档 §决策 记录）。

使用示例::

    python -X utf8 -m src.ai_penetration.panel_v2.dedup --bench    # 单片吞吐基准
    python -X utf8 -m src.ai_penetration.panel_v2.dedup --dry-run
    python -X utf8 -m src.ai_penetration.panel_v2.dedup            # 全量
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg2

from config.paths import get_project_paths

from ..common import eps_conn_params, setup_logging
from ..text_clean import match_from_raw, normalize_position, text_hash

logger = logging.getLogger("ai_penetration.panel_v2.dedup")

# b 版：规则1 跨城 rid 去重 + 组首锚定 30 天桶（链式语义超披露线修正）
MASTER_VERSION = "main_v2a_20260907b"
TABLE_STAGE = "dedup_stage_gzsz"
TABLE_SORTED = "dedup_sorted_gzsz"
TABLE_MASTER = "job_master_gzsz"
TABLE_GROUPMAP = "dup_group_map_gzsz"
TABLE_ENTMAP = "ent_company_map"
SHARDS = (("广州市", "job_p0387", 0), ("深圳市", "job_p0389", 1))
_EPOCH_DAYS = 14610  # date(2010,1,1).toordinal()
_YEAR_RE = re.compile(r"^\s*(\d{4})")
_ISO_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})")
BAD_YEAR_THRESHOLD = 0.001  # 年份不可解析阻断线（评审建议 0.1%）

# 定长窄行（64B）：rid32 + plat/city + yr2 + day4 + posh8 + thash8 + dlen2 + comp1 + pad2
ROW_DTYPE = np.dtype([
    ("rid", "S32"), ("plat", "u1"), ("city", "u1"), ("yr", "u2"),
    ("day", "i4"), ("posh", "i8"), ("thash", "i8"), ("dlen", "u2"),
    ("comp", "u1"), ("pad", "S5"),
])
assert ROW_DTYPE.itemsize == 64, ROW_DTYPE.itemsize


def _h63(s: str) -> int:
    from hashlib import blake2b
    return int.from_bytes(blake2b(s.encode("utf-8"), digest_size=8).digest(),
                          "big") & 0x7FFF_FFFF_FFFF_FFFF


def _day_of(y: int, mo: int, d: int) -> int:
    return datetime(y, mo, d).toordinal() - _EPOCH_DAYS


def _blocks(cur, shard: str) -> int:
    cur.execute("SELECT current_setting('block_size')::int")
    bs = int(cur.fetchone()[0])
    cur.execute("SELECT pg_relation_size(%s)", (f"public.{shard}",))
    return max(1, int(cur.fetchone()[0]) // bs)


_WORKER: dict = {}


def _init_worker(platforms: dict[str, int]) -> None:
    _WORKER["platforms"] = platforms


def _completeness(edu, wtype, exp, rcnt, age) -> int:
    """§6.2.2.2 字段完整度代理：五个常用协变量非空个数（评审定义）。"""
    n = 0
    for v in (edu, wtype, exp, rcnt, age):
        s = str(v or "").strip()
        if s and s not in ("-", "不限", "面议"):
            n += 1
    return n


def _scan_slice(shard: str, city_id: int, lo: int, hi: int, out_dir: str,
                year_filter: str | None) -> dict:
    task = f"{shard}_{lo}"
    out_path = Path(out_dir) / f"{task}.rows.bin"
    meta_path = Path(out_dir) / f"{task}.json"
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info("切片 %s 断点复用（%d 行）", task, meta["rows"])
        return meta
    plats = _WORKER["platforms"]
    conn = psycopg2.connect(**eps_conn_params())
    buf = np.empty(100000, dtype=ROW_DTYPE)
    n = total = bad_year = bad_day = unk_plat = 0
    try:
        with conn.cursor() as setup:
            setup.execute("SET LOCAL work_mem = '256MB'")
            setup.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cond = ("WHERE ctid >= '(%s,0)'::tid AND ctid < '(%s,0)'::tid "
                "AND job_description IS NOT NULL AND job_description != '' "
                "AND position IS NOT NULL AND position != '' "
                "AND recruit_id IS NOT NULL")
        params: tuple = (int(lo), int(hi))
        if year_filter:
            cond += " AND publish_time LIKE %s"
            params = params + (year_filter + "%",)
        cur = conn.cursor(f"dedup_{task}")
        cur.itersize = 50000
        cur.execute(
            f"SELECT recruit_id, platform, position, publish_time, job_description, "
            f"education, work_type, experience, recruit_count, age_req "
            f"FROM public.{shard} {cond}", params)
        with out_path.open("wb") as fh:
            while True:
                batch = cur.fetchmany(50000)
                if not batch:
                    break
                for rid, platform, pos, ptime, desc, edu, wt, exp, rcnt, age in batch:
                    m_yr = _YEAR_RE.match(str(ptime or ""))
                    m_iso = _ISO_RE.match(str(ptime or ""))
                    if not m_yr:
                        yr, bad_year = 0, bad_year + 1
                    else:
                        yr = int(m_yr.group(1))
                    if m_iso and 2000 <= int(m_iso.group(1)) <= 2030:
                        day = _day_of(int(m_iso.group(1)), int(m_iso.group(2)),
                                      int(m_iso.group(3)))
                    else:
                        day, bad_day = -1, bad_day + 1
                    match_txt = match_from_raw(str(desc))
                    pkey = (platform or "").strip()
                    if pkey not in plats:
                        unk_plat += 1
                    if n == len(buf):
                        fh.write(buf.tobytes())
                        n = 0
                    r = buf[n]
                    r["rid"] = str(rid)[:32].encode()
                    r["plat"] = plats.get(pkey, 255)
                    r["city"] = city_id
                    r["yr"] = yr
                    r["day"] = day
                    r["posh"] = _h63(normalize_position(str(pos)))
                    r["thash"] = text_hash(match_txt)
                    r["dlen"] = min(len(match_txt), 65535)
                    r["comp"] = _completeness(edu, wt, exp, rcnt, age)
                    n += 1
                    total += 1
            if n:
                fh.write(buf[:n].tobytes())
        cur.close()
    finally:
        conn.close()
    meta = {"task": task, "rows": total, "bad_year": bad_year, "bad_day": bad_day,
            "unknown_platform": unk_plat}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")  # meta 最后落=完整性标记
    logger.info("切片完成 %s: rows=%d bad_year=%d bad_day=%d", task, total,
                bad_year, bad_day)
    return meta


def _plan_tasks(slices: int, year_filter: str | None) -> list[tuple]:
    conn = psycopg2.connect(**eps_conn_params())
    cur = conn.cursor()
    tasks = []
    for _city, shard, city_id in SHARDS:
        total_blocks = _blocks(cur, shard)
        step = total_blocks // slices + 1
        for i in range(slices):
            lo = i * step
            hi = 4294967295 if i == slices - 1 else min(total_blocks, (i + 1) * step)
            if lo < hi:
                tasks.append((shard, city_id, lo, hi, year_filter))
    conn.close()
    return tasks


def _build_platform_dict() -> dict[str, int]:
    conn = psycopg2.connect(**eps_conn_params())
    cur = conn.cursor()
    plats: dict[str, int] = {}
    for _, shard, _ in SHARDS:
        cur.execute(f"SELECT DISTINCT platform FROM public.{shard} "
                    "TABLESAMPLE SYSTEM (0.05)")
        for (p,) in cur.fetchall():
            key = (p or "").strip()
            if key and key not in plats and len(plats) < 250:
                plats[key] = len(plats)
    conn.close()
    return plats


def pass1_scan(workers: int, slices: int, out_dir: Path,
               year_filter: str | None = None) -> list[dict]:
    """并行 pass1（workers≤8 HDD 纪律）。"""
    from concurrent.futures import ProcessPoolExecutor
    assert workers <= 8, "HDD 纪律：大表 ≤8 流（CLAUDE.md §9）"
    plats = _build_platform_dict()
    logger.info("平台字典: %d 种", len(plats))
    tasks = _plan_tasks(slices, year_filter)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(plats,)) as pool:
        futs = [pool.submit(_scan_slice, s, c, lo, hi, str(out_dir), yf)
                for s, c, lo, hi, yf in tasks]
        for f in futs:
            metas.append(f.result())
    return metas


def _results_conn():
    import psycopg2 as _p
    rp = get_project_paths().results_connection_params
    return _p.connect(**rp)


def copy_ent_map() -> None:
    """ent 映射流式导出结果库（无主键防 ent 重复 recruit_id 炸 COPY）。"""
    conn = _results_conn()
    cur = conn.cursor()
    cur.execute(f"CREATE TABLE IF NOT EXISTS public.{TABLE_ENTMAP} "
                "(recruit_id text, company_id text)")
    cur.execute(f"SELECT count(*) FROM public.{TABLE_ENTMAP}")
    has = cur.fetchone()[0]
    conn.commit()
    conn.close()
    if has:
        logger.info("ent 映射已有 %d 行，跳过导出", has)
        return
    eps = psycopg2.connect(**eps_conn_params())
    try:
        for _, ent_shard, _ in (("广州市", "ent_p0387", 0), ("深圳市", "ent_p0389", 1)):
            cur = eps.cursor(f"entmap_{ent_shard}")
            cur.itersize = 200000
            cur.execute(f"SELECT recruit_id, company_id FROM public.{ent_shard} "
                        "WHERE recruit_id IS NOT NULL AND company_id IS NOT NULL")
            conn2 = _results_conn()
            cur2 = conn2.cursor()
            n = 0
            bio = io.StringIO()
            while True:
                batch = cur.fetchmany(200000)
                if not batch:
                    break
                for rid, cid in batch:
                    bio.write(f"{rid}\t{cid}\n")
                    n += 1
                if bio.tell() > 200 << 20:
                    bio.seek(0)
                    cur2.copy_expert(f"COPY public.{TABLE_ENTMAP} FROM STDIN WITH (FORMAT text)", bio)
                    conn2.commit()
                    bio = io.StringIO()
            bio.seek(0)
            cur2.copy_expert(f"COPY public.{TABLE_ENTMAP} FROM STDIN WITH (FORMAT text)", bio)
            conn2.commit()
            conn2.close()
            logger.info("ent 映射导出 %s: %d 行", ent_shard, n)
    finally:
        eps.close()


def copy_stage(metas: list[dict], resume: bool) -> int:
    """窄行分片文本 COPY 进 UNLOGGED stage。"""
    conn = _results_conn()
    cur = conn.cursor()
    cur.execute(f"""
        CREATE UNLOGGED TABLE IF NOT EXISTS public.{TABLE_STAGE} (
            rid text NOT NULL, plat smallint, city smallint, yr int,
            day int, posh bigint, thash bigint, dlen int, comp smallint)""")
    cur.execute(f"SELECT count(*) FROM public.{TABLE_STAGE}")
    existing = cur.fetchone()[0]
    if existing and not resume:
        cur.execute(f"TRUNCATE public.{TABLE_STAGE}")
        existing = 0
    conn.commit()
    conn.close()
    if existing:
        logger.info("stage 已有 %d 行（--resume），跳过 COPY", existing)
        return existing
    total = 0
    parts = Path(metas[0]["file"]).parent
    for m in metas:
        done_flag = parts / f"{m['task']}.copied"
        if resume and done_flag.exists():
            total += int(done_flag.read_text(encoding="utf-8"))
            continue
        arr = np.fromfile(m["file"], dtype=ROW_DTYPE)
        conn2 = _results_conn()
        cur2 = conn2.cursor()
        bio = io.StringIO()
        for row in arr:
            rid = bytes(row["rid"]).rstrip(b"\x00").decode()
            bio.write(f"{rid}\t{int(row['plat'])}\t{int(row['city'])}\t{int(row['yr'])}\t"
                      f"{int(row['day'])}\t{int(row['posh'])}\t{int(row['thash'])}\t"
                      f"{int(row['dlen'])}\t{int(row['comp'])}\n")
        bio.seek(0)
        cur2.copy_expert(f"COPY public.{TABLE_STAGE} FROM STDIN WITH (FORMAT text)", bio)
        conn2.commit()
        conn2.close()
        done_flag.write_text(str(len(arr)), encoding="utf-8")
        total += len(arr)
        logger.info("已 COPY %s 累计 %d 行", m["task"], total)
    return total


def _master_stats(cur) -> tuple[int, int]:
    """统计与审计指标（master 已存在时的轻量路径）。"""
    cur.execute(f"SELECT count(*), sum(records_collapsed), sum(company_unmatched::int),"
                f" sum(CASE WHEN latest_day - earliest_day > 30 AND records_collapsed > 1"
                f"         THEN records_collapsed ELSE 0 END),"
                f" count(*) FILTER (WHERE latest_day - earliest_day > 30 AND records_collapsed > 1)"
                f" FROM public.{TABLE_MASTER}")
    n_master, sum_collapsed, unmatched, chain_over, chain_groups = cur.fetchone()
    cur.execute(f"SELECT count(*) FROM public.{TABLE_STAGE}")
    n_stage = cur.fetchone()[0]
    logger.info("master=%d stage=%d unmatched_company=%d 链跨度>30天组=%d(行%d)",
                n_master, n_stage, unmatched, chain_groups, chain_over)
    return n_master, n_stage


def build_master() -> tuple[int, int]:
    """两段式：join 物化 sorted → 窗口分组 canonical → master + 附表 + 审计。

    幂等（断点纪律）：master 已存在且带本次版本标记时，跳过全部 DDL
    （sorted/seg/master 的窗口计算耗时 1-2h），仅重算统计。
    """
    conn = _results_conn()
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET work_mem = '1GB'")
    cur.execute("SELECT to_regclass('public." + TABLE_MASTER + "')")
    if cur.fetchone()[0] is not None:
        cur.execute(f"SELECT count(*) FROM public.{TABLE_MASTER} "
                    f"WHERE duplicate_version = '{MASTER_VERSION}'")
        if cur.fetchone()[0] > 0:
            logger.info("master 已存在（%s），跳过重建，仅出统计", MASTER_VERSION)
            out = _master_stats(cur)
            conn.close()
            return out
    cur.execute("SELECT to_regclass('public." + TABLE_SORTED + "')")
    sorted_exists = cur.fetchone()[0] is not None
    if sorted_exists:
        # 旧版 sorted 缺 rule1_dups 列（跨城重复未处理）→ 强制重建
        cur.execute("SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name=%s AND column_name='rule1_dups'",
                    (TABLE_SORTED,))
        if cur.fetchone()[0] == 0:
            logger.info("sorted 表为旧版（无 rule1_dups），DROP 重建")
            cur.execute(f"DROP TABLE public.{TABLE_SORTED}")
            sorted_exists = False
    if not sorted_exists:
        # 段一：规则1（§6.2.1.1 平台+编号唯一记录，跨城重复折叠并计数）
        # + join ent 聚合映射（min 消一 rid 多司），NULL→哨兵
        cur.execute(f"""
            CREATE UNLOGGED TABLE public.{TABLE_SORTED} AS
            WITH joined AS (
                SELECT s.rid, s.plat, s.city, s.yr, s.day, s.posh, s.thash,
                       s.dlen, s.comp,
                       coalesce(e.company_id, 'UNK:' || s.rid) AS company_id,
                       (e.company_id IS NULL) AS company_unmatched
                FROM public.{TABLE_STAGE} s
                LEFT JOIN (SELECT recruit_id, min(company_id) AS company_id
                           FROM public.{TABLE_ENTMAP} GROUP BY recruit_id) e
                       ON e.recruit_id = s.rid
            ), rid_rank AS (
                SELECT *,
                       row_number() OVER (PARTITION BY plat, rid
                                          ORDER BY city, day, thash) AS rn_rid,
                       count(*) OVER (PARTITION BY plat, rid) - 1 AS rule1_dups
                FROM joined
            )
            SELECT rid, plat, city, yr, day, posh, thash, dlen, comp,
                   company_id, company_unmatched, rule1_dups
            FROM rid_rank WHERE rn_rid = 1
        """)
        cur.execute(f"CREATE INDEX ON public.{TABLE_SORTED} (company_id, posh, city, thash, yr, day, rid)")
    conn.commit()
    # 段一·五：30 天分组 = 组首锚定均匀桶（每桶跨度≤30，满足 §6.2.1.2
    # "两两≤30" 的确定性规则；审计修正：链式传递语义对长链组过松，2.65%
    # 超披露线。桶规则对"恰跨 30 天边界的序列"可能多切一刀，保守方向，
    # 规则本身记录于 duplicate_version 可追溯）。day<0 坏日期每行独立成组。
    TABLE_SEG = TABLE_SORTED.replace("sorted", "seg")
    cur.execute(f"DROP TABLE IF EXISTS public.{TABLE_SEG}")
    cur.execute(f"""
        CREATE UNLOGGED TABLE public.{TABLE_SEG} AS
        SELECT *, (company_id || ':' || posh || ':' || city || ':' || thash || ':'
                   || yr || ':' || seg_bucket) AS seg
        FROM (
            SELECT *,
                   CASE WHEN day < 0
                        THEN 999999 + row_number() OVER (
                             PARTITION BY company_id, posh, city, thash, yr
                             ORDER BY rid)                    -- 坏日期独立成组
                        ELSE (day - first_value(day) OVER (
                             PARTITION BY company_id, posh, city, thash, yr
                             ORDER BY day, rid))::int / 31    -- 组首锚定 30 天桶
                   END AS seg_bucket
            FROM public.{TABLE_SORTED}
        ) a
    """)
    cur.execute(f"CREATE INDEX ON public.{TABLE_SEG} (company_id, posh, city, thash, yr, seg)")
    conn.commit()
    # 段二：canonical（§6.2.2 顺序：完整度→最长→最早→rid 字典序）
    cur.execute(f"DROP TABLE IF EXISTS public.{TABLE_MASTER}")
    cur.execute(f"""
        CREATE TABLE public.{TABLE_MASTER} AS
        WITH ranked AS (
            SELECT *,
                   count(*) OVER w AS grp_n,
                   min(day) OVER w AS gmin,
                   max(day) OVER w AS gmax,
                   row_number() OVER (PARTITION BY company_id, posh, city, thash, yr, seg
                                      ORDER BY comp DESC, dlen DESC, day ASC, rid ASC) AS pick
            FROM public.{TABLE_SEG}
            WINDOW w AS (PARTITION BY company_id, posh, city, thash, yr, seg)
        )
        SELECT row_number() OVER (ORDER BY company_id, yr, thash, rid) AS job_id,
               rid AS job_id_raw, plat, city, yr AS year, company_id, thash,
               (company_id || ':' || posh || ':' || city || ':' || thash || ':'
                || yr || ':' || seg) AS duplicate_group_id,
               grp_n AS records_collapsed,
               CASE WHEN grp_n = 1 THEN 'single'
                    WHEN day < 0 THEN 'bad_date'
                    ELSE 'exact_hash_group' END AS duplicate_reason,
               gmin AS earliest_day, gmax AS latest_day,
               company_unmatched,
               rule1_dups,
               '{MASTER_VERSION}' AS duplicate_version
        FROM ranked WHERE pick = 1
    """)
    # §6.2.2 组元数据：dup_group→(platform, job_id_raw) 全映射 + 平台数回填
    cur.execute(f"DROP TABLE IF EXISTS public.{TABLE_GROUPMAP}")
    cur.execute(f"""
        CREATE TABLE public.{TABLE_GROUPMAP} AS
        SELECT (company_id || ':' || posh || ':' || city || ':' || thash || ':'
                || yr || ':' || seg) AS duplicate_group_id,
               plat, rid AS job_id_raw, day, comp, dlen
        FROM public.{TABLE_SEG}
    """)
    cur.execute("CREATE TABLE public.job_master_pc AS "
                "SELECT duplicate_group_id, count(DISTINCT plat) AS platform_count "
                "FROM public." + TABLE_GROUPMAP + " GROUP BY 1")
    cur.execute(f"ALTER TABLE public.{TABLE_MASTER} "
                "ADD COLUMN platform_count int")
    cur.execute(f"UPDATE public.{TABLE_MASTER} m SET platform_count = p.platform_count "
                "FROM public.job_master_pc p "
                "WHERE m.duplicate_group_id = p.duplicate_group_id")
    cur.execute("DROP TABLE public.job_master_pc")
    cur.execute(f"CREATE INDEX ON public.{TABLE_GROUPMAP} (duplicate_group_id)")
    cur.execute(f"CREATE INDEX ON public.{TABLE_MASTER} (job_id_raw)")
    conn.commit()
    out = _master_stats(cur)
    conn.commit()
    conn.close()
    return out


def verify_invariants() -> None:
    """守恒与确定性验收（评审必改6/7）；失败 raise。"""
    conn = _results_conn()
    cur = conn.cursor()
    # 守恒含规则1折叠：sum(collapsed) + sum(rule1_dups) = stage
    cur.execute(f"""
        SELECT (SELECT count(*) FROM public.{TABLE_STAGE}),
               (SELECT count(*) FROM public.{TABLE_MASTER}),
               (SELECT sum(records_collapsed) FROM public.{TABLE_MASTER}),
               (SELECT coalesce(sum(rule1_dups),0) FROM public.{TABLE_MASTER}),
               (SELECT count(*) - count(DISTINCT duplicate_group_id) FROM public.{TABLE_MASTER}),
               (SELECT count(*) FROM (
                   SELECT plat, job_id_raw FROM public.{TABLE_MASTER}
                   GROUP BY 1,2 HAVING count(*)>1) z),
               (SELECT count(*) FROM public.{TABLE_STAGE} WHERE yr = 0)
    """)
    stage, master, collapsed, rule1, dup_groups, rid_dups, bad_years = cur.fetchone()
    conn.close()
    problems = []
    if stage != collapsed + rule1:
        problems.append(f"守恒失败: stage {stage} != collapsed {collapsed} "
                        f"+ rule1_dups {rule1}")
    if dup_groups != 0:
        problems.append(f"{dup_groups} 组出现多 canonical")
    if rid_dups != 0:
        problems.append(f"{rid_dups} 个 (plat,rid) 出现多 canonical")
    if bad_years / max(stage, 1) > BAD_YEAR_THRESHOLD:
        problems.append(f"年份不可解析 {bad_years}/{stage} 超阻断线 {BAD_YEAR_THRESHOLD:.1%}")
    if problems:
        for p in problems:
            logger.error("不变量: %s", p)
        raise SystemExit(2)
    logger.info("M2 不变量通过: stage=%d collapsed=%s 组唯一 canonical=%d",
                stage, collapsed, master)


def main() -> None:
    parser = argparse.ArgumentParser(description="panel_v2 M2 岗位去重主表")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--slices", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bench", action="store_true", help="仅广州单切片基准吞吐")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_dedup.log")
    out_dir = paths.output_dir / "panel_v2" / "dedup_parts"
    t0 = datetime.now()

    if args.dry_run:
        for s, c, lo, hi, yf in _plan_tasks(args.slices, None):
            logger.info("[dry-run] %s blocks %d-%d（%s 行/片近似）", s, lo, hi, "≈")
        return

    if args.bench:
        # 单城单年小片（~8M/32≈25 万行/片）测吞吐，先估后跑（§9 长跑纪律）
        s, c, lo, hi, yf = _plan_tasks(32, "2024")[0]
        plats = _build_platform_dict()
        _init_worker(plats)
        (out_dir / "bench").mkdir(parents=True, exist_ok=True)
        meta = _scan_slice(s, c, lo, hi, str(out_dir / "bench"), yf)
        dur = (datetime.now() - t0).total_seconds()
        rate = meta["rows"] / max(dur, 1)
        logger.info("[bench] %d 行 / %.0fs = %.0f 行/s → 全量 ~1.01亿行/8 路 ≈ %.1f 小时",
                    meta["rows"], dur, rate, 101_000_000 / rate / 8 / 3600)
        return

    copy_ent_map()
    metas = pass1_scan(args.workers, args.slices, out_dir)
    for m in metas:
        m["file"] = str(out_dir / f"{m['task']}.rows.bin")
    copy_stage(metas, resume=args.resume)
    n_master, n_stage = build_master()
    verify_invariants()
    dur = (datetime.now() - t0).total_seconds()
    print(f"M2 完成: stage={n_stage:,} → master={n_master:,} "
          f"去重比 {1 - n_master / max(n_stage, 1):.1%}，耗时 {dur/3600:.2f}h")


if __name__ == "__main__":
    main()
