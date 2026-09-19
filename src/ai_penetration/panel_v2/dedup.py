"""panel_v2 M2：招聘广告去重主表构建（指南 §6.2.1 主规则，评审修正版）。

数据事实（2026-09-07 实测+评审确认）：广深 recruit_id 平台内唯一（v2b 起
规则1 跨城折叠兜底残余重复）；job↔ent join 覆盖高；ent.recruit_id 有重复
（聚合取 min(company_id)，确定性选择；未命中数入披露并断言）。
所有写入仅在结果库；eps 只读。

流程（2026-09-09 审计修复版：断点版号绑定 + WAL 强制 + 谓词单源）：
1. ``copy_ent_map``：ent (recruit_id, company_id) 流式导出结果库（无主键，
   join 时 GROUP BY 聚合去重）。
2. ``pass1_scan``：ctid 切片（≤8 路，HDD 纪律）扫 job 表全行（准入谓词
   ``ADMISSION_WHERE`` 单源定义，scan 阶段复用防漂移），Python 侧算
   match 文本 hash、pos_norm_hash、日期、描述长度、**字段完整度**
   （education/work_type/experience/recruit_count/age_req 非空数——评审代理
   定义，§6.2.2.2 的可辩护实现）；年份/日期不可解析先保留在 stage，随后进入
   `dedup_invalid_date_gzsz` 隔离表，不进入正式 master；rid 超 32 字节**硬失败**；定长窄行
   分片落盘，断点复用校验 MASTER_VERSION+切片范围。
3. ``copy_stage``：文本 COPY 进常规 WAL stage 表（启动时强制
   ``SET LOGGED``——`CREATE IF NOT EXISTS` 不改变既有表持久性，审计实证
   2026-09-07 的"WAL 化修复"未生效）；.copied 标志带版号+行数，resume 时
   三方对账 count(stage)==Σflags==Σmetas。
4. ``build_master``：PG 侧两段式——stage LEFT JOIN 聚合后的 ent_map 物化
   sorted_stage（company 未命中→哨兵 'UNK:<rid>'，禁 NULL 共组，规则4）；
   组首锚定 30 天桶（规则5 修订版：链式传递语义 2.65% 超"两两≤30"披露线，
   桶规则构造性保证组内两两≤30，year 入分区键天然不跨年）与 §6.2.2
   canonical 选择，物化 job_master_gzsz；另建 dup_group_map（§6.2.2 平台
   数/编号映射，仅含规则1 幸存行，被折叠行以聚合差值披露——指南"跨年互标"
   与全量映射明细为已申报偏离）。
5. ``verify_invariants``：count(stage)==Σmetas==Σflags、
   Σ(collapsed)=groupmap 行数、(plat,rid) 全量唯一、plat=255==Σmeta
   unknown_platform、long_rid==0、company 未命中披露、bad_year>0.1% 阻断、
   组跨度>30 天=0 断言（桶规则推论）。

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
# c 版：2022 数据治理——描述有效长度 >=10 准入（blank 率 18.3% 污染修复）
MASTER_VERSION = "main_v2a_20260919e_platformhash"
TABLE_STAGE = "dedup_stage_gzsz"
TABLE_SORTED = "dedup_sorted_gzsz"
TABLE_MASTER = "job_master_gzsz"
TABLE_GROUPMAP = "dup_group_map_gzsz"
TABLE_ENTMAP = "ent_company_map"
TABLE_ISOLATED = "dedup_invalid_date_gzsz"
SHARDS = (("广州市", "job_p0387", 0), ("深圳市", "job_p0389", 1))
_EPOCH_DAYS = 14610  # date(2010,1,1).toordinal()
_YEAR_RE = re.compile(r"^\s*(\d{4})")
_ISO_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})")
BAD_YEAR_THRESHOLD = 0.001  # 年份不可解析阻断线（评审建议 0.1%）
# 准入谓词**单源**（审计 D3：dedup/scan 两处手抄同文才碰巧互证，任一侧
# 改动即失去闭环）。scan.py 必须 import 本常量，禁止再抄写。
ADMISSION_WHERE = ("AND job_description IS NOT NULL "
                   "AND length(trim(job_description)) >= 10 "
                   "AND position IS NOT NULL AND position != '' "
                   "AND recruit_id IS NOT NULL")
# 平台字典采样种子（审计 D6：无 seed 时字典规模跨运行漂移，255 兜底组
# 成员随之变化；固定采样保证跨运行一致）
PLATFORM_SAMPLE_PCT = 0.05
PLATFORM_SAMPLE_SEED = 42

# 定长窄行（80B）：rid32 + jid/phash + plat/city + yr/day + posh/thash + dlen/comp/pad
ROW_DTYPE = np.dtype([
    ("rid", "S32"), ("jid", "i8"), ("phash", "i8"), ("plat", "u1"), ("city", "u1"),
    ("yr", "u2"), ("day", "i4"), ("posh", "i8"), ("thash", "i8"),
    ("dlen", "u2"), ("comp", "u1"), ("pad", "S5"),
])
if ROW_DTYPE.itemsize != 80:
    raise RuntimeError(f"ROW_DTYPE 尺寸异常: {ROW_DTYPE.itemsize} != 80")


def _stable_job_id_sha256(platform: str, raw_job_id: str) -> str:
    """指南 §3.2 的完整稳定岗位编号。"""
    import hashlib
    payload = (
        str(platform or "").strip()
        + "\x1f"
        + str(raw_job_id or "").strip()
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_job_id(platform: str, raw_job_id: str) -> int:
    """SHA256(source_platform | job_id_raw) 的 63-bit 计算代理键。

    完整 SHA256 公式固定；截取 63 bit 仅为保持下游 int64 高效表示，
    master 构建后必须做全量碰撞检查。
    """
    import hashlib
    payload = (
        str(platform or "").strip()
        + "\x1f"
        + str(raw_job_id or "").strip()
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _h63(s: str) -> int:
    from hashlib import blake2b
    return int.from_bytes(blake2b(s.encode("utf-8"), digest_size=8).digest(),
                          "big") & 0x7FFF_FFFF_FFFF_FFFF


def _stable_platform_hash(platform: str) -> int:
    """平台真实字符串的稳定 63-bit 标识；去重语义不依赖采样字典。"""
    return _h63("platform\x1f" + str(platform or "").strip())


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
        # 断点复用三校验（审计 D2：旧版仅凭文件存在性复用，--slices 改变
        # 时同名任务范围不同会被静默截空；准入规则变更需重扫）
        if (meta.get("version") == MASTER_VERSION and meta.get("lo") == lo
                and meta.get("hi") == hi):
            logger.info("切片 %s 断点复用（%d 行，版号一致）", task, meta["rows"])
            return meta
        logger.warning("切片 %s 断点版号/范围不符（meta=%s/%s-%s 现=%s/%s-%s），"
                       "重扫", task, meta.get("version"), meta.get("lo"),
                       meta.get("hi"), MASTER_VERSION, lo, hi)
    plats = _WORKER["platforms"]
    conn = psycopg2.connect(**eps_conn_params())
    buf = np.empty(100000, dtype=ROW_DTYPE)
    n = total = bad_year = bad_day = unk_plat = long_rid = 0
    try:
        with conn.cursor() as setup:
            setup.execute("SET LOCAL work_mem = '256MB'")
            setup.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cond = ("WHERE ctid >= '(%s,0)'::tid AND ctid < '(%s,0)'::tid "
                + ADMISSION_WHERE)
        params: tuple = (int(lo), int(hi))
        if year_filter:
            cond += " AND publish_time LIKE %s"
            params = params + (year_filter + "%",)
        cur = conn.cursor(f"dedup_{task}")
        cur.itersize = 50000
        # 2022 数据治理：blank 描述率 18.3%（trim 后 <10 字符），" \n" 等
        # 非空串漏过 !='' 过滤污染主样本——统一按有效长度 >=10 准入
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
                    srid = str(rid).encode("utf-8")
                    if len(srid) > 32:  # 审计 D11：S32 超长为静默截断，
                        long_rid += 1   # 两条 rid 可同键致规则1误折叠
                        srid = srid[:32]
                    r["rid"] = srid
                    r["jid"] = _stable_job_id(pkey, str(rid))
                    r["phash"] = _stable_platform_hash(pkey)
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
            "unknown_platform": unk_plat, "long_rid": long_rid,
            "version": MASTER_VERSION, "lo": int(lo), "hi": int(hi)}
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
                    f"TABLESAMPLE SYSTEM ({PLATFORM_SAMPLE_PCT}) "
                    f"REPEATABLE ({PLATFORM_SAMPLE_SEED})")  # 审计 D6：定种
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
    if workers > 8:
        raise ValueError("HDD 纪律：大表 ≤8 流（CLAUDE.md §9）")
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
    """ent 映射按 city+recruit_id 导出；多企业键不做任意 min 裁决。"""
    conn = _results_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT to_regclass(%s)", (f"public.{TABLE_ENTMAP}",)
    )
    exists = cur.fetchone()[0] is not None
    if exists:
        cur.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s "
            "AND column_name='city'",
            (TABLE_ENTMAP,),
        )
        if cur.fetchone()[0] == 0:
            cur.execute(f"DROP TABLE public.{TABLE_ENTMAP}")
            exists = False
    if not exists:
        cur.execute(
            f"CREATE TABLE public.{TABLE_ENTMAP} "
            "(city smallint NOT NULL, recruit_id text, company_id text)"
        )
        conn.commit()
    cur.execute(f"SELECT count(*) FROM public.{TABLE_ENTMAP}")
    has = cur.fetchone()[0]
    conn.commit()
    conn.close()
    if has:
        logger.info("ent 映射已有 %d 行，跳过导出", has)
        return

    eps = psycopg2.connect(**eps_conn_params())
    try:
        ent_specs = (("ent_p0387", 0), ("ent_p0389", 1))
        for ent_shard, city_id in ent_specs:
            cur = eps.cursor(f"entmap_{ent_shard}")
            cur.itersize = 200000
            cur.execute(
                f"SELECT recruit_id, company_id FROM public.{ent_shard} "
                "WHERE recruit_id IS NOT NULL AND company_id IS NOT NULL"
            )
            conn2 = _results_conn()
            cur2 = conn2.cursor()
            n = 0
            bio = io.StringIO()
            while True:
                batch = cur.fetchmany(200000)
                if not batch:
                    break
                for rid, cid in batch:
                    bio.write(f"{city_id}\t{rid}\t{cid}\n")
                    n += 1
                if bio.tell() > 200 << 20:
                    bio.seek(0)
                    cur2.copy_expert(
                        f"COPY public.{TABLE_ENTMAP} FROM STDIN "
                        "WITH (FORMAT text)",
                        bio,
                    )
                    conn2.commit()
                    bio = io.StringIO()
            bio.seek(0)
            cur2.copy_expert(
                f"COPY public.{TABLE_ENTMAP} FROM STDIN WITH (FORMAT text)",
                bio,
            )
            conn2.commit()
            conn2.close()
            logger.info("ent 映射导出 %s: %d 行", ent_shard, n)
    finally:
        eps.close()


def _ensure_logged(cur, table: str) -> None:
    """强制常规 WAL 持久性（审计 D1：CREATE TABLE IF NOT EXISTS 不会把
    已存在的 UNLOGGED 表转为 logged——2026-09-07 的"WAL 化修复"因此从未
    生效）。仅在 relpersistence=='u' 时执行 SET LOGGED（logged→logged
    会白付重写成本）。"""
    cur.execute("SELECT relpersistence FROM pg_class WHERE relname = %s",
                (table,))
    row = cur.fetchone()
    if row is not None and row[0] == "u":
        logger.warning("表 %s 仍为 UNLOGGED（历史遗留），执行 SET LOGGED", table)
        cur.execute(f"ALTER TABLE public.{table} SET LOGGED")


def copy_stage(metas: list[dict], resume: bool) -> int:
    """窄行分片文本 COPY 进 stage（常规 WAL 表 + 启动强制 SET LOGGED）。

    断点纪律（审计 D2/D10 修订）：.copied 标志为 JSON{version,rows}；
    resume+非空 stage 时先做 count==Σ有效标志 对账，不符即阻断（防
    part-k 提交后崩溃产生前缀 stage）；stage 被重启清空时全部标志作废
    重灌。结束时无条件断言 count(stage)==Σmetas（本模块绝对守恒锚）。
    """
    conn = _results_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT to_regclass(%s)", (f"public.{TABLE_STAGE}",)
    )
    stage_exists = cur.fetchone()[0] is not None
    if stage_exists:
        cur.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s "
            "AND column_name IN ('jid','phash')",
            (TABLE_STAGE,),
        )
        has_current_identity = cur.fetchone()[0] == 2
        if not has_current_identity:
            if resume:
                conn.close()
                raise SystemExit(
                    "旧 stage 缺稳定 jid/phash 列，不能 --resume；请无 --resume 重跑"
                )
            cur.execute(f"DROP TABLE public.{TABLE_STAGE}")
            stage_exists = False
    if not stage_exists:
        cur.execute(f"""
            CREATE TABLE public.{TABLE_STAGE} (
                rid text NOT NULL, jid bigint NOT NULL, phash bigint NOT NULL,
                plat smallint, city smallint,
                yr int, day int, posh bigint, thash bigint, dlen int, comp smallint)""")
    _ensure_logged(cur, TABLE_STAGE)
    cur.execute(f"SELECT count(*) FROM public.{TABLE_STAGE}")
    existing = cur.fetchone()[0]
    parts = Path(metas[0]["file"]).parent
    want_total = sum(int(m["rows"]) for m in metas)
    verified: set[str] = set()
    if not resume:
        if existing:
            cur.execute(f"TRUNCATE public.{TABLE_STAGE}")
        existing = 0
    elif existing:
        # resume+非空：标志三态——有效（版号+行数匹配 metas）/畸形（旧格式
        # 或版号不符，来源不可辨→阻断）/缺失（视为未灌，允许补灌）。
        # 有效标志求和必须 == count(stage)，否则 stage 含来历不明行→阻断；
        # 崩溃于"提交后、写标志前"的少量重复由末尾绝对锚 stage==Σmetas 兜住。
        flag_total = 0
        malformed: list[str] = []
        verified: set[str] = set()
        for m in metas:
            f = parts / f"{m['task']}.copied"
            if not f.exists():
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                malformed.append(m["task"])
                continue
            if d.get("version") != MASTER_VERSION \
                    or int(d.get("rows", -1)) != int(m["rows"]):
                malformed.append(m["task"])
            else:
                flag_total += int(d["rows"])
                verified.add(m["task"])
        if malformed or flag_total != existing:
            conn.close()
            raise SystemExit(
                f"--resume 对账失败：畸形/混代标志={malformed[:5]}，"
                f"Σ有效标志={flag_total} != stage={existing}。"
                f"请去掉 --resume 全量重灌，或人工核查")
        logger.info("stage resume 对账通过：%d 行 == Σ有效标志（%d/%d 片）",
                    existing, len(verified), len(metas))
    else:
        # resume+空表（重启清空场景）：标志不可信，全部作废重灌
        for m in metas:
            f = parts / f"{m['task']}.copied"
            if f.exists():
                f.unlink()
        logger.info("stage 为空，作废全部 .copied 标志重灌")
    conn.commit()
    conn.close()
    total = existing
    for m in metas:
        done_flag = parts / f"{m['task']}.copied"
        if m["task"] in verified:
            continue  # resume 对账已证明该片已在 stage 且行数/版号匹配
        arr = np.fromfile(m["file"], dtype=ROW_DTYPE)
        conn2 = _results_conn()
        cur2 = conn2.cursor()
        bio = io.StringIO()
        for row in arr:
            rid = bytes(row["rid"]).rstrip(b"\x00").decode()
            bio.write(
                f"{rid}\t{int(row['jid'])}\t{int(row['plat'])}\t"
                f"{int(row['city'])}\t{int(row['yr'])}\t{int(row['day'])}\t"
                f"{int(row['posh'])}\t{int(row['thash'])}\t"
                f"{int(row['dlen'])}\t{int(row['comp'])}\n"
            )
        bio.seek(0)
        cur2.copy_expert(f"COPY public.{TABLE_STAGE} FROM STDIN WITH (FORMAT text)", bio)
        conn2.commit()
        conn2.close()
        done_flag.write_text(
            json.dumps({"version": MASTER_VERSION, "rows": len(arr)}),
            encoding="utf-8")
        total += len(arr)
        logger.info("已 COPY %s 累计 %d 行", m["task"], total)
    # 绝对守恒锚：stage == Σmetas（审计 D10——原 verify 缺这一方）
    conn3 = _results_conn()
    cur3 = conn3.cursor()
    cur3.execute(f"SELECT count(*) FROM public.{TABLE_STAGE}")
    n_stage = cur3.fetchone()[0]
    conn3.close()
    if n_stage != want_total:
        raise SystemExit(f"stage 守恒失败: count={n_stage} != Σmetas={want_total}")
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
    cur.execute("SET maintenance_work_mem = '2GB'")
    _ensure_logged(cur, TABLE_SORTED)  # 历史 UNLOGGED 遗留防御（审计 D1）
    _ensure_logged(cur, TABLE_SORTED.replace("sorted", "seg"))
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
        # 旧版缺 rule1_dups 列 或 表被 PG 重启清空（原 UNLOGGED 版本）→ 重建
        cur.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s "
            "AND column_name IN ('rule1_dups','phash')",
            (TABLE_SORTED,),
        )
        has_col = cur.fetchone()[0] == 2
        cur.execute(f"SELECT EXISTS(SELECT 1 FROM public.{TABLE_SORTED} LIMIT 1)")
        nonempty = cur.fetchone()[0]
        if not has_col or not nonempty:
            logger.info("sorted 表旧版/空（重启清空），DROP 重建")
            cur.execute(f"DROP TABLE public.{TABLE_SORTED}")
            sorted_exists = False
    if not sorted_exists:
        # §3.1：无法完整解析日期的记录进入隔离表，不参与正式按年样本。
        cur.execute(f"DROP TABLE IF EXISTS public.{TABLE_ISOLATED}")
        cur.execute(f"""
            CREATE TABLE public.{TABLE_ISOLATED} AS
            SELECT * FROM public.{TABLE_STAGE}
            WHERE yr NOT BETWEEN 2014 AND 2025 OR day < 0
        """)
        conn.commit()
        # 段一：规则1（§6.2.1.1 平台+编号唯一记录，跨城重复折叠并计数）
        # + join ent 城市内映射；一 rid 多司不任意裁决，NULL/冲突→哨兵
        cur.execute(f"""
            CREATE TABLE public.{TABLE_SORTED} AS
            WITH joined AS (
                SELECT s.rid, s.jid, s.phash, s.plat, s.city, s.yr, s.day, s.posh, s.thash,
                       s.dlen, s.comp,
                       coalesce(e.company_id, 'UNK:' || s.rid) AS company_id,
                       (e.company_id IS NULL) AS company_unmatched
                FROM public.{TABLE_STAGE} s
                LEFT JOIN (
                    SELECT city, recruit_id,
                           CASE WHEN count(DISTINCT company_id)=1
                                THEN min(company_id) ELSE NULL END AS company_id
                    FROM public.{TABLE_ENTMAP}
                    GROUP BY city, recruit_id
                ) e
                  ON e.city = s.city AND e.recruit_id = s.rid
                WHERE s.yr BETWEEN 2014 AND 2025 AND s.day >= 0
            ), rid_rank AS (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY jid
                           ORDER BY comp DESC, dlen DESC, (day < 0) ASC,
                                    day ASC, rid ASC, city ASC, thash ASC
                       ) AS rn_rid,
                       count(*) OVER (PARTITION BY jid) - 1 AS rule1_dups
                FROM joined
            )
            SELECT rid, jid, phash, plat, city, yr, day, posh, thash, dlen, comp,
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
        CREATE TABLE public.{TABLE_SEG} AS
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
                   row_number() OVER (
                       PARTITION BY company_id, posh, city, thash, yr, seg
                       ORDER BY comp DESC, dlen DESC, (day < 0) ASC,
                                day ASC, rid ASC, jid ASC
                   ) AS pick
            FROM public.{TABLE_SEG}
            WINDOW w AS (PARTITION BY company_id, posh, city, thash, yr, seg)
        )
        SELECT jid AS job_id,
               rid AS job_id_raw, phash, plat, city, yr AS year, company_id, thash,
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
               phash AS platform_hash, plat, rid AS job_id_raw, day, comp, dlen
        FROM public.{TABLE_SEG}
    """)
    cur.execute("CREATE TABLE public.job_master_pc AS "
                "SELECT duplicate_group_id, count(DISTINCT platform_hash) AS platform_count "
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


def verify_invariants(metas: list[dict] | None = None) -> None:
    """守恒与确定性验收（评审必改6/7 + 审计 D4/D6/D11 补强）；失败 raise。"""
    conn = _results_conn()
    cur = conn.cursor()
    # 守恒核对完全走常规表（不依赖 sorted/seg 中间态，master 幂等跳过后
    # 依然可验证）：groupmap 行恒等于 sorted 应有行数（规则1去重后）
    #   stage - groupmap = 规则1折叠行数（记录并阈值监控）
    #   Σmaster.records_collapsed = groupmap 行数（分桶守恒）
    cur.execute(f"""
        SELECT (SELECT count(*) FROM public.{TABLE_STAGE}),
               (SELECT count(*) FROM public.{TABLE_ISOLATED}),
               (SELECT count(*) FROM public.{TABLE_GROUPMAP}),
               (SELECT count(*) FROM public.{TABLE_MASTER}),
               (SELECT sum(records_collapsed) FROM public.{TABLE_MASTER}),
               (SELECT count(*) - count(DISTINCT duplicate_group_id) FROM public.{TABLE_MASTER}),
               (SELECT count(*) FROM (
                   SELECT phash, job_id_raw FROM public.{TABLE_MASTER}
                   GROUP BY 1,2 HAVING count(*)>1) z),
               (SELECT count(*) FROM public.{TABLE_STAGE} WHERE yr = 0),
               (SELECT count(*) FROM public.{TABLE_STAGE} WHERE plat = 255),
               (SELECT count(*) FROM public.{TABLE_MASTER}
                WHERE latest_day - earliest_day > 30 AND records_collapsed > 1),
               (SELECT count(*) - count(DISTINCT job_id)
                FROM public.{TABLE_MASTER})
    """)
    (stage, isolated, map_n, master, collapsed, dup_groups, rid_dups, bad_years,
     unk_plat_stage, span_over_groups, stable_id_collisions) = cur.fetchone()
    conn.close()
    eligible_stage = stage - isolated
    rule1_sum = eligible_stage - map_n
    problems: list[str] = []
    if rule1_sum < 0:
        problems.append(f"groupmap {map_n} > stage {stage}（不可能状态）")
    # 片内计数三方对账（审计 D6/D10/D11：unknown_platform/long_rid 原只
    # log 不断言；组跨度必须 ≤30——组首锚定桶规则的构造性推论，违例=实现缺陷）
    if metas:
        unk_meta = sum(int(m.get("unknown_platform", 0)) for m in metas)
        if unk_plat_stage != unk_meta:
            problems.append(f"stage plat255={unk_plat_stage} != "
                            f"Σmeta unknown_platform={unk_meta}")
        long_rid = sum(int(m.get("long_rid", 0)) for m in metas)
        if long_rid:
            problems.append(f"{long_rid} 条 rid 超 32 字节被定长槽截断"
                            f"（须清理源数据后重跑）")
    if span_over_groups:
        problems.append(f"{span_over_groups} 组跨度>30 天（桶规则违例，实现缺陷）")
    if stable_id_collisions:
        problems.append(
            f"稳定 job_id 发生 {stable_id_collisions} 个 SHA256-63 碰撞，拒绝发布")
    if collapsed != map_n:
        problems.append(f"分桶守恒失败: Σcollapsed {collapsed} != groupmap {map_n}")
    if rule1_sum > eligible_stage * 0.02:
        problems.append(
            f"规则1折叠 {rule1_sum} 超 eligible stage 2%（rid 重复异常）"
        )
    logger.info("规则1折叠行数（stage-groupmap）: %d", rule1_sum)
    if dup_groups != 0:
        problems.append(f"{dup_groups} 组出现多 canonical")
    if rid_dups != 0:
        problems.append(f"{rid_dups} 个 (platform_hash,rid) 出现多 canonical")
    if bad_years > isolated:
        problems.append(
            f"yr=0 行 {bad_years} 大于日期隔离行 {isolated}（隔离逻辑异常）"
        )
    if problems:
        for p in problems:
            logger.error("不变量: %s", p)
        raise SystemExit(2)
    logger.info(
        "M2 不变量通过: stage=%d isolated_date=%d eligible=%d "
        "collapsed=%s canonical=%d",
        stage, isolated, eligible_stage, collapsed, master,
    )


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
    verify_invariants(metas)
    dur = (datetime.now() - t0).total_seconds()
    print(f"M2 完成: stage={n_stage:,} → master={n_master:,} "
          f"去重比 {1 - n_master / max(n_stage, 1):.1%}，耗时 {dur/3600:.2f}h")


if __name__ == "__main__":
    main()
