"""§6.2.0 + 8/24 handoff：构建正式 dictionary discovery frame。

目标：
- 对源招聘语料建立全局 DISTINCT(source_platform, text_hash) 文本语料；
- 每组保留可复现代表文本，并保存出现次数、企业数、年份范围和岗位ID列表；
- 只使用源数据自带行业字段；
- 岗位组使用确定性规范化 position，不训练职业分类器；
- 企业规模和技术/非技术分层必须在 discovery_strata_v1.yaml 显式冻结；
- 用冻结 A + v3 pre-discovery 正式技能表计算 baseline matched_skill_count。

本模块只构建候选发现样本框，不参与最终 AI 岗得分。
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import logging
import os
import pickle
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from config.paths import get_project_paths, load_config_yaml
from ..common import eps_conn_params, setup_logging
from ..text_clean import match_from_raw, normalize_position, text_hash
from .anchors import match_all_versions
from .dedup import ADMISSION_WHERE, SHARDS, _stable_job_id, _results_conn
from .governance import governed_skill_map, normalize_term
from .lexicon import _load_atier_alias_records, build_union_lexicon

STAGE = "_dictionary_discovery_stage_v1"
FINAL = "_dictionary_discovery_corpus_v1"
CONFIG_NAME = "discovery_strata_v1.yaml"

logger = logging.getLogger("ai_penetration.panel_v2.discovery_frame")


def _unlink_with_retry(path: Path, attempts: int = 6) -> None:
    """删除临时文件，被短暂占用时重试；最终失败只告警、不抛异常。

    实测：刚写完的 GB 级文件会被 Windows 的索引/防病毒服务短暂持有，立即
    unlink 会抛 PermissionError（实测 19 GB 的临时 CSV 在进程退出后即可正常
    删除，说明锁是瞬时的而非持久）。而清理失败**不应终结整个构建**——此时
    parquet 产物已完整写出，收尾清理不是交付物；原实现把二者绑在一起，导致
    一次成功的构建被收尾动作判为失败，白白重跑数小时的扫描。
    """
    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if i < attempts - 1:
                time.sleep(1.5 * (i + 1))
                continue
            logger.warning(
                "临时文件仍被占用，保留 %s —— 产物已写出，不阻断构建", path
            )


PROGRESS_NAME = "discovery_frame_progress.json"


def _guard_single_instance() -> None:
    """单实例保护：已有更早启动的 discovery_frame 进程在跑则本进程退出。

    防止两个 build 并发互踩：第二个进程的 DROP/CREATE stage 会毁掉第一个
    进程已写入的数据（2026-09-24 曾发生两轮构建重叠）。实现沿用
    fused_national._guard_single_instance 的既有模式。
    """
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
            if "discovery_frame" in cmd and "pytest" not in cmd:
                if p.create_time() < my_start:
                    logger.warning(
                        "检测到更早启动的 discovery_frame PID=%s，本实例退出"
                        "（单实例保护）", p.pid,
                    )
                    raise SystemExit(0)
        except (psutil.Error, OSError):
            continue


def _progress_path() -> Path:
    """断点文件路径（与 corpus 同目录，构建成功后删除）。"""
    return (
        get_project_paths().output_dir / "dictionary" / PROGRESS_NAME
    )


def _load_progress() -> dict:
    """读取断点；文件缺失/损坏时返回空 dict（视为无断点）。"""
    path = _progress_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("断点文件损坏，忽略并从零开始: %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _save_progress(progress: dict) -> None:
    """原子写入断点文件（先写 tmp 再替换，避免半写断点误导 resume）。"""
    path = _progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)


def _table_exists(table: str) -> bool:
    conn = _results_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
        return cur.fetchone()[0] is not None
    finally:
        conn.close()


def _drop_stage() -> None:
    """构建成功后清理 stage 中间表（纯中间态，非交付物，实测可达 38GB）。"""
    conn = _results_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS public.{STAGE}")
        conn.commit()
    finally:
        conn.close()


def _delete_shard_rows(city: str) -> None:
    """删除某分片已写入 stage 的行（该分片上次中断、本次重扫时防重复）。"""
    conn = _results_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM public.{STAGE} WHERE source_city = %s", (city,)
        )
        deleted = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    if deleted:
        logger.info(
            "清除中断分片 %s 的部分数据: %s 行", city, deleted
        )


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _discovery_id(platform: str, text_hash_value: int) -> str:
    """§6.2.0 唯一文本组身份，不复用可能多文本的原始岗位编号。"""
    payload = (
        str(platform) + "\x1f" + str(int(text_hash_value))
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _norm_meta(value, missing: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return text if text else missing


def _load_config() -> tuple[dict, Path]:
    paths = get_project_paths()
    path = paths.config_dir / CONFIG_NAME
    cfg = load_config_yaml(CONFIG_NAME)
    source_cfg = cfg.get("source", {})
    industry_source = str(source_cfg.get("industry_source", "")).strip()
    industry_col = str(source_cfg.get("industry_column", "")).strip()
    company_source = str(source_cfg.get("company_size_source", "")).strip()
    company_col = str(source_cfg.get("company_size_column", "")).strip()
    tech_regex = str(
        cfg.get("tech_flag", {}).get("position_regex", "")
    ).strip()
    bad = []
    if industry_source not in {"job", "ent"}:
        bad.append("source.industry_source")
    if not industry_col or industry_col == "TO_BE_CONFIRMED":
        bad.append("source.industry_column")
    if company_source not in {"job", "ent"}:
        bad.append("source.company_size_source")
    if not company_col or company_col == "TO_BE_CONFIRMED":
        bad.append("source.company_size_column")
    if not tech_regex or tech_regex == "TO_BE_CONFIRMED":
        bad.append("tech_flag.position_regex")
    if bad:
        raise RuntimeError(
            "正式 discovery 分层配置尚未冻结: " + ", ".join(bad)
        )
    bins = cfg.get("text_length", {}).get("bins", [])
    labels = cfg.get("text_length", {}).get("labels", [])
    if len(bins) != len(labels) + 1:
        raise ValueError("text_length bins 必须比 labels 多 1")
    if any(float(b) >= float(n) for b, n in zip(bins, bins[1:])):
        raise ValueError("text_length bins 必须严格递增")
    re.compile(tech_regex)
    return cfg, path


def _columns(cur, table: str) -> set[str]:
    cur.execute(
        """SELECT column_name
           FROM information_schema.columns
           WHERE table_schema='public' AND table_name=%s""",
        (table,),
    )
    return {str(r[0]) for r in cur.fetchall()}


def _validate_source_fields(cfg: dict) -> None:
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        source_cfg = cfg["source"]
        for _city, job_table, _city_id in SHARDS:
            ent_table = job_table.replace("job_", "ent_", 1)
            job_cols = _columns(cur, job_table)
            ent_cols = _columns(cur, ent_table)
            for logical in ("industry", "company_size"):
                source = str(source_cfg[f"{logical}_source"])
                col = str(source_cfg[f"{logical}_column"])
                if source not in {"job", "ent"}:
                    raise ValueError(
                        f"{logical}_source 必须为 job/ent，当前={source!r}"
                    )
                cols = job_cols if source == "job" else ent_cols
                if col not in cols:
                    raise RuntimeError(
                        f"{ent_table if source == 'ent' else job_table} "
                        f"缺配置字段 {col!r}（{logical}）"
                    )
            needed_job = {
                "recruit_id", "platform", "position",
                "publish_time", "job_description",
            }
            missing = needed_job - job_cols
            if missing:
                raise RuntimeError(
                    f"{job_table} 缺 discovery 必需字段: "
                    + ", ".join(sorted(missing))
                )
    finally:
        conn.close()


def _verify_source_snapshot(audit: dict) -> None:
    """开始 discovery 前重新核对数据库/表行数，避免快照标签漂移。"""
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        # 关闭本会话的 PG 内部并行扫描。这 4 张表各 4700-5400 万行，count(*)
        # 会为其启动多个并行 worker，实测连续两次导致 postmaster 终止（服务端
        # 崩溃并进入恢复模式），且第二次发生时无任何并发客户端，说明问题在
        # 该阶段本身。会话级设置只作用于本连接的这几次计数：仍是精确计数，
        # 不改服务器配置，也不改校验强度。
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SELECT current_database()")
        database = str(cur.fetchone()[0])
        if database != str(audit.get("database", "")):
            raise RuntimeError(
                f"当前数据库 {database!r} != source audit {audit.get('database')!r}"
            )
        tables = audit.get("tables")
        if not isinstance(tables, dict) or not tables:
            raise RuntimeError("source audit manifest 缺 tables 快照统计")
        drift = []
        for table, meta in tables.items():
            t0 = time.time()
            cur.execute(f"SELECT count(*) FROM public.{table}")
            now = int(cur.fetchone()[0])
            logger.info(
                "快照核对 %s: rows=%s 用时 %.1f 分钟",
                table, now, (time.time() - t0) / 60,
            )
            expected = int(meta.get("rows", -1))
            if now != expected:
                drift.append(f"{table}:{expected}->{now}")
        if drift:
            raise RuntimeError(
                "源数据库已偏离审计快照，拒绝 discovery: "
                + ", ".join(drift[:10])
            )
    finally:
        conn.close()


def _v3_governance_path() -> Path:
    """pre-discovery v3 治理表路径；缺失即 fail-closed。"""
    path = (
        get_project_paths().output_dir / "dictionary"
        / "skill_legacy_graded_BCD_v3.csv"
    )
    if not path.exists():
        raise RuntimeError(
            "缺 pre-discovery v3 治理表；请先完成 legacy_freq + lexicon_llm merge"
        )
    return path


def _pre_discovery_lexicon():
    path = _v3_governance_path()
    grades = pd.read_csv(path, encoding="utf-8-sig")
    mapping = governed_skill_map(grades)
    formal = grades[grades.final_grade.isin(["A", "B", "C"])]
    ambiguous = {
        normalize_term(term)
        for term, flag in zip(formal.term, formal.t2_ambig)
        if str(flag).strip().lower() in {"1", "true", "yes"}
    }
    lex = build_union_lexicon(
        legacy_terms=sorted(mapping),
        aliases=_load_atier_alias_records(),
        legacy_id_map=mapping,
        legacy_ambiguous_keys=ambiguous,
    )
    return lex, path


def _bin_length(length: int, bins: list, labels: list[str]) -> str:
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        if float(lo) <= length < float(hi):
            return str(label)
    raise RuntimeError(f"text length {length} 未落入配置分箱")


def _source_query(
    conn,
    job_table: str,
    ent_table: str,
    cfg: dict,
) -> str:
    source = cfg["source"]
    job_cols = _columns(conn.cursor(), job_table)
    ent_cols = _columns(conn.cursor(), ent_table)
    industry_col = str(source["industry_column"])
    size_col = str(source["company_size_column"])

    ent_fields = []
    if source["industry_source"] == "ent":
        ent_fields.append(("industry", industry_col))
    if source["company_size_source"] == "ent":
        ent_fields.append(("company_size", size_col))

    ent_select = ["recruit_id"]
    for alias, col in ent_fields:
        if col not in ent_cols:
            raise RuntimeError(f"{ent_table} 缺字段 {col}")
        ent_select.append(
            "CASE WHEN count(DISTINCT {0}::text)=1 "
            "THEN min({0}::text) ELSE NULL END AS {1}".format(
                col, alias
            )
        )
    if "company_id" in ent_cols:
        ent_select.append(
            "CASE WHEN count(DISTINCT company_id::text)=1 "
            "THEN min(company_id::text) ELSE NULL END AS company_id"
        )
    else:
        ent_select.append("NULL::text AS company_id")

    ent_sql = (
        "SELECT " + ", ".join(ent_select)
        + f" FROM public.{ent_table} GROUP BY recruit_id"
    )
    industry_expr = (
        f"j.{industry_col}::text"
        if source["industry_source"] == "job"
        else "e.industry"
    )
    size_expr = (
        f"j.{size_col}::text"
        if source["company_size_source"] == "job"
        else "e.company_size"
    )
    for col, origin in (
        (industry_col, source["industry_source"]),
        (size_col, source["company_size_source"]),
    ):
        if origin == "job" and col not in job_cols:
            raise RuntimeError(f"{job_table} 缺配置字段 {col}")
    return f"""
        SELECT j.recruit_id, j.platform, j.position, j.publish_time,
               j.job_description, {industry_expr} AS industry,
               {size_expr} AS company_size, e.company_id
        FROM public.{job_table} j
        LEFT JOIN ({ent_sql}) e ON e.recruit_id=j.recruit_id
        WHERE 1=1 {ADMISSION_WHERE.replace('job_description','j.job_description').replace('position','j.position').replace('recruit_id','j.recruit_id')}
    """


def _create_stage() -> None:
    conn = _results_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS public.{STAGE}")
        cur.execute(f"""
            CREATE TABLE public.{STAGE} (
                job_id bigint NOT NULL,
                year int NOT NULL,
                industry text NOT NULL,
                position_group text NOT NULL,
                company_size text NOT NULL,
                text_length_bin text NOT NULL,
                tech_flag smallint NOT NULL,
                platform text NOT NULL,
                anchor_main smallint NOT NULL,
                matched_skill_count int NOT NULL,
                text_hash bigint NOT NULL,
                description text NOT NULL,
                company_id text,
                source_city text NOT NULL,
                source_raw_id text NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _copy_rows(rows: list[tuple]) -> None:
    if not rows:
        return
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(rows)
    buf.seek(0)
    conn = _results_conn()
    try:
        cur = conn.cursor()
        cur.copy_expert(
            f"COPY public.{STAGE} FROM STDIN WITH (FORMAT csv)",
            buf,
        )
        conn.commit()
    finally:
        conn.close()


def _process_rows(
    rows, city: str, bins: list, labels: list[str], tech, lex,
) -> tuple[list[tuple], dict]:
    """单批源行的清洗/哈希/匹配（纯函数，串行与并行路径共用）。

    语义与串行版内循环逐字一致：年份白名单 2014-2025，无效年行计入
    invalid_year 跳过；其余行按 stage 表列序构造元组。

    Args:
        rows: 游标批次 ((rid, platform, position, publish_time, desc,
            industry, company_size, company_id), ...)。
        city: 分片城市名（写入 source_city 列）。
        bins: 文本长度分箱边界。
        labels: 分箱标签。
        tech: 技术/非技术岗位名正则（已编译）。
        lex: union 词表（需提供 extract()）。

    Returns:
        (stage 行元组列表, {"rows_seen","rows_kept","invalid_year"} 增量)。
    """
    out: list[tuple] = []
    stats = {"rows_seen": 0, "rows_kept": 0, "invalid_year": 0}
    for (
        rid, platform, position, publish_time, desc,
        industry, company_size, company_id,
    ) in rows:
        stats["rows_seen"] += 1
        year_s = str(publish_time or "")[:4]
        if not year_s.isdigit() or not 2014 <= int(year_s) <= 2025:
            stats["invalid_year"] += 1
            continue
        raw = str(desc)
        match = match_from_raw(raw)
        h = int(text_hash(match))
        pos_group = normalize_position(str(position))
        hits = match_all_versions(match)
        matched = len(lex.extract(match))
        raw_platform = str(platform or "").strip()
        platform_s = _norm_meta(platform, "PLATFORM_MISSING")
        out.append((
            int(_stable_job_id(raw_platform, str(rid))),
            int(year_s),
            _norm_meta(industry, "INDUSTRY_MISSING"),
            pos_group or "POSITION_MISSING",
            _norm_meta(company_size, "COMPANY_SIZE_MISSING"),
            _bin_length(len(match), bins, labels),
            int(bool(tech.search(pos_group))),
            platform_s,
            int(hits["main"].flag),
            int(matched),
            h,
            raw,
            _norm_meta(company_id, ""),
            city,
            str(rid),
        ))
        stats["rows_kept"] += 1
    return out, stats


_WORKER_CTX: dict = {}


def _worker_init(cfg: dict, lex_pickle: bytes | None) -> None:
    """进程池 worker 初始化：反序列化父进程词表；不可 pickle 时自建。

    Args:
        cfg: 冻结的 discovery_strata 配置。
        lex_pickle: 父进程 pickle 的 union 词表；None 时 worker 经
            _pre_discovery_lexicon 从源库重建（每 worker 一次性，秒级）。
    """
    if lex_pickle is not None:
        lex = pickle.loads(lex_pickle)
    else:
        lex, _unused = _pre_discovery_lexicon()
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        lex=lex,
        bins=[float(x) for x in cfg["text_length"]["bins"]],
        labels=[str(x) for x in cfg["text_length"]["labels"]],
        tech=re.compile(str(cfg["tech_flag"]["position_regex"]), re.I),
    )


def _process_rows_worker(item: tuple) -> tuple:
    """worker 侧入口：解包 (batch, city) 后调用共用纯函数。"""
    batch, city = item
    return _process_rows(
        batch, city, _WORKER_CTX["bins"], _WORKER_CTX["labels"],
        _WORKER_CTX["tech"], _WORKER_CTX["lex"],
    )


def _iter_processed(batches, city: str, cfg: dict, lex, workers: int):
    """按批产出 (stage_rows, stats)；workers=1 串行，>1 进程池并行（保序）。

    设计针对本项目历史并行失败的三个机制（逐项绕开）：
    - 仍由主进程单一命名游标顺序读：PG 侧查询形态、ent 聚合次数与
      连接负载完全不变（历史坑：多客户端切片使 5000 万行 ent GROUP BY
      每路重复一次、HDD 多流互抢磁头、PG 并行 worker 曾致 postmaster 崩溃）；
    - 实测瓶颈在 Python 逐行处理（扫描游标会话持续 ClientRead 等
      客户端），故只并行 CPU 部分；
    - 有界在途批次（workers×2）避免物化全量输入；按提交顺序收集，
      stage 行序与串行完全一致，断点/对账语义不变。

    Args:
        batches: 批次可迭代对象（每批一批源行元组）。
        city: 分片城市名。
        cfg: 冻结配置。
        lex: union 词表（workers>1 时优先 pickle 传给 worker）。
        workers: 并行进程数；<=1 走串行纯函数。

    Yields:
        (stage 行元组列表, stats 增量)，与批次顺序一致。
    """
    bins = [float(x) for x in cfg["text_length"]["bins"]]
    labels = [str(x) for x in cfg["text_length"]["labels"]]
    if workers <= 1:
        tech = re.compile(str(cfg["tech_flag"]["position_regex"]), re.I)
        for batch in batches:
            yield _process_rows(batch, city, bins, labels, tech, lex)
        return
    try:
        lex_pickle: bytes | None = pickle.dumps(lex)
        logger.info(
            "并行扫描：词表经 pickle 传递（%.1f MB），workers=%s",
            len(lex_pickle) / 1048576, workers,
        )
    except Exception:  # noqa: BLE001 - 词表不可 pickle 时退回 worker 自建
        lex_pickle = None
        logger.info("并行扫描：词表不可 pickle，worker 各自从源库重建")
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(cfg, lex_pickle),
    )
    max_pending = workers * 2
    try:
        pending = collections.deque()
        it = iter(batches)
        exhausted = False
        while True:
            while not exhausted and len(pending) < max_pending:
                batch = next(it, None)
                if batch is None:
                    exhausted = True
                    break
                pending.append(
                    pool.submit(_process_rows_worker, (batch, city))
                )
            if not pending:
                break
            yield pending.popleft().result()
    except BaseException:
        # 消费侧中断（异常/Ctrl+C）时取消排队批次，避免 with-exit 排空
        # 整个队列（进程版同样会"假死排空"）。
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def _scan_source(cfg: dict, lex, progress: dict, workers: int = 1) -> dict:
    """全量扫描源表写入 stage；按分片（城市）断点续扫。

    每个分片完成即把 done 标志与计数落盘到断点文件；中断后重跑跳过已完成
    分片，只清除并重扫中断的那个分片——不再像旧实现那样整轮从零开始
    （实测一轮全量扫描 7-8 小时）。分片内每 50 万行打一条进度日志。

    Args:
        cfg: 冻结的 discovery_strata_v1 配置。
        lex: union 词表（build_union_lexicon 产物）。
        progress: 断点 dict（就地更新并周期性落盘）。
        workers: 扫描处理并行进程数（AIPEN_SCAN_WORKERS，默认 1 串行）。

    Returns:
        各分片合计的 {rows_seen, rows_kept, invalid_year}。断点按 job_table
        记、清理按 source_city 删——当前 SHARDS 两者一一对应；未来若同城
        多表（GD_SHARDS 模式）需改为按表清理。
    """
    totals = {"rows_seen": 0, "rows_kept": 0, "invalid_year": 0}
    conn = psycopg2.connect(**eps_conn_params())
    try:
        for city, job_table, _city_id in SHARDS:
            shard_state = progress["shards"].get(job_table, {})
            if shard_state.get("done"):
                logger.info(
                    "分片 %s(%s) 已完成（断点跳过）: rows_seen=%s "
                    "rows_kept=%s",
                    city, job_table,
                    shard_state.get("rows_seen"),
                    shard_state.get("rows_kept"),
                )
                for key in totals:
                    totals[key] += int(shard_state.get(key, 0))
                continue
            if shard_state.get("started"):
                logger.info(
                    "分片 %s(%s) 上次中断，清除部分数据后重扫",
                    city, job_table,
                )
                _delete_shard_rows(city)
            progress["shards"][job_table] = {"started": True}
            _save_progress(progress)
            t_shard = time.time()
            stats = {"rows_seen": 0, "rows_kept": 0, "invalid_year": 0}
            next_milestone = 500_000
            logger.info(
                "分片开始 %s(%s) workers=%s", city, job_table, workers,
            )
            ent_table = job_table.replace("job_", "ent_", 1)
            query = _source_query(conn, job_table, ent_table, cfg)
            cur = conn.cursor(f"discovery_{job_table}")
            cur.itersize = 20000
            cur.execute(query)
            buf: list[tuple] = []

            def _batches():
                while True:
                    batch = cur.fetchmany(20000)
                    if not batch:
                        return
                    yield batch

            for stage_rows, delta in _iter_processed(
                _batches(), city, cfg, lex, workers,
            ):
                stats["rows_seen"] += delta["rows_seen"]
                stats["rows_kept"] += delta["rows_kept"]
                stats["invalid_year"] += delta["invalid_year"]
                buf.extend(stage_rows)
                if len(buf) >= 50000:
                    _copy_rows(buf)
                    buf = []
                if stats["rows_seen"] >= next_milestone:
                    elapsed = time.time() - t_shard
                    logger.info(
                        "扫描 %s 进行中: rows_seen=%s rows_kept=%s "
                        "速率=%.0f 行/s 已用=%.1f 分钟",
                        city, stats["rows_seen"], stats["rows_kept"],
                        stats["rows_seen"] / max(elapsed, 1e-6),
                        elapsed / 60,
                    )
                    next_milestone += 500_000
            if buf:
                _copy_rows(buf)
                buf = []
            cur.close()
            progress["shards"][job_table] = {"done": True, **stats}
            _save_progress(progress)
            logger.info(
                "分片完成 %s(%s): rows_seen=%s rows_kept=%s "
                "invalid_year=%s 用时=%.1f 分钟",
                city, job_table, stats["rows_seen"], stats["rows_kept"],
                stats["invalid_year"], (time.time() - t_shard) / 60,
            )
            for key in totals:
                totals[key] += stats[key]
    finally:
        conn.close()
    return totals


def _materialize_final() -> int:
    conn = _results_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS public.{FINAL}")
        cur.execute(f"""
            CREATE TABLE public.{FINAL} AS
            WITH agg AS (
                SELECT platform, text_hash,
                       count(*)::bigint AS original_occurrences,
                       count(DISTINCT NULLIF(company_id,''))::bigint
                           AS enterprise_count,
                       min(year) AS year_min,
                       max(year) AS year_max,
                       jsonb_agg(job_id ORDER BY job_id)::text AS job_ids_json
                FROM public.{STAGE}
                GROUP BY platform, text_hash
            ),
            rep AS (
                SELECT DISTINCT ON (platform, text_hash)
                       *
                FROM public.{STAGE}
                ORDER BY platform, text_hash,
                         (industry <> 'INDUSTRY_MISSING') DESC,
                         (company_size <> 'COMPANY_SIZE_MISSING') DESC,
                         year ASC, job_id ASC
            )
            SELECT r.*, a.original_occurrences, a.enterprise_count,
                   a.year_min, a.year_max, a.job_ids_json
            FROM rep r
            JOIN agg a USING (platform, text_hash)
        """)
        cur.execute(
            f"CREATE UNIQUE INDEX ON public.{FINAL} (platform, text_hash)"
        )
        cur.execute(f"SELECT count(*) FROM public.{FINAL}")
        n = int(cur.fetchone()[0])
        conn.commit()
        return n
    finally:
        conn.close()


def _export_table(table: str, out: Path, columns: list[str],
                  types: dict[str, pa.DataType]) -> None:
    tmp = out.with_suffix(".csv.tmp")
    conn = _results_conn()
    copy_ok = False
    try:
        cur = conn.cursor()
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            cur.copy_expert(
                "COPY (SELECT " + ", ".join(columns)
                + f" FROM public.{table} ORDER BY platform, text_hash) "
                "TO STDOUT WITH (FORMAT csv, HEADER true)",
                fh,
            )
        copy_ok = True
    finally:
        conn.close()
        if not copy_ok:
            # COPY 中途失败时 tmp 是半写的孤儿文件（实测可达 19GB），
            # 旧实现此路径无任何清理。
            _unlink_with_retry(tmp)
    try:
        reader = pcsv.open_csv(
            tmp,
            read_options=pcsv.ReadOptions(block_size=1 << 24),
            # 岗位描述正文含换行符：PG 的 COPY TO CSV 会正确加引号，但 pyarrow
            # 默认不接受单元格内换行（newlines_in_values=False），会报
            # "CSV parser got out of sync with chunker" 并解析错位。开启后
            # pyarrow 无法再按块并行读，代价是这一遍转换变慢——正确性优先。
            parse_options=pcsv.ParseOptions(newlines_in_values=True),
            convert_options=pcsv.ConvertOptions(column_types=types),
        )
    except BaseException:
        _unlink_with_retry(tmp)
        raise
    writer = None
    try:
        for batch in reader:
            table_batch = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(
                    out, table_batch.schema, compression="zstd"
                )
            writer.write_table(table_batch)
    finally:
        if writer is not None:
            writer.close()
        # reader 持有 tmp 的句柄。Windows 下不先关闭就 unlink 会抛
        # PermissionError，而且它发生在 except 路径上会**掩盖真正的异常**
        # （本次即被掩盖，只看到句柄占用而看不到 CSV 解析失败）。
        reader.close()
        _unlink_with_retry(tmp)
    if writer is None:
        raise RuntimeError(f"{table} 导出为空")


def build(*, fresh: bool = False, reverify: bool = False) -> Path:
    """构建正式 discovery frame；带分片断点、进度日志与产物复用。

    Args:
        fresh: 忽略断点与已有 manifest，从零重建。
        reverify: 强制重跑源快照行数核对（默认同一审计哈希只核对一次）。

    Returns:
        manifest 路径。
    """
    paths = get_project_paths()
    setup_logging(paths.log_dir / "discovery_frame.log")
    _guard_single_instance()
    t_start = time.time()
    logger.info(
        "discovery frame 构建开始（fresh=%s reverify=%s）", fresh, reverify
    )

    cfg, cfg_path = _load_config()
    logger.info("分层配置冻结校验通过: %s", cfg_path.name)
    _validate_source_fields(cfg)
    logger.info("源表字段校验通过（%s 个分片）", len(SHARDS))

    audit_manifest = (
        paths.output_dir / "data_audit" / "source_db_manifest_v1.json"
    )
    if not audit_manifest.exists():
        raise RuntimeError("缺 source_db_manifest_v1.json；先运行 source_audit")
    audit = json.loads(audit_manifest.read_text(encoding="utf-8"))
    if audit.get("status") != "formal_pass":
        raise RuntimeError("source_db_manifest_v1 未 formal_pass")
    audit_sha = _sha(audit_manifest)

    v3_path = _v3_governance_path()
    v3_sha = _sha(v3_path)
    cfg_sha = _sha(cfg_path)

    out_dir = paths.output_dir / "dictionary"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "discovery_frame_manifest_v1.json"

    # 幂等重跑：同输入且已 formal_pass → 直接返回。旧实现无条件重建，
    # 9.4GB corpus 已在而 manifest 缺失时仍会整轮重扫（实测白跑约 8 小时）。
    # --reverify 时仍进入早退检查之外（用户显式要求重核快照）；两个 parquet
    # 任一被删时早退不生效（返回的 manifest 必须对应在盘产物）。
    if (
        not fresh
        and not reverify
        and manifest_path.exists()
    ):
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
        if (
            previous.get("status") == "formal_pass"
            and previous.get("source_audit_sha256") == audit_sha
            and previous.get("strata_config_sha256") == cfg_sha
            and previous.get("pre_discovery_governance_sha256") == v3_sha
        ):
            corpus_path_early = out_dir / "dictionary_discovery_corpus_v1.parquet"
            frame_path_early = out_dir / "discovery_frame_v1.parquet"
            if corpus_path_early.exists() and frame_path_early.exists():
                logger.info(
                    "manifest 已 formal_pass 且三个输入哈希未变，跳过构建: %s",
                    manifest_path,
                )
                return manifest_path
            logger.info(
                "manifest 有效但 parquet 产物缺失，全量重建（corpus/frame "
                "必须在盘）"
            )
        elif previous:
            logger.info("manifest 存在但输入已变化或未通过，全量重建")

    # 源快照核对（结果缓存）：门禁是 4 张大表共约 2 亿行的精确 count，
    # 同一审计哈希核对过一次后不再重付（eps 为只读源库，策略上无写入方）。
    progress = _load_progress() if not fresh else {}
    if not reverify and progress.get("verified_audit_sha256") == audit_sha:
        logger.info(
            "源快照核对跳过（同审计哈希已于 %s 核对过；--reverify 强制重核）",
            progress.get("verified_at", "未知时间"),
        )
    else:
        t0 = time.time()
        _verify_source_snapshot(audit)
        progress["verified_audit_sha256"] = audit_sha
        progress["verified_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        _save_progress(progress)
        logger.info(
            "源快照核对通过，用时 %.1f 分钟", (time.time() - t0) / 60
        )

    # 断点仅在三个输入哈希一致时可续用。
    resume_valid = (
        not fresh
        and progress.get("audit_sha256") == audit_sha
        and progress.get("strata_config_sha256") == cfg_sha
        and progress.get("pre_discovery_governance_sha256") == v3_sha
        and isinstance(progress.get("shards"), dict)
    )
    if not resume_valid:
        progress = {
            "audit_sha256": audit_sha,
            "strata_config_sha256": cfg_sha,
            "pre_discovery_governance_sha256": v3_sha,
            "verified_audit_sha256": progress.get("verified_audit_sha256"),
            "verified_at": progress.get("verified_at"),
            "shards": {},
        }
        logger.info(
            "断点不可复用（首次构建/输入变化/--fresh），从零扫描"
        )

    lex, _v3_used = _pre_discovery_lexicon()
    logger.info("union 词表构建完成")

    if resume_valid and _table_exists(STAGE):
        logger.info("resume：复用已有 stage 表（不 DROP）")
    else:
        # 先落盘清空 shards 再 DROP/CREATE stage：若顺序相反，在 DROP 清空
        # 数据之后、断点落盘之前崩溃，磁盘断点仍记着旧分片 done，下次重跑
        # 会跳过"已完成"分片而 stage 已空——静默丢掉整个分片的数据。
        progress["shards"] = {}
        _save_progress(progress)
        _create_stage()
        logger.info("stage 表已重建（全量扫描起点）")

    scan_workers = max(1, int(os.getenv("AIPEN_SCAN_WORKERS", "1")))
    logger.info("扫描并行度: %s（AIPEN_SCAN_WORKERS 可覆盖）", scan_workers)
    scan_stats = _scan_source(cfg, lex, progress, workers=scan_workers)
    logger.info(
        "扫描合计: rows_seen=%s rows_kept=%s invalid_year=%s",
        scan_stats["rows_seen"], scan_stats["rows_kept"],
        scan_stats["invalid_year"],
    )

    corpus_path = out_dir / "dictionary_discovery_corpus_v1.parquet"
    corpus_cols = [
        "job_id AS representative_job_id", "year", "industry", "position_group", "company_size",
        "text_length_bin", "tech_flag", "platform", "anchor_main",
        "matched_skill_count", "text_hash", "description",
        "company_id", "source_city", "source_raw_id",
        "original_occurrences", "enterprise_count",
        "year_min", "year_max", "job_ids_json",
    ]
    types = {
        "representative_job_id": pa.int64(), "year": pa.int32(),
        "industry": pa.string(), "position_group": pa.string(),
        "company_size": pa.string(), "text_length_bin": pa.string(),
        "tech_flag": pa.int16(), "platform": pa.string(),
        "anchor_main": pa.int16(), "matched_skill_count": pa.int32(),
        "text_hash": pa.int64(), "description": pa.string(),
        "company_id": pa.string(), "source_city": pa.string(),
        "source_raw_id": pa.string(), "original_occurrences": pa.int64(),
        "enterprise_count": pa.int64(), "year_min": pa.int32(),
        "year_max": pa.int32(), "job_ids_json": pa.string(),
    }

    corpus_reusable = (
        resume_valid
        and progress.get("corpus_sha256")
        and corpus_path.exists()
        and _sha(corpus_path) == progress["corpus_sha256"]
    )
    if corpus_reusable:
        logger.info("corpus parquet 与断点记录一致，跳过物化与导出")
        n = int(progress.get("n_unique_platform_texts", 0))
    else:
        if resume_valid and progress.get("final_built") and _table_exists(FINAL):
            logger.info("FINAL 表已物化（断点），跳过 DISTINCT ON 重物化")
            conn = _results_conn()
            try:
                cur = conn.cursor()
                cur.execute(f"SELECT count(*) FROM public.{FINAL}")
                n = int(cur.fetchone()[0])
            finally:
                conn.close()
        else:
            t0 = time.time()
            n = _materialize_final()
            progress["final_built"] = True
            progress["n_unique_platform_texts"] = n
            _save_progress(progress)
            logger.info(
                "FINAL 物化完成: %s 组，用时 %.1f 分钟",
                n, (time.time() - t0) / 60,
            )
        t0 = time.time()
        _export_table(FINAL, corpus_path, corpus_cols, types)
        progress["corpus_sha256"] = _sha(corpus_path)
        progress.pop("final_built", None)
        _save_progress(progress)
        logger.info(
            "corpus parquet 导出完成（zstd），用时 %.1f 分钟",
            (time.time() - t0) / 60,
        )

    frame_path = out_dir / "discovery_frame_v1.parquet"
    source_cols = [
        "representative_job_id", "year", "industry", "position_group",
        "company_size", "text_length_bin", "tech_flag", "platform",
        "anchor_main", "matched_skill_count", "text_hash", "description",
    ]
    t0 = time.time()
    pf = pq.ParquetFile(corpus_path)
    writer = None
    try:
        for rb in pf.iter_batches(batch_size=200000, columns=source_cols):
            tbl = pa.Table.from_batches([rb])
            platforms = tbl["platform"].to_pylist()
            hashes = tbl["text_hash"].to_pylist()
            did = pa.array(
                [
                    _discovery_id(p, h)
                    for p, h in zip(platforms, hashes)
                ],
                pa.string(),
            )
            tbl = tbl.add_column(0, "job_id", did)
            if writer is None:
                writer = pq.ParquetWriter(
                    frame_path, tbl.schema, compression="zstd"
                )
            writer.write_table(tbl)
    finally:
        if writer is not None:
            writer.close()
        pf.close()
    if writer is None:
        raise RuntimeError("discovery frame 为空")
    logger.info(
        "frame parquet 写出完成，用时 %.1f 分钟", (time.time() - t0) / 60
    )

    manifest = {
        "status": "formal_pass",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "global_distinct_platform_text_hash_representative_v2",
        "discovery_job_id_formula": "SHA256(platform | text_hash)",
        "representative_rule": [
            "industry_present_desc", "company_size_present_desc",
            "year_asc", "stable_job_id_asc",
        ],
        "source_snapshot_id": audit.get("snapshot_id"),
        "source_audit_sha256": audit_sha,
        "strata_config_sha256": cfg_sha,
        "pre_discovery_governance_sha256": v3_sha,
        "scan_stats": scan_stats,
        "scan_workers": scan_workers,
        "n_unique_platform_texts": n,
        "corpus_sha256": _sha(corpus_path),
        "frame_sha256": _sha(frame_path),
        "corpus_path": str(corpus_path),
        "frame_path": str(frame_path),
        "build_wall_seconds": round(time.time() - t_start, 1),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 收尾：stage 是纯中间态（实测可达 38GB），构建成功后清理；断点文件
    # 同步删除（manifest 的幂等检查取代它成为重跑门槛）。用重试删除：
    # 此刻若被 Windows 索引/防病毒短暂持有，裸 unlink 会让一次成功的构建
    # 以非零退出码收场。
    _drop_stage()
    _unlink_with_retry(_progress_path())
    logger.info(
        "构建完成: manifest=%s（stage 已清理，断点文件已删除）总用时 %.1f 小时",
        manifest_path, (time.time() - t_start) / 3600,
    )
    return manifest_path


def main() -> None:
    """CLI 入口。

    原实现直接调用 build() 而完全不解析 argv，于是任何参数——包括 --help，
    或一个拼错的选项——都会被静默忽略并启动整轮构建（实测：查参数时误触发了
    一次约 6 小时的全语料扫描）。此处补上 argparse，使参数错误能立即暴露。
    """
    ap = argparse.ArgumentParser(
        description="构建正式 dictionary discovery frame（§6.2.0 + handoff）"
    )
    ap.add_argument(
        "--fresh", action="store_true",
        help="忽略断点与已有 manifest，从零重建",
    )
    ap.add_argument(
        "--reverify", action="store_true",
        help="强制重跑源快照行数核对（默认同一审计哈希只核对一次）",
    )
    args = ap.parse_args()
    build(fresh=args.fresh, reverify=args.reverify)


if __name__ == "__main__":
    main()
