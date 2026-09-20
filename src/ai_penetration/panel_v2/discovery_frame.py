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

import csv
import hashlib
import io
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
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
    company_col = str(
        cfg.get("source", {}).get("company_size_column", "")
    ).strip()
    tech_regex = str(
        cfg.get("tech_flag", {}).get("position_regex", "")
    ).strip()
    bad = []
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


def _pre_discovery_lexicon():
    paths = get_project_paths()
    path = (
        paths.output_dir / "dictionary"
        / "skill_legacy_graded_BCD_v3.csv"
    )
    if not path.exists():
        raise RuntimeError(
            "缺 pre-discovery v3 治理表；请先完成 legacy_freq + lexicon_llm merge"
        )
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


def _scan_source(cfg: dict, lex) -> dict:
    bins = [float(x) for x in cfg["text_length"]["bins"]]
    labels = [str(x) for x in cfg["text_length"]["labels"]]
    tech = re.compile(str(cfg["tech_flag"]["position_regex"]), re.I)
    stats = {"rows_seen": 0, "rows_kept": 0, "invalid_year": 0}
    conn = psycopg2.connect(**eps_conn_params())
    try:
        for city, job_table, _city_id in SHARDS:
            ent_table = job_table.replace("job_", "ent_", 1)
            query = _source_query(conn, job_table, ent_table, cfg)
            cur = conn.cursor(f"discovery_{job_table}")
            cur.itersize = 20000
            cur.execute(query)
            buf: list[tuple] = []
            while True:
                batch = cur.fetchmany(20000)
                if not batch:
                    break
                for (
                    rid, platform, position, publish_time, desc,
                    industry, company_size, company_id,
                ) in batch:
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
                    platform_s = _norm_meta(platform, "PLATFORM_MISSING")
                    buf.append((
                        int(_stable_job_id(platform_s, str(rid))),
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
                    if len(buf) >= 50000:
                        _copy_rows(buf)
                        buf = []
                if buf:
                    _copy_rows(buf)
                    buf = []
            cur.close()
    finally:
        conn.close()
    return stats


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
    try:
        cur = conn.cursor()
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            cur.copy_expert(
                "COPY (SELECT " + ", ".join(columns)
                + f" FROM public.{table} ORDER BY platform, text_hash) "
                "TO STDOUT WITH (FORMAT csv, HEADER true)",
                fh,
            )
    finally:
        conn.close()
    reader = pcsv.open_csv(
        tmp,
        read_options=pcsv.ReadOptions(block_size=1 << 24),
        convert_options=pcsv.ConvertOptions(column_types=types),
    )
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
        tmp.unlink(missing_ok=True)
    if writer is None:
        raise RuntimeError(f"{table} 导出为空")


def build() -> Path:
    paths = get_project_paths()
    setup_logging(paths.log_dir / "discovery_frame.log")
    cfg, cfg_path = _load_config()
    _validate_source_fields(cfg)
    audit_manifest = (
        paths.output_dir / "data_audit" / "source_db_manifest_v1.json"
    )
    if not audit_manifest.exists():
        raise RuntimeError("缺 source_db_manifest_v1.json；先运行 source_audit")
    audit = json.loads(audit_manifest.read_text(encoding="utf-8"))
    if audit.get("status") != "formal_pass":
        raise RuntimeError("source_db_manifest_v1 未 formal_pass")

    lex, v3_path = _pre_discovery_lexicon()
    _create_stage()
    scan_stats = _scan_source(cfg, lex)
    n = _materialize_final()

    out_dir = paths.output_dir / "dictionary"
    out_dir.mkdir(parents=True, exist_ok=True)
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
    _export_table(FINAL, corpus_path, corpus_cols, types)

    frame_path = out_dir / "discovery_frame_v1.parquet"
    source_cols = [
        "representative_job_id", "year", "industry", "position_group",
        "company_size", "text_length_bin", "tech_flag", "platform",
        "anchor_main", "matched_skill_count", "text_hash", "description",
    ]
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
    if writer is None:
        raise RuntimeError("discovery frame 为空")

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
        "source_audit_sha256": _sha(audit_manifest),
        "strata_config_sha256": _sha(cfg_path),
        "pre_discovery_governance_sha256": _sha(v3_path),
        "scan_stats": scan_stats,
        "n_unique_platform_texts": n,
        "corpus_sha256": _sha(corpus_path),
        "frame_sha256": _sha(frame_path),
        "corpus_path": str(corpus_path),
        "frame_path": str(frame_path),
    }
    manifest_path = out_dir / "discovery_frame_manifest_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    build()


if __name__ == "__main__":
    main()
