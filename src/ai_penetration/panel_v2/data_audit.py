"""指南 §5：正式源数据审计与源库 provenance。

只读 eps；生成阶段一要求的审计报告、字段字典、年度岗位数、重复诊断和
data_audit_manifest_v1.json。未知/可选字段按实际 schema 记录，不猜测。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

from config.paths import get_project_paths
from ..common import eps_conn_params
from .dedup import SHARDS

REQUIRED_JOB_FIELDS = {
    "recruit_id", "publish_time", "job_description", "position", "platform"
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _columns(cur, table: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s
        ORDER BY ordinal_position
        """,
        (table,),
    )
    return pd.DataFrame(
        cur.fetchall(), columns=["column_name", "data_type", "is_nullable"]
    )


def run() -> Path:
    paths = get_project_paths()
    out = paths.report_dir / "data_audit"
    out.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(**eps_conn_params())
    cur = conn.cursor()
    cur.execute("SET statement_timeout=0")
    cur.execute(
        "SELECT current_database(), current_user, version(), "
        "coalesce(inet_server_addr()::text,'local')"
    )
    dbname, dbuser, pg_version, server_addr = cur.fetchone()

    field_frames = []
    year_rows = []
    dup_rows = []
    table_fingerprints = []
    blockers = []
    warnings = []

    for city, table, _city_id in SHARDS:
        cols = _columns(cur, table)
        cols.insert(0, "city", city)
        cols.insert(1, "table", table)
        field_frames.append(cols)
        available = set(cols.column_name)
        missing = sorted(REQUIRED_JOB_FIELDS - available)
        if missing:
            blockers.append(f"{table} 缺必需字段: {', '.join(missing)}")
            continue

        cur.execute(
            f"""
            SELECT count(*)::bigint,
                   min(publish_time)::text,
                   max(publish_time)::text,
                   count(*) FILTER (WHERE job_description IS NULL)::bigint,
                   count(*) FILTER (
                       WHERE job_description IS NOT NULL
                         AND length(trim(job_description)) < 20
                   )::bigint,
                   count(*) FILTER (
                       WHERE recruit_id IS NULL OR trim(recruit_id::text)=''
                   )::bigint
            FROM public.{table}
            """
        )
        n, tmin, tmax, n_missing_desc, n_short, n_missing_rid = cur.fetchone()
        table_fingerprints.append({
            "city": city, "table": table, "rows": int(n),
            "min_publish_time": tmin, "max_publish_time": tmax,
        })
        if n_missing_rid:
            blockers.append(f"{table} 原始岗位编号缺失 {int(n_missing_rid)} 行")
        if n_missing_desc:
            warnings.append(f"{table} 描述 NULL {int(n_missing_desc)} 行")
        if n_short:
            warnings.append(f"{table} 描述可见长度<20 {int(n_short)} 行")

        cur.execute(
            f"""
            SELECT substr(publish_time,1,4) AS year,
                   count(*)::bigint AS n_jobs,
                   count(*) FILTER (
                       WHERE job_description IS NULL
                          OR trim(job_description)=''
                   )::bigint AS description_missing,
                   round(avg(length(coalesce(job_description,'')))::numeric,2)
                       AS avg_description_length
            FROM public.{table}
            GROUP BY 1 ORDER BY 1
            """
        )
        for year, nj, miss, avglen in cur.fetchall():
            year_rows.append({
                "city": city, "table": table, "year": year,
                "n_jobs": int(nj), "description_missing": int(miss),
                "avg_description_length": float(avglen or 0),
            })

        # 规则1诊断：同平台+原始编号重复。
        cur.execute(
            f"""
            SELECT count(*)::bigint, coalesce(sum(n-1),0)::bigint
            FROM (
                SELECT platform, recruit_id, count(*)::bigint n
                FROM public.{table}
                WHERE recruit_id IS NOT NULL
                GROUP BY 1,2 HAVING count(*)>1
            ) x
            """
        )
        dup_groups, dup_extra = cur.fetchone()

        # 完全相同规范文本的精确比例需要 Python 清洗；这里先记录原始描述
        # 完全相同的 DB 级下界，正式 dedup 的 text_hash 诊断由 M2 再输出。
        cur.execute(
            f"""
            SELECT count(*)::bigint, coalesce(sum(n-1),0)::bigint
            FROM (
                SELECT md5(job_description) h, count(*)::bigint n
                FROM public.{table}
                WHERE job_description IS NOT NULL
                GROUP BY 1 HAVING count(*)>1
            ) x
            """
        )
        text_groups, text_extra = cur.fetchone()
        dup_rows.append({
            "city": city, "table": table, "raw_rows": int(n),
            "platform_raw_id_duplicate_groups": int(dup_groups),
            "platform_raw_id_extra_rows": int(dup_extra),
            "raw_description_duplicate_groups": int(text_groups),
            "raw_description_extra_rows": int(text_extra),
        })

    fields = pd.concat(field_frames, ignore_index=True)
    fields_csv = out / "data_field_dictionary.csv"
    fields.to_csv(fields_csv, index=False, encoding="utf-8-sig")
    # 指南点名 xlsx；同时保留 CSV 便于无 Excel 环境审计。
    fields_xlsx = out / "data_field_dictionary.xlsx"
    try:
        fields.to_excel(fields_xlsx, index=False)
    except ImportError as exc:
        blockers.append(
            "缺 openpyxl，无法生成指南 §5.3 data_field_dictionary.xlsx"
        )

    years = pd.DataFrame(year_rows)
    years_csv = out / "year_job_counts.csv"
    years.to_csv(years_csv, index=False, encoding="utf-8-sig")
    dups = pd.DataFrame(dup_rows)
    dups_csv = out / "duplicate_diagnostics.csv"
    dups.to_csv(dups_csv, index=False, encoding="utf-8-sig")

    # 年份硬门：无法解析/超 2014—2025 要进入问题清单。
    if not years.empty:
        parsed = pd.to_numeric(years.year, errors="coerce")
        bad_year = parsed.isna() | ~parsed.between(2014, 2025)
        if bad_year.any():
            blockers.append(
                f"存在 {int(bad_year.sum())} 个 city×year 非法年份桶"
            )

    report = out / "data_audit_report.md"
    report.write_text(
        "\n".join([
            "# 招聘数据阶段一审计",
            "",
            f"- generated_at: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"- database: {dbname}",
            f"- server: {server_addr}",
            f"- postgres: {pg_version}",
            "",
            "## 阻断项",
            *(["- 无"] if not blockers else [f"- {x}" for x in blockers]),
            "",
            "## 警告",
            *(["- 无"] if not warnings else [f"- {x}" for x in warnings]),
            "",
            "## 分片快照",
            pd.DataFrame(table_fingerprints).to_markdown(index=False),
            "",
            "## 重复诊断",
            dups.to_markdown(index=False) if not dups.empty else "无",
        ]),
        encoding="utf-8",
    )
    conn.close()

    artifacts = [report, fields_csv, years_csv, dups_csv]
    if fields_xlsx.exists():
        artifacts.append(fields_xlsx)
    manifest = {
        "status": "formal_pass" if not blockers else "failed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "database": dbname,
        "database_user": dbuser,
        "server_addr": server_addr,
        "postgres_version": pg_version,
        "table_fingerprints": table_fingerprints,
        "blockers": blockers,
        "warnings": warnings,
        "artifact_sha256": {p.name: _sha(p) for p in artifacts},
    }
    manifest_path = out / "data_audit_manifest_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if blockers:
        raise SystemExit(2)
    return manifest_path


def main() -> None:
    argparse.ArgumentParser(description="指南 §5 数据审计").parse_args()
    run()


if __name__ == "__main__":
    main()
