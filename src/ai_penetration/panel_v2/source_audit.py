"""指南 §5：源招聘数据库只读审计与快照身份登记。

正式运行必须由执行者提供可追溯的 --snapshot-id（数据库备份名、快照ID或
交付方固定数据版本号）。代码不会把当前日期/库名伪装成不可变快照。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime

import pandas as pd
import psycopg2

from config.paths import get_project_paths
from ..common import eps_conn_params
from .dedup import SHARDS

JOB_REQUIRED = {"recruit_id", "publish_time", "job_description", "position", "platform"}

JOB_EXPECTED = {
    "recruit_id": "原始岗位编号",
    "publish_time": "发布日期",
    "job_description": "岗位描述",
    "position": "岗位名称",
    "platform": "招聘平台",
    "education": "学历",
    "work_type": "工作类型",
    "experience": "经验",
    "recruit_count": "招聘人数",
    "age_req": "年龄要求",
}
ENT_EXPECTED = {
    "recruit_id": "岗位编号关联键",
    "company_id": "企业ID",
    "industry_code": "行业编码",
}


def _one(cur, sql: str):
    cur.execute(sql)
    return cur.fetchone()[0]


def _columns(cur, table: str) -> list[tuple]:
    cur.execute(
        """SELECT column_name, data_type, is_nullable
           FROM information_schema.columns
           WHERE table_schema='public' AND table_name=%s
           ORDER BY ordinal_position""",
        (table,),
    )
    return cur.fetchall()


def audit(snapshot_id: str):
    snapshot_id = str(snapshot_id).strip()
    if not snapshot_id or snapshot_id.lower() in {
        "latest", "current", "unknown", "none", "to_be_confirmed"
    }:
        raise ValueError(
            "--snapshot-id 必须是外部可追溯的固定数据版本/备份标识，"
            "不能用 latest/current/unknown"
        )

    conn = psycopg2.connect(**eps_conn_params())
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SHOW server_version")
    server_version = str(cur.fetchone()[0])
    cur.execute("SELECT current_database()")
    database = str(cur.fetchone()[0])

    field_rows, year_rows, dup_rows = [], [], []
    blocking, warnings = [], []
    table_stats = {}

    for city, job_table, _city_id in SHARDS:
        ent_table = job_table.replace("job_", "ent_", 1)
        cols = _columns(cur, job_table)
        col_names = {x[0] for x in cols}
        for name, dtype, nullable in cols:
            field_rows.append({
                "city": city, "table": job_table, "column": name,
                "data_type": dtype, "nullable": nullable,
                "expected_role": JOB_EXPECTED.get(name, ""),
            })
        missing_required = sorted(JOB_REQUIRED - col_names)
        if missing_required:
            blocking.append(
                job_table + " 缺主链必需字段: " + ", ".join(missing_required)
            )
        missing_optional = sorted((set(JOB_EXPECTED) - JOB_REQUIRED) - col_names)
        if missing_optional:
            warnings.append(
                job_table + " 缺审计辅助字段: " + ", ".join(missing_optional)
            )

        ecols = _columns(cur, ent_table)
        e_names = {x[0] for x in ecols}
        for name, dtype, nullable in ecols:
            field_rows.append({
                "city": city, "table": ent_table, "column": name,
                "data_type": dtype, "nullable": nullable,
                "expected_role": ENT_EXPECTED.get(name, ""),
            })
        if "company_id" not in e_names:
            warnings.append(ent_table + " 缺 company_id，去重/LOO能力下降")
        if "industry_code" not in e_names:
            warnings.append(ent_table + " 缺 industry_code，正式分层发现无法完成")

        total = _one(cur, f"SELECT count(*) FROM public.{job_table}")
        cur.execute(
            f"""SELECT
                count(*) FILTER (WHERE recruit_id IS NULL),
                count(*) FILTER (WHERE position IS NULL OR trim(position)=''),
                count(*) FILTER (
                    WHERE job_description IS NULL OR trim(job_description)=''
                ),
                count(*) FILTER (
                    WHERE job_description IS NOT NULL
                      AND length(trim(job_description)) < 20
                ),
                min(publish_time), max(publish_time)
                FROM public.{job_table}"""
        )
        (missing_id, missing_pos, missing_desc, short_desc,
         min_date, max_date) = cur.fetchone()
        table_stats[job_table] = {
            "city": city, "rows": int(total),
            "missing_recruit_id": int(missing_id),
            "missing_position": int(missing_pos),
            "missing_description": int(missing_desc),
            "short_description_lt20": int(short_desc),
            "min_publish_time": str(min_date),
            "max_publish_time": str(max_date),
        }
        if total and missing_desc == total:
            blocking.append(job_table + " 岗位描述整体缺失")

        cur.execute(
            f"""SELECT substr(publish_time,1,4) AS y, count(*)
                FROM public.{job_table}
                GROUP BY 1 ORDER BY 1"""
        )
        for year, n in cur.fetchall():
            year_s = str(year or "")
            valid = year_s.isdigit() and 2014 <= int(year_s) <= 2025
            year_rows.append({
                "city": city, "table": job_table,
                "year_raw": year_s, "n_jobs": int(n),
                "valid_2014_2025": valid,
            })

        cur.execute(
            f"""SELECT count(*), coalesce(sum(n-1),0) FROM (
                SELECT platform, recruit_id, count(*) n
                FROM public.{job_table}
                WHERE recruit_id IS NOT NULL
                GROUP BY 1,2 HAVING count(*)>1
            ) x"""
        )
        dup_groups, dup_extra = cur.fetchone()
        cur.execute(
            f"""SELECT count(*) FROM (
                SELECT md5(job_description) h
                FROM public.{job_table}
                WHERE job_description IS NOT NULL
                GROUP BY 1 HAVING count(*)>1
            ) x"""
        )
        exact_desc_groups = cur.fetchone()[0]
        cur.execute(
            f"""SELECT count(*) FROM (
                SELECT md5(job_description) h
                FROM public.{job_table}
                WHERE job_description IS NOT NULL
                GROUP BY 1
                HAVING count(DISTINCT platform)>1
            ) x"""
        )
        cross_platform_desc_groups = cur.fetchone()[0]
        dup_rows.append({
            "city": city, "table": job_table,
            "platform_rawid_duplicate_groups": int(dup_groups),
            "platform_rawid_extra_rows": int(dup_extra),
            "exact_raw_description_duplicate_groups":
                int(exact_desc_groups),
            "cross_platform_exact_raw_description_groups":
                int(cross_platform_desc_groups),
        })

    conn.close()

    paths = get_project_paths()
    out = paths.output_dir / "data_audit"
    out.mkdir(parents=True, exist_ok=True)
    years_path = out / "year_job_counts.csv"
    dup_path = out / "duplicate_diagnostics.csv"
    fields_path = out / "data_field_dictionary.xlsx"
    pd.DataFrame(year_rows).to_csv(
        years_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(dup_rows).to_csv(
        dup_path, index=False, encoding="utf-8-sig"
    )
    try:
        pd.DataFrame(field_rows).to_excel(fields_path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "生成指南要求的 data_field_dictionary.xlsx 需要 openpyxl"
        ) from exc

    year_df = pd.DataFrame(year_rows)
    invalid_n = int(
        year_df.loc[~year_df.valid_2014_2025, "n_jobs"].sum()
    )
    if invalid_n:
        warnings.append(
            f"2014–2025 外/不可解析年份共 {invalid_n} 行，正式主样本须隔离"
        )

    report_path = out / "data_audit_report.md"
    lines = [
        "# 源招聘数据审计",
        "",
        "- snapshot_id: " + snapshot_id,
        "- database: " + database,
        "- PostgreSQL: " + server_version,
        "- 生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "## 阻断项",
    ]
    lines += ["- 无"] if not blocking else ["- ❌ " + x for x in blocking]
    lines += ["", "## 警告"]
    lines += ["- 无"] if not warnings else ["- ⚠️ " + x for x in warnings]
    lines += [
        "", "## 普通统计",
        "- year counts: " + years_path.name,
        "- duplicate diagnostics: " + dup_path.name,
        "- field dictionary: " + fields_path.name,
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    files = [years_path, dup_path, fields_path, report_path]
    manifest = {
        "status": "formal_pass" if not blocking else "failed",
        "snapshot_id": snapshot_id,
        "database": database,
        "server_version": server_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tables": table_stats,
        "blocking": blocking,
        "warnings": warnings,
        "artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    manifest_path = out / "source_db_manifest_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if blocking:
        raise SystemExit(2)
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(description="指南 §5 源招聘库正式审计")
    ap.add_argument("--snapshot-id", required=True)
    args = ap.parse_args()
    audit(args.snapshot_id)


if __name__ == "__main__":
    main()
