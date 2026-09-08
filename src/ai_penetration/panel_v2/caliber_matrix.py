"""口径归因矩阵：把 v1 面板率与 v2 主指标的差异分解到可量化的格。

同一抽样上同时计算四个口径（消除数据可得性差异）：
  cell1  v1 规则（A ∪ B: avgω≥0.15 且 maxω≥0.5，自建词表+ω 快照）× 全量广告（不去重）
  cell2  v1 规则 × master 去重文本（本模块实测）
  cell3  v2 主指标 >0.05 × master（来自发布分类表）
  cell4  v2 严格 >0.15 × master（同上）
cell1→cell2 = 去重/文本构成贡献；cell2→cell4/3 = 判据阈值与词表贡献。

eps 只读；输出 output/reports/caliber_matrix_<ts>.md/csv。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.caliber_matrix --sample-pct 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import DEFAULT_OMEGA_SNAPSHOT, eps_conn_params, resolve_artifact_path, setup_logging
from ..ai_scoring import is_ai_job
from ..load_guangdong import GD_SHARDS
from ..skill_ai_anchor import build_skill_regex, extract_skills_fast, load_merged_skills
from ..text_clean import match_from_raw
from .dedup import _h63

logger = logging.getLogger("ai_penetration.panel_v2.caliber")

MASTER_NPY = "output/panel_v2/pass2/master_npy"


def main() -> None:
    parser = argparse.ArgumentParser(description="v1/v2 口径归因矩阵")
    parser.add_argument("--sample-pct", type=float, default=2.0)
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "caliber_matrix.log")

    keys = np.load(f"{MASTER_NPY}/key.npy", mmap_mode="r")
    myear = np.load(f"{MASTER_NPY}/year.npy", mmap_mode="r")
    omega = json.loads(resolve_artifact_path(
        DEFAULT_OMEGA_SNAPSHOT, artifact="ωsAI 快照").read_text(encoding="utf-8"))
    regex = build_skill_regex(load_merged_skills(include_llm=True))

    conn = psycopg2.connect(**eps_conn_params())
    cur = conn.cursor()
    cur.execute("SET statement_timeout=0")
    stat = {"all": [0, 0, 0], "mst": [0, 0, 0]}  # [n, a, fused]
    for city, shard in (("广州市", "job_p0387"), ("深圳市", "job_p0389")):
        cur.execute(
            f"SELECT recruit_id, position, job_description FROM public.{shard} "
            f"TABLESAMPLE SYSTEM ({args.sample_pct}) "
            "WHERE substr(publish_time,1,4)=%s "
            "  AND job_description IS NOT NULL AND length(trim(job_description))>=10 "
            "  AND position IS NOT NULL AND position != '' AND recruit_id IS NOT NULL",
            (str(args.year),))
        while True:
            batch = cur.fetchmany(50000)
            if not batch:
                break
            for rid, pos, desc in batch:
                k = _h63(str(rid))
                i = int(np.searchsorted(keys, k))
                in_m = i < len(keys) and int(keys[i]) == k
                a = is_ai_job(str(pos), str(desc))
                sk = extract_skills_fast(str(desc), regex)
                scored = [omega[s] for s in sk if s in omega]
                b = bool(scored) and sum(scored) / len(scored) >= 0.15 and max(scored) >= 0.5
                f = a or b
                stat["all"][0] += 1
                stat["all"][1] += a
                stat["all"][2] += f
                if in_m:
                    stat["mst"][0] += 1
                    stat["mst"][1] += a
                    stat["mst"][2] += f
    conn.close()

    cls = pq.read_table(
        paths.output_dir / "release" / "panel_v2" / "job_ai_classification.parquet",
        columns=["year", "aijob_main_annual_raw_005", "aijob_main_annual_raw_015"]
    ).to_pandas()
    cy = cls[cls.year == args.year]
    v2 = {
        "v2_005": float(cy.aijob_main_annual_raw_005.mean()),
        "v2_015": float(cy.aijob_main_annual_raw_015.mean()),
    }
    rows = [
        ("cell1", f"v1 规则 × 当年全量广告（不去重）",
         stat["all"][2] / max(stat["all"][0], 1), stat["all"][0]),
        ("cell2", f"v1 规则 × 当年 master 去重文本",
         stat["mst"][2] / max(stat["mst"][0], 1), stat["mst"][0]),
        ("cell3", "v2 主指标 >0.05 × master（发布值）", v2["v2_005"], len(cy)),
        ("cell4", "v2 严格 >0.15 × master（发布值）", v2["v2_015"], len(cy)),
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [f"# 口径归因矩阵（{args.year}，抽样 {args.sample_pct}%）", "",
             "| 格 | 口径 | AI 率 | n |", "|---|---|---|---|"]
    lines += [f"| {c} | {lab} | {r:.4%} | {n:,} |" for c, lab, r, n in rows]
    d12 = rows[0][2] and (rows[1][2] / rows[0][2] - 1)
    d24 = rows[3][2] / max(rows[1][2], 1e-12)
    lines += ["",
              f"- 去重/文本构成效应（cell1→cell2）：{d12:+.1%}",
              f"- 判据+词表效应（cell2→cell4，同为严阈值口径倍率）：×{d24:.2f}",
              f"- 阈值效应（cell4→cell3）：×{rows[2][2]/max(rows[3][2],1e-12):.2f}",
              ""]
    out = paths.report_dir / f"caliber_matrix_{stamp}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
