"""金标准候选分层抽样：为人工/模型复核产出五层岗位样本（供 precision/recall 代理评估）。

分层（2024，基于已发布分类表 + 抽样重算）：
  S1_increment    v2>0.05 判 AI 但 v1 规则不判（宽口径增量核心带）
  S2_highconf     v2 得分 >0.30（高置信 AI 带）
  S3_borderline   v2 得分 0.03~0.05 未过线（边界漏检代理）
  S4_between      v2>0.15 未过但 >0.05 过（严宽之间）
  S5_negative     v2=0 随机（基线误报率对照）
每层随机定额抽样，输出岗位名+描述片段+全部判定，交复核方阅读标注。
用法::
    python -X utf8 -m src.ai_penetration.panel_v2.gold_sample
"""
from __future__ import annotations

import argparse
import json
import logging
import random

import numpy as np
import pandas as pd
import psycopg2
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..ai_scoring import is_ai_job
from ..common import (DEFAULT_OMEGA_SNAPSHOT, eps_conn_params,
                      resolve_artifact_path, setup_logging)
from ..skill_ai_anchor import build_skill_regex, extract_skills_fast
from .dedup import _h63

logger = logging.getLogger("ai_penetration.panel_v2.gold")

QUOTA = {"S1_increment": 60, "S2_highconf": 40, "S3_borderline": 40,
         "S4_between": 40, "S5_negative": 40}


def main() -> None:
    parser = argparse.ArgumentParser(description="金标准分层抽样")
    parser.add_argument("--sample-pct", type=float, default=0.6)
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "gold_sample.log")

    keys = np.load("output/panel_v2/pass2/master_npy/key.npy", mmap_mode="r")
    jids = np.load("output/panel_v2/pass2/master_npy/job_id.npy", mmap_mode="r")
    cls = pq.read_table(
        paths.output_dir / "release" / "panel_v2" / "job_ai_classification.parquet",
        columns=["job_id", "aijob_main_annual_raw_005",
                 "aijob_main_annual_raw_015"]).to_pandas()
    cls = cls.sort_values("job_id").reset_index(drop=True)
    score = pq.read_table(
        paths.output_dir / "release" / "panel_v2" / "job_ai_score"
        / "main_annual_raw.parquet",
        columns=["job_id", "ai_score"]).to_pandas() \
        if (paths.output_dir / "release" / "panel_v2" / "job_ai_score"
            / "main_annual_raw.parquet").exists() else None
    if score is None:
        raise SystemExit("job_ai_score dataset 已被合并删除？改用发布单文件抽样读取")
    score = score.sort_values("job_id")["ai_score"].to_numpy()
    # cls 行序即 job_id-1（job_id 从 1 连续）
    flag005 = cls.aijob_main_annual_raw_005.to_numpy()
    flag015 = cls.aijob_main_annual_raw_015.to_numpy()

    omega = json.loads(resolve_artifact_path(
        DEFAULT_OMEGA_SNAPSHOT, artifact="ωsAI").read_text(encoding="utf-8"))
    regex = build_skill_regex(__import__(
        "src.ai_penetration.skill_ai_anchor", fromlist=["load_merged_skills"]
    ).load_merged_skills(include_llm=True))

    buckets: dict[str, list] = {k: [] for k in QUOTA}
    n_scan = 0
    conn = psycopg2.connect(**eps_conn_params())
    cur = conn.cursor()
    cur.execute("SET statement_timeout=0")
    rng = random.Random(20260908)
    for _, shard in ((0, "job_p0387"), (1, "job_p0389")):
        cur.execute(
            f"SELECT recruit_id, position, job_description FROM public.{shard} "
            f"TABLESAMPLE SYSTEM ({args.sample_pct}) "
            f"WHERE substr(publish_time,1,4)='{args.year}' "
            "AND job_description IS NOT NULL AND length(trim(job_description))>=10 "
            "AND position IS NOT NULL AND position != '' AND recruit_id IS NOT NULL")
        while True:
            batch = cur.fetchmany(50000)
            if not batch:
                break
            for rid, pos, desc in batch:
                n_scan += 1
                k = _h63(str(rid))
                i = int(np.searchsorted(keys, k))
                if i >= len(keys) or int(keys[i]) != k:
                    continue
                j = int(jids[i]) - 1
                s = float(score[j])
                a = is_ai_job(str(pos), str(desc))
                sk = extract_skills_fast(str(desc), regex)
                scored = [omega[x] for x in sk if x in omega]
                b = bool(scored) and sum(scored) / len(scored) >= 0.15 and max(scored) >= 0.5
                v1f = a or b
                f005, f015 = flag005[j], flag015[j]
                if f005 and not v1f:
                    tag = "S1_increment"
                elif s > 0.30:
                    tag = "S2_highconf"
                elif 0.03 <= s <= 0.05:
                    tag = "S3_borderline"
                elif f005 and not f015:
                    tag = "S4_between"
                elif not f005 and rng.random() < 0.02:
                    tag = "S5_negative"
                else:
                    continue
                if len(buckets[tag]) < QUOTA[tag] * 3 and rng.random() < 0.5:
                    buckets[tag].append({
                        "stratum": tag, "job_id": j + 1, "year": args.year,
                        "v2_score": round(s, 4), "flag005": int(f005),
                        "flag015": int(f015), "v1_a": int(a), "v1_b": int(b),
                        "position": str(pos)[:50],
                        "desc_head": str(desc)[:260].replace("\n", " ")})
    conn.close()
    rows = []
    for tag, lst in buckets.items():
        rows.extend(random.Random(9).sample(lst, min(QUOTA[tag], len(lst))))
    out = pd.DataFrame(rows)
    path = paths.report_dir / "gold_sample_2024.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"扫描 {n_scan:,} 行 -> 五层共 {len(out)} 条 -> {path.name}")
    print(out.stratum.value_counts().to_dict())


if __name__ == "__main__":
    main()
