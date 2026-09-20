"""互斥得分带的无偏验证样本（补充验证，不改变主方法）。

与历史 gold_sample.py 不同：本脚本直接从完整发布 frame 中抽样，不使用
TABLESAMPLE、不使用重叠优先层、不预截候选池；同一得分带内每个岗位等概率。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from config.paths import get_project_paths
from ..common import eps_conn_params
from ..text_clean import match_from_raw, text_hash
from .dedup import SHARDS


BANDS = (
    ("gt_030", lambda s: s > 0.30),
    ("015_030", lambda s: (s > 0.15) & (s <= 0.30)),
    ("005_015", lambda s: (s > 0.05) & (s <= 0.15)),
    ("003_005", lambda s: (s >= 0.03) & (s <= 0.05)),
    ("le_003", lambda s: s < 0.03),
)


def _rank(job_id: int, seed: int) -> int:
    raw = f"{seed}|{int(job_id)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def select_band_sample(frame: pd.DataFrame, quota: int,
                       seed: int) -> pd.DataFrame:
    if frame.job_id.duplicated().any():
        raise ValueError("完整抽样 frame job_id 不唯一")
    rows = []
    covered = pd.Series(False, index=frame.index)
    for band, predicate in BANDS:
        mask = predicate(frame.ai_score)
        if (covered & mask).any():
            raise RuntimeError(f"得分带定义重叠: {band}")
        covered |= mask
        sub = frame[mask].copy()
        n_frame = len(sub)
        sub["_rank"] = sub.job_id.map(lambda j: _rank(j, seed))
        picked = sub.nsmallest(min(quota, n_frame), "_rank").drop(columns="_rank")
        picked["band"] = band
        picked["band_frame_n"] = n_frame
        picked["seed"] = seed
        picked["inclusion_probability"] = (
            min(quota, n_frame) / n_frame if n_frame else 0.0
        )
        rows.append(picked)
    out = pd.concat(rows, ignore_index=True)
    if out.job_id.duplicated().any():
        raise RuntimeError("互斥得分带样本出现重复 job_id")
    return out


def _attach_canonical_text(sample: pd.DataFrame) -> pd.DataFrame:
    import psycopg2
    from .scan import _results_conn

    rc = _results_conn()
    try:
        cur = rc.cursor()
        ids = sample.job_id.astype(int).tolist()
        meta = []
        for i in range(0, len(ids), 5000):
            chunk = ids[i:i + 5000]
            cur.execute(
                "SELECT job_id, job_id_raw, city, thash "
                "FROM public.job_master_gzsz WHERE job_id = ANY(%s)",
                (chunk,),
            )
            meta.extend(cur.fetchall())
    finally:
        rc.close()
    m = pd.DataFrame(meta, columns=["job_id", "job_id_raw", "city", "thash"])
    out = sample.merge(m, on="job_id", how="left", validate="one_to_one")
    if out.job_id_raw.isna().any():
        raise RuntimeError("样本 job_id 无法回到 canonical master")

    source = psycopg2.connect(**eps_conn_params())
    text_rows = []
    try:
        for _city_name, shard, city_id in SHARDS:
            need = out[out.city == city_id]
            rids = need.job_id_raw.astype(str).tolist()
            if not rids:
                continue
            want = {
                str(r.job_id_raw): (int(r.job_id), int(r.thash))
                for r in need.itertuples()
            }
            for i in range(0, len(rids), 2000):
                chunk = rids[i:i + 2000]
                cur = source.cursor()
                cur.execute(
                    f"SELECT recruit_id, position, job_description "
                    f"FROM public.{shard} WHERE recruit_id = ANY(%s)",
                    (chunk,),
                )
                for rid, pos, desc in cur.fetchall():
                    target = want.get(str(rid))
                    if target is None:
                        continue
                    jid, thash = target
                    match = match_from_raw(str(desc))
                    if text_hash(match) != thash:
                        continue
                    text_rows.append({
                        "job_id": jid,
                        "position": str(pos),
                        "job_description": str(desc),
                    })
    finally:
        source.close()

    text_df = pd.DataFrame(text_rows).drop_duplicates("job_id")
    out = out.merge(text_df, on="job_id", how="left", validate="one_to_one")
    if out.job_description.isna().any():
        raise RuntimeError(
            f"{int(out.job_description.isna().sum())} 个样本无法回填 canonical 原文"
        )
    return out


def run(rel_name: str, year: int, quota: int, seed: int) -> Path:
    paths = get_project_paths()
    rel = paths.output_dir / "release" / rel_name
    cls = pq.read_table(
        rel / "job_ai_classification.parquet",
        columns=["job_id", "year", "aijob_main_annual_raw_005",
                 "aijob_main_annual_raw_015"],
    ).to_pandas()
    score_path = rel / "job_ai_score" / "main_annual_raw.parquet"
    score = pq.read_table(score_path, columns=["job_id", "ai_score"]).to_pandas()
    frame = cls.merge(score, on="job_id", how="left", validate="one_to_one")
    frame = frame[frame.year == year].copy()
    if frame.ai_score.isna().all():
        raise RuntimeError("目标年份无可用 main annual raw score")
    sample = select_band_sample(frame, quota=quota, seed=seed)
    sample = _attach_canonical_text(sample)

    out = paths.report_dir / f"gold_band_sample_{year}_seed{seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False, encoding="utf-8-sig")
    manifest = {
        "sample_file": str(out),
        "seed": seed,
        "year": year,
        "quota_per_band": quota,
        "band_frame_sizes": sample.groupby("band").band_frame_n.first().to_dict(),
        "sampling": "full_frame_equal_probability_within_mutually_exclusive_band",
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="无偏得分带验证样本")
    ap.add_argument("--rel", default="panel_v2i")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--quota", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()
    run(args.rel, args.year, args.quota, args.seed)


if __name__ == "__main__":
    main()
