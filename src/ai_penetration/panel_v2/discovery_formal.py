"""指南 §9：正式分层候选发现的确定性抽样与饱和门。

本模块只负责“如何抽、何时停、如何留痕”，不替代 Qwen 抽取本身。
输入 discovery_frame.parquet 必须由 canonical 岗位构造，并显式带齐交接要求
的分层字段；字段不全时 fail-closed，而不是退化成 TABLESAMPLE。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import get_project_paths

SEED = 20260822
BASE_PER_YEAR = 10_000
ROUND_N = 10_000
REQUIRED = {
    "job_id", "year", "industry", "position_group", "company_size",
    "text_length_bin", "tech_flag", "platform", "anchor_main",
    "matched_skill_count", "text_hash", "description",
}
STRATA = [
    "industry", "position_group", "company_size",
    "text_length_bin", "tech_flag", "platform",
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_frame(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(
            "正式候选发现 frame 缺交接要求分层字段: " + ", ".join(missing)
        )
    if df.job_id.duplicated().any():
        raise ValueError("discovery frame job_id 不唯一")
    if df.text_hash.isna().any():
        raise ValueError("discovery frame text_hash 缺失")
    if (df.description.astype(str).str.len() < 10).any():
        raise ValueError("discovery frame 含无效描述")
    if df.year.isna().any():
        raise ValueError("discovery frame year 缺失")


def _rank(df: pd.DataFrame, seed: int) -> pd.Series:
    key = (
        df.job_id.astype(str)
        + "|"
        + df.text_hash.astype(str)
        + f"|{int(seed)}"
    )
    return key.map(
        lambda x: int.from_bytes(
            hashlib.blake2b(x.encode("utf-8"), digest_size=8).digest(), "big"
        )
    )


def _proportional_take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    n = min(n, len(df))
    work = df.copy()
    work["_stratum"] = work[STRATA].fillna("MISSING").astype(str).agg("|".join, axis=1)
    sizes = work.groupby("_stratum").size().sort_index()
    raw = sizes / sizes.sum() * n
    alloc = np.floor(raw).astype(int)
    remainder = n - int(alloc.sum())
    if remainder:
        frac = (raw - alloc).sort_values(ascending=False, kind="stable")
        for key in frac.index[:remainder]:
            alloc.loc[key] += 1
    parts = []
    for i, (key, take_n) in enumerate(alloc.items()):
        if take_n <= 0:
            continue
        g = work[work["_stratum"] == key].copy()
        g["_rank"] = _rank(g, seed + i)
        parts.append(g.nsmallest(min(int(take_n), len(g)), "_rank"))
    out = pd.concat(parts, ignore_index=False) if parts else work.iloc[0:0]
    if len(out) < n:
        remain = work.loc[~work.index.isin(out.index)].copy()
        remain["_rank"] = _rank(remain, seed + 991)
        out = pd.concat([out, remain.nsmallest(n - len(out), "_rank")])
    return out.drop(columns=["_stratum", "_rank"], errors="ignore").head(n)


def baseline_sample(df: pd.DataFrame, per_year: int = BASE_PER_YEAR,
                    seed: int = SEED) -> pd.DataFrame:
    """§9.2：逐年最多1万；行业先分配 min(50,n)，再跨全部分层比例补齐。"""
    validate_frame(df)
    parts = []
    for year in sorted(df.year.astype(int).unique()):
        y = df[df.year.astype(int) == int(year)].copy()
        target = min(per_year, len(y))
        picked = []
        for j, industry in enumerate(sorted(y.industry.fillna("INDUSTRY_MISSING").astype(str).unique())):
            g = y[y.industry.fillna("INDUSTRY_MISSING").astype(str) == industry].copy()
            k = min(50, len(g), target - sum(len(x) for x in picked))
            if k <= 0:
                break
            g["_rank"] = _rank(g, seed + year * 101 + j)
            picked.append(g.nsmallest(k, "_rank").drop(columns="_rank"))
        first = pd.concat(picked) if picked else y.iloc[0:0]
        remain_n = target - len(first)
        remain = y.loc[~y.index.isin(first.index)]
        extra = _proportional_take(remain, remain_n, seed + year * 1009)
        parts.append(pd.concat([first, extra]))
    out = pd.concat(parts).drop_duplicates("job_id").reset_index(drop=True)
    out["discovery_round"] = 0
    out["sampling_component"] = "baseline_year_industry_multistrata"
    return out


def incremental_round(df: pd.DataFrame, selected_ids: set,
                      round_no: int, n: int = ROUND_N,
                      seed: int = SEED) -> pd.DataFrame:
    """§9.2.1：40%低覆盖 + 30%主锚点 + 30%普通，岗位不得重复。"""
    validate_frame(df)
    pool = df[~df.job_id.isin(selected_ids)].copy()
    quotas = [
        ("low_coverage", 0.40, pool[pool.matched_skill_count <= 0]),
        ("main_anchor", 0.30, pool[pool.anchor_main == 1]),
        ("general", 0.30, pool),
    ]
    chosen = []
    used: set = set()
    for i, (name, share, cand) in enumerate(quotas):
        cand = cand[~cand.job_id.isin(used)]
        take = int(round(n * share))
        part = _proportional_take(cand, take, seed + round_no * 10007 + i)
        part = part.copy()
        part["sampling_component"] = name
        chosen.append(part)
        used.update(part.job_id.tolist())
    out = pd.concat(chosen) if chosen else pool.iloc[0:0]
    if len(out) < min(n, len(pool)):
        remain = pool[~pool.job_id.isin(used)]
        extra = _proportional_take(
            remain, min(n, len(pool)) - len(out),
            seed + round_no * 10007 + 999,
        ).copy()
        extra["sampling_component"] = "quota_shortfall_fill"
        out = pd.concat([out, extra])
    out = out.drop_duplicates("job_id").head(n).reset_index(drop=True)
    out["discovery_round"] = int(round_no)
    return out


def saturation_pass(metrics: pd.DataFrame) -> bool:
    """§9.2.2：最后连续两轮都满足全部四个停止条件。"""
    required = {
        "round", "new_standard_concepts", "coverage_gain_pp",
        "all_years_major_industries_covered", "anchor_uncovered_stable",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError("round metrics 缺列: " + ", ".join(sorted(missing)))
    if len(metrics) < 2:
        return False
    last = metrics.sort_values("round").tail(2)
    ok = (
        (last.new_standard_concepts < 5)
        & (last.coverage_gain_pp < 0.1)
        & last.all_years_major_industries_covered.astype(bool)
        & last.anchor_uncovered_stable.astype(bool)
    )
    return bool(ok.all())


def finalize(frame_path: Path, selected_path: Path, metrics_path: Path) -> Path:
    df = pd.read_parquet(frame_path)
    validate_frame(df)
    selected = pd.read_csv(selected_path)
    if selected.job_id.duplicated().any():
        raise ValueError("候选发现样本存在重复 job_id")
    metrics = pd.read_csv(metrics_path)
    passed = saturation_pass(metrics)
    manifest = {
        "status": "formal_pass" if passed else "incomplete",
        "seed": SEED,
        "sampling_protocol": "guide_9.1_9.2_9.2.1_9.2.2_v1",
        "frame_sha256": _sha(frame_path),
        "selected_sha256": _sha(selected_path),
        "round_metrics_sha256": _sha(metrics_path),
        "n_frame": len(df),
        "n_selected": len(selected),
        "rounds": int(metrics["round"].max()) if len(metrics) else 0,
        "required_strata": STRATA,
    }
    out = get_project_paths().output_dir / "dictionary" / "formal_discovery_manifest_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit(2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="指南 §9 正式候选发现协议")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--frame", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("round")
    r.add_argument("--frame", type=Path, required=True)
    r.add_argument("--selected", type=Path, required=True)
    r.add_argument("--round", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--frame", type=Path, required=True)
    f.add_argument("--selected", type=Path, required=True)
    f.add_argument("--metrics", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "baseline":
        df = pd.read_parquet(args.frame)
        out = baseline_sample(df)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
    elif args.cmd == "round":
        df = pd.read_parquet(args.frame)
        selected = pd.read_csv(args.selected)
        out = incremental_round(df, set(selected.job_id), args.round)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
    else:
        finalize(args.frame, args.selected, args.metrics)


if __name__ == "__main__":
    main()
