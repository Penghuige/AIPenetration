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
BASE_PER_YEAR_INDUSTRY = 10_000
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
    if df[["platform", "text_hash"]].duplicated().any():
        raise ValueError(
            "discovery frame 必须是 §6.2.0 DISTINCT(platform,text_hash) 语料"
        )
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


def baseline_sample(
    df: pd.DataFrame,
    per_year_industry: int = BASE_PER_YEAR_INDUSTRY,
    seed: int = SEED,
) -> pd.DataFrame:
    """8/24 handoff：每个 year×industry 单元最多 1 万条，不放回。

    单元内部再按岗位组、企业规模、文本长度、技术/非技术、平台做比例分层，
    兼容指南 §9.1 的其余分层维度。
    """
    validate_frame(df)
    parts = []
    work = df.copy()
    work["_industry"] = (
        work.industry.fillna("INDUSTRY_MISSING").astype(str)
        .str.strip().replace("", "INDUSTRY_MISSING")
    )
    grouped = work.groupby([work.year.astype(int), "_industry"], sort=True)
    for idx, ((year, _industry), group) in enumerate(grouped):
        target = min(per_year_industry, len(group))
        sampled = _proportional_take(
            group.drop(columns=["_industry"]),
            target,
            seed + int(year) * 1009 + idx,
        )
        sampled = sampled.copy()
        sampled["sampling_component"] = "baseline_year_industry"
        parts.append(sampled)
    if parts:
        out = pd.concat(parts, ignore_index=True)
    else:
        out = work.iloc[0:0].drop(columns=["_industry"]).copy()
    if out.job_id.duplicated().any():
        raise RuntimeError("基准分层样本出现重复 job_id")
    out["discovery_round"] = 0
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
    """8/24 handoff：连续两轮新增有效标准概念 <5 即满足饱和。

    coverage_gain_pp 仍保存为监测指标，但任何覆盖率派生条件都不参与停止。
    年份×行业抽样完整性由 finalize 直接从 frame/selected 验证。
    """
    required = {"round", "new_standard_concepts", "coverage_gain_pp"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError("round metrics 缺列: " + ", ".join(sorted(missing)))
    if len(metrics) < 2:
        return False
    last = metrics.sort_values("round").tail(2)
    return bool((last.new_standard_concepts < 5).all())


def finalize(
    frame_path: Path,
    selected_path: Path,
    metrics_path: Path,
    candidate_audit_path: Path,
    governance_path: Path,
) -> Path:
    """只有“抽样饱和 + 候选已接入最终治理表”同时满足才 formal_pass。"""
    df = pd.read_parquet(frame_path)
    validate_frame(df)
    selected = pd.read_csv(selected_path)
    if selected.job_id.duplicated().any():
        raise ValueError("候选发现样本存在重复 job_id")
    metrics = pd.read_csv(metrics_path)
    passed = saturation_pass(metrics)

    # 后期 handoff 的基准规则是每个 year×industry 单元都进入样本。
    frame_units = {
        (int(y), str(i).strip() or "INDUSTRY_MISSING")
        for y, i in zip(
            df.year,
            df.industry.fillna("INDUSTRY_MISSING"),
        )
    }
    if not {"year", "industry"}.issubset(selected.columns):
        raise ValueError("selected sample 缺 year/industry，无法验收分层覆盖")
    selected_units = {
        (int(y), str(i).strip() or "INDUSTRY_MISSING")
        for y, i in zip(
            selected.year,
            selected.industry.fillna("INDUSTRY_MISSING"),
        )
    }
    missing_units = frame_units - selected_units
    if missing_units:
        raise ValueError(
            "基准发现样本未覆盖全部 year×industry 单元: "
            + ", ".join(map(str, sorted(missing_units)[:10]))
        )

    audit = pd.read_csv(candidate_audit_path)
    required_audit = {
        "term", "final_grade", "final_skill_id",
        "source_round", "evidence_count", "span_valid",
    }
    missing = required_audit - set(audit.columns)
    if missing:
        raise ValueError(
            "candidate audit 缺列: " + ", ".join(sorted(missing))
        )
    if not set(audit.final_grade.astype(str)) <= {"A", "B", "C", "D"}:
    if {"decision", "source_round"}.issubset(audit.columns):
        derived = (
            audit[
                (audit.decision.astype(str) == "NEW_CONCEPT")
                & audit.final_grade.isin(["B", "C"])
            ]
            .groupby("source_round")
            .size()
            .to_dict()
        )
        for row in metrics.itertuples(index=False):
            expected_new = int(derived.get(int(row.round), 0))
            if int(row.new_standard_concepts) != expected_new:
                raise ValueError(
                    f"round {int(row.round)} new_standard_concepts="
                    f"{int(row.new_standard_concepts)} != audit-derived "
                    f"{expected_new}"
                )
        raise ValueError("candidate audit 含未知 final_grade")
    formal = audit[audit.final_grade.isin(["A", "B", "C"])].copy()
    if (formal.evidence_count <= 0).any() or not formal.span_valid.astype(bool).all():
        raise ValueError("A/B/C 候选存在无有效原文证据记录")
    if formal.final_skill_id.isna().any() or (
        formal.final_skill_id.astype(str).str.len() == 0
    ).any():
        raise ValueError("A/B/C 候选存在空 final_skill_id")

    governance = pd.read_csv(governance_path, encoding="utf-8-sig")
    req_g = {"term", "final_grade", "final_skill_id"}
    if not req_g <= set(governance.columns):
        raise ValueError("最终治理表缺 term/final_grade/final_skill_id")
    joined = formal.merge(
        governance[list(req_g)],
        on="term",
        how="left",
        suffixes=("_audit", "_gov"),
        validate="one_to_one",
    )
    mismatch = (
        joined.final_grade_gov.isna()
        | (joined.final_grade_audit.astype(str) != joined.final_grade_gov.astype(str))
        | (joined.final_skill_id_audit.astype(str)
           != joined.final_skill_id_gov.astype(str))
    )
    if mismatch.any():
        bad = joined.loc[mismatch, "term"].astype(str).head(10).tolist()
        raise ValueError(
            "正式发现的 A/B/C 候选未正确接入最终治理表: " + ", ".join(bad)
        )

    manifest = {
        "status": "formal_pass" if passed else "incomplete",
        "seed": SEED,
        "sampling_protocol": "handoff_20260824_year_industry_10k_saturation_v2",
        "frame_sha256": _sha(frame_path),
        "selected_sha256": _sha(selected_path),
        "round_metrics_sha256": _sha(metrics_path),
        "candidate_audit_sha256": _sha(candidate_audit_path),
        "governance_sha256": _sha(governance_path),
        "n_frame": len(df),
        "n_selected": len(selected),
        "n_candidates_audited": len(audit),
        "n_formal_candidates": len(formal),
        "rounds": int(metrics["round"].max()) if len(metrics) else 0,
        "required_strata": STRATA,
        "year_industry_units": len(frame_units),
        "coverage_gain_role": "monitor_only_not_stop_or_admission",
    }
    out = (
        get_project_paths().output_dir
        / "dictionary"
        / "formal_discovery_manifest_v1.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    f.add_argument("--candidate-audit", type=Path, required=True)
    f.add_argument("--governance", type=Path, required=True)
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
        finalize(args.frame, args.selected, args.metrics,
                 args.candidate_audit, args.governance)


if __name__ == "__main__":
    main()
