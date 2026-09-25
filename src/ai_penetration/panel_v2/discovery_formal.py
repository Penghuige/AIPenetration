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
# §9.2.3 行业保底与决策点 A 的"主要行业"量化定义（guideline clarification，
# 记入 manifest）
INDUSTRY_FLOOR = 50
MAJOR_INDUSTRY_MIN = 100
COVERAGE_STOP_PP = 0.1
ANCHOR_UNCOVERED_STOP_PP = 0.1
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


def _norm_industry(value) -> str:
    """行业字段的确定性清洗（§9.1：空白归 INDUSTRY_MISSING，strip）。"""
    text = str(value if value is not None else "").strip()
    return text if text else "INDUSTRY_MISSING"


def _industry_quota(sizes: "pd.Series[int]", n: int) -> dict[str, int]:
    """§9.2.3-4 年内行业配额：保底 + 比例 + 差额补齐。

    年内首先向每个非空行业分配 ``min(50, 该行业岗位数)`` 条；剩余名额按
    该行业在本年度**剩余**岗位中的比例分配；舍入造成的差额按行业剩余
    岗位数**由大到小**补齐。保底总额超过样本名额（行业极多而名额少，
    指南未覆盖的情形）时，按行业岗位数由大到小保留保底并截断到 n
    （guideline clarification）。

    Args:
        sizes: {行业键: 该年岗位数}。
        n: 该年样本名额（min(1 万, 年量)）。

    Returns:
        {行业键: 配额}，合计恰为 min(n, 全部岗位数)。
    """
    if n <= 0 or sizes.empty:
        return {}
    floors = {k: min(INDUSTRY_FLOOR, int(v)) for k, v in sizes.items()}
    total_floor = sum(floors.values())
    if total_floor >= n:
        quotas: dict[str, int] = {}
        acc = 0
        for key in sorted(sizes.index, key=lambda k: (-int(sizes[k]), str(k))):
            take = min(floors[key], n - acc)
            if take <= 0:
                break
            quotas[key] = take
            acc += take
        return quotas
    quotas = dict(floors)
    remaining = n - total_floor
    residual = {k: int(sizes[k]) - floors[k] for k in sizes.index}
    total_residual = sum(residual.values())
    if remaining <= 0 or total_residual <= 0:
        return quotas
    raw = {
        k: remaining * v / total_residual
        for k, v in residual.items() if v > 0
    }
    alloc = {k: int(v) for k, v in raw.items()}
    diff = remaining - sum(alloc.values())
    order = sorted(alloc, key=lambda k: (-residual[k], str(k)))
    i = 0
    while diff > 0 and order:
        alloc[order[i % len(order)]] += 1
        diff -= 1
        i += 1
    for k, v in alloc.items():
        quotas[k] += v
    return quotas


def baseline_sample(
    df: pd.DataFrame,
    per_year: int = BASE_PER_YEAR,
    seed: int = SEED,
) -> pd.DataFrame:
    """§9.1/§9.2.1-9.2.5 基准样本：按年分层，每年最多 1 万条，不放回。

    年内实行指南两阶段行业配额（:func:`_industry_quota`：保底
    min(50,·) + 剩余岗位比例 + 差额按剩余岗位数由大到小补齐）；行业配额
    内部再按岗位组、企业规模、文本长度、技术/非技术、平台做比例分层
    （行业子集内 industry 维恒定，分层自动退化为其余五维）。

    口径决策记录：指南 §9.1 三处均写"每年抽取 1 万条、最多形成 12 万条基准
    招聘文本"。曾按 8/24 后续裁决实现为"每个 year×industry 单元最多 1 万条"，
    但实测本语料 11527 个单元平均约 1449 行，等于全量取走 9,168,049 条
    （指南口径的 101 倍、约 1.5 年抽取），不可行，故回归指南口径。
    min(50) 行业保底此前未实现、现按指南补齐；偏差仍记入 manifest 的
    guideline_deviations。
    """
    validate_frame(df)
    parts = []
    work = df.copy()
    work["_industry"] = work.industry.map(_norm_industry)
    grouped = work.groupby(work.year.astype(int), sort=True)
    for idx, (year, group) in enumerate(grouped):
        target = min(per_year, len(group))
        sizes = group.groupby("_industry").size()
        quotas = _industry_quota(sizes, target)
        seed_year = seed + int(year) * 1009 + idx
        year_parts = []
        for j, (ind, quota_n) in enumerate(sorted(quotas.items())):
            sub = group[group["_industry"] == ind].drop(columns=["_industry"])
            take = _proportional_take(
                sub, quota_n, seed_year + j * 101
            )
            take = take.copy()
            take["sampling_component"] = "baseline_year"
            year_parts.append(take)
        if year_parts:
            sampled = pd.concat(year_parts, ignore_index=True)
        else:
            sampled = group.iloc[0:0].drop(columns=["_industry"])
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
    """§9.2.2 默认饱和标准（四条件，连续两轮）。

    1. 每 1 万条新增有效标准概念 <5；
    2. 覆盖率提升 <0.1pp（覆盖率缺失/NaN 时 fail-closed 不饱和——
       覆盖率必须每轮真实计算，见 coverage 模块）；
    3. 各年份与主要行业入批由 finalize 从 frame/selected 验证（§9.2.2.3）；
    4. 主锚点命中岗位的未覆盖比例不再明显下降（最后两轮降幅
       <0.1pp，决策点 A 的 guideline clarification）。
    """
    required = {
        "round", "new_standard_concepts", "coverage_gain_pp",
        "anchor_uncovered_pp",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError("round metrics 缺列: " + ", ".join(sorted(missing)))
    incremental = metrics[metrics["round"].astype(int) > 0].copy()
    if len(incremental) < 2:
        return False
    last = incremental.sort_values("round").tail(2)
    if int(last["round"].iloc[1]) != int(last["round"].iloc[0]) + 1:
        return False
    if not (last.new_standard_concepts < 5).all():
        return False
    cov = pd.to_numeric(last.coverage_gain_pp, errors="coerce")
    if cov.isna().any() or not (cov < COVERAGE_STOP_PP).all():
        return False
    unc = pd.to_numeric(last.anchor_uncovered_pp, errors="coerce")
    if unc.isna().any():
        return False
    drop = float(unc.iloc[0]) - float(unc.iloc[1])
    return drop < ANCHOR_UNCOVERED_STOP_PP


def build_round_metrics(
    selected_path: Path,
    candidate_audit_path: Path,
    out_path: Path,
    coverage_path: Path | None = None,
) -> Path:
    """从候选审计自动生成停止指标；覆盖率仅作为监测字段。"""
    audit = pd.read_csv(candidate_audit_path)
    required = {"source_round", "decision", "final_grade"}
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(
            "candidate audit 缺 metrics 字段: "
            + ", ".join(sorted(missing))
        )
    selected = pd.read_csv(selected_path)
    if "discovery_round" not in selected.columns:
        raise ValueError("selected sample 缺 discovery_round")
    rounds = sorted(
        {int(x) for x in selected.discovery_round.dropna().astype(int)}
    )
    if not rounds:
        raise ValueError("selected sample 无 discovery_round")
    if rounds[0] != 0 or rounds != list(range(rounds[-1] + 1)):
        raise ValueError(
            f"discovery_round 必须从0连续递增，当前={rounds}"
        )
    accepted = audit[
        (audit.decision.astype(str) == "NEW_CONCEPT")
        & audit.final_grade.isin(["B", "C"])
    ].copy()
    new_counts = (
        accepted.groupby(accepted.source_round.astype(int))
        .size().to_dict()
    )
    coverage: dict[int, float] = {}
    anchor_unc: dict[int, float] = {}
    if coverage_path is not None:
        cov = pd.read_csv(coverage_path)
        if not {"round", "coverage_gain_pp", "anchor_uncovered_pp"} <= set(cov.columns):
            raise ValueError(
                "coverage monitor CSV 必须含 round,coverage_gain_pp,"
                "anchor_uncovered_pp"
            )
        if cov["round"].astype(int).duplicated().any():
            raise ValueError("coverage monitor round 重复")
        coverage = {
            int(r): float(v)
            for r, v in zip(cov["round"], cov["coverage_gain_pp"])
        }
        anchor_unc = {
            int(r): float(v)
            for r, v in zip(cov["round"], cov["anchor_uncovered_pp"])
        }
    rows = [
        {
            "round": r,
            "new_standard_concepts": int(new_counts.get(r, 0)),
            "coverage_gain_pp": coverage.get(r, np.nan),
            "anchor_uncovered_pp": anchor_unc.get(r, np.nan),
        }
        for r in rounds
    ]
    out = pd.DataFrame(rows).sort_values("round")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

def finalize(
    frame_path: Path,
    frame_manifest_path: Path,
    selected_path: Path,
    metrics_path: Path,
    candidate_audit_path: Path,
    governance_path: Path,
) -> Path:
    """只有“抽样饱和 + 候选已接入最终治理表”同时满足才 formal_pass。"""
    df = pd.read_parquet(frame_path)
    validate_frame(df)
    if not frame_manifest_path.exists():
        raise FileNotFoundError(
            f"缺 discovery frame manifest: {frame_manifest_path}"
        )
    frame_manifest = json.loads(
        frame_manifest_path.read_text(encoding="utf-8")
    )
    if frame_manifest.get("status") != "formal_pass":
        raise ValueError("discovery frame manifest 未 formal_pass")
    if frame_manifest.get("frame_sha256") != _sha(frame_path):
        raise ValueError(
            "discovery frame 与 frame manifest 哈希不一致"
        )
    selected = pd.read_csv(selected_path)
    if selected.job_id.duplicated().any():
        raise ValueError("候选发现样本存在重复 job_id")
    metrics = pd.read_csv(metrics_path)
    passed = saturation_pass(metrics)

    # 分层验收（与 2026-09-24 采用的"每年 1 万"口径一致，依据见
    # baseline_sample 的口径决策记录；旧的 year×industry 全单元覆盖校验属于
    # 8/24 裁决口径，在该口径下不可行已作废——比例分层不保证稀疏单元入样，
    # 旧校验会使 formal_pass 永远不可达）：
    # ① 每个出现在 frame 的年份都必须有样本；
    # ② 样本带 sampling_component 时，基准轮每年条数必须精确等于
    #    min(1 万, 该年 frame 行数)。
    frame_years = df.year.astype(int)
    if "year" not in selected.columns:
        raise ValueError("selected sample 缺 year，无法验收分层覆盖")
    sel_years = selected.year.astype(int)
    missing_years = sorted(
        set(frame_years.unique()) - set(sel_years.unique())
    )
    if missing_years:
        raise ValueError(
            "发现样本缺整年份样本（每年 1 万口径）: " + str(missing_years)
        )
    if "sampling_component" in selected.columns:
        baseline = selected[
            selected.sampling_component.astype(str) == "baseline_year"
        ]
        frame_counts = frame_years.value_counts()
        for year, frame_n in frame_counts.items():
            target = min(BASE_PER_YEAR, int(frame_n))
            got = int((baseline.year.astype(int) == int(year)).sum())
            if got != target:
                raise ValueError(
                    f"基准轮 {int(year)} 年样本 {got} 条 != 目标 {target} 条"
                    f"（每年 1 万口径，frame 该年 {int(frame_n)} 行）"
                )
    # §9.2.2.3：各年份和主要行业均已至少进入一个批次。"主要行业"按
    # 决策点 A 定义为该年 frame 岗位数 >=100 的行业（guideline
    # clarification；保底抽样下自然满足，此校验 fail-closed 防回归）。
    if "industry" in selected.columns:
        frame_ind = df.industry.map(_norm_industry)
        sel_ind = selected.industry.map(_norm_industry)
        major_missing = []
        for year in sorted(set(frame_years.unique())):
            in_year = frame_ind[frame_years == year]
            majors = set(in_year.value_counts()[
                lambda s: s >= MAJOR_INDUSTRY_MIN
            ].index)
            covered = set(sel_ind[frame_years == year].unique())
            for ind in sorted(majors - covered):
                major_missing.append(f"{int(year)}×{ind}")
        if major_missing:
            raise ValueError(
                "主要行业（该年 >=100 岗位）未进入任何批次: "
                + ", ".join(major_missing[:10])
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
        raise ValueError("candidate audit 含未知 final_grade")

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
        "sampling_protocol": "guideline_9_1_per_year_10k_saturation_v3",
        "guideline_deviations": [
            {
                "field": "baseline_sampling_caliber",
                "guideline_value": "每年 1 万（指南 §9.1 三处明文）",
                "adopted_value": "每年 1 万（行业保底+比例配额，§9.2.3-4 逐字实现）",
                "reason": (
                    "8/24 裁决的 year×industry 单元口径实测等于全量取走 "
                    "9,168,049 条（11527 单元平均约 1449 行），不可行；"
                    "2026-09-24 复核改回指南口径；2026-09-25 补齐 min(50) "
                    "行业保底与差额补齐（此前未实现）。依据详见 "
                    "discovery_formal.baseline_sample docstring。"
                ),
            },
            {
                "field": "guideline_9_2_2_saturation_conditions",
                "guideline_value": (
                    "四条件：新增<5、覆盖率提升<0.1pp、年份与主要行业入批、"
                    "锚点未覆盖比例不再明显下降"
                ),
                "adopted_value": (
                    "四条件全部实现（2026-09-25 起；此前 8/24 裁决曾将"
                    "覆盖率降级为仅监测）"
                ),
                "reason": (
                    "guideline clarification（决策点 A，用户批准）：\"不再明显"
                    "下降\"量化为最后两轮降幅 <0.1pp；\"主要行业\"定义为该年 "
                    "frame 岗位数 >=100 的行业。覆盖率由 coverage 模块每轮对"
                    "全体去重岗位确定性匹配计算，缺失时 fail-closed。"
                ),
            },
            {
                "field": "guideline_9_2_3_floor_overflow",
                "guideline_value": "指南未覆盖（保底总额超过样本名额的情形）",
                "adopted_value": "按行业岗位数由大到小保留保底并截断到名额",
                "reason": (
                    "行业极多而名额少时 min(50) 保底总额可能溢出；截断规则"
                    "为本项目补充定义，大行业优先。"
                ),
            },
        ],
        "frame_sha256": _sha(frame_path),
        "frame_manifest_sha256": _sha(frame_manifest_path),
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
        "baseline_caliber": "per_year_10k_proportional",
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
    m = sub.add_parser("metrics")
    m.add_argument("--selected", type=Path, required=True)
    m.add_argument("--candidate-audit", type=Path, required=True)
    m.add_argument("--coverage", type=Path)
    m.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--frame", type=Path, required=True)
    f.add_argument("--frame-manifest", type=Path, required=True)
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
        new_round = incremental_round(
            df, set(selected.job_id), args.round
        )
        out = pd.concat(
            [selected, new_round], ignore_index=True, sort=False
        )
        if out.job_id.duplicated().any():
            raise RuntimeError("累计 discovery selected 出现重复 job_id")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
    elif args.cmd == "metrics":
        build_round_metrics(
            args.selected, args.candidate_audit,
            args.out, args.coverage
        )
    else:
        finalize(args.frame, args.frame_manifest, args.selected, args.metrics,
                 args.candidate_audit, args.governance)


if __name__ == "__main__":
    main()
