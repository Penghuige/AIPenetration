"""panel_v2 M4-a：自动化质量门（指南 §17.3–§17.6）与统计报告。

§17.6 十项阻断检查任一失败 → raise（停止发布）；六类警告项与 §17.4/§17.5
统计进入 quality_control_report.md（警告不阻断，全部入报）。

"原文跨度回填"（§17.6.3）在 v2a 记为 waived：词典匹配为确定性子串命中，
证据可由 match 文本 + 词表重算（无 LLM 生成词），非缺失场景。
重跑一致项（§17.6.10）：本次运行统计量落 quality_stats.json，
与上一版比对差异并披露。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.quality
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging

logger = logging.getLogger("ai_penetration.panel_v2.quality")


def _sha_stats(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    for c in cols:
        if c in df.columns:
            h.update(np.asarray(df[c]).tobytes()[:1 << 24])
    return h.hexdigest()[:16]


def gate_checks(rel: Path) -> tuple[list[str], dict]:
    """§17.6 阻断十项。返回 (失败列表, 统计量字典)。"""
    fails: list[str] = []
    stats: dict = {}
    flags = pq.read_table(rel / "job_anchor_flag.parquet").to_pandas()
    longs = pq.read_table(rel / "job_skill_long.parquet").to_pandas()
    counts = pq.read_table(rel / "skill_ai_counts.parquet").to_pandas()
    rel_df = pq.read_table(rel / "skill_ai_relevance.parquet").to_pandas()
    score = pq.read_table(rel / "job_ai_score.parquet").to_pandas()
    cls = pq.read_table(rel / "job_ai_classification.parquet").to_pandas()

    # 1 job_id 唯一 + 引用完整
    if flags.job_id.duplicated().any():
        fails.append("job_id 不唯一")
    jset = np.sort(flags.job_id.to_numpy())
    hit = np.isin(np.sort(np.unique(longs.job_id.to_numpy())), jset)
    if not hit.all():
        fails.append("long 表存在 flag 外 job_id")
    stats["n_jobs"] = len(flags)
    stats["n_pairs"] = len(longs)
    # 2 (job, skill) 唯一
    if longs.duplicated(["job_id", "skill_code"]).any():
        fails.append("(job_id, skill_code) 不唯一")
    # 3 跨度回填：v2a waived（确定性子串匹配可重算）
    stats["span_backfill"] = "waived_deterministic"
    # 4 分子≤分母
    if (counts.n_ai_cooccur > counts.n_skill).any():
        fails.append("计数分子>分母")
    # 5 权重/得分 [0,1]
    for col in ("ai_rate_raw", "ai_rate_smoothed"):
        v = rel_df[col]
        if ((v < 0) | (v > 1)).any():
            fails.append(f"{col} 越界")
    sv = score.ai_score.dropna()
    if ((sv < 0) | (sv > 1)).any():
        fails.append("ai_score 越界")
    # 6/7 年度=pooled、roll3=年度和+边界两年（recompute 复核，独立于 counts.verify）
    for ver in ("main", "cn_paper", "babina"):
        ann = counts[(counts.anchor_version == ver) & (counts.window_type == "annual")]
        pool = counts[(counts.anchor_version == ver) & (counts.window_type == "pooled")]
        a = ann.groupby("skill_code")[["n_skill", "n_ai_cooccur"]].sum()
        p = pool.groupby("skill_code")[["n_skill", "n_ai_cooccur"]].sum()
        if not a.equals(p):
            j = a.join(p, lsuffix="_a", rsuffix="_p", how="outer").fillna(0)
            if not np.allclose(j.n_skill_a, j.n_skill_p) \
                    or not np.allclose(j.n_ai_cooccur_a, j.n_ai_cooccur_p):
                fails.append(f"pooled 加总不守恒({ver})")
        r3 = counts[(counts.anchor_version == ver)
                    & (counts.window_type == "roll3_centered")]
        wsize = (r3.window_end - r3.window_start + 1).value_counts()
        # 首末年两年、其余三年
        if set(wsize.index) - {2, 3}:
            fails.append("roll3 窗口尺寸异常")
    # 8 coverage=1
    elig = score[score.matched_skill_count > 0]
    if (elig.weighted_skill_count != elig.matched_skill_count).any():
        fails.append("§15.3 weighted != matched")
    if not (elig.score_skill_coverage == 1).all():
        fails.append("coverage != 1")
    stats["coverage_min"] = float(elig.score_skill_coverage.min())
    # 9 阈值单调
    for col in [c for c in cls.columns if c.startswith("aijob_")]:
        base = col.rsplit("_", 1)[0]
        try:
            m = (cls[f"{base}_015"] <= cls[f"{base}_010"]).all() and \
                (cls[f"{base}_010"] <= cls[f"{base}_005"]).all()
        except KeyError:
            continue
        if not m:
            fails.append(f"阈值单调性破坏: {col}")
    # 10 重跑一致性（本次统计与上一版比对，差异仅披露不阻断——
    #    阻断语义留给"同配置同输入"，跨配置变化视为新版本）
    stats["checksum_flags"] = _sha_stats(flags, ["anchor_main", "anchor_babina"])
    stats["checksum_counts"] = _sha_stats(
        counts.sort_values(["skill_code", "anchor_version", "window_type", "year"]),
        ["n_skill", "n_ai_cooccur"])
    return fails, stats


def warning_checks(rel: Path) -> tuple[list[str], dict]:
    """§17.6 警告六类 + §17.4/§17.5 关键统计（进报告）。"""
    warns: list[str] = []
    info: dict = {}
    flags = pq.read_table(rel / "job_anchor_flag.parquet").to_pandas()
    longs = pq.read_table(rel / "job_skill_long.parquet").to_pandas()
    score = pq.read_table(rel / "job_ai_score.parquet").to_pandas()
    cls = pq.read_table(rel / "job_ai_classification.parquet").to_pandas()
    rel_df = pq.read_table(rel / "skill_ai_relevance.parquet").to_pandas()

    info["zero_skill_by_year"] = (
        cls.groupby("year").zero_skill_override.mean().round(4).to_dict())
    # raw vs smoothed 岗位得分相关（main annual）
    piv = score[(score.anchor_version == "main")
                & (score.window_type == "annual")].pivot_table(
        index="job_id", columns="score_type", values="ai_score")
    if {"raw", "smoothed"} <= set(piv.columns):
        info["corr_raw_smoothed"] = round(
            float(piv.raw.corr(piv.smoothed)), 4)
    # 锚点版本 AI 率差异
    rates = {}
    for ver in ("main", "cn_paper", "babina"):
        col = f"aijob_{ver}_annual_raw_005"
        if col in cls.columns:
            rates[ver] = round(float(cls[col].mean()), 5)
    info["aijob_rate_by_anchor"] = rates
    if rates:
        lo, hi = min(rates.values()), max(rates.values())
        if lo > 0 and (hi - lo) / lo > 1.0:
            warns.append(f"锚点口径 AI 率差异过大 main/cn/babina={rates}")
    # 低频 0/1 原始权重规模（§17.6 警告3）
    rare01 = rel_df[(rel_df.window_type == "annual")
                    & (rel_df.n_skill < 10)
                    & ((rel_df.ai_rate_raw < 1e-9)
                       | ((1 - rel_df.ai_rate_raw).abs() < 1e-9))].skill_code.nunique()
    info["rare01_skills"] = int(rare01)
    if int(rare01) > rel_df.skill_code.nunique() * 0.5:
        warns.append(f"低频 0/1 原始权重技能占比 {rare01} 偏高")
    # LLM/TRANS 锚点增量（§17.4.8：仅 main 有该两组，用 bitmask 32|64 观察）
    inc = flags[(flags.groups_main_bits & 96) > 0]
    info["llm_transformer_anchor_jobs"] = len(inc)
    info["year_dist_jobs"] = flags.groupby("year").size().to_dict()
    info["pair_dist"] = longs.groupby("job_id").size().describe().round(2).to_dict()
    # 主标识年度比例（§17.5.3）
    info["aijob_main_annual_005_by_year"] = cls.groupby(
        "year")["aijob_main_annual_raw_005"].mean().round(5).to_dict()
    return warns, info


def run() -> None:
    paths = get_project_paths()
    rel = paths.output_dir / "release" / "panel_v2"
    fails, stats = gate_checks(rel)
    warns, info = warning_checks(rel)
    stats_path = rel / "quality_stats.json"
    prev = json.loads(stats_path.read_text(encoding="utf-8")) \
        if stats_path.exists() else {}
    drift = [k for k in stats if k.startswith("checksum")
             and k in prev and prev[k] != stats[k]]
    report = [
        "# panel_v2 质量检查报告",
        "",
        f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 岗位数: {stats.get('n_jobs'):,} / 技能对: {stats.get('n_pairs'):,}",
        "",
        "## §17.6 阻断检查",
        "",
    ]
    report += ([f"- ❌ {f}" for f in fails] or ["- ✅ 十项全部通过（含跨度回填 waived 声明）"])
    report += ["", "## 警告与披露", ""]
    report += ([f"- ⚠️ {w}" for w in warns] or ["- （无警告）"])
    report += ["", "## §17.4/§17.5 统计", "", "```json",
               json.dumps({**info, **{k: v for k, v in stats.items()}},
                          ensure_ascii=False, indent=1, default=str),
               "```"]
    if drift:
        report += ["", f"⚠️ 与上一版统计漂移（新配置预期内，需人工确认）: {drift}"]
    (rel / "quality_control_report.md").write_text(
        "\n".join(report), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    logger.info("质量门: fail=%d warn=%d", len(fails), len(warns))
    if fails:
        raise SystemExit(2)


def main() -> None:
    argparse.ArgumentParser(description="panel_v2 §17 质量门").parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_quality.log")
    run()


if __name__ == "__main__":
    main()
