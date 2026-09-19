"""panel_v2 M4-a：自动化质量门（指南 §17.3–§17.6）与统计报告。

§17.6 十项阻断检查任一失败 → raise（停止发布）；警告项与 §17.4/§17.5
统计进入 quality_control_report.md（警告不阻断）。

**警告六类覆盖披露（2026-09-09 审计修订）**——指南要求六类警告全部入报并
附年份/行业/岗位/技能明细表，本实现覆盖情况如实如下：
① 低频 0/1 原始权重（已实现，rare01_skills）；② 锚点口径差异（已实现，
exposure_rate_by_anchor）；③ LLM/TRANS 增量（已实现，anchor_jobs）；
④ 年份无技能比例（以 zero_skill_by_year 等价实现，行业维度缺失）；
⑤ 歧义锚点×行业集中度（**未实现**：master 无行业字段，需数据源增强）；
⑥ 年度断点（**未实现**：判据未定义，留交接方澄清）。
明细表（逐岗位/逐技能清单）未生成——阻断级检查均在全量上验证，警告触发时
的量级可由 quality_stats.json 反查。⑤⑥为正式偏离，随全国重跑申报。

"原文跨度回填"（§17.6.3）在 v2a 记为 waived：词典匹配为确定性子串命中，
证据可由 match 文本 + 词表重算（无 LLM 生成词），非缺失场景。
重跑一致项（§17.6.10）：同一发布目录已经存在 quality_stats.json 时，
本次运行必须与上一轮保持相同行数和关键统计量；漂移属于阻断错误。

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
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from .anchors import ANCHOR_RULES_VERSION

logger = logging.getLogger("ai_penetration.panel_v2.quality")

QUALITY_STATS_SCHEMA_VERSION = 1


def _sha_stats(df: pd.DataFrame, cols: list[str]) -> str:
    """对指定关键列做稳定的全行校验和。

    使用 pandas 的确定性逐行哈希，避免 object dtype 的 ``ndarray.tobytes()``
    把 Python 对象地址写入摘要而导致跨进程不稳定；同时把列名和 dtype 纳入摘要。
    """
    use = [c for c in cols if c in df.columns]
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update("|".join(f"{c}:{df[c].dtype}" for c in use).encode())
    if use:
        hashed = pd.util.hash_pandas_object(df[use], index=False).to_numpy(np.uint64)
        h.update(np.ascontiguousarray(hashed).tobytes())
    return h.hexdigest()[:16]


def _persist_quality_stats(stats_path: Path, stats: dict, fails: list[str]) -> Path:
    """质量门通过后才推进 canonical baseline；失败运行只写旁路快照。"""
    payload = dict(stats)
    payload["_schema_version"] = QUALITY_STATS_SCHEMA_VERSION
    encoded = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    if fails:
        failed = stats_path.with_name(
            f"{stats_path.stem}.failed_{datetime.now():%Y%m%d_%H%M%S}{stats_path.suffix}"
        )
        failed.write_text(encoded, encoding="utf-8")
        return failed
    tmp = stats_path.with_name(f".{stats_path.name}.tmp")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(stats_path)
    return stats_path


def _rerun_drift(prev: dict, stats: dict) -> list[str]:
    """返回 §17.6.10 同目录重跑发生漂移的关键统计项。"""
    if not prev:
        return []
    keys = (
        "n_jobs",
        "n_pairs",
        "checksum_flags",
        "checksum_counts",
        "main_annual_raw_005_rate",
        "main_annual_raw_015_rate",
        "zero_skill_rate",
    )
    return [k for k in keys if k in prev and k in stats and prev[k] != stats[k]]


def _check_span_evidence(path: Path) -> tuple[int, set[str]]:
    """流式校验 §11/§17.6 的岗位技能证据字段与跨度内部一致性。"""
    required = {
        "surface_form", "start", "end", "mention_count",
        "match_method", "ambiguity_flag",
    }
    pf = pq.ParquetFile(path)
    missing = required - set(pf.schema_arrow.names)
    if missing:
        return 0, missing
    bad = 0
    for rb in pf.iter_batches(
        batch_size=1_000_000,
        columns=["surface_form", "start", "end", "mention_count"],
    ):
        tbl = rb.to_pydict()
        surfaces = tbl["surface_form"]
        starts = tbl["start"]
        ends = tbl["end"]
        mentions = tbl["mention_count"]
        for surface, start, end, mention in zip(
            surfaces, starts, ends, mentions
        ):
            if (
                surface is None
                or start is None
                or end is None
                or mention is None
                or start < 0
                or end <= start
                or mention < 1
                or len(surface) != end - start
            ):
                bad += 1
    return bad, set()


def gate_checks(rel: Path) -> tuple[list[str], dict]:
    """§17.6 阻断十项。返回 (失败列表, 统计量字典)。"""
    fails: list[str] = []
    stats: dict = {}
    flags = pq.read_table(rel / "job_anchor_flag.parquet").to_pandas()
    long_path = rel / "job_skill_long.parquet"
    longs = pq.read_table(
        long_path, columns=["job_id", "skill_code"]
    ).to_pandas()
    counts = pq.read_table(rel / "skill_ai_counts.parquet").to_pandas()
    rel_df = pq.read_table(rel / "skill_ai_relevance.parquet").to_pandas()
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

    # 2 (job, skill) 唯一：int64 复合键全局查重（4 亿规模内存可行）
    ck = (longs.job_id.to_numpy(np.int64) << 32) | longs.skill_code.to_numpy(np.int64)
    if np.unique(ck).size != ck.size:
        fails.append("(job_id, skill_code) 不唯一")
    del ck

    # 3 原文跨度证据：扫描时已强制 surface == match_text[start:end]；
    # 发布质量门再独立检查 schema 与跨度内部一致性，禁止 waived。
    bad_span, missing_span = _check_span_evidence(long_path)
    if missing_span:
        fails.append(
            "§17.6.3 job_skill_long 缺证据字段: "
            + ", ".join(sorted(missing_span))
        )
        stats["span_backfill"] = "missing_columns"
    elif bad_span:
        fails.append(f"§17.6.3 原文跨度/mention_count 无效（{bad_span} 行）")
        stats["span_backfill"] = f"failed:{bad_span}"
    else:
        stats["span_backfill"] = "scan_verified_and_release_rechecked"

    # 4 分子≤分母
    if (counts.n_ai_cooccur > counts.n_skill).any():
        fails.append("计数分子>分母")

    # 5 权重 [0,1]（score 越界在 dataset 分块检查，见 8）
    for col in ("ai_rate_raw", "ai_rate_smoothed"):
        v = rel_df[col]
        if ((v < 0) | (v > 1)).any():
            fails.append(f"{col} 越界")

    # 6/7 年度=pooled；roll3=年度窗口和 + 两年/三年窗口
    for ver in ("main", "cn_paper", "babina"):
        ann = counts[(counts.anchor_version == ver) & (counts.window_type == "annual")]
        pool = counts[(counts.anchor_version == ver) & (counts.window_type == "pooled")]
        a = ann.groupby("skill_code")[["n_skill", "n_ai_cooccur"]].sum()
        p = pool.groupby("skill_code")[["n_skill", "n_ai_cooccur"]].sum()
        j = a.join(p, lsuffix="_a", rsuffix="_p", how="outer").fillna(0)
        if not np.allclose(j.n_skill_a, j.n_skill_p) \
                or not np.allclose(j.n_ai_cooccur_a, j.n_ai_cooccur_p):
            fails.append(f"pooled 加总不守恒({ver})")

        years = np.sort(ann.year.unique())
        n_y = len(years)
        nmat = np.zeros((int(counts.skill_code.max()) + 1, n_y), np.int64)
        cm = np.zeros_like(nmat)
        yix = np.searchsorted(years, ann.year.to_numpy())
        nmat[ann.skill_code.to_numpy(), yix] = ann.n_skill.to_numpy()
        cm[ann.skill_code.to_numpy(), yix] = ann.n_ai_cooccur.to_numpy()
        r3 = counts[(counts.anchor_version == ver)
                    & (counts.window_type == "roll3_centered")]
        for i, y in enumerate(years):
            lo, hi = max(0, i - 1), min(n_y - 1, i + 1)
            wn = nmat[:, lo:hi + 1].sum(axis=1)
            wc = cm[:, lo:hi + 1].sum(axis=1)
            exp = r3[r3.year == y]
            got_n = np.zeros(nmat.shape[0], np.int64)
            got_c = np.zeros_like(got_n)
            got_n[exp.skill_code.to_numpy()] = exp.n_skill.to_numpy()
            got_c[exp.skill_code.to_numpy()] = exp.n_ai_cooccur.to_numpy()
            if not (np.array_equal(wn, got_n) and np.array_equal(wc, got_c)):
                fails.append(f"roll3 年度窗口和校验失败({ver}/{y})")
                break
        wsize = (r3.window_end - r3.window_start + 1).value_counts()
        if set(wsize.index) - {2, 3}:
            fails.append("roll3 窗口尺寸异常")

    # 8 coverage 实测
    cov_min, score_lo, score_hi, n_cov_bad = 1.0, 0.0, 1.0, 0
    ds = pq.ParquetDataset(rel / "job_ai_score")
    for frag in ds.fragments:
        df = frag.to_table(columns=[
            "matched_skill_count", "weighted_skill_count",
            "score_skill_coverage", "ai_score"]).to_pandas()
        elig = df[df.matched_skill_count > 0]
        if elig.empty:
            continue
        n_cov_bad += int((elig.weighted_skill_count != elig.matched_skill_count).sum())
        cov_min = min(cov_min, float(elig.score_skill_coverage.min()))
        s = df.ai_score.dropna()
        if len(s):
            score_lo = min(score_lo, float(s.min()))
            score_hi = max(score_hi, float(s.max()))
    if n_cov_bad:
        fails.append(f"§15.3 weighted != matched（{n_cov_bad} 岗位）")
    if cov_min < 1:
        fails.append(f"coverage<1（min={cov_min}）")
    if score_lo < 0 or score_hi > 1:
        fails.append(f"ai_score 越界 [{score_lo},{score_hi}]")
    stats["coverage_min"] = cov_min

    # 9 阈值单调
    for col in [c for c in cls.columns if c.startswith("aijob_")
                and c.endswith("_015")]:
        base = col[:-4]
        monotonic = (
            int((cls[f"{base}_015"] <= cls[f"{base}_010"]).sum()) == len(cls)
            and int((cls[f"{base}_010"] <= cls[f"{base}_005"]).sum()) == len(cls)
        )
        if not monotonic:
            fails.append(f"阈值单调性破坏: {base}")

    # 10 重跑一致性：稳定关键统计落盘，run() 与同目录上一轮比较并阻断漂移
    stats["anchor_rules_version"] = ANCHOR_RULES_VERSION
    flags_sorted = flags.sort_values("job_id")
    stats["checksum_flags"] = _sha_stats(
        flags_sorted,
        ["job_id", "year", "anchor_main", "anchor_cn_paper",
         "anchor_babina", "groups_main_bits",
         "matched_anchor_groups_main", "matched_anchor_terms_main"],
    )
    counts_sorted = counts.sort_values(
        ["skill_code", "anchor_version", "window_type", "year"]
    )
    stats["checksum_counts"] = _sha_stats(
        counts_sorted,
        ["skill_code", "anchor_version", "window_type", "year",
         "window_start", "window_end", "n_skill", "n_ai_cooccur"],
    )
    stats["main_annual_raw_005_rate"] = float(
        cls["aijob_main_annual_raw_005"].mean()
    )
    stats["main_annual_raw_015_rate"] = float(
        cls["aijob_main_annual_raw_015"].mean()
    )
    stats["zero_skill_rate"] = float(cls["zero_skill_override"].mean())

    # §16.4 版本转换矩阵
    stats["transition_matrix"] = {}
    for thr, fname in (("005", "exposure"), ("015", "primary")):
        trio = pd.DataFrame({
            "annual": cls[f"aijob_main_annual_raw_{thr}"],
            "pooled": cls[f"aijob_main_pooled_raw_{thr}"],
            "roll3": cls[f"aijob_main_roll3_centered_raw_{thr}"],
        })
        trans = {}
        for a, b in (("annual", "pooled"), ("annual", "roll3"),
                     ("pooled", "roll3")):
            trans[f"{a}x{b}"] = pd.crosstab(trio[a], trio[b]).to_dict()
        (rel / f"transition_matrix_main_raw{thr}.json").write_text(
            json.dumps(trans, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        stats[f"transition_matrix_{fname}"] = trans
    return fails, stats


def warning_checks(rel: Path) -> tuple[list[str], dict]:
    """§17.6 警告六类 + §17.4/§17.5 关键统计（进报告）。"""
    warns: list[str] = []
    info: dict = {}
    flags = pq.read_table(rel / "job_anchor_flag.parquet").to_pandas()
    longs = pq.read_table(
        rel / "job_skill_long.parquet", columns=["job_id"]
    ).to_pandas()
    cls = pq.read_table(rel / "job_ai_classification.parquet").to_pandas()
    rel_df = pq.read_table(rel / "skill_ai_relevance.parquet").to_pandas()

    info["zero_skill_by_year"] = (
        cls.groupby("year").zero_skill_override.mean().round(4).to_dict()
    )
    raw_s = pq.read_table(
        rel / "job_ai_score" / "main_annual_raw.parquet",
        columns=["job_id", "ai_score"],
    )["ai_score"].to_pandas()
    sm_s = pq.read_table(
        rel / "job_ai_score" / "main_annual_smoothed.parquet",
        columns=["ai_score"],
    )["ai_score"].to_pandas()
    info["corr_raw_smoothed"] = round(float(raw_s.corr(sm_s)), 4)

    rates = {}
    for ver in ("main", "cn_paper", "babina"):
        col = f"aijob_{ver}_annual_raw_005"
        if col in cls.columns:
            rates[ver] = round(float(cls[col].mean()), 5)
    info["exposure_rate_by_anchor"] = rates
    if rates:
        lo, hi = min(rates.values()), max(rates.values())
        if lo > 0 and (hi - lo) / lo > 1.0:
            warns.append(f"锚点口径 AI 率差异过大 main/cn/babina={rates}")

    rare01 = rel_df[
        (rel_df.window_type == "annual")
        & (rel_df.n_skill < 10)
        & ((rel_df.ai_rate_raw < 1e-9)
           | ((1 - rel_df.ai_rate_raw).abs() < 1e-9))
    ].skill_code.nunique()
    info["rare01_skills"] = int(rare01)
    if int(rare01) > rel_df.skill_code.nunique() * 0.5:
        warns.append(f"低频 0/1 原始权重技能占比 {rare01} 偏高")

    inc = flags[(flags.groups_main_bits & 96) > 0]
    info["llm_transformer_anchor_jobs"] = len(inc)
    info["year_dist_jobs"] = flags.groupby("year").size().to_dict()
    info["pair_dist"] = longs.groupby("job_id").size().describe().round(2).to_dict()
    info["primary_015_by_year"] = cls.groupby(
        "year")["aijob_main_annual_raw_015"].mean().round(5).to_dict()
    info["exposure_005_by_year"] = cls.groupby(
        "year")["aijob_main_annual_raw_005"].mean().round(5).to_dict()
    return warns, info


def run(rel: Path | None = None) -> None:
    """质量门全流程（rel 默认 release/panel_v2；v2d 传 alt 目录）。"""
    paths = get_project_paths()
    if rel is None:
        rel = paths.output_dir / "release" / "panel_v2"
    fails, stats = gate_checks(rel)
    warns, info = warning_checks(rel)
    stats_path = rel / "quality_stats.json"
    prev = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.exists() else {}
    )
    drift = _rerun_drift(prev, stats)
    if drift:
        fails.append(
            "§17.6.10 同一发布目录重跑关键统计漂移: " + ", ".join(drift)
        )

    report = [
        "# panel_v2 质量检查报告",
        "",
        f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 岗位数: {stats.get('n_jobs'):,} / 技能对: {stats.get('n_pairs'):,}",
        "",
        "## §17.6 阻断检查",
        "",
    ]
    report += (
        [f"- ❌ {f}" for f in fails]
        or ["- ✅ 十项全部通过（含跨度回填 waived 声明）"]
    )
    report += ["", "## 警告与披露", ""]
    report += ([f"- ⚠️ {w}" for w in warns] or ["- （无警告）"])
    report += [
        "", "## §17.4/§17.5 统计", "", "```json",
        json.dumps(
            {**info, **{k: v for k, v in stats.items()}},
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        "```",
    ]
    (rel / "quality_control_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    persisted = _persist_quality_stats(stats_path, stats, fails)
    if fails:
        logger.error(
            "质量门失败：canonical baseline 未更新；本次统计旁路保存到 %s",
            persisted,
        )
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
