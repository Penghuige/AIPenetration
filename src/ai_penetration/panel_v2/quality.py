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

"原文跨度回填"（§17.6.3）为阻断项：正式 job_skill_long 必须持久化
surface/start/end 等证据，并由 span_verified=1 证明扫描时可直接回填。
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
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from .anchors import ANCHOR_RULES_VERSION

logger = logging.getLogger("ai_penetration.panel_v2.quality")

CHECKSUM_SCHEMA_VERSION = 2


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


def _rerun_drift(prev: dict, stats: dict) -> list[str]:
    """返回 §17.6.10 真正的数据/结果漂移项。

    checksum schema 升级本身不是数据漂移：若 schema 不同，只比较与算法无关
    的行数和关键比例；这些相同则允许成功迁移到新 schema。
    """
    if not prev:
        return []
    stable_keys = (
        "n_jobs",
        "n_pairs",
        "main_annual_raw_005_rate",
        "main_annual_raw_015_rate",
        "zero_skill_rate",
    )
    stable_drift = [
        k for k in stable_keys
        if k in prev and k in stats and prev[k] != stats[k]
    ]
    if prev.get("checksum_schema_version") != stats.get("checksum_schema_version"):
        return stable_drift
    checksum_keys = ("checksum_flags", "checksum_counts")
    return stable_drift + [
        k for k in checksum_keys
        if k in prev and k in stats and prev[k] != stats[k]
    ]


def gate_checks(rel: Path) -> tuple[list[str], dict]:
    """§17.6 阻断十项。返回 (失败列表, 统计量字典)。"""
    fails: list[str] = []
    stats: dict = {}
    flags = pq.read_table(rel / "job_anchor_flag.parquet").to_pandas()
    longs = pq.read_table(rel / "job_skill_long.parquet").to_pandas()
    counts = pq.read_table(rel / "skill_ai_counts.parquet").to_pandas()
    rel_df = pq.read_table(rel / "skill_ai_relevance.parquet").to_pandas()
    cls = pq.read_table(rel / "job_ai_classification.parquet").to_pandas()
    text_path = rel / "job_text_clean.parquet"
    text_rows = None
    if not text_path.exists():
        fails.append("缺少 §6.4 job_text_clean.parquet")
    else:
        pf_text = pq.ParquetFile(text_path)
        text_rows = int(pf_text.metadata.num_rows)
        text_cols = set(pf_text.schema_arrow.names)
        required_text = {
            "job_id", "job_id_sha256", "job_id_raw", "source_platform", "year",
            "job_description_raw", "job_description_clean",
            "job_description_match", "text_hash",
        }
        missing_text = required_text - text_cols
        if missing_text:
            fails.append(
                "job_text_clean 缺三态文本字段: "
                + ", ".join(sorted(missing_text))
            )
        if {"job_id", "job_id_sha256"} <= text_cols:
            # 只读身份列；raw/clean/match 大文本无需进入 Pandas 内存。
            ids = pq.read_table(
                text_path, columns=["job_id", "job_id_sha256"]
            ).to_pandas()
            if ids.job_id.duplicated().any():
                fails.append("job_text_clean job_id 不唯一")
            sha = ids.job_id_sha256.astype(str)
            if sha.duplicated().any():
                fails.append("完整 SHA256 stable job_id 不唯一")
            if not sha.str.fullmatch(r"[0-9a-f]{64}").all():
                fails.append("job_id_sha256 格式非法")

    # §18 正式发布词典必须能独立解释正式长表。
    concept_path = rel / "skill_concept_v1.parquet"
    alias_path = rel / "skill_alias_v1.parquet"
    source_path = rel / "source_skill_record_v1.parquet"
    if not concept_path.exists() or not alias_path.exists():
        fails.append("缺少 §18 正式 skill_concept/skill_alias 发布件")
    else:
        concepts = pq.read_table(concept_path).to_pandas()
        aliases = pq.read_table(alias_path).to_pandas()
        required_concept_cols = {
            "skill_id", "canonical_zh", "canonical_en", "skill_type",
            "skill_category", "confidence_tier", "dictionary_version",
            "valid_from", "valid_to",
        }
        missing_concept_cols = sorted(required_concept_cols - set(concepts.columns))
        if missing_concept_cols:
            fails.append(
                "正式概念词典缺 §7.2 字段: " + ", ".join(missing_concept_cols)
            )
        required_alias_cols = {
            "alias_id", "skill_id", "alias", "alias_normalized", "language",
            "source", "matching_rule", "ambiguity_flag", "confidence_tier",
            "dictionary_version",
        }
        missing_alias_cols = sorted(required_alias_cols - set(aliases.columns))
        if missing_alias_cols:
            fails.append(
                "正式别名词典缺 §7.3 字段: " + ", ".join(missing_alias_cols)
            )
        concept_ids = set(concepts.skill_id.astype(str))
        if concepts.skill_id.astype(str).duplicated().any():
            fails.append("正式词典 skill_id 不唯一")
        if aliases.alias_id.astype(str).duplicated().any():
            fails.append("正式别名 alias_id 不唯一")
        dangling_alias = set(aliases.skill_id.astype(str)) - concept_ids
        if dangling_alias:
            fails.append(f"正式别名存在无概念引用（{len(dangling_alias)} skill_id）")
        if "skill_id" in longs.columns:
            dangling_long = set(longs.skill_id.astype(str)) - concept_ids
            if dangling_long:
                fails.append(
                    f"job_skill_long 含正式词典外 skill_id（{len(dangling_long)}）"
                )
            if longs[["job_id", "skill_id"]].duplicated().any():
                fails.append("(job_id, skill_id) 不唯一")

        if not source_path.exists():
            fails.append("缺少 §7.4.1 source_skill_record_v1.parquet")
        else:
            source_records = pq.read_table(source_path).to_pandas()
            required_source_cols = {
                "source_name", "source_version", "source_skill_id",
                "source_label", "source_description", "source_category",
                "internal_skill_id", "mapping_type", "mapping_evidence",
            }
            missing_source_cols = sorted(
                required_source_cols - set(source_records.columns)
            )
            if missing_source_cols:
                fails.append(
                    "source_skill_record 缺字段: "
                    + ", ".join(missing_source_cols)
                )
            dangling_source = (
                set(source_records.internal_skill_id.astype(str)) - concept_ids
                if "internal_skill_id" in source_records.columns else set()
            )
            if dangling_source:
                fails.append(
                    f"source_skill_record 存在正式概念外引用（{len(dangling_source)}）"
                )

    # 1 job_id 唯一 + 引用完整
    anchor_detail_cols = {"matched_anchor_groups_main", "matched_anchor_terms_main"}
    missing_anchor_detail = anchor_detail_cols - set(flags.columns)
    if missing_anchor_detail:
        fails.append("job_anchor_flag 缺命中明细: " + ", ".join(sorted(missing_anchor_detail)))
    if flags.job_id.duplicated().any():
        fails.append("job_id 不唯一")
    jset = np.sort(flags.job_id.to_numpy())
    hit = np.isin(np.sort(np.unique(longs.job_id.to_numpy())), jset)
    if not hit.all():
        fails.append("long 表存在 flag 外 job_id")
    stats["n_jobs"] = len(flags)
    stats["n_pairs"] = len(longs)
    if text_rows is not None and text_rows != len(flags):
        fails.append(
            f"job_text_clean 行数 {text_rows} != job_anchor_flag {len(flags)}")

    # 2 (job, skill) 唯一：job_id 为稳定 63-bit 哈希，禁止位移打包
    # （左移会 int64 溢出并制造伪碰撞/漏碰撞）。
    if longs.duplicated(["job_id", "skill_code"]).any():
        fails.append("(job_id, skill_code) 不唯一")

    # 3 原文跨度回填：指南 §11.1.4/§17.6 为阻断项，不允许 waiver。
    evidence_cols = {
        "skill_id", "surface_form", "start", "end", "mention_count",
        "match_method", "ambiguity_flag", "confidence_tier",
        "dictionary_version", "span_verified",
        "covered_candidate_count", "covered_candidates",
    }
    missing_evidence = sorted(evidence_cols - set(longs.columns))
    if missing_evidence:
        fails.append("job_skill_long 缺少交接要求证据字段: "
                     + ", ".join(missing_evidence))
        stats["span_backfill"] = "failed_missing_columns"
    else:
        bad_span = (
            (longs["start"] < 0)
            | (longs["end"] <= longs["start"])
            | (longs["mention_count"] < 1)
            | (longs["covered_candidate_count"] < 0)
            | (longs["span_verified"] != 1)
            | (longs["surface_form"].astype(str).str.len()
               != (longs["end"] - longs["start"]))
        )
        if bad_span.any():
            fails.append(f"原文跨度回填失败（{int(bad_span.sum())} 行）")
            stats["span_backfill"] = "failed"
        else:
            stats["span_backfill"] = "verified"
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
    stats["checksum_schema_version"] = CHECKSUM_SCHEMA_VERSION
    flags_sorted = flags.sort_values("job_id")
    stats["checksum_flags"] = _sha_stats(
        flags_sorted,
        ["job_id", "year", "anchor_main", "anchor_cn_paper",
         "anchor_babina", "groups_main_bits"],
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
    longs = pq.read_table(rel / "job_skill_long.parquet").to_pandas()
    cls = pq.read_table(rel / "job_ai_classification.parquet").to_pandas()
    rel_df = pq.read_table(rel / "skill_ai_relevance.parquet").to_pandas()

    zero_by_year = (
        cls.groupby("year", as_index=False)
        .agg(n_jobs=("job_id", "size"),
             zero_skill_rate=("zero_skill_override", "mean"))
    )
    zero_by_year.to_csv(
        rel / "quality_detail_zero_skill_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    info["zero_skill_by_year"] = dict(zip(
        zero_by_year.year.astype(int),
        zero_by_year.zero_skill_rate.round(4),
    ))
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
    pd.DataFrame(
        [{"anchor_version": k, "main_005_rate": v} for k, v in rates.items()]
    ).to_csv(
        rel / "quality_detail_anchor_rates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if rates:
        lo, hi = min(rates.values()), max(rates.values())
        if lo > 0 and (hi - lo) / lo > 1.0:
            warns.append(f"锚点口径 AI 率差异过大 main/cn/babina={rates}")

    rare01_rows = rel_df[
        (rel_df.window_type == "annual")
        & (rel_df.n_skill < 10)
        & ((rel_df.ai_rate_raw < 1e-9)
           | ((1 - rel_df.ai_rate_raw).abs() < 1e-9))
    ].copy()
    rare01_rows.to_csv(
        rel / "quality_detail_rare01_skills.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rare01 = rare01_rows.skill_code.nunique()
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

    if "confidence_tier" in longs.columns:
        tier_detail = (
            longs.groupby(["year", "confidence_tier"], as_index=False)
            .size().rename(columns={"size": "matched_pairs"})
        )
        tier_detail.to_csv(
            rel / "quality_detail_tier_contribution.csv",
            index=False,
            encoding="utf-8-sig",
        )
        info["tier_pair_share"] = (
            longs.confidence_tier.value_counts(normalize=True).round(4).to_dict()
        )
    else:
        warns.append("§17 警告明细不完整：job_skill_long 缺 confidence_tier，无法输出 A/B/C 构成")

    # 指南没有给这两类 warning 的自动阈值；不得擅自伪造判据。
    warns.append(
        "§17 待人工判据：歧义锚点×非技术行业集中度尚无“非技术行业”正式定义；"
        "需在 job_context/行业口径冻结后执行"
    )
    warns.append(
        "§17 待人工判据：年度 AI 比例“无法由覆盖变化解释的断点”未给数值阈值；"
        "已输出年度 AI 率与 zero-skill 明细供判读"
    )
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
        or ["- ✅ 十项全部通过（含原文跨度回填验证）"]
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
    if fails:
        # 失败运行不得推进 §17.6.10 的成功基线。否则第一次真实漂移会把
        # quality_stats.json 覆盖成新值，第二次原样重跑就会“新对新”误通过。
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_stats = rel / f"quality_stats.failed_{stamp}.json"
        failed_stats.write_text(json.dumps(stats, indent=1), encoding="utf-8")
        logger.error(
            "质量门失败；成功基线未更新。候选统计写入 %s", failed_stats.name
        )
    else:
        tmp = rel / ".quality_stats.json.tmp"
        tmp.write_text(json.dumps(stats, indent=1), encoding="utf-8")
        tmp.replace(stats_path)
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
