"""panel_v2 M3-d：岗位 AI 相关度与判定（指南 §15–§16 + §14.3 留一）。

- §15.1 岗位得分 = 去重技能共现率简单平均（pair 级 gather+bincount 向量实现）；
- §15.2 年份连接：annual 连岗位年、pooled 连全期、roll3 连岗位年窗口；
- §15.3 coverage=1 硬断言（cnt>0 岗位 weighted==matched）；
- §15.4 raw 与 smoothed 双得分，主可比用 raw；
- §16.1 三阈值严格大于；§16.2 零技能岗位：保留记录、得分缺失、
  三标识=0、zero_skill_override=1；§16.4 主标识 aijob_main_annual_raw_005；
- §14.3 企业留一（仅 main、三窗口、原始率）：
  w^-f = (c_{s,t}-c_{s,f,t})/(n_{s,t}-n_{s,f,t})，分母≤0 该技能无留一权重
  （得分按有权重技能取均值，覆盖率仅描述，§15.3.5）。

产物（release/panel_v2/）：job_ai_score.parquet（长表）、
job_ai_classification.parquet（宽表标识）、job_ai_score_loo.parquet。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from .anchors import ANCHOR_VERSIONS

logger = logging.getLogger("ai_penetration.panel_v2.scoring")

THRESHOLDS = (0.05, 0.10, 0.15)
VERSIONS = tuple(ANCHOR_VERSIONS)
WINDOWS = ("annual", "pooled", "roll3_centered")


def _unit_key(win: str, yidx: int, n_years: int) -> int:
    """(win, yidx) → 列号：annual 0..n-1，pooled n，roll3 n+1..2n。"""
    if win == "annual":
        return yidx
    if win == "pooled":
        return n_years
    return n_years + 1 + yidx


def _dense_weights(rel: pd.DataFrame, years: np.ndarray, n_skill: int):
    """(ver, scoretype) → dense 矩阵 [n_cols, n_skill]。"""
    n_y = len(years)
    cols = 2 * n_y + 1
    out: dict[tuple[str, str], np.ndarray] = {}
    for ver in VERSIONS:
        for st in ("raw", "smoothed"):
            # NaN 填充：真缺失（连接/版本错误）在得分层可检出（B2，§15.3.2
            # 禁止静默置 0）；合法 w=0.0 由 counts 全技能保留写入覆盖
            mat = np.full((cols, n_skill), np.nan)
            sub = rel[rel.anchor_version == ver]
            col = np.empty(len(sub), np.int64)
            win_idx = sub.window_type.to_numpy()
            col[win_idx == "annual"] = [
                _unit_key("annual", i, n_y)
                for i in np.searchsorted(years, sub.year.to_numpy()[win_idx == "annual"])]
            col[win_idx == "pooled"] = n_y
            col[win_idx == "roll3_centered"] = [
                _unit_key("roll3", i, n_y)
                for i in np.searchsorted(years, sub.year.to_numpy()[win_idx == "roll3_centered"])]
            src = sub.ai_rate_raw.to_numpy() if st == "raw" \
                else sub.ai_rate_smoothed.to_numpy()
            mat[col, sub.skill_code.to_numpy()] = src
            out[(ver, st)] = mat
    return out


def _job_loo_scores(pair_job, pair_skill, pair_yidx, pair_comp, anchored,
                    unit_dense, n_years, n_jobs_total):
    """§14.3 留一（main raw，三窗口）——pair 级实现，返回 job×窗口均分与覆盖率。

    Args:
        pair_job/pair_skill/pair_yidx: long 表向量化列。
        pair_comp: pair 对应企业编码。
        anchored: pair 所属岗位是否 main 锚点。
        unit_dense: win -> (n_unit, c_unit) pair 级单位整数计数。
        n_years: 年份数。
    """
    n_y = n_years
    n_comp = int(pair_comp.max()) + 1
    key = ((pair_skill.astype(np.int64) * n_comp + pair_comp) << 4) | pair_yidx
    order = np.argsort(key, kind="stable")
    k_uniq, starts = np.unique(key[order], return_index=True)
    n_sf = np.diff(np.append(starts, len(key)))  # 哨兵=总 pair 数（非 uniq 数）
    anchored_sorted = anchored[order]
    starts_of = np.repeat(np.arange(len(k_uniq)), n_sf)
    c_sf = np.bincount(starts_of, weights=anchored_sorted.astype(np.float64),
                       minlength=len(k_uniq))
    triple_idx = np.searchsorted(k_uniq, key)
    # 窗口聚合：按 (s,f) 分组
    sf_key = k_uniq >> 4
    sf_change = np.r_[True, sf_key[1:] != sf_key[:-1]]
    sf_start = np.flatnonzero(sf_change)
    sf_id = np.cumsum(sf_change) - 1
    pooled_n = np.add.reduceat(n_sf, sf_start)
    pooled_c = np.add.reduceat(c_sf, sf_start)
    n_pairs = len(key)
    out = {}
    base_key = (pair_skill.astype(np.int64) * n_comp + pair_comp) << 4
    yidx_lo = np.clip(pair_yidx - 1, 0, n_y - 1).astype(np.int64)
    yidx_hi = np.clip(pair_yidx + 1, 0, n_y - 1).astype(np.int64)
    for win in WINDOWS:
        if win == "annual":
            tn, tc = n_sf[triple_idx], c_sf[triple_idx]
        elif win == "pooled":
            tn, tc = pooled_n[sf_id[triple_idx]], pooled_c[sf_id[triple_idx]]
        else:  # roll3 窗口 {y-1,y,y+1}：边界年去重，防同 triple 双计（§13.3）
            tn = np.zeros(n_pairs)
            tc = np.zeros(n_pairs)
            terms = [(pair_yidx.astype(np.int64), None),
                     (yidx_lo, pair_yidx.astype(np.int64)),
                     (yidx_hi, pair_yidx.astype(np.int64))]
            for yy, excl in terms:
                if excl is not None:
                    yy = np.where(yy == excl, -1, yy)  # 与主年重复则跳过
                kk = base_key | yy
                valid = (yy >= 0)
                ii = np.searchsorted(k_uniq, kk)
                ii_c = np.clip(ii, 0, max(len(k_uniq) - 1, 0))
                hit = valid & (k_uniq[ii_c] == kk)
                tn += np.where(hit, n_sf[ii_c], 0)
                tc += np.where(hit, c_sf[ii_c], 0)
        # 单位（main×win×岗位年）总计数：dense 按 pair 展开
        nn, cc = unit_dense(win)
        with np.errstate(invalid="ignore", divide="ignore"):
            w_loo = (cc - tc) / (nn - tn)
        bad = (nn - tn) <= 0
        w_loo = np.where(bad, np.nan, w_loo)
        valid = ~np.isnan(w_loo)
        s_sum = np.bincount(pair_job[valid], weights=w_loo[valid],
                            minlength=n_jobs_total)
        s_cnt = np.bincount(pair_job[valid], minlength=n_jobs_total)
        cnt_all = np.bincount(pair_job, minlength=n_jobs_total)
        score = np.divide(s_sum, s_cnt, out=np.full(n_jobs_total, np.nan),
                          where=s_cnt > 0)
        out[win] = (score, (s_cnt / np.maximum(cnt_all, 1)).astype(np.float32))
    return out


def run(rel_dir: Path, bench: bool = False) -> None:
    """全量得分与判定。"""
    flags = pq.read_table(rel_dir / "job_anchor_flag.parquet").to_pandas()
    longs = pq.read_table(rel_dir / "job_skill_long.parquet").to_pandas()
    firm = pq.read_table(rel_dir / "job_firm.parquet").to_pandas()
    rel = pq.read_table(rel_dir / "skill_ai_relevance.parquet").to_pandas()

    jobs = np.sort(flags.job_id.to_numpy())
    if bench:
        keep_jobs = jobs[:100000]
        flags = flags[flags.job_id.isin(keep_jobs)]
        longs = longs[longs.job_id.isin(keep_jobs)]
        firm = firm[firm.job_id.isin(keep_jobs)]
        jobs = keep_jobs
        logger.info("[bench] 截取 %d 岗位试算", len(jobs))
    j_of = np.searchsorted(jobs, longs.job_id.to_numpy())
    s_of = longs.skill_code.to_numpy(np.int32)
    y_of = longs.year.to_numpy(np.int32)
    n_skill = int(rel.skill_code.max()) + 1
    years = np.sort(np.unique(flags.year.to_numpy()))
    ymin = int(years[0])
    yidx_pairs = y_of - ymin
    # job → company 映射（firm 每 job 一行，按 job_id 对齐，防行序错配）
    company_of_job = np.zeros(len(jobs), np.int32)
    firm_job_arr = firm.job_id.to_numpy()
    assert np.array_equal(np.sort(firm_job_arr), np.unique(firm_job_arr)), \
        "firm 表 job 重复"
    company_of_job[np.searchsorted(jobs, firm_job_arr)] = \
        firm.company_code.to_numpy(np.int32)

    anchor_arrays = {}
    for ver in VERSIONS:
        arr = np.zeros(len(jobs), np.int8)
        arr[np.searchsorted(jobs, flags.job_id.to_numpy())] = \
            flags[f"anchor_{ver}"].to_numpy(np.int8)
        anchor_arrays[ver] = arr
    dense = _dense_weights(rel, years, n_skill)
    n_y = len(years)

    # job 属性表
    job_year = np.zeros(len(jobs), np.int32)
    job_year[np.searchsorted(jobs, flags.job_id.to_numpy())] = \
        flags.year.to_numpy(np.int32) - ymin
    matched = np.bincount(j_of, minlength=len(jobs)).astype(np.int32)

    # B1：每个 (ver,win,scoretype) 单元即算即落盘（dataset 分区），
    # 不累积 18 个全量帧；B2：weighted/coverage 由 isfinite 实测。
    assert n_y <= 16, "留一 triple 键打包假设 yidx<16（M1 防扩年后静默错配）"
    score_dir = rel_dir / "job_ai_score"
    score_dir.mkdir(parents=True, exist_ok=True)
    cls = pd.DataFrame({"job_id": jobs, "year": job_year + ymin})
    n_jobs = len(jobs)
    for ver in VERSIONS:
        for win in WINDOWS:
            for st in ("raw", "smoothed"):
                unit = np.array([_unit_key(win, y, n_y)
                                 for y in range(n_y)])[job_year[j_of]] \
                    if win != "pooled" else np.full(len(j_of), n_y)
                w = dense[(ver, st)][unit, s_of]
                finite = np.isfinite(w)
                sw = np.bincount(j_of, weights=np.where(finite, w, 0.0),
                                 minlength=n_jobs)
                weighted = np.bincount(j_of[finite], minlength=n_jobs)
                bad = (matched > 0) & (weighted != matched)
                assert not bad.any(), \
                    f"§15.3.2 阻断：{ver}/{win}/{st} 有 {int(bad.sum())} 岗位技能权重缺失"
                cov = np.divide(sw, weighted, out=np.full(n_jobs, np.nan),
                                where=weighted > 0)
                pq.write_table(
                    pa.Table.from_pandas(pd.DataFrame({
                        "job_id": jobs, "anchor_version": ver, "window_type": win,
                        "score_type": st, "ai_score": cov,
                        "matched_skill_count": matched,
                        "weighted_skill_count": weighted.astype(np.int32),
                        "score_skill_coverage": np.divide(
                            weighted, matched, out=np.zeros(n_jobs),
                            where=matched > 0),
                        "score_eligible": (matched > 0).astype(np.int8),
                    })),
                    score_dir / f"{ver}_{win}_{st}.parquet")
                cname = f"aijob_{ver}_{win}_{st}"
                for thr in THRESHOLDS:
                    cls[f"{cname}_{'005' if thr == 0.05 else ('010' if thr == 0.1 else '015')}"] = \
                        (cov > thr).astype(np.int8)
    # §16.2 零技能岗位：标识归 0 + override（得分保持缺失）
    zero = matched == 0
    cls["zero_skill_override"] = zero.astype(np.int8)
    cls["window_period_available"] = np.int8(1)
    cls["score_eligible"] = (~zero).astype(np.int8)
    for c in cls.columns:
        if c.startswith("aijob_"):
            cls[c] = np.where(zero, 0, cls[c])

    # §14.3 留一（main raw）：单位整数计数 dense 从 counts 重建（向量）
    anchored_main = anchor_arrays["main"] > 0
    pair_anchored = anchored_main[j_of]
    counts_df = pd.read_parquet(rel_dir / "skill_ai_counts.parquet")
    cm = counts_df[counts_df.anchor_version == "main"]
    dense_n = np.zeros((2 * n_y + 1, n_skill), np.float64)
    dense_c = np.zeros_like(dense_n)
    win_arr = cm.window_type.to_numpy()
    yr_idx = np.searchsorted(years, cm.year.to_numpy())
    cols = np.where(win_arr == "annual", yr_idx,
                    np.where(win_arr == "pooled", n_y, n_y + 1 + yr_idx))
    dense_n[cols, cm.skill_code.to_numpy()] = cm.n_skill.to_numpy(np.float64)
    dense_c[cols, cm.skill_code.to_numpy()] = cm.n_ai_cooccur.to_numpy(np.float64)

    def unit_dense(win):
        if win == "pooled":
            return dense_n[n_y][s_of], dense_c[n_y][s_of]
        base = 0 if win == "annual" else n_y + 1
        return (dense_n[base + yidx_pairs, s_of],
                dense_c[base + yidx_pairs, s_of])

    loo = _job_loo_scores(j_of, s_of, yidx_pairs, company_of_job[j_of],
                          pair_anchored, unit_dense, n_y, len(jobs))
    loo_frame = pd.DataFrame({"job_id": jobs, "year": job_year + ymin})
    for win, (score, cov) in loo.items():
        loo_frame[f"loo_main_{win}_raw"] = score
        loo_frame[f"loo_main_{win}_coverage"] = cov

    out = rel_dir
    pq.write_table(pa.Table.from_pandas(cls), out / "job_ai_classification.parquet",
                   compression="zstd")
    pq.write_table(pa.Table.from_pandas(loo_frame),
                   out / "job_ai_score_loo.parquet", compression="zstd")
    exp5 = cls["aijob_main_annual_raw_005"].mean()
    prim = cls["aijob_main_annual_raw_015"].mean()
    logger.info("scoring 完成: 岗位 %d，主指标(>0.15) %.4f%%，暴露率(>0.05) %.4f%%，"
                "零技能 %.2f%%", len(jobs), prim * 100, exp5 * 100, zero.mean() * 100)
    print(f"job_ai_score 已按 18 单元分区写入 {score_dir.name}/；"
          f"主指标 aijob_main_annual_raw_015 = {prim:.4%}；"
          f"暴露率 aijob_main_annual_raw_005 = {exp5:.4%}；零技能 {zero.mean():.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="panel_v2 §15-16 得分与判定")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_scoring.log")
    run(paths.output_dir / "release" / "panel_v2", bench=args.bench)


if __name__ == "__main__":
    main()
