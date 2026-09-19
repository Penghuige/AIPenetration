"""panel_v2 M3-b：岗位—技能共现计数（指南 §13）。

从 pass2 产物（job_anchor_flag + job_skill_long）计算三套锚点 × 三种窗口
（annual/pooled/roll3_centered）的整数分子分母计数，实现顺序严格遵循
§13.6（先整数计数、比率一律由整数重相除、64 位整数/64 位浮点），
并执行 §13.7 全部不变量（任一失败阻断）。

产物 skill_ai_counts.parquet（§13.5）：
(skill_code, anchor_version, window_type, year, window_start, window_end,
 n_skill, n_ai_cooccur)

roll3 边界（§13.3）：2014→{2014,2015}，末年→{末年-1,末年}（数据末年 2024）。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from .anchors import ANCHOR_VERSIONS

logger = logging.getLogger("ai_penetration.panel_v2.counts")

WINDOWS = ("annual", "pooled", "roll3_centered")


def load_inputs(rel_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """读两个 parquet，job_id 重映射为连续行号。

    Returns:
        (job_idx_of_pair, skill_code, year, flags_by_version)
    """
    flags = pq.read_table(rel_dir / "job_anchor_flag.parquet").to_pandas()
    longs = pq.read_table(rel_dir / "job_skill_long.parquet",
                          columns=["job_id", "year", "skill_code"]).to_pandas()
    jid_arr = flags["job_id"].to_numpy()
    order = np.argsort(jid_arr)
    job_ids = jid_arr[order]
    long_ids = longs["job_id"].to_numpy()
    pos = np.searchsorted(job_ids, long_ids)
    valid = pos < len(job_ids)
    if not valid.all() or not np.all(job_ids[pos[valid]] == long_ids[valid]):
        raise RuntimeError("long 表存在 flag 表之外的 job_id")
    flag_idx = {v: flags[f"anchor_{v}"].to_numpy()[order]
                for v in ANCHOR_VERSIONS}
    return pos.astype(np.int64), longs["skill_code"].to_numpy(np.int32), \
        longs["year"].to_numpy(np.int16), flag_idx


def compute_counts(rel_dir: Path) -> pd.DataFrame:
    """§13.1–§13.3 全部计数 + §13.6 实现顺序。"""
    job_idx, skill, year, flags = load_inputs(rel_dir)
    # §13.6.9 计算前基线日志
    # 标签修正（2026-09-09 审计：unique([job_idx, year]) 是岗位×年组合数，
    # 不是"岗位×技能×年"——键里根本没有 skill）
    logger.info("计数基线: 岗位=%d 岗位-技能行=%d 岗位×年组合数=%d",
                len(flags["main"]) if isinstance(flags, dict) else 0,
                len(skill), len(np.unique(np.stack([job_idx, year], axis=1), axis=0)))
    years = np.sort(np.unique(year))
    n_y = len(years)
    yidx = np.searchsorted(years, year)
    n_skill_max = int(skill.max()) + 1
    cells = skill.astype(np.int64) * n_y + yidx
    out_frames = []
    for ver in ANCHOR_VERSIONS:
        anchored = flags[ver][job_idx].astype(bool)
        mask = cells[anchored]
        n_total = np.bincount(cells, minlength=n_skill_max * n_y)
        n_cooc = np.bincount(mask, minlength=n_skill_max * n_y)
        ann = pd.DataFrame({
            "skill_code": np.repeat(np.arange(n_skill_max), n_y),
            "year": np.tile(years, n_skill_max),
            "n_skill": n_total.reshape(n_skill_max, n_y).ravel(),
            "n_ai_cooccur": n_cooc.reshape(n_skill_max, n_y).ravel(),
        })
        ann = ann[ann["n_skill"] > 0]
        ann["anchor_version"] = ver
        ann["window_type"] = "annual"
        ann["window_start"] = ann["year"]
        ann["window_end"] = ann["year"]
        out_frames.append(ann)
        # pooled：整数加总（§13.6.6 不从比率算）
        pool_n = n_total.reshape(n_skill_max, n_y).sum(axis=1)
        pool_c = n_cooc.reshape(n_skill_max, n_y).sum(axis=1)
        keep = pool_n > 0
        out_frames.append(pd.DataFrame({
            "skill_code": np.nonzero(keep)[0], "year": -1,
            "n_skill": pool_n[keep], "n_ai_cooccur": pool_c[keep],
            "anchor_version": ver, "window_type": "pooled",
            "window_start": years[0], "window_end": years[-1],
        }))
        # roll3_centered：窗口整数计数=年度计数之和（§13.7 第4条构造性满足）
        nt = n_total.reshape(n_skill_max, n_y)
        nc = n_cooc.reshape(n_skill_max, n_y)
        cum_n = np.cumsum(nt, axis=1)
        cum_c = np.cumsum(nc, axis=1)

        def _win(i: int) -> tuple[np.ndarray, np.ndarray]:
            lo = max(0, i - 1)
            hi = min(n_y - 1, i + 1)
            wn = cum_n[:, hi] - (cum_n[:, lo - 1] if lo > 0 else 0)
            wc = cum_c[:, hi] - (cum_c[:, lo - 1] if lo > 0 else 0)
            return wn, wc

        for i, y in enumerate(years):
            wn, wc = _win(i)
            keep = wn > 0
            out_frames.append(pd.DataFrame({
                "skill_code": np.nonzero(keep)[0], "year": int(y),
                "n_skill": wn[keep], "n_ai_cooccur": wc[keep],
                "anchor_version": ver, "window_type": "roll3_centered",
                "window_start": int(max(years[0], y - 1)),
                "window_end": int(min(years[-1], y + 1)),
            }))
    counts = pd.concat(out_frames, ignore_index=True)
    verify_counts(counts)
    return counts


def verify_counts(counts: pd.DataFrame) -> None:
    """§13.7 计数不变量；任何失败均显式阻断。"""
    g = counts.groupby(
        ["anchor_version", "window_type", "year", "skill_code"]
    )
    if int(g.size().max()) != 1:
        raise RuntimeError("同一 skill×窗口×锚点出现多行")
    bad = counts[counts["n_ai_cooccur"] > counts["n_skill"]]
    if not bad.empty:
        raise RuntimeError(f"{len(bad)} 行分子>分母")
    if not (counts["n_skill"] > 0).all():
        raise RuntimeError("n_skill<=0")
    for ver in ANCHOR_VERSIONS:
        ann = counts[
            (counts.anchor_version == ver)
            & (counts.window_type == "annual")
        ]
        pool = counts[
            (counts.anchor_version == ver)
            & (counts.window_type == "pooled")
        ]
        a = ann.groupby("skill_code")[["n_skill", "n_ai_cooccur"]].sum()
        p = pool.set_index("skill_code")[["n_skill", "n_ai_cooccur"]]
        joined = a.join(
            p, how="outer", lsuffix="_a", rsuffix="_p"
        ).fillna(0)
        ok = (
            np.array_equal(
                joined["n_skill_a"].to_numpy(np.int64),
                joined["n_skill_p"].to_numpy(np.int64),
            )
            and np.array_equal(
                joined["n_ai_cooccur_a"].to_numpy(np.int64),
                joined["n_ai_cooccur_p"].to_numpy(np.int64),
            )
        )
        if not ok:
            raise RuntimeError(f"pooled 年度加总不守恒 ({ver})")
    logger.info("§13.7 计数不变量全部通过")


def main() -> None:
    ap = argparse.ArgumentParser(description="panel_v2 §13 共现计数")
    ap.add_argument("--rel", default="panel_v2", help="release 子目录名")
    args = ap.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_counts.log")
    rel = paths.output_dir / "release" / args.rel
    counts = compute_counts(rel)
    counts.to_parquet(rel / "skill_ai_counts.parquet", index=False)
    logger.info("skill_ai_counts: %d 行", len(counts))
    print(f"计数完成: {len(counts):,} 行（9 组窗口×版本 × 技能）")


if __name__ == "__main__":
    main()
