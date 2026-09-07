"""scoring 端到端小 fixture：主标识、零技能覆盖、留一合理性。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ai_penetration.panel_v2.scoring import run


def _fixture(tmp_path):
    # 5 岗位 2 年。job1/2/3 有技能且 job1,3 锚点；job4 有技能无锚点；job5 无技能
    flags = pa.table({
        "job_id": pa.array([1, 2, 3, 4, 5], pa.int64()),
        "year": pa.array([2014, 2014, 2015, 2015, 2014], pa.int32()),
        "anchor_main": pa.array([1, 0, 1, 0, 0], pa.int8()),
        "anchor_cn_paper": pa.array([1, 0, 0, 0, 0], pa.int8()),
        "anchor_babina": pa.array([1, 0, 1, 0, 0], pa.int8()),
        "groups_main_bits": pa.array([1, 0, 3, 0, 0], pa.int16()),
    })
    longs = pa.table({
        "job_id": pa.array([1, 1, 2, 3, 4], pa.int64()),
        "year": pa.array([2014, 2014, 2014, 2015, 2015], pa.int32()),
        "skill_code": pa.array([0, 1, 1, 0, 1], pa.int32()),
    })
    firm = pa.table({
        "job_id": pa.array([1, 2, 3, 4], pa.int64()),
        "year": pa.array([2014, 2014, 2015, 2015], pa.int32()),
        "company_code": pa.array([0, 1, 0, 2], pa.int32()),
    })
    for t, name in ((flags, "job_anchor_flag"), (longs, "job_skill_long"),
                    (firm, "job_firm")):
        pq.write_table(t, tmp_path / f"{name}.parquet")
    # 手工权重表（main/cn/babina × annual 14/15 + pooled + roll3 14/15，
    # raw 与 smoothed 两列）——直接构造 counts 由 relevance 产出的等价物
    cells = []
    for ver in ("main", "cn_paper", "babina"):
        for (win, y, ws, we) in (("annual", 2014, 2014, 2014),
                                 ("annual", 2015, 2015, 2015),
                                 ("pooled", -1, 2014, 2015),
                                 ("roll3_centered", 2014, 2014, 2015),
                                 ("roll3_centered", 2015, 2014, 2015)):
            cells.append((ver, win, y, ws, we))
    rows = []
    # skill0: n/c 各单元 (14: 1/1 15: 1/1 pooled 2/2 roll3 2/2) → rate 1
    # skill1: 全部单元 n 2 c 0 → rate 0
    for ver, win, y, ws, we in cells:
        if win == "annual":
            n0 = c0 = 1 if y in (2014, 2015) else 1
            n1, c1 = 2, 0
            if y == 2014:
                n1 = 2
            n0 = 1
        elif win == "pooled":
            n0, c0, n1, c1 = 2, 2, 2, 0
        else:
            n0, c0, n1, c1 = 2, 2, 2, 0
        rows.append({"skill_code": 0, "year": y, "window_type": win,
                     "window_start": ws, "window_end": we,
                     "anchor_version": ver, "n_skill": n0,
                     "n_ai_cooccur": c0,
                     "ai_rate_raw": c0 / n0,
                     "ai_rate_smoothed": (c0 + 0.5) / (n0 + 1.0),
                     "confidence_tier": "A"})
        rows.append({"skill_code": 1, "year": y, "window_type": win,
                     "window_start": ws, "window_end": we,
                     "anchor_version": ver, "n_skill": n1,
                     "n_ai_cooccur": c1,
                     "ai_rate_raw": c1 / n1,
                     "ai_rate_smoothed": (c1 + 0.5) / (n1 + 1.0),
                     "confidence_tier": "B"})
    pd.DataFrame(rows).to_parquet(tmp_path / "skill_ai_relevance.parquet")
    # counts（main 子集供留一）
    pd.DataFrame([
        {"skill_code": s, "year": y, "n_skill": n, "n_ai_cooccur": c,
         "anchor_version": "main", "window_type": w,
         "window_start": y if w != "pooled" else 2014,
         "window_end": y if w != "pooled" else 2015}
        for s in (0, 1)
        for (w, y, n, c) in (
            ("annual", 2014, (1, 2)[s], (1, 0)[s]),
            ("annual", 2015, (1, 0)[s], (1, 0)[s]),
            ("pooled", -1, (2, 2)[s], (2, 0)[s]),
            ("roll3_centered", 2014, (2, 2)[s], (2, 0)[s]),
            ("roll3_centered", 2015, (2, 2)[s], (2, 0)[s]))
    ]).to_parquet(tmp_path / "skill_ai_counts.parquet")
    return tmp_path


def test_scoring_end_to_end(tmp_path):
    run(_fixture(tmp_path))
    cls = pd.read_parquet(tmp_path / "job_ai_classification.parquet")
    score = pd.read_parquet(tmp_path / "job_ai_score.parquet")
    # job1: 含 skill0 rate=1 → 得分 0.5(两技能均值) → >0.05 标识 1
    r = cls[cls.job_id == 1].iloc[0]
    assert r.aijob_main_annual_raw_005 == 1
    # job5 零技能：标识 0 + override 1；得分缺失
    r5 = cls[cls.job_id == 5].iloc[0]
    assert r5.zero_skill_override == 1 and r5.aijob_main_annual_raw_005 == 0
    s5 = score[(score.job_id == 5) & (score.anchor_version == "main")
               & (score.window_type == "annual") & (score.score_type == "raw")]
    assert np.isnan(s5.ai_score.iloc[0])
    # 得分表行数 = 5 jobs × 9 单元 × 2 类型
    assert len(score) == 5 * 9 * 2
    # coverage 恒 1（有技能岗位）
    assert (score[score.matched_skill_count > 0]
            .score_skill_coverage == 1.0).all()
    loo = pd.read_parquet(tmp_path / "job_ai_score_loo.parquet")
    # job3(company0,skill0,2015) 与 job1(company0,2014,skill0)：
    # annual 2015 n_{s,f,y}=0/1 → 留一分母 0 → 得分缺失（有权重技能=skill1 rate0）
    l3 = loo[loo.job_id == 3].iloc[0]
    assert np.isnan(l3.loo_main_annual_raw)  # skill0 无留一权重，其余仅 skill1?
    assert l3.loo_main_annual_coverage >= 0.0
