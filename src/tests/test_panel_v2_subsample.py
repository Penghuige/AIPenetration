"""零技能子样本曲线模块的离线测试（合成数据，不连库/不读发布件）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_penetration.panel_v2.subsample_curve import curve_frame


def _frames():
    cls = pd.DataFrame({
        "job_id": [1, 2, 3, 4, 5, 6],
        "year": [2015, 2016, 2016, 2016, 2016, 2016],
        "zero_skill_override": [0, 1, 0, 0, 0, 0],
        "aijob_main_annual_raw_005": [1, 0, 1, 1, 0, 0],
        "aijob_main_annual_raw_015": [0, 0, 0, 1, 0, 0],
    })
    score = pd.DataFrame({
        "job_id": [1, 2, 3, 4, 5, 6],
        "ai_score": [0.9, np.nan, 0.4, 0.05, 0.02, np.nan],
    })
    return cls, score


def test_curve_frame_excludes_pre_min_year_and_recomputes_matched():
    cls, score = _frames()
    fr = curve_frame(cls, score, y_min=2016)
    assert list(fr.year) == [2016]
    row = fr.iloc[0]
    assert row.n == 5
    assert row.zero_skill_rate == 0.2          # 5 岗中 1 个零技能
    assert row.r005_all == 0.4                 # 2/5 主标识
    assert row.r005_matched == round(2 / 4, 5)  # 有技能子样本 2/4
    # 零技能岗位（NaN 分）在 hi030 中计 0，不产生 NaN 传播
    assert row.hi030_all == 0.2                # 仅 job 3（0.4>0.3）
    assert not fr.isna().any().any()


def test_curve_frame_deterministic():
    cls, score = _frames()
    pd.testing.assert_frame_equal(curve_frame(cls, score),
                                  curve_frame(cls, score))
