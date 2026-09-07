"""§14 平滑的单测：拟合方向、回退、原始权重不被覆盖。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_penetration.panel_v2.relevance import (
    beta_binomial_fit_unit,
    compute_relevance,
)


def test_fit_finds_low_prior_for_sparse_signal():
    n = np.concatenate([np.full(300, 50.0), np.full(20, 500.0)])
    c = np.concatenate([np.zeros(300), np.full(20, 350.0)])
    a, b, status = beta_binomial_fit_unit(n, c)
    assert status.startswith(("fitted", "jeffreys"))
    assert a / (a + b) < 0.5  # 先验均值接近全局低频共现水平


def test_fit_fallback_on_tiny_unit():
    a, b, status = beta_binomial_fit_unit(np.array([5.0, 6.0]), np.array([1.0, 2.0]))
    assert (a, b) == (0.5, 0.5) and status == "jeffreys:few_skills"


def test_relevance_keeps_raw_and_smoothed(tmp_path):
    counts = pd.DataFrame({
        "skill_code": [0, 1, 0, 1, 0],
        "year": [2014, 2014, 2015, 2015, -1],
        "n_skill": [10, 10, 12, 12, 22],
        "n_ai_cooccur": [8, 1, 10, 2, 18],
        "anchor_version": ["main"] * 5,
        "window_type": ["annual", "annual", "annual", "annual", "pooled"],
        "window_start": [2014, 2014, 2015, 2015, 2014],
        "window_end": [2014, 2014, 2015, 2015, 2015],
    })
    rel = compute_relevance(counts, np.array(["A", "B"], dtype=object))
    row = rel[(rel.window_type == "annual") & (rel.skill_code == 0)
              & (rel.year == 2014)].iloc[0]
    assert row.ai_rate_raw == 0.8
    assert row.ai_rate_smoothed <= 0.8          # 收缩方向
    assert row.smoothing_status.startswith(("fitted", "jeffreys"))
    assert set(rel.columns) >= {"alpha", "beta", "rare_lt10", "confidence_tier",
                                "ai_rate_raw", "ai_rate_smoothed",
                                "smoothing_status"}
    # pooled 行保留且由整数构成（22 分母）
    p = rel[rel.window_type == "pooled"]
    assert p.n_skill.item() == 22 and p.n_ai_cooccur.item() == 18
