"""Beta-Binomial 平滑的机制性测试：噪声极端值收缩、稳定信号保留。"""
from __future__ import annotations

from src.ai_penetration.compare_atier_gz2024 import beta_binomial_fit, smooth_omega


def test_smooth_shrinks_small_sample_extreme_value():
    """首轮假阳性根因回归：n=25 全共现的 ω=1.0 在合理先验下必须被压到 <0.5。"""
    counts = {f"s{i}": (25, 25) for i in range(3)}
    out = smooth_omega(counts, alpha=0.5, beta=50.0, min_count=5)
    assert all(w < 0.5 for w in out.values()), out


def test_smooth_keeps_stable_signal():
    """大样本高共现概念（真 AI 技能）平滑后仍须过 max>=0.5 门槛。"""
    out = smooth_omega({"ml": (5000, 4000)}, alpha=0.5, beta=50.0, min_count=5)
    assert out["ml"] >= 0.5


def test_smooth_drops_low_frequency_below_min_count():
    out = smooth_omega({"rare": (4, 4)}, alpha=0.5, beta=50.0, min_count=5)
    assert out == {}


def test_smooth_monotone_in_evidence():
    """同比例 80% 共现，样本越大平滑后 ω 越高。"""
    counts = {"small": (20, 16), "big": (2000, 1600)}
    out = smooth_omega(counts, alpha=1.0, beta=10.0, min_count=5)
    assert out["big"] > out["small"]


def test_beta_fit_returns_positive_params_or_jeffres_fallback():
    counts = {}
    for i in range(300):  # 主体：低频低共现（模拟通用技能）
        counts[f"g{i}"] = (30, 1)
    for i in range(10):  # 少量真 AI 技能
        counts[f"a{i}"] = (500, 300)
    alpha, beta = beta_binomial_fit(counts)
    assert alpha > 0 and beta > 0
    prior_mean = alpha / (alpha + beta)
    assert prior_mean < 0.2  # 全局先验应接近锚点岗位占比（低）


def test_beta_fit_fallback_on_tiny_input():
    alpha, beta = beta_binomial_fit({"x": (3, 1)})
    assert (alpha, beta) == (0.5, 0.5)
