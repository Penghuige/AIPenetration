"""discovery_formal 口径忠实化回归：§9.2.1-9.2.5 抽样与 §9.2.2 饱和条件。

指南算例：年内先向每个非空行业分配 min(50, 该行业岗位数)；剩余名额按
行业剩余岗位数比例分配；舍入差额按行业剩余岗位数由大到小补齐（§9.2.3-4）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ai_penetration.panel_v2.discovery_formal import (
    BASE_PER_YEAR,
    _industry_quota,
    baseline_sample,
    saturation_pass,
)


def _frame(rows):
    cols = [
        "job_id", "year", "industry", "position_group", "company_size",
        "text_length_bin", "tech_flag", "platform", "anchor_main",
        "matched_skill_count", "text_hash", "description",
    ]
    return pd.DataFrame(rows, columns=cols)


def _year_rows(year, industries, start=0):
    """构造某年按行业数量的合成 frame 行。"""
    rows = []
    i = start
    for ind, n in industries.items():
        for _ in range(n):
            rows.append([
                f"j{year}-{i}", year, ind, "engineer", "100",
                "long", 1, "51job", 0, 0, 1000 + i, f"desc {i} " + "x" * 20,
            ])
            i += 1
    return rows


def test_industry_quota_guideline_example():
    # A:1000, B:30, C:80，n=200：
    # 保底 A50+B30+C50=130；剩余 70 按 A950/C30 比例 → A67.86→67、C2.14→2；
    # 差额 1 按剩余岗位数由大到小补 A → A118 / B30 / C52
    sizes = pd.Series({"A": 1000, "B": 30, "C": 80})
    assert _industry_quota(sizes, 200) == {"A": 118, "B": 30, "C": 52}


def test_industry_quota_small_year_takes_all():
    # §9.2.2：年份岗位不足 1 万时取全部——行业岗位数小于保底时保底即全取
    sizes = pd.Series({"A": 10, "B": 5})
    assert _industry_quota(sizes, 15) == {"A": 10, "B": 5}


def test_industry_quota_floor_overflow_largest_first():
    # 保底总额超过样本名额（行业极多）：按行业岗位数由大到小保留保底并截断
    sizes = pd.Series({"A": 200, "B": 100, "C": 50})
    assert _industry_quota(sizes, 100) == {"A": 50, "B": 50}


def test_industry_quota_deterministic_on_ties():
    sizes = pd.Series({"B": 60, "A": 60})
    # 并列行业按键稳定序补差，结果跨调用一致
    q1 = _industry_quota(sizes, 130)
    q2 = _industry_quota(pd.Series({"A": 60, "B": 60}), 130)
    assert q1 == q2


def test_baseline_per_year_exact_and_major_industries_covered():
    rows = _year_rows(2020, {"A": 300, "B": 60, "C": 5})
    rows += _year_rows(2021, {"A": 40}, start=1000)
    df = _frame(rows)
    out = baseline_sample(df, per_year=100)
    # 每年精确 = min(100, 年量)
    counts = out.groupby(out.year.astype(int)).size().to_dict()
    assert counts == {2020: 100, 2021: 40}
    # 主要行业（≥100 岗位）全部入样
    got_2020 = set(out[out.year == 2020].industry)
    assert "A" in got_2020 and "B" in got_2020
    # 不放回、无重复
    assert not out.job_id.duplicated().any()


def test_baseline_respects_industry_floor():
    # 小行业 B(60) 保底 50，且其剩余 10 条岗位仍按比例分得 1 条 → B=51
    rows = _year_rows(2020, {"A": 5000, "B": 60})
    out = baseline_sample(_frame(rows), per_year=1000)
    counts = out[out.year == 2020].groupby("industry").size().to_dict()
    assert counts["B"] == 51
    assert counts["A"] == 949
    assert sum(counts.values()) == 1000


def test_saturation_requires_coverage_and_anchor_conditions():
    # §9.2.2 恢复为四条件：新增覆盖 <5 之外，覆盖率提升 <0.1pp 与
    # 锚点未覆盖比例不再明显下降（连续两轮降幅 <0.1pp）都必须满足
    base = {"round": [1, 2], "new_standard_concepts": [1, 1]}
    # 覆盖率仍在明显提升 → 不饱和
    rising = pd.DataFrame({
        **base, "coverage_gain_pp": [0.30, 0.25],
        "anchor_uncovered_pp": [10.0, 9.5],
    })
    assert saturation_pass(rising) is False
    # 全部满足 → 饱和
    ok = pd.DataFrame({
        **base, "coverage_gain_pp": [0.05, 0.04],
        "anchor_uncovered_pp": [8.00, 7.95],
    })
    assert saturation_pass(ok) is True
    # 锚点未覆盖仍在明显下降（0.5pp > 0.1pp）→ 不饱和
    anchor_falling = pd.DataFrame({
        **base, "coverage_gain_pp": [0.05, 0.04],
        "anchor_uncovered_pp": [10.0, 9.5],
    })
    assert saturation_pass(anchor_falling) is False
    # coverage 缺失（NaN）→ fail-closed 不饱和
    missing = pd.DataFrame({
        **base, "coverage_gain_pp": [np.nan, 0.04],
        "anchor_uncovered_pp": [8.0, 7.95],
    })
    assert saturation_pass(missing) is False
