"""AI 职业渗透率分析单元测试。"""
import pandas as pd

from src.ai_penetration.compute_penetration import (
    identify_ai_new_occupations,
    compute_penetration,
    compute_quarterly_penetration,
)
from src.ai_penetration.load_guangdong import _build_position_stats, GD_SHARDS


def test_gd_shards_covers_21_cities():
    assert len(GD_SHARDS) == 21
    assert "广州市" in GD_SHARDS
    assert "深圳市" in GD_SHARDS


def test_build_position_stats_groups_by_position_year_quarter():
    rows = [
        ("Java开发工程师", "2021-03-15"),
        ("Java开发工程师", "2021-06-10"),
        ("Java开发工程师", "2022-01-20"),
        ("大模型算法工程师", "2022-12-05"),
        ("大模型算法工程师", "2023-02-01"),
        ("大模型算法工程师", "2023-03-01"),
        ("大模型算法工程师", "2023-05-01"),
        ("", "2022-01-01"),  # 空岗位名应被忽略
    ]
    df = _build_position_stats(rows)
    assert sorted(df["position"].unique().tolist()) == sorted(
        ["Java开发工程师", "大模型算法工程师"]
    )
    # Java: 2021Q1=1, 2021Q2=1, 2022Q1=1
    java = df[df["position"] == "Java开发工程师"]
    assert len(java) == 3
    assert java[java["year"] == 2021].shape[0] == 2
    assert java[java["quarter"] == 1].shape[0] == 2
    # 空岗位名不在结果中
    assert "" not in df["position"].tolist()


from src.ai_penetration.standardize_jobs import (
    build_standardize_prompt,
    parse_standardize_result,
)


def test_build_standardize_prompt_contains_positions():
    prompt = build_standardize_prompt(["大模型算法工程师", "Java开发工程师"])
    assert "大模型算法工程师" in prompt
    assert "职业大类" in prompt


def test_parse_standardize_result_extracts_fields():
    text = '{"occupation_name":"大模型算法工程师","occupation_category":"IT/软件"}'
    parsed = parse_standardize_result(text)
    assert parsed["occupation_name"] == "大模型算法工程师"
    assert parsed["occupation_category"] == "IT/软件"


def test_parse_standardize_result_empty_name_falls_back():
    text = '{"occupation_name":"","occupation_category":"其他"}'
    parsed = parse_standardize_result(text)
    assert parsed["occupation_name"] == ""
    assert parsed["occupation_category"] == "其他"


def _make_stats():
    # position: (year, quarter, count)
    return pd.DataFrame([
        {"position": "Java开发工程师", "year": 2021, "quarter": 1, "count": 10},
        {"position": "Java开发工程师", "year": 2023, "quarter": 1, "count": 8},
        {"position": "大模型算法工程师", "year": 2022, "quarter": 4, "count": 3},
        {"position": "大模型算法工程师", "year": 2023, "quarter": 1, "count": 6},
        {"position": "大模型算法工程师", "year": 2023, "quarter": 2, "count": 4},
        {"position": "提示词工程师", "year": 2023, "quarter": 1, "count": 7},
    ])


def test_identify_ai_new_occupations():
    stats = _make_stats()
    occ_map = {
        "Java开发工程师": {"occupation_name": "Java开发工程师", "occupation_category": "IT/软件"},
        "大模型算法工程师": {"occupation_name": "大模型算法工程师", "occupation_category": "AI/人工智能"},
        "提示词工程师": {"occupation_name": "提示词工程师", "occupation_category": "AI/人工智能"},
    }
    ai_new = identify_ai_new_occupations(stats, occ_map, min_post_ai_count=5)
    assert "大模型算法工程师" in ai_new      # 2021 前零出现 + 2022+ 共 13 次
    assert "提示词工程师" in ai_new           # 2021 前零出现 + 2023 共 7 次
    assert "Java开发工程师" not in ai_new     # 2021 已出现，非 AI 新职业


def test_identify_ai_new_requires_min_count():
    stats = _make_stats()
    occ_map = {
        "大模型算法工程师": {"occupation_name": "大模型算法工程师", "occupation_category": "AI"},
    }
    # 大模型算法工程师 post-2022 共 13 次，阈值 14 时不计入
    ai_new = identify_ai_new_occupations(stats, occ_map, min_post_ai_count=14)
    assert "大模型算法工程师" not in ai_new


def test_compute_penetration():
    stats = _make_stats()
    occ_map = {
        "Java开发工程师": {"occupation_name": "Java开发工程师", "occupation_category": "IT/软件"},
        "大模型算法工程师": {"occupation_name": "大模型算法工程师", "occupation_category": "AI/人工智能"},
        "提示词工程师": {"occupation_name": "提示词工程师", "occupation_category": "AI/人工智能"},
    }
    ai_new = {"大模型算法工程师", "提示词工程师"}
    pen = compute_penetration(stats, occ_map, ai_new)
    # 2023 年总职业 3 个（Java/大模型/提示词），AI 新职业 2 个
    row23 = pen[pen["year"] == 2023]
    total_all = row23["total_occupations"].sum()
    ai_all = row23["ai_occupations"].sum()
    assert total_all == 3
    assert ai_all == 2
    assert abs(ai_all / total_all - 2 / 3) < 1e-9


def test_compute_quarterly_penetration():
    stats = _make_stats()
    occ_map = {
        "大模型算法工程师": {"occupation_name": "大模型算法工程师", "occupation_category": "AI"},
        "提示词工程师": {"occupation_name": "提示词工程师", "occupation_category": "AI"},
    }
    ai_new = {"大模型算法工程师", "提示词工程师"}
    q = compute_quarterly_penetration(stats, occ_map, ai_new)
    assert "quarter" in q.columns
    # 2023Q1 总职业 2 个，AI 2 个
    row = q[(q["year"] == 2023) & (q["quarter"] == 1)]
    assert row["total_occupations"].iloc[0] == 2


from pathlib import Path
from src.ai_penetration.report import build_matrix_csv, generate_report


def test_build_matrix_csv(tmp_path):
    stats = _make_stats()
    occ_map = {
        "Java开发工程师": {"occupation_name": "Java开发工程师", "occupation_category": "IT/软件"},
        "大模型算法工程师": {"occupation_name": "大模型算法工程师", "occupation_category": "AI"},
        "提示词工程师": {"occupation_name": "提示词工程师", "occupation_category": "AI"},
    }
    ai_new = {"大模型算法工程师", "提示词工程师"}
    out = Path(tmp_path) / "matrix.csv"
    result = build_matrix_csv(stats, occ_map, ai_new, out)
    assert result.exists()
    text = result.read_text(encoding="utf-8-sig")
    assert "occupation" in text
    assert "is_ai" in text
    assert "大模型算法工程师" in text


def test_generate_report(tmp_path):
    """验证 generate_report 总体渗透率为 sum(ai)/sum(total) 而非 ratio 求和。"""
    pen_yearly = pd.DataFrame([
        {"year": 2023, "category": "IT/软件", "total_occupations": 10, "ai_occupations": 2, "penetration": 0.2},
        {"year": 2023, "category": "AI/人工智能", "total_occupations": 5, "ai_occupations": 3, "penetration": 0.6},
    ])
    pen_quarterly = pd.DataFrame([
        {"year": 2023, "quarter": 1, "category": "IT/软件", "total_occupations": 8, "ai_occupations": 1, "penetration": 0.125},
    ])
    ai_new = {"大模型算法工程师", "提示词工程师"}
    matrix_path = Path(tmp_path) / "matrix.csv"
    matrix_path.write_text("occupation,year_2023,is_ai\n", encoding="utf-8-sig")
    output_dir = Path(tmp_path) / "reports"

    report_path = generate_report(
        pen_yearly=pen_yearly,
        pen_quarterly=pen_quarterly,
        ai_new=ai_new,
        matrix_path=matrix_path,
        output_dir=output_dir,
        timestamp="20260806_test",
    )

    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "总体渗透率" in text

    # 从 report 表格中解析 2023 年总体渗透率：sum(ai)/sum(total) = 5/15 ≈ 0.3333
    # 不应是 sum(ratio) = 0.2+0.6 = 0.8
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("| 2023"):
            parts = [p.strip() for p in line.split("|")]
            # 格式：| 年份 | 总职业数 | AI新职业数 | 渗透率 |
            reported_pen = float(parts[4])
            expected = round(5 / 15, 4)  # 报告使用 .4f 格式化
            assert abs(reported_pen - expected) < 1e-9, (
                f"总体渗透率应为 5/15={5/15:.4f}，实际 {reported_pen:.4f}"
            )
            assert reported_pen <= 1.0, f"渗透率 {reported_pen} 应 ≤1"
            break


from src.ai_penetration.stream_penetration import (
    _checkpoint_path,
    _load_checkpoint,
    _save_checkpoint,
)


def test_stream_checkpoint_roundtrip_and_corrupt_tolerant():
    """验证逐年断点保存/加载往返一致，且损坏文件容错。"""
    shard = "job_p_test_checkpoint"
    cp = _checkpoint_path(shard)
    try:
        stats = {
            2014: {"total": 100, "keyword_ai": 1, "aiskill_ai": 2, "combined_ai": 3},
            2015: {"total": 200, "keyword_ai": 2, "aiskill_ai": 3, "combined_ai": 5},
        }
        _save_checkpoint(shard, stats)
        loaded = _load_checkpoint(shard)
        assert loaded == stats
        assert all(isinstance(k, int) for k in loaded)
        # 损坏文件容错返回空
        cp.write_text("{bad json", encoding="utf-8")
        assert _load_checkpoint(shard) == {}
    finally:
        if cp.exists():
            cp.unlink()


from src.ai_penetration.penetration_detail import (
    _agg_key,
    _bucket,
    _parse_period,
    build_detail_dataframe,
    _industry_label,
)


def test_parse_period_handles_clean_and_dirty_dates():
    """验证日期解析：正常月份、半年/季度、脏月份、无法解析。"""
    assert _parse_period("2024-03-15") == (2024, 1, 1)
    assert _parse_period("2024-07-01") == (2024, 2, 3)
    assert _parse_period("2024-12-31") == (2024, 2, 4)
    assert _parse_period("2024-13-01") == (2024, 0, 0)  # 脏月份仅保留年份
    assert _parse_period("abc") == (0, 0, 0)


def test_industry_label_defaults_unknown():
    """验证行业标签归类到大类层面，空值归为未知。"""
    assert _industry_label("C34") == "C34通用设备制造业"
    assert _industry_label("") == "未知"
    assert _industry_label(None) == "未知"


def test_detail_dataframe_three_granularities():
    """验证宽表构建：三粒度、期间标签、渗透度计算。"""
    agg = {}
    b = _bucket(agg, _agg_key("年度", 2024, 0, 0, "广州市", "C34通用设备制造业"))
    b["total"] += 600
    b["keyword_ai"] += 5
    b = _bucket(agg, _agg_key("季度", 2024, 0, 1, "广州市", "C34通用设备制造业"))
    b["total"] += 600
    b["combined_ai"] += 2
    df = build_detail_dataframe(agg)
    assert len(df) == 2
    q = df[df["时间粒度"] == "季度"].iloc[0]
    assert q["期间"] == "Q1"
    y = df[df["时间粒度"] == "年度"].iloc[0]
    assert abs(y["AI岗位渗透度_关键词"] - 5 / 600) < 1e-9


from src.ai_penetration.industry_classification import classify_industry


def test_classify_industry_major_class_level():
    """验证 GB/T 编码归类到大类层面（门类字母 + 大类名）。"""
    assert classify_industry("C22") == "C22造纸和纸制品业"
    assert classify_industry("C2211") == "C22造纸和纸制品业"  # 深层编码归到大类
    assert classify_industry("C1492") == "C14食品制造业"
    assert classify_industry("I65") == "I65软件和信息技术服务业"
    assert classify_industry("L7212") == "L72商务服务业"
    assert classify_industry("A") == "A农、林、牧、渔业"  # 仅门类
    assert classify_industry("") == "未知"
    assert classify_industry(None) == "未知"


def test_detail_dataframe_threshold_excludes_small_cells():
    """验证样本量阈值：总发布数低于阈值时剔除该组合。"""
    from src.ai_penetration.penetration_detail import MIN_TOTAL_THRESHOLD

    agg = {}
    # 低于阈值的组合应被剔除
    b = _bucket(agg, _agg_key("年度", 2024, 0, 0, "广州市", "C16烟草制品业"))
    b["total"] += 100  # < MIN_TOTAL_THRESHOLD
    b["combined_ai"] += 1
    # 高于阈值的组合保留
    b = _bucket(agg, _agg_key("年度", 2024, 0, 0, "广州市", "C39计算机、通信和其他电子设备制造业"))
    b["total"] += MIN_TOTAL_THRESHOLD
    b["combined_ai"] += 10
    df = build_detail_dataframe(agg)
    assert len(df) == 1
    assert df.iloc[0]["行业"] == "C39计算机、通信和其他电子设备制造业"
    assert df.iloc[0]["AI岗位渗透度_复合方法"] == 10 / MIN_TOTAL_THRESHOLD
