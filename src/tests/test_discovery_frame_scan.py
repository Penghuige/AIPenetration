"""discovery_frame 扫描并行化回归：纯函数语义 + 并行/串行逐值等价。

对应设计：瓶颈实测在 Python 逐行处理（游标会话持续 ClientRead），并行只
分摊 CPU 部分，查询形态不变；并行路径必须与串行路径输出完全一致。
"""
from __future__ import annotations

import datetime

from src.ai_penetration.panel_v2.discovery_frame import (
    _iter_processed,
    _process_rows,
)
from src.ai_penetration.panel_v2.anchors import match_all_versions
from src.ai_penetration.text_clean import match_from_raw, normalize_position

CFG = {
    "text_length": {"bins": [0, 10, 100], "labels": ["short", "long"]},
    "tech_flag": {"position_regex": "算法|AI"},
}
BINS = [0.0, 10.0, 100.0]
LABELS = ["short", "long"]
CITY = "测试市"


class _FakeLex:
    """extract() 恒返回固定集合的可 pickle 桩词表。"""

    def __init__(self, hits=("s1", "s2")):
        self._hits = tuple(hits)

    def extract(self, text: str) -> set:
        return set(self._hits)


def _rows():
    return [
        # (rid, platform, position, publish_time, desc, industry, size, cid)
        ("r1", "51job", "算法工程师", datetime.date(2020, 5, 1),
         "python developer with machine learning skill", "I1", "100", "c1"),
        ("r2", "boss", "数据分析师", "2023-07-01",
         "data analysis sql reporting", "I2", None, None),
        ("r3", "zhaopin", "销售", None, "sell products", "I3", "50", "c3"),
        ("r4", "liepin", "工程师", datetime.date(2026, 1, 1),
         "future job out of range", "I4", "50", "c4"),
        ("r5", None, "AI 产品经理", datetime.date(2014, 1, 1),
         "ai product design", None, "", "c5"),
    ]


def test_process_rows_semantics():
    import re

    tech = re.compile(CFG["tech_flag"]["position_regex"], re.I)
    rows, stats = _process_rows(
        _rows(), CITY, BINS, LABELS, tech, _FakeLex()
    )
    # r3（publish_time=None）与 r4（2026 越界）被年白名单过滤
    assert stats == {
        "rows_seen": 5, "rows_kept": 3, "invalid_year": 2,
    }
    assert len(rows) == 3
    first = rows[0]
    # 列序：job_id, year, industry, position_group, company_size, bin,
    #       tech_flag, platform, anchor, matched, hash, raw, company_id,
    #       city, rid
    assert first[1] == 2020
    assert first[2] == "I1"
    assert first[3] == normalize_position("算法工程师")
    assert first[4] == "100"
    assert first[6] == 1  # 岗位名含"算法"→技术岗
    assert first[7] == "51job"
    expected_flag = match_all_versions(
        match_from_raw("python developer with machine learning skill")
    )["main"].flag
    assert first[8] == int(expected_flag)
    assert first[9] == 2  # 假词表两个技能
    assert first[11] == str(match_from_raw(
        "python developer with machine learning skill"
    ))
    assert first[13] == CITY
    assert first[14] == "r1"
    # r5：platform=None → PLATFORM_MISSING；industry=None → INDUSTRY_MISSING
    third = rows[2]
    assert third[7] == "PLATFORM_MISSING"
    assert third[2] == "INDUSTRY_MISSING"
    assert third[1] == 2014


def test_serial_iter_processed_matches_direct():
    batches = [_rows()[:2], _rows()[2:4], _rows()[4:]]
    out = list(_iter_processed(batches, CITY, CFG, _FakeLex(), workers=1))
    rows_flat = [r for rows, _s in out for r in rows]
    stats_all = {
        k: sum(s[k] for _r, s in out)
        for k in ("rows_seen", "rows_kept", "invalid_year")
    }
    assert len(out) == 3
    assert stats_all == {
        "rows_seen": 5, "rows_kept": 3, "invalid_year": 2,
    }
    assert [r[14] for r in rows_flat] == ["r1", "r2", "r5"]


def test_parallel_matches_serial_value_for_value():
    """并行（workers=2）与串行（workers=1）输出逐值相等——部署硬门。"""
    batches = [_rows()[:1], _rows()[1:2], _rows()[2:3], _rows()[3:4],
               _rows()[4:]]
    serial = list(
        _iter_processed(batches, CITY, CFG, _FakeLex(), workers=1)
    )
    parallel = list(
        _iter_processed(batches, CITY, CFG, _FakeLex(), workers=2)
    )
    assert parallel == serial
