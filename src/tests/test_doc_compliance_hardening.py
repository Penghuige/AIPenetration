"""文档一致性加固的离线回归测试。"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.ai_penetration.common import setup_logging
from src.ai_penetration.panel_v2.quality import _rerun_drift, _sha_stats


def test_rerun_drift_blocks_key_stat_changes():
    """指南 §17.6.10 的关键统计漂移必须可被识别。"""
    prev = {
        "n_jobs": 100,
        "n_pairs": 200,
        "checksum_flags": "aaa",
        "checksum_counts": "bbb",
        "main_annual_raw_005_rate": 0.1,
        "main_annual_raw_015_rate": 0.02,
        "zero_skill_rate": 0.15,
    }
    same = dict(prev)
    assert _rerun_drift(prev, same) == []

    changed = dict(prev, n_pairs=201, main_annual_raw_005_rate=0.11)
    assert set(_rerun_drift(prev, changed)) == {
        "n_pairs", "main_annual_raw_005_rate"
    }


def test_sha_stats_is_order_sensitive_but_repeatable():
    """稳定摘要覆盖字符串键和值列，重复计算结果一致。"""
    df = pd.DataFrame({
        "skill_code": [1, 2],
        "anchor_version": ["main", "babina"],
        "n_skill": [10, 20],
    })
    cols = ["skill_code", "anchor_version", "n_skill"]
    a = _sha_stats(df, cols)
    b = _sha_stats(df.copy(), cols)
    c = _sha_stats(df.iloc[::-1].reset_index(drop=True), cols)
    assert a == b
    assert a != c


def test_setup_logging_can_add_file_after_console_only(tmp_path: Path):
    """先仅终端初始化，后续仍能补装目标文件 handler。"""
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    try:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        setup_logging()
        log_path = tmp_path / "nested" / "run.log"
        setup_logging(log_path)
        setup_logging(log_path)

        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == log_path.resolve()
        ]
        assert len(file_handlers) == 1
        assert any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in root.handlers
        )
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        for handler in old_handlers:
            root.addHandler(handler)
        root.setLevel(old_level)
