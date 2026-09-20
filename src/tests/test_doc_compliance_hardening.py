"""文档一致性与复现加固的离线回归测试。"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.paths import get_project_paths
from src.ai_penetration.common import setup_logging
from src.ai_penetration.panel_v2.anchors import ANCHOR_VERSIONS
from src.ai_penetration.panel_v2.quality import _rerun_drift, _sha_stats
from src.ai_penetration.panel_v2.relevance import MIN_FIT_N
from src.ai_penetration.panel_v2.reproducibility import write_run_manifest
from src.ai_penetration.panel_v2.scoring import THRESHOLDS, VERSIONS, WINDOWS


def test_rerun_drift_blocks_key_stat_changes():
    """指南 §17.6.10 的关键统计漂移必须可被识别。"""
    prev = {
        "checksum_schema_version": 2,
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
    added: list[logging.Handler] = []
    try:
        for handler in old_handlers:
            root.removeHandler(handler)

        setup_logging()
        added.extend(h for h in root.handlers if h not in old_handlers)
        log_path = tmp_path / "nested" / "run.log"
        setup_logging(log_path)
        added.extend(
            h for h in root.handlers if h not in old_handlers and h not in added
        )
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
            if handler not in old_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in old_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(old_level)


def _registered_inline_list(text: str, key: str) -> tuple[str, ...]:
    """读取 panel_v2.yaml 中简单的行内列表，避免测试依赖 PyYAML。"""
    match = re.search(rf"^\s*{re.escape(key)}:\s*\[([^\]]*)\]", text, re.MULTILINE)
    assert match is not None, f"panel_v2.yaml 缺少 {key} 登记"
    return tuple(x.strip() for x in match.group(1).split(",") if x.strip())


def test_registered_panel_config_matches_runtime_constants():
    """口径登记文件不得与真正执行的常量静默漂移。"""
    text = (get_project_paths().config_dir / "panel_v2.yaml").read_text(encoding="utf-8")
    anchors = _registered_inline_list(text, "anchors")
    windows = _registered_inline_list(text, "windows")
    thresholds = tuple(float(x) for x in _registered_inline_list(text, "thresholds"))

    assert anchors == tuple(ANCHOR_VERSIONS)
    assert VERSIONS == tuple(ANCHOR_VERSIONS)
    assert windows == WINDOWS
    assert thresholds == THRESHOLDS
    assert re.search(rf"^\s*min_fit_n:\s*{MIN_FIT_N}\s*$", text, re.MULTILINE)
    assert re.search(
        r"^\s*primary_score:\s*aijob_main_annual_raw_005(?:\s|#|$)",
        text,
        re.MULTILINE,
    )


def test_run_manifest_binds_inputs_outputs_and_hashes(tmp_path: Path):
    """指南 §4.2 总账必须记录输入/输出哈希、行数与代码版本。"""
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    out = tmp_path / "output.csv"
    out.write_text("id,value\n1,x\n2,y\n", encoding="utf-8")
    release = tmp_path / "release"

    target = write_run_manifest(
        release,
        run_id="test_run",
        started_at=datetime(2026, 9, 12, 12, 0, 0),
        input_paths=[inp],
        output_paths=[out],
        dictionary_version="dict-test",
        anchor_version="main@test",
    )
    manifest = json.loads(target.read_text(encoding="utf-8"))

    assert manifest["run_id"] == "test_run"
    assert manifest["dictionary_version"] == "dict-test"
    assert manifest["anchor_version"] == "main@test"
    assert manifest["status"] == "complete"
    assert len(manifest["input_file_paths"]) == 1
    assert len(manifest["output_file_paths"]) == 1
    input_key = manifest["input_file_paths"][0]
    output_key = manifest["output_file_paths"][0]
    assert manifest["input_row_counts"][input_key] == 2
    assert manifest["output_row_counts"][output_key] == 2
    assert len(manifest["input_file_sha256"][input_key]) == 64
    assert len(manifest["output_file_sha256"][output_key]) == 64
    assert manifest["git_commit_or_code_hash"]
    assert manifest["config_file_sha256"]
