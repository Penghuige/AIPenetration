"""discovery_extract 断点容错回归。

对应审查发现：强杀进程在 flush 中途留下半行 JSON 时，resume 读取无容错，
每次重启都 JSONDecodeError——断点文件反而成为重跑障碍。
"""
from __future__ import annotations

import json

import pytest

from src.ai_penetration.panel_v2.discovery_extract import (
    _load_existing_records,
)


def _line(rec: dict) -> str:
    return json.dumps(rec, ensure_ascii=False)


def test_missing_file_returns_empty(tmp_path):
    assert _load_existing_records(tmp_path / "nope.jsonl") == []


def test_torn_tail_line_dropped(tmp_path):
    p = tmp_path / "chunk_results.jsonl"
    good = {"record_key": "a", "status": "success"}
    p.write_text(
        _line(good) + "\n" + '{"record_key": "b", "status": "suc',
        encoding="utf-8",
    )
    out = _load_existing_records(p)
    assert out == [good]


def test_corrupt_middle_line_raises(tmp_path):
    p = tmp_path / "chunk_results.jsonl"
    good = {"record_key": "a", "status": "success"}
    p.write_text(
        _line(good) + "\n" + '{"broken' + "\n" + _line(good),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="第 2 行"):
        _load_existing_records(p)


def test_invalid_utf8_tail_recovered(tmp_path):
    p = tmp_path / "chunk_results.jsonl"
    good = {"record_key": "a", "status": "success"}
    p.write_bytes(
        (_line(good) + "\n").encode("utf-8") + b'{"k": "\xff\xfe'
    )
    out = _load_existing_records(p)
    assert out == [good]


def test_torn_tail_truncated_for_clean_append(tmp_path):
    # 撕裂行只被"读时忽略"而不截断文件的话，resume 的第一笔 append 会粘在
    # 撕裂片段后生成新的粘结行，第二次中断后断点文件即不可用。
    p = tmp_path / "chunk_results.jsonl"
    good = {"record_key": "a", "status": "success"}
    p.write_text(_line(good) + "\n" + '{"torn', encoding="utf-8")
    _load_existing_records(p)
    assert p.read_text(encoding="utf-8") == _line(good) + "\n"
    # 截断后再追加应保持逐行完整
    with p.open("a", encoding="utf-8") as fh:
        fh.write(_line({"record_key": "b"}) + "\n")
    assert _load_existing_records(p) == [good, {"record_key": "b"}]


def test_record_containing_u2028_is_one_line(tmp_path):
    # json.dumps(ensure_ascii=False) 不转义 U+2028/U+2029/\x85；
    # 按 str.splitlines() 读会把单条记录切成两行误报损坏。
    p = tmp_path / "chunk_results.jsonl"
    rec = {"record_key": "a", "evidence": "第一段 第二段"}
    p.write_text(_line(rec) + "\n", encoding="utf-8")
    assert _load_existing_records(p) == [rec]
