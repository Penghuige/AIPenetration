"""llm client 修复回归：超时分类、4xx 不重试、响应清洗边界。

对应审查发现：requests 的 Timeout 不是内建 TimeoutError 子类且 _post_chat
先包成 RuntimeError（调用方 except TimeoutError 为死代码）；``` 全局剥离会
改写 JSON 字符串内容；成对 <think> 标签会剥坏合法 JSON。
"""
from __future__ import annotations

import json

import pytest
import requests

from src.model_platform import llm as llm_mod
from src.model_platform.llm import (
    LLMTimeout,
    OpenAICompatClient,
    extract_json_from_response,
)


# ---------------------------------------------------------------------------
# extract_json_from_response 边界
# ---------------------------------------------------------------------------

def test_fenced_json_at_both_ends_stripped():
    assert extract_json_from_response('```json\n{"a": 1}\n```') == {"a": 1}


def test_backticks_inside_string_preserved():
    # JD 常含 markdown 代码块；旧实现全局删除 ``` 会把 surface 改写。
    payload = '{"surface": "```python", "evidence": "code ```py``` here"}'
    assert extract_json_from_response(payload) == json.loads(payload)


def test_paired_think_tags_inside_string_preserved():
    # 旧实现的成对 <think>...</think> 剥离会把两个字段整个吞掉。
    payload = '{"a": "<think>", "b": "</think>"}'
    assert extract_json_from_response(payload) == json.loads(payload)


def test_leading_think_segment_stripped():
    assert extract_json_from_response(
        '<think>推理内容</think>\n{"a": 1}'
    ) == {"a": 1}


def test_trailing_close_think_after_json_stripped():
    assert extract_json_from_response('{"a": 1}\n</think>') == {"a": 1}


def test_first_balanced_object_recovered_from_trailing_text():
    # 贪婪 \{.*\} 在尾随文字含 } 时无法回收首个 JSON。
    assert extract_json_from_response(
        '{"a": 1} explanation with } brace'
    ) == {"a": 1}


def test_garbage_returns_none():
    assert extract_json_from_response("no json at all") is None


# ---------------------------------------------------------------------------
# _post_chat：超时分类与 4xx 不重试
# ---------------------------------------------------------------------------

def _client(retry: int = 2) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="http://localhost:9/v1", api_key="EMPTY", retry=retry
    )


class _FakeResp:
    def __init__(self, status: int):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code}", response=self
            )

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def test_read_timeout_raises_llmtimeout_subclass_of_builtin(monkeypatch):
    calls = []

    def fake_post(*a, **kw):
        calls.append(1)
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    with pytest.raises(LLMTimeout) as exc_info:
        _client(retry=2)._post_chat({"model": "m"})
    # 调用方 except TimeoutError 必须能接住（旧实现抛 RuntimeError）。
    assert isinstance(exc_info.value, TimeoutError)
    assert len(calls) == 2  # retry=2 次后放弃


def test_http_4xx_fails_fast_without_retry(monkeypatch):
    calls = []

    def fake_post(*a, **kw):
        calls.append(1)
        return _FakeResp(400)

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    with pytest.raises(RuntimeError, match="400"):
        _client(retry=3)._post_chat({"model": "m"})
    assert len(calls) == 1  # 确定性失败不重试


def test_http_429_is_retried(monkeypatch):
    # 429 是限流（暂态），重试是正确行为；归入 4xx 快速失败会误伤。
    seq = [_FakeResp(429), _FakeResp(200)]

    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **kw: seq.pop(0))
    out = _client(retry=2)._post_chat({"model": "m"})
    assert out["choices"][0]["message"]["content"] == "ok"
    assert seq == []


def test_transient_connection_error_retries_then_succeeds(monkeypatch):
    seq = [
        requests.exceptions.ConnectionError("reset"),
        requests.exceptions.ConnectionError("reset"),
        _FakeResp(200),
    ]

    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **kw: seq.pop(0))
    out = _client(retry=3)._post_chat({"model": "m"})
    assert out["choices"][0]["message"]["content"] == "ok"
    assert seq == []
