"""统一 LLM 调用入口。

默认后端为 OpenAI-compatible API（本机 WSL vLLM 或其他兼容服务），
地址 / 模型 / key 由 `config/model_runtime.yaml` 提供，环境变量可覆盖。
不加载任何本地模型权重。

环境变量覆盖:
    AIPEN_LLM_BASE_URL — OpenAI-compatible API 地址
    AIPEN_LLM_MODEL    — 模型名
    AIPEN_LLM_API_KEY  — API key
    AIPEN_LLM_TIMEOUT  — 单次请求超时秒数（默认 600）
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import requests

from config.paths import load_config_yaml

logger = logging.getLogger(__name__)

# vLLM 0.20 起移除了 guided_json / guided_regex / guided_choice / guided_grammar
# 请求参数，受控解码改为顶层 structured_outputs={"json": <schema>}。旧字段仍能
# 通过请求体校验，但会被**静默忽略**并退化为无约束生成——模型照常返回 200，
# 调用方却拿到不合 schema 的输出。调用方按策略语义传 guided_json（与
# config/model_config_v1.yaml 的冻结项同名），在此适配为服务端接受的字段。
_GUIDED_DECODING_KEYS = (
    "guided_json", "guided_regex", "guided_choice", "guided_grammar",
)


def _adapt_guided_decoding(extra: dict[str, Any]) -> dict[str, Any]:
    """把旧式 guided_* 字段适配为 vLLM 当前的 structured_outputs 字段。

    Args:
        extra: 调用方传入的 extra_payload。

    Returns:
        适配后的请求体字段；不含 guided_* 时返回原字典。

    Raises:
        ValueError: 同时指定 guided_* 与 structured_outputs，或使用了在
            vLLM 0.20 已无对应字段的受控解码类型（宁可显式失败，也不静默降级）。
    """
    present = [k for k in _GUIDED_DECODING_KEYS if extra.get(k) is not None]
    if not present:
        return extra
    if len(present) > 1:
        raise ValueError(
            "extra_payload 一次只能指定一种受控解码: " + ", ".join(present)
        )
    if "structured_outputs" in extra:
        raise ValueError(
            f"extra_payload 不能同时指定 {present[0]} 与 structured_outputs"
        )
    if present[0] != "guided_json":
        raise ValueError(
            f"{present[0]} 在 vLLM 0.20 已无对应字段，请改用 guided_json"
        )
    adapted = {
        k: v for k, v in extra.items() if k not in _GUIDED_DECODING_KEYS
    }
    adapted["structured_outputs"] = {"json": extra["guided_json"]}
    return adapted


class LLMTimeout(TimeoutError):
    """LLM 请求超时（连接/读取超时重试耗尽后抛出）。

    继承内建 ``TimeoutError``，使 discovery_extract / model_benchmark 等调用方
    既有的 ``except TimeoutError`` 分流真正生效——requests 的
    ``Timeout`` 体系不是内建 ``TimeoutError`` 的子类，旧实现把它包成
    ``RuntimeError`` 抛出，超时全部被误计入 runtime_failed。
    """


class LLMClient(Protocol):
    """统一 LLM client 协议。"""

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> str:
        """生成文本。"""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """生成并解析 JSON。"""

    def batch_complete_text(
        self,
        prompt_pairs: Sequence[tuple[str, str]],
        *,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """批量生成文本。"""


def _strip_llm_wrappers(text: str) -> str:
    """剥离模型输出首尾的推理标签与 markdown 围栏，不动字符串内部内容。

    只处理三种形态：开头的完整 ``<think>…</think>`` 推理段（或裸闭合标签）、
    结尾的 ``</think>`` 残留、以及**首尾成对**的 ``` 围栏。全局删除会改写
    JSON 字符串里恰好含这些字面量的合法内容（实测 JD 常含 ``` 代码块、
    讨论 LLM 标签的正文可含成对 think 标签），一律不做。

    Args:
        text: LLM 原始回复文本。

    Returns:
        去掉首尾包装后的文本。
    """
    s = (text or "").strip()
    m = re.match(
        r"^<think(?:ing)?>.*?</think(?:ing)?>\s*", s, flags=re.DOTALL
    )
    if m is None:
        m = re.match(r"^</think(?:ing)?>\s*", s)
    if m is not None:
        s = s[m.end():]
    s = re.sub(r"\s*</think(?:ing)?>\s*$", "", s)
    m = re.match(r"^```(?:json)?\s*", s)
    if m is not None and s.endswith("```"):
        s = s[m.end():]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return s


def _first_balanced_json(s: str) -> dict[str, Any] | list[Any] | None:
    """从文本中提取首个括号平衡、可解析的 JSON 对象/数组。

    比 ``re.search(r"\\{.*\\}")`` 的贪婪匹配多回收一类输入：JSON 之后跟有
    含 ``}`` 的尾随文字时，贪婪匹配吞到最后的 ``}`` 导致整段不可解析；
    平衡扫描（跳过字符串字面量）只取首个完整块。

    Args:
        s: 已剥包装的文本。

    Returns:
        解析结果；不存在可解析块时返回 None。
    """
    for opener, closer in (("{", "}"), ("[", "]")):
        i = s.find(opener)
        if i < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(s)):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(s[i:j + 1])
                    except (json.JSONDecodeError, TypeError):
                        break
                    if isinstance(parsed, (dict, list)):
                        return parsed
                    break
        # 当前 opener 找不到平衡块时继续尝试另一种（如文本只有数组）。
    return None


def extract_json_from_response(text: str) -> dict[str, Any] | list[Any] | None:
    """从 LLM 回复中提取 JSON 对象或数组。

    兼容 ```json 代码块包裹、前后推理标签和首尾附带说明文字的情况；
    字符串内部出现的围栏/标签字面量原样保留。

    Args:
        text: LLM 原始回复文本。

    Returns:
        解析出的 dict/list；无法解析时返回 None。
    """
    cleaned = _strip_llm_wrappers(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return _first_balanced_json(cleaned)


@dataclass
class OpenAICompatClient:
    """OpenAI-compatible chat completions HTTP client。"""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen3-8B"
    retry: int = 2
    timeout: float = 600.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送一次 chat completions 请求，按失败类别分流重试。

        - 连接错误 / 5xx：指数退避重试，耗尽后 RuntimeError；
        - 4xx：确定性失败（如 prompt+max_tokens 超 vLLM 上限的 400），
          立即 RuntimeError 不重试——重试只会原样失败并白烧超时预算；
        - 超时：重试耗尽后抛 :class:`LLMTimeout`（内建 TimeoutError 子类），
          使调用方 ``except TimeoutError`` 的超时分流真正生效。

        Args:
            payload: chat completions 请求体。

        Returns:
            服务端 JSON 响应 dict。

        Raises:
            LLMTimeout: 超时重试耗尽。
            RuntimeError: 其余失败重试耗尽，或 4xx 立即失败。
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.retry) + 1):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout as exc:
                last_error = exc
                logger.warning(
                    "LLM 请求超时 attempt=%s/%s timeout=%ss error=%s",
                    attempt, self.retry, self.timeout, exc,
                )
                if attempt >= max(1, self.retry):
                    raise LLMTimeout(
                        f"LLM 请求超时（{self.timeout}s，重试 "
                        f"{self.retry} 次后放弃）: {exc}"
                    ) from exc
            except requests.exceptions.HTTPError as exc:
                status = getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                if status is not None and 400 <= status < 500:
                    raise RuntimeError(
                        f"LLM 请求被拒绝（HTTP {status}，确定性失败不重试）: "
                        f"{exc}"
                    ) from exc
                last_error = exc
                logger.warning(
                    "LLM 请求失败 attempt=%s/%s error=%s",
                    attempt, self.retry, exc,
                )
                if attempt >= max(1, self.retry):
                    raise RuntimeError(
                        f"LLM 请求失败: {last_error}"
                    ) from last_error
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "LLM 请求失败 attempt=%s/%s error=%s",
                    attempt, self.retry, exc,
                )
                if attempt >= max(1, self.retry):
                    raise RuntimeError(
                        f"LLM 请求失败: {last_error}"
                    ) from last_error
            if attempt < self.retry:
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"LLM 请求失败: {last_error}") from last_error

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> str:
        """生成一段文本回复。

        Args:
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。
            strength: 后端强度偏好；轻量实现仅保留参数兼容，不做路由。
            max_output_tokens: 最大输出 token 数。
            reasoning_effort: 推理努力程度；轻量实现忽略。
            temperature: 采样温度。
            extra_payload: 追加到请求体的额外字段（如
                ``{"chat_template_kwargs": {"enable_thinking": false}}``）。
                其中 ``guided_json`` 会按当前 vLLM 版本适配为
                ``structured_outputs``，见 :func:`_adapt_guided_decoding`。

        Returns:
            助手回复文本。

        Raises:
            RuntimeError: 重试后仍请求失败或响应结构异常。
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = int(max_output_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if extra_payload:
            payload.update(_adapt_guided_decoding(extra_payload))
        response = self._post_chat(payload)
        try:
            return str(response["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM 响应结构异常: {response}") from exc

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """生成并解析 JSON 回复。

        Raises:
            ValueError: LLM 返回不是合法 JSON。
        """
        text = self.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            strength=strength,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            extra_payload=extra_payload,
        )
        parsed = extract_json_from_response(text)
        if parsed is None:
            raise ValueError("LLM 返回不是合法 JSON")
        return parsed

    def batch_complete_text(
        self,
        prompt_pairs: Sequence[tuple[str, str]],
        *,
        strength: str = "cheap",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """顺序批量生成文本。"""
        return [
            self.complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                strength=strength,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            )
            for system_prompt, user_prompt in prompt_pairs
        ]


def create_llm_client(
    backend: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> LLMClient:
    """创建统一 LLM client。

    Args:
        backend: 后端名称；当前仅支持 ``openai_compat``（默认）。
        config_path: 可选覆盖 `config/model_runtime.yaml` 路径。

    Returns:
        LLMClient: 配置好的 client 实例。

    Raises:
        ValueError: 指定了不支持的后端名称。
    """
    import os

    del backend  # 轻量实现只有一种后端，保留参数以维持调用方兼容
    cfg = (
        load_config_yaml(str(config_path))
        if config_path is not None
        else load_config_yaml("model_runtime.yaml")
    )

    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    return OpenAICompatClient(
        base_url=os.getenv("AIPEN_LLM_BASE_URL") or str(llm_cfg.get("base_url", "http://localhost:8000/v1")),
        api_key=os.getenv("AIPEN_LLM_API_KEY") or str(llm_cfg.get("api_key", "EMPTY")),
        model=os.getenv("AIPEN_LLM_MODEL") or str(llm_cfg.get("model", "Qwen3-8B")),
        retry=int(os.getenv("AIPEN_LLM_RETRY") or llm_cfg.get("retry", 2)),
        timeout=float(
            os.getenv("AIPEN_LLM_TIMEOUT")
            or llm_cfg.get("timeout", 600.0)
        ),
    )
