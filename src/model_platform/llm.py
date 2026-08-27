"""统一 LLM 调用入口。

默认后端为 OpenAI-compatible API（本机 WSL vLLM 或其他兼容服务），
地址 / 模型 / key 由 `config/model_runtime.yaml` 提供，环境变量可覆盖。
不加载任何本地模型权重。

环境变量覆盖:
    AIPEN_LLM_BASE_URL — OpenAI-compatible API 地址
    AIPEN_LLM_MODEL    — 模型名
    AIPEN_LLM_API_KEY  — API key
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


def extract_json_from_response(text: str) -> dict[str, Any] | list[Any] | None:
    """从 LLM 回复中提取 JSON 对象或数组。

    兼容 ```json 代码块包裹和前后附带说明文字的情况。

    Args:
        text: LLM 原始回复文本。

    Returns:
        解析出的 dict/list；无法解析时返回 None。
    """
    cleaned = re.sub(r"```(?:json)?|```", "", (text or "").strip())
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    # 尝试提取首个平衡的 {...} 或 [...]
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, (dict, list)):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                continue
    return None


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
        """发送一次 chat completions 请求，带指数退避重试。"""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.retry) + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "LLM 请求失败 attempt=%s/%s error=%s", attempt, self.retry, exc
                )
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
    ) -> str:
        """生成一段文本回复。

        Args:
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。
            strength: 后端强度偏好；轻量实现仅保留参数兼容，不做路由。
            max_output_tokens: 最大输出 token 数。
            reasoning_effort: 推理努力程度；轻量实现忽略。
            temperature: 采样温度。

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
    )
