"""岗位名标准化模块。

将 distinct position 通过 LLM 标准化为 {职业名, 职业大类}，
供渗透率计算按职业与职业大类聚合。
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.model_platform.llm import LLMClient

logger = logging.getLogger("ai_penetration.standardize")

STANDARDIZE_SYSTEM_PROMPT = """/no_think
你是职业分类专家。请把招聘岗位名称标准化为《职业大典》的职业名，并归入职业大类。

规则：
1. 岗位名规范化：去掉公司名、地点、薪资、编号等噪声，保留职业核心（如"高级大模型算法工程师(广州)"→"大模型算法工程师"）。
2. 若岗位对应大典已有职业，用规范名称；若是新兴职业，用行业通用名称。
3. 从以下职业大类中选择一个：IT/软件、AI/人工智能、生产制造、销售/营销、客服/行政、物流/供应链、建筑/工程、医疗/医药、教育/培训、金融/财务、新媒体/电商、新能源/环保、其他。
4. 岗位名过于笼统无法判断时，occupation_name 留空字符串。

只能输出一行合法 JSON：{"occupation_name":"职业名","occupation_category":"职业大类"}"""


def build_standardize_prompt(positions: list[str]) -> str:
    """构建一批岗位名的标准化 user prompt。

    Args:
        positions: 去重后的岗位名列表（建议 ≤20 个）。

    Returns:
        user prompt 字符串。
    """
    lines = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(positions))
    return (
        f"请标准化以下 {len(positions)} 个岗位名称：\n\n{lines}\n\n"
        '对每个岗位输出 JSON 数组，每项形如 '
        '{"occupation_name":"职业名","occupation_category":"职业大类"}。'
        "职业大类必须从给定选项中选择。"
    )


def parse_standardize_result(text: str) -> dict:
    """解析单条 LLM 标准化输出。

    Args:
        text: LLM 返回的 JSON 文本（可能含代码块包裹）。

    Returns:
        {"occupation_name": str, "occupation_category": str}；
        解析失败时两字段均为空字符串。
    """
    cleaned = re.sub(r"```json|```", "", text or "").strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        # 尝试提取第一个 JSON 对象
        m = re.search(r"\{[^{}]*\}", cleaned)
        if not m:
            return {"occupation_name": "", "occupation_category": ""}
        try:
            parsed = json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            return {"occupation_name": "", "occupation_category": ""}
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    return {
        "occupation_name": str(parsed.get("occupation_name", "")).strip(),
        "occupation_category": str(parsed.get("occupation_category", "")).strip(),
    }


def standardize_positions(
    position_stats: pd.DataFrame,
    client: LLMClient,
    batch_size: int = 20,
    max_workers: int = 8,
) -> dict[str, dict]:
    """批量标准化 distinct position。

    Args:
        position_stats: Task 1 输出的统计表（含 position 列）。
        client: LLM 客户端。
        batch_size: 每个 prompt 处理的岗位数。
        max_workers: 并发数。

    Returns:
        position -> {"occupation_name": str, "occupation_category": str} 映射。
        标准化失败/空结果时，occupation_name 回退为原岗位名，category 为 "其他"。
    """
    positions = sorted(position_stats["position"].unique().tolist())
    batches = [
        positions[i : i + batch_size] for i in range(0, len(positions), batch_size)
    ]
    result_map: dict[str, dict] = {}

    def _process(batch: list[str]) -> list[dict]:
        try:
            prompt = build_standardize_prompt(batch)
            resp = client.complete_text(
                system_prompt=STANDARDIZE_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_output_tokens=1024,
                temperature=0.0,
            )
            # 尝试解析为数组或单个对象
            parsed = _parse_batch_response(resp)
            return parsed
        except Exception as exc:  # noqa: BLE001
            logger.warning("批次标准化失败: %s", exc)
            return [{} for _ in batch]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, b): b for b in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                results = future.result()
            except Exception:  # noqa: BLE001
                results = [{} for _ in batch]
            # LLM 可能因 max_tokens 截断或解析失败返回少于批大小的条目，
            # 缺失岗位补空结果（回退原岗位名），避免静默丢项扭曲渗透率分母
            if len(results) != len(batch):
                logger.warning(
                    "批次结果数 %d != 岗位数 %d，缺失部分回退原岗位名",
                    len(results), len(batch),
                )
                results = list(results) + [{}] * (len(batch) - len(results))
            for pos, item in zip(batch, results):
                occ = str(item.get("occupation_name", "")).strip()
                cat = str(item.get("occupation_category", "")).strip()
                result_map[pos] = {
                    "occupation_name": occ or pos,
                    "occupation_category": cat or "其他",
                }
    logger.info("标准化完成: %d 个岗位", len(result_map))
    return result_map


def _parse_batch_response(text: str) -> list[dict]:
    """解析 LLM 批量输出（JSON 数组或单对象）。"""
    cleaned = re.sub(r"```json|```", "", text or "").strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return [{}]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [
            p if isinstance(p, dict) else {}
            for p in parsed
        ]
    return [{}]
