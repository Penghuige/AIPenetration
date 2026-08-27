"""LLM 生成 AI 技能别名（扩充技能词典的别名/同义词/缩写变体）。

背景：技能词典每概念一个规范词，但招聘文本中同一技能有多种写法
（深度学习/Deep Learning/DL、PyTorch/pytorch/Torch、自然语言处理/NLP）。
论文 9.9 万技能中大量是别名。为高 ωsAI 的 AI 技能生成别名可显著提升召回。

流程：
1. 加载当前 LLM 技能词典 + ωsAI 分数（取 AI 相关技能）
2. 分批调用 LLM 生成每个技能的别名/同义词/缩写/大小写变体
3. 合并别名回词典（dicts/ai_skill_terms_llm.txt 追加）
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from config.paths import get_project_paths

from .common import DEFAULT_OMEGA_SNAPSHOT
from src.model_platform.llm import create_llm_client

logger = logging.getLogger("ai_penetration.llm_alias")

_SYSTEM_PROMPT = (
    "你是中文招聘文本技能词典专家。给定一个技术技能词，列出它在招聘岗位描述中"
    "常见的**别名、同义词、英文缩写、大小写变体、中英混合写法**。\n"
    "规则：\n"
    "1. 只输出该技能在真实招聘文本中会出现的写法，不输出泛化描述\n"
    "2. 中文技能补英文对应（如深度学习→Deep Learning/DL）\n"
    "3. 英文技能补常见大小写/缩写变体（如 PyTorch→pytorch/Pytorch）\n"
    "4. 输出 JSON 对象：{技能名: [别名1, 别名2, ...]}，别名勿与原名重复\n"
    "5. 无常见别名的技能给空数组 []\n"
)


def main() -> None:
    """别名生成入口。"""
    parser = argparse.ArgumentParser(description="LLM 生成 AI 技能别名")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--max-skills", type=int, default=300,
                        help="处理的技能数（按 ωsAI 从高到低取前 N）")
    parser.add_argument("--omega-file", type=str,
                        default=DEFAULT_OMEGA_SNAPSHOT)
    args = parser.parse_args()

    paths = get_project_paths()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.project_root / "logs" / "llm_alias.log",
                                encoding="utf-8"),
        ],
    )

    # 加载 LLM 词典 + ωsAI，取 AI 相关技能
    llm_dict = [
        line.strip()
        for line in (paths.project_root / "dicts" / "ai_skill_terms_llm.txt")
        .read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    omega_path = Path(args.omega_file)
    if not omega_path.exists():
        omega_path = paths.project_root / args.omega_file
    omega = json.loads(omega_path.read_text(encoding="utf-8"))
    # 排序：优先有 ωsAI 分的，再按分数降序
    scored = sorted(llm_dict, key=lambda s: -omega.get(s, 0.0))
    # 只取有 ωsAI 分的技能（没分=样本中未出现，生成别名意义小）
    scored = [s for s in scored if s in omega]
    skills = scored[: args.max_skills]
    logger.info("处理 %d 个 AI 技能（ωsAI 从高到低）", len(skills))

    client = create_llm_client()
    alias_map: dict[str, list[str]] = {}
    batches = [skills[i:i + args.batch_size] for i in range(0, len(skills), args.batch_size)]
    logger.info("共 %d 批", len(batches))
    for bi, batch in enumerate(batches, 1):
        user_prompt = (
            "为以下技能生成别名，输出 JSON 对象：\n\n"
            + json.dumps(batch, ensure_ascii=False)
            + "\n\n格式 {\"技能\": [\"别名1\",\"别名2\"]}"
        )
        try:
            res = client.complete_json(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                strength="cheap",
                max_output_tokens=800,
            )
            if isinstance(res, dict):
                for k, v in res.items():
                    if isinstance(v, list):
                        alias_map[k] = [str(a).strip() for a in v if str(a).strip()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("批次 %d 失败: %s", bi, str(exc)[:100])
        logger.info("批次 %d/%d 完成", bi, len(batches))

    # 输出
    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"skill_aliases_{ts}.json"
    out_path.write_text(json.dumps(alias_map, ensure_ascii=False, indent=1), encoding="utf-8")
    total_aliases = sum(len(v) for v in alias_map.values())
    logger.info("完成: %d 技能生成 %d 个别名, 结果: %s", len(alias_map), total_aliases, out_path)

    # 预览 Top 别名
    print("别名样例:")
    for k, v in list(alias_map.items())[:15]:
        print("  %s -> %s" % (k, v[:5]))


if __name__ == "__main__":
    main()
