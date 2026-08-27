"""用 LLM 从招聘描述挖掘 AI 技能词（扩充技能词典，无人工标注）。

背景：通用技能词典（约 2000 项）AI 技能覆盖不足，需数据驱动扩充。
用 LLM 从含 AI 锚点的岗位描述中抽取具体的 AI 技术技能，多条描述合并为一次
LLM 调用（减少调用次数，vLLM 单次约 50-60s）。

流程：
1. 抽样含 AI 锚点（深度学习/机器学习/视觉/语言/识别等）的描述
2. 每 batch 条合并为一个 prompt，调 LLM 抽取 JSON 技能数组
3. 合并去重、统计频次、标记"是否已在新词典中"
4. 保存到 output/reports/llm_ai_skills.json

输出包含：抽取出的技能词、出现频次、是否已存在于现有词典。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime

import psycopg2

from config.paths import get_project_paths
from src.model_platform.llm import create_llm_client

from .common import eps_conn_params
from .skill_dictionary import load_ai_skill_terms, load_skill_names

logger = logging.getLogger("ai_penetration.llm_mining")

# AI 锚点：用于筛选含 AI 内容的描述
_AI_ANCHORS = ("深度学习", "机器学习", "计算机视觉", "自然语言", "语音识别",
               "大模型", "神经网络", "图像识别", "智能体", "强化学习")

_SYSTEM_PROMPT = (
    "你是招聘数据技能抽取专家。从一批岗位描述中抽取出现的【人工智能相关的"
    "具体技术技能】。规则：\n"
    "1. 只抽取具体的、可检索的技术/工具/模型/框架/算法名称，如 PyTorch、BERT、"
    "目标检测、LoRA、RAG、Transformer、数据标注\n"
    "2. 不要抽取泛化描述（如\"AI算法\"\"人工智能技术\"\"智能化\"）或软素质"
    "（沟通、团队合作）\n"
    "3. 同义表达合并为一个规范词（如\"深度学习\"与\"Deep Learning\"只留\"深度学习\"）\n"
    "4. 只输出 JSON 数组，如 [\"技能1\",\"技能2\"]，不要任何其他文字"
)


def _sample_descriptions(
    city_shards: list[str],
    year_start: int,
    year_end: int,
    n: int,
) -> list[str]:
    """抽样含 AI 锚点的岗位描述（多年份平均分配）。

    Args:
        city_shards: 城市分片表名列表。
        year_start: 起始年份。
        year_end: 结束年份。
        n: 抽样总条数。

    Returns:
        岗位描述列表。
    """
    params = eps_conn_params()
    anchor_like = " OR ".join(
        f"job_description LIKE '%%{a}%%'" for a in _AI_ANCHORS
    )
    years = list(range(year_start, year_end + 1))
    per_year = max(1, n // (len(years) * len(city_shards)))
    descs: list[str] = []
    conn = psycopg2.connect(**params)
    try:
        cur = conn.cursor()
        for shard in city_shards:
            for year in years:
                sql = f"""
                    SELECT job_description FROM public.{shard}
                    WHERE substr(publish_time, 1, 4) = %s
                      AND ({anchor_like})
                      AND job_description IS NOT NULL AND job_description != ''
                      AND position IS NOT NULL AND position != ''
                    ORDER BY random() LIMIT %s
                """
                cur.execute(sql, (str(year), int(per_year)))
                for (d,) in cur.fetchall():
                    descs.append(str(d)[:2000])  # 截断控制 prompt 长度
    finally:
        conn.close()
    logger.info("抽样描述 %d 条（%d-%d 年）", len(descs), year_start, year_end)
    return descs


def _mine_batch(client, batch: list[str]) -> list[str]:
    """用 LLM 抽取一批描述中的 AI 技能。

    Args:
        client: LLM 客户端。
        batch: 描述列表。

    Returns:
        AI 技能词列表。
    """
    user_prompt = "以下是 %d 条岗位描述：\n\n%s\n\n请抽取其中出现的AI技术技能，输出JSON数组。" % (
        len(batch),
        "\n\n---\n\n".join(f"[{i+1}] {d[:1200]}" for i, d in enumerate(batch)),
    )
    try:
        res = client.complete_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            strength="cheap",
            max_output_tokens=500,
        )
        if isinstance(res, list):
            return [str(s).strip() for s in res if str(s).strip()]
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("批次抽取失败: %s", str(exc)[:100])
        return []


def main() -> None:
    """LLM AI 技能挖掘入口。"""
    parser = argparse.ArgumentParser(description="LLM 挖掘 AI 技能词")
    parser.add_argument("--year-start", type=int, default=2016)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    args = parser.parse_args()

    paths = get_project_paths()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.project_root / "logs" / "llm_skill_mining.log",
                                encoding="utf-8"),
        ],
    )

    from .load_guangdong import GD_SHARDS

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    shards = [GD_SHARDS[c] for c in cities if c in GD_SHARDS]

    descs = _sample_descriptions(
        shards, args.year_start, args.year_end, args.sample_size
    )

    existing = set(load_skill_names()) | set(load_ai_skill_terms())
    logger.info("现有词典 %d 项", len(existing))

    client = create_llm_client()
    counter: Counter = Counter()
    batches = [descs[i:i + args.batch_size] for i in range(0, len(descs), args.batch_size)]
    logger.info("共 %d 个批次", len(batches))
    for bi, batch in enumerate(batches, 1):
        skills = _mine_batch(client, batch)
        for s in skills:
            counter[s] += 1
        logger.info("批次 %d/%d 完成, 累计技能 %d 个", bi, len(batches), len(counter))
        # 断点：每批保存进度
        _save_progress(counter, existing, paths)

    # 输出
    rows = [
        {"skill": s, "freq": n, "in_existing": s in existing}
        for s, n in counter.most_common()
    ]
    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"llm_ai_skills_{ts}.json"
    out_path.write_text(
        json.dumps({"total_descs": len(descs), "skills": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    new_skills = [r["skill"] for r in rows if not r["in_existing"]]
    logger.info("完成: 抽取 %d 技能, 其中新技能 %d 个", len(rows), len(new_skills))
    logger.info("结果: %s", out_path)
    print(f"新技能样例: {new_skills[:30]}")


def _save_progress(counter: Counter, existing: set[str], paths) -> None:
    """保存挖掘进度（断点）。"""
    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"skill": s, "freq": n, "in_existing": s in existing}
            for s, n in counter.most_common()]
    (out_dir / "llm_ai_skills_progress.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
