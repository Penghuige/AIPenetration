"""广州/深圳全量 AI 渗透率（分批流式，不超内存）。

逐市分批读取岗位描述（每批限行数），用正则合并匹配 AI 技能词典（一次扫描
匹配全部词），累计每市每年的 keyword / aiskill 命中计数，最终输出渗透率。

- 内存安全：每批只保留一批数据，只累计计数（小结构）
- 加速：AI 技能词合并为单个正则，避免逐词子串匹配
- 覆盖全量：不抽样、不设上限，逐批读完整个分片表
- 逐年断点：每年完成后立即写 JSON（output/ai_penetration/stream_checkpoint_{shard}.json），
  中断后重启跳过已完成年份续跑，报告写成功后清理断点

使用示例::

    python -m src.ai_penetration.stream_penetration --cities "广州市,深圳市" --batch-size 100000
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.paths import get_project_paths

from .ai_scoring import (
    AI_SCORE_THRESHOLD,
    build_prefilter_regex,
    match_keywords_scored,
    match_skills_scored,
)
from .common import eps_conn_params, setup_logging
from .keyword_penetration import load_ai_keywords
from .load_guangdong import GD_SHARDS, get_eps_engine
from .skill_dictionary import load_ai_skill_terms

logger = logging.getLogger("ai_penetration.stream")


# 逐年断点保存：每年完成后立即写 JSON，中断后重启跳过已完成年份，避免全量白跑
# 路径按分片表区分，写入 output/ai_penetration/stream_checkpoint_{shard}.json


def _checkpoint_path(shard: str) -> Path:
    """返回某分片表的断点文件路径。

    Args:
        shard: 分片表名。

    Returns:
        断点 JSON 路径。
    """
    out_dir = get_project_paths().output_dir / "ai_penetration"
    return out_dir / f"stream_checkpoint_{shard}.json"


def _load_checkpoint(shard: str) -> dict[int, dict]:
    """加载断点：{年份: {total, keyword_ai, aiskill_ai, combined_ai}}。

    Args:
        shard: 分片表名。

    Returns:
        已完成的年份统计；无断点或损坏时返回空 dict。
    """
    path = _checkpoint_path(shard)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        stats = {int(k): v for k, v in data.items()}
        logger.info("%s 加载断点: 已完成 %d 年 %s", shard, len(stats), sorted(stats))
        return stats
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("%s 断点文件损坏，忽略: %s", path, exc)
        return {}


def _save_checkpoint(shard: str, stats: dict[int, dict]) -> None:
    """保存断点（每年完成后调用）。

    Args:
        shard: 分片表名。
        stats: 当前城市全部年份统计。
    """
    path = _checkpoint_path(shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")


def stream_city(
    engine,
    shard: str,
    batch_size: int,
    kw_regex: re.Pattern,
    ai_regex: re.Pattern,
) -> dict:
    """分批流式处理单个城市，累计每年的加权命中计数。

    Args:
        engine: eps 数据库 engine。
        shard: 分片表名。
        batch_size: 每批行数。
        kw_regex: 关键词快筛合并正则。
        ai_regex: AI 技能快筛合并正则。

    Returns:
        {year: {"total": int, "keyword_ai": int, "aiskill_ai": int, "combined_ai": int}}。
    """
    stats: dict[int, dict] = defaultdict(
        lambda: {"total": 0, "keyword_ai": 0, "aiskill_ai": 0, "combined_ai": 0},
        _load_checkpoint(shard),
    )
    # 分年份处理：每年一个独立 psycopg2 server-side cursor（命名游标），
    # fetchmany 分批读取。单次游标规模 = 单年数据，避免长连接大游标压垮 PG。
    # 每年完成后立即写断点，中断后可跳过已完成年份续跑。
    import psycopg2

    params = eps_conn_params()
    years = range(2014, 2025)
    for year in years:
        if year in stats:
            logger.info("%s %d 年已有断点，跳过", shard, year)
            continue
        sql = f"""
            SELECT position, job_description
            FROM public.{shard}
            WHERE substr(publish_time, 1, 4) = %s
              AND job_description IS NOT NULL AND job_description != ''
              AND position IS NOT NULL AND position != ''
        """
        processed = 0
        conn = psycopg2.connect(**params)
        try:
            cur = conn.cursor(f"ai_stream_{shard}_{year}")  # 命名游标 = server-side
            cur.execute(sql, (str(year),))
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                for position, desc in batch:
                    s = stats[year]
                    s["total"] += 1
                    pos = str(position or "")
                    d = str(desc or "")
                    # 快筛：合并正则一次扫描，无命中则跳过详细计分（绝大多数岗位）
                    if not kw_regex.search(pos) and not ai_regex.search(d):
                        continue
                    _, kw_score = match_keywords_scored(pos)
                    _, sk_score = match_skills_scored(d)
                    if kw_score >= AI_SCORE_THRESHOLD:
                        s["keyword_ai"] += 1
                    if sk_score >= AI_SCORE_THRESHOLD:
                        s["aiskill_ai"] += 1
                    if (kw_score + sk_score) >= AI_SCORE_THRESHOLD:
                        s["combined_ai"] += 1
                processed += len(batch)
                logger.info("%s %d 年 已处理 %d 条", shard, year, processed)
            cur.close()
        finally:
            conn.close()
        logger.info("%s %d 年完成: %d 条", shard, year, processed)
        _save_checkpoint(shard, dict(stats))
    return dict(stats)


def write_stream_report(
    pens: dict[str, pd.DataFrame],
    ai_terms: list[str],
    kw_terms: list[str],
    timestamp: str,
    cities: list[str],
    out_dir: Path,
) -> Path:
    """生成全量渗透率报告（含方法说明）。

    Args:
        pens: {城市名: 按年渗透率 DataFrame}，键含"合计"。
        ai_terms: AI 技能词列表。
        kw_terms: AI 关键词列表。
        timestamp: 时间戳（用于文件名）。
        cities: 参与分析的城市列表。
        out_dir: 报告输出目录。

    Returns:
        报告文件路径。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"ai_penetration_stream_{timestamp}.md"

    merged = pens.get("合计")
    total_rows = int(merged["total"].sum()) if merged is not None else 0

    lines = [
        "# 广州/深圳全量 AI 渗透率（加权判定，全量不抽样）",
        "",
        f"- 分析时间：{timestamp}",
        f"- 数据范围：{'、'.join(cities)}（全量，分批流式，不抽样，共 {total_rows:,} 行）",
        f"- AI 关键词词典：{len(kw_terms)} 词（dicts/ai_occupation_keywords.txt）",
        f"- AI 技能词典：{len(ai_terms)} 词（dicts/ai_skill_terms.txt）",
        "",
        "## 一、方法说明",
        "",
        "### 判定模型：加权评分",
        "",
        "岗位是否 AI 相关 = 岗位名命中 AI 关键词 + 岗位描述命中 AI 技术技能词，两部分加权：",
        "",
        "```",
        "score = 岗位名关键词得分 + 岗位描述技能词得分",
        "AI 判定：score >= 3",
        "```",
        "",
        "权重分层（词典中以 `词|权重` 标注）：",
        "",
        "| 层级 | 权重 | 说明 | 示例 |",
        "|---|---|---|---|",
        "| 特定 AI 技术词 | 3 | 单独出现基本即 AI | PyTorch、SLAM、RAG、轨迹预测 |",
        "| 通用强词 | 2 | 公司简介/模板常用，需佐证 | 深度学习、大模型、人脸识别、AIGC |",
        "| 弱词 | 1 | 歧义大，需共同佐证 | AI设计(排除Adobe)、内容安全、特征工程、岗位名中的\"AI\" |",
        "",
        "三个口径：",
        "- **关键词率（方法一）**：岗位名关键词得分 >= 3",
        "- **AI技能率（方法三）**：描述技能词得分 >= 3",
        "- **综合率**：关键词 + 技能合计 >= 3（两法并集，含弱信号互证）",
        "",
        "### 关键规则",
        "",
        "- 纯英文词（RAG/LLM/AI）一律词边界匹配，避免英文子串误报（如 RAG 误匹配 BEVERAGE）",
        "- \"AI设计\"后紧跟\"软件\"时排除（=Adobe Illustrator，非 AI）",
        "- 剔除岗位名中含公司名的括号（如\"电气工程师(机器人公司)\"）",
        "- 快筛合并正则 + 命中后详细计分，全量数据可流式处理（内存安全）",
        "- 逐年断点保存，中断可续跑",
        "",
        "## 二、结果",
        "",
    ]
    for city in pens:
        df = pens[city]
        lines.extend([
            f"### {city} 按年渗透率",
            "",
            "| 年份 | 总岗位 | 关键词AI | AI技能AI | 综合AI | 关键词率 | AI技能率 | 综合率 |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for _, r in df.iterrows():
            lines.append(
                f"| {int(r['year'])} | {int(r['total'])} | {int(r['keyword_ai'])} | "
                f"{int(r['aiskill_ai'])} | {int(r['combined_ai'])} | "
                f"{r['keyword_rate']:.4f} | {r['aiskill_rate']:.4f} | {r['combined_rate']:.4f} |"
            )
        lines.append("")

    # 趋势要点
    lines.extend(["## 三、趋势要点", ""])
    if merged is not None:
        mr = merged.sort_values("year")
        first = mr[mr["combined_ai"] > 0]
        if not first.empty:
            start_yr = int(first["year"].iloc[0])
            start_rate = first["combined_rate"].iloc[0] * 100
            last = mr.iloc[-1]
            last_rate = last["combined_rate"] * 100
            lines.append(
                f"- 渗透率自 {start_yr} 年起步（{start_rate:.2f}%），到 {int(last['year'])} 年达 "
                f"{last_rate:.2f}%"
            )
            gz = pens.get("广州市")
            sz = pens.get("深圳市")
            if gz is not None and sz is not None:
                gz24 = gz[gz["year"] == 2024]["combined_rate"]
                sz24 = sz[sz["year"] == 2024]["combined_rate"]
                if not gz24.empty and not sz24.empty:
                    ratio = sz24.iloc[0] / gz24.iloc[0] if gz24.iloc[0] else 0
                    lines.append(
                        f"- 2024 年深圳综合率 {sz24.iloc[0]*100:.2f}% 约为广州 "
                        f"{gz24.iloc[0]*100:.2f}% 的 {ratio:.1f} 倍"
                    )
            # 单法对比（最近 3 年）
            recent = mr[mr["year"] >= 2022]
            for _, r in recent.iterrows():
                lines.append(
                    f"- {int(r['year'])}：关键词率 {r['keyword_rate']*100:.2f}% vs "
                    f"AI技能率 {r['aiskill_rate']*100:.2f}%，综合率 {r['combined_rate']*100:.2f}%"
                )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已写入: %s", report_path)
    return report_path


def main() -> None:
    """广州/深圳全量流式渗透率入口。"""
    parser = argparse.ArgumentParser(description="广州/深圳全量 AI 渗透率（流式）")
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    parser.add_argument("--batch-size", type=int, default=100000)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_stream.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    ai_terms = load_ai_skill_terms()
    kw_terms = list(load_ai_keywords())
    kw_regex = build_prefilter_regex(kw_terms)
    ai_regex = build_prefilter_regex(ai_terms)
    logger.info("AI 技能词典 %d 词、关键词 %d 词，已合并快筛正则", len(ai_terms), len(kw_terms))

    engine = get_eps_engine()
    # 各市分开统计：{city: {year: stats}}
    per_city: dict[str, dict] = {}
    for city in cities:
        shard = GD_SHARDS.get(city)
        if not shard:
            logger.warning("未知城市: %s", city)
            continue
        logger.info("处理 %s（%s）全量...", city, shard)
        city_stats = stream_city(engine, shard, args.batch_size, kw_regex, ai_regex)
        per_city[city] = city_stats
        logger.info("%s 完成: %d 年", city, len(city_stats))

    def _to_pen(stats: dict[int, dict]) -> pd.DataFrame:
        """将 {year: stats} 转为渗透率 DataFrame。"""
        rows = []
        for yr in sorted(stats):
            s = stats[yr]
            n = s["total"]
            rows.append({
                "year": yr,
                "total": n,
                "keyword_ai": s["keyword_ai"],
                "aiskill_ai": s["aiskill_ai"],
                "combined_ai": s["combined_ai"],
                "keyword_rate": s["keyword_ai"] / n if n else 0.0,
                "aiskill_rate": s["aiskill_ai"] / n if n else 0.0,
                "combined_rate": s["combined_ai"] / n if n else 0.0,
            })
        return pd.DataFrame(rows)

    # 各市单独 + 合并
    pens: dict[str, pd.DataFrame] = {}
    for city, stats in per_city.items():
        pens[city] = _to_pen(stats)
        print(f"\n=== {city} 渗透率 ===")
        print(pens[city][["year", "total", "keyword_ai", "aiskill_ai", "combined_ai",
                          "keyword_rate", "aiskill_rate", "combined_rate"]].to_string(index=False))

    merged_stats: dict[int, dict] = defaultdict(
        lambda: {"total": 0, "keyword_ai": 0, "aiskill_ai": 0, "combined_ai": 0}
    )
    for stats in per_city.values():
        for yr, s in stats.items():
            acc = merged_stats[yr]
            acc["total"] += s["total"]
            acc["keyword_ai"] += s["keyword_ai"]
            acc["aiskill_ai"] += s["aiskill_ai"]
            acc["combined_ai"] += s["combined_ai"]
    pens["合计"] = _to_pen(merged_stats)
    print("\n=== 合计渗透率 ===")
    print(pens["合计"][["year", "total", "keyword_ai", "aiskill_ai", "combined_ai",
                        "keyword_rate", "aiskill_rate", "combined_rate"]].to_string(index=False))

    # 输出报告（含方法说明）
    out_dir = paths.output_dir / "reports"
    write_stream_report(pens, ai_terms, kw_terms, timestamp, cities, out_dir)

    # 报告成功写入后清理断点，避免下次重复跳过
    for city in per_city:
        shard = GD_SHARDS.get(city)
        if not shard:
            continue
        cp = _checkpoint_path(shard)
        if cp.exists():
            cp.unlink()
            logger.info("已清理断点: %s", cp)


if __name__ == "__main__":
    main()
