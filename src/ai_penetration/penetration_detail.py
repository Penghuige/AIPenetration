"""细分粒度 AI 渗透率明细（年度/半年度/季度 × 城市 × 行业 × 三方法）。

在 stream_penetration 按年汇总基础上，进一步细分并输出宽表：

- 时间粒度：年度 / 半年度 / 季度
- 维度：城市、行业（来自 ent 企业表 industry_code，经 recruit_id 关联）
- 三方法：
  - 关键词（方法一）：岗位名命中 AI 关键词（得分 >= 阈值）
  - 共现率（AI技能词典法，替代已弃用的技能共现度法）：岗位描述命中 AI 技能词
  - 复合方法：关键词 + 技能合计得分 >= 阈值（加权并集）

输出：
- 宽表 CSV（含"时间粒度"列，统一三粒度）
- 季度明细 CSV

逐年断点保存，中断可续跑。行业映射一次性加载进内存（recruit_id → industry_code）。

使用示例::

    python -m src.ai_penetration.penetration_detail --cities "广州市,深圳市" --batch-size 100000
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.paths import get_project_paths

from .common import eps_conn_params, setup_logging
from .ai_scoring import (
    AI_SCORE_THRESHOLD,
    build_prefilter_regex,
    match_keywords_scored,
    match_skills_scored,
)
from .industry_classification import classify_industry
from .keyword_penetration import load_ai_keywords
from .load_guangdong import GD_SHARDS, get_eps_engine
from .skill_dictionary import load_ai_skill_terms

logger = logging.getLogger("ai_penetration.detail")

# 输出列
_DETAIL_COLUMNS = [
    "时间粒度", "年份", "期间", "城市", "行业",
    "总发布数",
    "AI岗位发布数_关键词", "AI岗位发布数_共现率", "AI岗位发布数_复合方法",
    "AI岗位渗透度_关键词", "AI岗位渗透度_共现率", "AI岗位渗透度_复合方法",
]

_GRANULARITIES = ("年度", "半年度", "季度")

# 样本量阈值：城市-行业-时间段总发布数低于此值则剔除，不计算渗透度
# 依据：广州2024 大类年发布数 median≈2万、P25≈2200，阈值500只剔除
#       极小行业尾部（烟草/渔业/采矿等 <500），主流行业远高于此；
#       且保证渗透度估计的基本统计稳定性（N≥500 时典型率 SE≈0.3-0.5pp）。
MIN_TOTAL_THRESHOLD = 500




def _parse_period(publish_time: str) -> tuple[int, int, int]:
    """解析发布时间的 (年, 上半年标记, 季度)。

    Args:
        publish_time: 形如 'YYYY-MM-DD'。

    Returns:
        (year, half, quarter)；无法解析时返回 (0, 0, 0)。
    """
    m = re.match(r"^(\d{4})-(\d{2})", str(publish_time or ""))
    if not m:
        return 0, 0, 0
    year = int(m.group(1))
    month = int(m.group(2))
    if month < 1 or month > 12:  # 脏数据月份，仅能确定年份
        return year, 0, 0
    half = 1 if month <= 6 else 2
    quarter = (month - 1) // 3 + 1
    return year, half, quarter


def _industry_label(code: str) -> str:
    """行业编码归类到大类层面（门类字母 + 大类名）。

    Args:
        code: ent 表 industry_code。

    Returns:
        大类标签，如 "C22造纸和纸制品业"；空值/无法识别时返回 "未知"。
    """
    return classify_industry(code)


def _agg_key(granularity: str, year: int, half: int, quarter: int, city: str, industry: str) -> str:
    """聚合键（JSON 可序列化字符串）。

    Args:
        granularity: 时间粒度（年度/半年度/季度）。
        year: 年份。
        half: 上半年标记（1/2，不适用时 0）。
        quarter: 季度（1-4，不适用时 0）。
        city: 城市。
        industry: 行业。

    Returns:
        管道分隔的字符串键。
    """
    return f"{granularity}|{year}|{half}|{quarter}|{city}|{industry}"


def _bucket(agg: dict[str, dict], key: str) -> dict:
    """获取或创建聚合桶。

    Args:
        agg: 聚合字典。
        key: 聚合键。

    Returns:
        {total, keyword_ai, aiskill_ai, combined_ai} 桶。
    """
    return agg.setdefault(key, {"total": 0, "keyword_ai": 0, "aiskill_ai": 0, "combined_ai": 0})


def _checkpoint_path() -> Path:
    """返回断点文件路径。"""
    return get_project_paths().output_dir / "ai_penetration" / "detail_checkpoint.json"


def _load_checkpoint() -> tuple[dict[str, dict], set[str]]:
    """加载断点（agg + 已完成 city|year 集合）。

    Returns:
        (agg, completed)。
    """
    path = _checkpoint_path()
    if not path.exists():
        return {}, set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        agg = data.get("agg", {})
        completed = set(data.get("completed", []))
        logger.info("加载断点: %d 聚合桶, %d 个城市年份", len(agg), len(completed))
        return agg, completed
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("断点损坏，忽略: %s", exc)
        return {}, set()


def _save_checkpoint(agg: dict[str, dict], completed: set[str]) -> None:
    """保存断点。"""
    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"agg": agg, "completed": sorted(completed)}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def stream_city_detail(
    engine,
    shard: str,
    batch_size: int,
    kw_regex: re.Pattern,
    ai_regex: re.Pattern,
    city: str,
    agg: dict[str, dict],
    completed: set[str],
) -> None:
    """分批流式处理单个城市，聚合到三粒度 × 行业。

    行业经 SQL LEFT JOIN ent 企业表按 recruit_id 关联（利用其索引），
    避免将整张 ent 表载入 Python 内存。

    Args:
        engine: eps 数据库 engine。
        shard: 岗位分片表名。
        batch_size: 每批行数。
        kw_regex: 关键词快筛正则。
        ai_regex: AI 技能快筛正则。
        city: 城市名。
        agg: 聚合字典（就地更新）。
        completed: 已完成 city|year 集合（就地更新）。
    """
    import psycopg2

    params = eps_conn_params()
    ent_shard = shard.replace("job_", "ent_", 1)
    years = range(2014, 2025)
    for year in years:
        cp_key = f"{city}|{year}"
        if cp_key in completed:
            logger.info("%s %d 年已有断点，跳过", city, year)
            continue
        sql = f"""
            SELECT j.position, j.job_description, j.publish_time, e.industry_code
            FROM public.{shard} j
            LEFT JOIN LATERAL (
                SELECT industry_code FROM public.{ent_shard}
                WHERE recruit_id = j.recruit_id
                LIMIT 1
            ) e ON true
            WHERE substr(j.publish_time, 1, 4) = %s
              AND j.job_description IS NOT NULL AND j.job_description != ''
              AND j.position IS NOT NULL AND j.position != ''
        """
        processed = 0
        conn = psycopg2.connect(**params)
        try:
            # 会话级优化：大表排序/哈希 work_mem（去重后的 ent 子查询 join 需哈希内存）
            setup_cur = conn.cursor()
            setup_cur.execute("SET LOCAL work_mem = '1GB'")
            setup_cur.close()
            cur = conn.cursor(f"ai_detail_{shard}_{year}")  # 命名游标 = server-side
            cur.execute(sql, (str(year),))
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                for position, desc, pub, ind_code in batch:
                    pos = str(position or "")
                    d = str(desc or "")
                    yr, half, quarter = _parse_period(pub)
                    if yr == 0:
                        yr = year  # 无法解析日期，归入当年
                    # 粒度桶：年度必含；半年度/季度仅在月份可解析时计入
                    periods = [("年度", yr, 0, 0)]
                    if half and quarter:
                        periods.append(("半年度", yr, half, 0))
                        periods.append(("季度", yr, 0, quarter))
                    industry = _industry_label(ind_code)
                    for gran, y, h, q in periods:
                        _bucket(agg, _agg_key(gran, y, h, q, city, industry))["total"] += 1
                    # 快筛：无 AI 命中则跳过计分
                    if not kw_regex.search(pos) and not ai_regex.search(d):
                        continue
                    _, kw_score = match_keywords_scored(pos)
                    _, sk_score = match_skills_scored(d)
                    kw_ai = kw_score >= AI_SCORE_THRESHOLD
                    sk_ai = sk_score >= AI_SCORE_THRESHOLD
                    comb_ai = (kw_score + sk_score) >= AI_SCORE_THRESHOLD
                    if not (kw_ai or sk_ai or comb_ai):
                        continue
                    for gran, y, h, q in periods:
                        b = _bucket(agg, _agg_key(gran, y, h, q, city, industry))
                        if kw_ai:
                            b["keyword_ai"] += 1
                        if sk_ai:
                            b["aiskill_ai"] += 1
                        if comb_ai:
                            b["combined_ai"] += 1
                processed += len(batch)
                logger.info("%s %d 年 已处理 %d 条", city, year, processed)
            cur.close()
        finally:
            conn.close()
        logger.info("%s %d 年完成: %d 条", city, year, processed)
        completed.add(cp_key)
        _save_checkpoint(agg, completed)


def build_detail_dataframe(agg: dict[str, dict]) -> pd.DataFrame:
    """将聚合字典转为宽表 DataFrame。

    Args:
        agg: 聚合字典。

    Returns:
        宽表 DataFrame（含"时间粒度"列）。
    """
    rows = []
    for key, s in agg.items():
        granularity, year, half, quarter, city, industry = key.split("|")
        if granularity == "年度":
            period = "全年"
        elif granularity == "半年度":
            period = "上半年" if half == "1" else "下半年"
        else:
            period = f"Q{quarter}"
        n = s["total"]
        if n < MIN_TOTAL_THRESHOLD:
            continue  # 样本不足：剔除该组合，不计算渗透度
        kw, sk, comb = s["keyword_ai"], s["aiskill_ai"], s["combined_ai"]
        rows.append({
            "时间粒度": granularity,
            "年份": int(year),
            "期间": period,
            "城市": city,
            "行业": industry,
            "总发布数": n,
            "AI岗位发布数_关键词": kw,
            "AI岗位发布数_共现率": sk,
            "AI岗位发布数_复合方法": comb,
            "AI岗位渗透度_关键词": kw / n if n else 0.0,
            "AI岗位渗透度_共现率": sk / n if n else 0.0,
            "AI岗位渗透度_复合方法": comb / n if n else 0.0,
        })
    df = pd.DataFrame(rows, columns=_DETAIL_COLUMNS)
    if not df.empty:
        df = (
            df.sort_values(["时间粒度", "年份", "期间", "城市", "行业"])
            .reset_index(drop=True)
        )
    return df


def main() -> None:
    """细分粒度渗透率明细入口。"""
    parser = argparse.ArgumentParser(description="细分粒度 AI 渗透率明细")
    parser.add_argument("--cities", type=str, default="广州市,深圳市",
                        help="城市列表，逗号分隔")
    parser.add_argument("--batch-size", type=int, default=100000,
                        help="每批行数")
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_detail.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    ai_terms = load_ai_skill_terms()
    kw_terms = list(load_ai_keywords())
    kw_regex = build_prefilter_regex(kw_terms)
    ai_regex = build_prefilter_regex(ai_terms)
    logger.info("AI 技能词典 %d 词、关键词 %d 词，已合并快筛正则",
                len(ai_terms), len(kw_terms))

    engine = get_eps_engine()
    agg, completed = _load_checkpoint()
    for city in cities:
        shard = GD_SHARDS.get(city)
        if not shard:
            logger.warning("未知城市: %s", city)
            continue
        logger.info("处理 %s（%s）...", city, shard)
        stream_city_detail(
            engine, shard, args.batch_size, kw_regex, ai_regex,
            city, agg, completed,
        )
        logger.info("%s 完成", city)

    df = build_detail_dataframe(agg)
    logger.info("明细总行数: %d", len(df))

    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_path = out_dir / f"ai_penetration_detail_{timestamp}.csv"
    df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    logger.info("宽表已写入: %s", wide_path)

    quarterly_path = out_dir / f"ai_penetration_quarterly_{timestamp}.csv"
    df[df["时间粒度"] == "季度"].to_csv(quarterly_path, index=False, encoding="utf-8-sig")
    logger.info("季度明细已写入: %s", quarterly_path)

    # 清理断点
    cp = _checkpoint_path()
    if cp.exists():
        cp.unlink()
        logger.info("已清理断点: %s", cp)

    # 终端预览（合计口径，忽略行业维度，按城市合计）
    preview = df.groupby(["时间粒度", "年份", "期间", "城市"], as_index=False)[
        ["总发布数", "AI岗位发布数_关键词", "AI岗位发布数_共现率",
         "AI岗位发布数_复合方法"]
    ].sum()
    print("\n合计（跨行业合计）预览:")
    print(preview[preview["时间粒度"] == "年度"].to_string(index=False))


if __name__ == "__main__":
    main()
