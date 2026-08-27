"""AI 职业渗透率分析 CLI 入口。

使用示例::

    python -m src.ai_penetration --min-post-ai-count 5

参数:
    --min-post-ai-count: AI 后最低出现次数阈值（默认 5）。
    --limit-per-shard: 每张分片表最多加载行数，0=全量（调试用）。
    --batch-size: 每个 LLM prompt 的岗位数（默认 20）。
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from config.paths import get_project_paths
from src.model_platform.llm import create_llm_client

from .common import setup_logging
from .load_guangdong import get_eps_engine, load_gd_position_stats
from .standardize_jobs import standardize_positions
from .compute_penetration import (
    compute_penetration,
    compute_quarterly_penetration,
    identify_ai_new_occupations,
)
from .report import build_matrix_csv, generate_report




def run_skill_mode(args: argparse.Namespace) -> None:
    """运行技能 AI 共现度流程。

    Args:
        args: 解析后的命令行参数（skill_mode / skill_cities / skill_max_jobs）。
    """
    import json

    from .skill_data import load_job_skill_data
    from .compute_skill_penetration import (
        compute_skill_ai_scores_from_df,
        compute_job_relevance_series,
    )

    logger = logging.getLogger("ai_penetration.main")
    engine = get_eps_engine()
    cities = (
        [c.strip() for c in args.skill_cities.split(",")]
        if args.skill_cities else None
    )
    df = load_job_skill_data(engine, cities=cities, max_jobs=args.skill_max_jobs)
    scores = compute_skill_ai_scores_from_df(df)
    rel = compute_job_relevance_series(df, scores)

    # 输出技能表
    out_dir = get_project_paths().output_dir / "reports"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"skill_ai_scores_{ts}.json"
    json.dump(scores, scores_path.open("w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # 相关度分布
    logger.info(
        "岗位 AI 相关度分布: mean=%.3f, median=%.3f, >0.5占比=%.1f%%",
        rel.mean(), rel.median(), (rel > 0.5).mean() * 100,
    )
    logger.info("技能 AI 共现度表已输出: %s", scores_path)


def main() -> None:
    """主入口。"""
    parser = argparse.ArgumentParser(description="AI 职业渗透率分析")
    parser.add_argument("--min-post-ai-count", type=int, default=5)
    parser.add_argument("--limit-per-shard", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--llm-workers", type=int, default=8)
    parser.add_argument("--skill-mode", action="store_true", default=False,
                        help="运行技能 AI 共现度分析流程")
    parser.add_argument("--skill-cities", type=str, default=None,
                        help="限定城市，逗号分隔，如 广州市,深圳市")
    parser.add_argument("--skill-max-jobs", type=int, default=200000,
                        help="技能模式最多加载岗位数")
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration.log")
    logger = logging.getLogger("ai_penetration.main")

    if args.skill_mode:
        logger.info("进入技能 AI 共现度模式")
        run_skill_mode(args)
        logger.info("技能 AI 共现度流程完成")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = paths.output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("步骤1: 加载广东省数据...")
    engine = get_eps_engine()
    position_stats = load_gd_position_stats(
        engine, limit_per_shard=args.limit_per_shard
    )
    logger.info(
        "position 统计表: %d 行, %d 个岗位",
        len(position_stats), position_stats["position"].nunique(),
    )

    logger.info("步骤2: LLM 标准化岗位名...")
    client = create_llm_client()
    position_occ_map = standardize_positions(
        position_stats, client,
        batch_size=args.batch_size,
        max_workers=args.llm_workers,
    )

    logger.info("步骤3: 识别 AI 新职业...")
    ai_new = identify_ai_new_occupations(
        position_stats, position_occ_map,
        min_post_ai_count=args.min_post_ai_count,
    )
    logger.info("AI 新职业: %d 个", len(ai_new))

    logger.info("步骤4: 计算渗透率...")
    pen_yearly = compute_penetration(position_stats, position_occ_map, ai_new)
    pen_quarterly = compute_quarterly_penetration(
        position_stats, position_occ_map, ai_new
    )

    logger.info("步骤5: 生成报告...")
    matrix_path = output_dir / f"ai_penetration_matrix_{timestamp}.csv"
    build_matrix_csv(position_stats, position_occ_map, ai_new, matrix_path)
    report_path = generate_report(
        pen_yearly, pen_quarterly, ai_new, matrix_path, output_dir, timestamp
    )
    logger.info("报告: %s", report_path)


if __name__ == "__main__":
    main()
