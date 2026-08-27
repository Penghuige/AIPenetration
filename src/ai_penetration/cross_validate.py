"""关键词匹配 vs 技能共现度的交叉验证。

对同一批岗位分别用两种方法判定是否 AI 相关：
- 关键词法：岗位名命中 AI 关键词词典（二值）
- 技能法：岗位技能平均 AI 共现度 ≥ 阈值（二值）

计算两方法的一致性（重合率 / Cohen's kappa / 相关性），
并输出不一致样本供人工抽检。

使用示例::

    python -m src.ai_penetration.cross_validate --cities "广州市,深圳市" --max-jobs 50000
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

import pandas as pd

from config.paths import get_project_paths

from .common import setup_logging
from .load_guangdong import get_eps_engine
from .skill_data import _extract_ai_skills, load_job_skill_data
from .compute_skill_penetration import compute_skill_ai_scores_from_df, compute_job_relevance_series

logger = logging.getLogger("ai_penetration.cross_validate")




def compute_agreement(
    keyword_flags: list[bool],
    skill_relevance: list[float],
    threshold: float = 0.5,
) -> dict:
    """计算两方法判定的混淆矩阵与一致性指标。

    Args:
        keyword_flags: 关键词法判定（是否 AI）。
        skill_relevance: 技能法岗位 AI 相关度（连续量）。
        threshold: 技能法判定阈值。

    Returns:
        含混淆矩阵 / 重合率 / Cohen's kappa / 相关系数的字典。
    """
    import numpy as np
    from sklearn.metrics import cohen_kappa_score

    kw = np.array([1 if f else 0 for f in keyword_flags])
    sk = np.array([1 if r >= threshold else 0 for r in skill_relevance])

    n = len(kw)
    both_ai = int(((kw == 1) & (sk == 1)).sum())
    kw_only = int(((kw == 1) & (sk == 0)).sum())
    sk_only = int(((kw == 0) & (sk == 1)).sum())
    both_not = int(((kw == 0) & (sk == 0)).sum())

    agreement = (both_ai + both_not) / n if n else 0.0
    kappa = cohen_kappa_score(kw, sk) if n else 0.0
    # 连续相关：关键词二值与技能相关度的相关性
    corr = float(pd.Series(kw).corr(pd.Series(skill_relevance))) if n else 0.0

    return {
        "n": n,
        "both_ai": both_ai,
        "keyword_only": kw_only,
        "skill_only": sk_only,
        "both_not": both_not,
        "agreement": round(agreement, 4),
        "kappa": round(kappa, 4),
        "corr": round(corr, 4),
        "keyword_ai_rate": round(kw.sum() / n, 4) if n else 0.0,
        "skill_ai_rate": round(sk.sum() / n, 4) if n else 0.0,
    }


def pick_disagreements(
    df: pd.DataFrame,
    skill_relevance: pd.Series,
    threshold: float = 0.5,
    sample: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """抽样两方法判定不一致的岗位（含关键词与技能相关度）。

    Args:
        df: 含 position 与 is_ai_seed 的岗位数据。
        skill_relevance: 技能法岗位 AI 相关度 Series。
        threshold: 技能法判定阈值。
        sample: 每个不一致类型抽多少条。
        seed: 随机种子。

    Returns:
        不一致样本 DataFrame，列 position / is_ai_seed / skill_relevance / conflict_type。
    """
    out = df.copy()
    out["skill_relevance"] = skill_relevance.values
    kw_ai = out["is_ai_seed"].astype(bool)
    sk_ai = out["skill_relevance"] >= threshold

    kw_only = out[kw_ai & ~sk_ai].copy()
    kw_only["conflict_type"] = "关键词AI但技能低分"
    sk_only = out[~kw_ai & sk_ai].copy()
    sk_only["conflict_type"] = "技能高分但关键词未命中"

    kw_only = kw_only.sample(min(sample, len(kw_only)), random_state=seed)
    sk_only = sk_only.sample(min(sample, len(sk_only)), random_state=seed)
    cols = ["position", "is_ai_seed", "skill_relevance", "conflict_type"]
    return pd.concat([kw_only, sk_only])[cols]


def main() -> None:
    """交叉验证入口。"""
    parser = argparse.ArgumentParser(description="关键词 vs 技能共现度交叉验证")
    parser.add_argument("--cities", type=str, default="广州市,深圳市")
    parser.add_argument("--max-jobs", type=int, default=50000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample", type=int, default=30)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "ai_penetration_cross_validate.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    engine = get_eps_engine()
    df = load_job_skill_data(engine, cities=cities, max_jobs=args.max_jobs)
    logger.info("加载岗位 %d 条", len(df))

    # 三视角判定
    ai_scores = compute_skill_ai_scores_from_df(df)
    skill_relevance = compute_job_relevance_series(df, ai_scores)
    # AI 技能词典法：描述命中 AI 技能词 → AI
    ai_skill_flags = df["description"].apply(lambda d: len(_extract_ai_skills(d)) > 0)
    ai_skill_rate = float(ai_skill_flags.mean())

    # 两两一致性
    result_kw_skill = compute_agreement(
        df["is_ai_seed"].tolist(), skill_relevance.tolist(), threshold=args.threshold
    )
    result_kw_aik = compute_agreement(
        df["is_ai_seed"].tolist(),
        ai_skill_flags.astype(int).tolist(),
        threshold=0.5,
    )
    logger.info("关键词 vs 技能共现度: %s", result_kw_skill)
    logger.info("关键词 vs AI技能词典: %s", result_kw_aik)
    logger.info("AI 技能词典 AI 占比: %.4f", ai_skill_rate)

    # 输出报告
    out_dir = paths.output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"ai_cross_validate_{timestamp}.md"
    lines = [
        "# AI 渗透率三方法交叉验证",
        "",
        f"- 时间：{timestamp}",
        f"- 数据：{len(df)} 条岗位（{'、'.join(cities)}）",
        f"- 技能共现度阈值：{args.threshold}",
        "",
        "## 方法说明",
        "",
        "- **关键词法**：岗位名命中 AI 关键词词典（`ai_occupation_keywords.txt`）",
        "- **技能共现度法**：岗位技能平均 AI 共现度 ≥ 阈值",
        "- **AI 技能词典法**：岗位描述命中 AI 技能词典（`ai_skill_terms.txt`）",
        "",
        "## 一致性指标",
        "",
        "### 关键词 vs 技能共现度",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 总岗位 | {result_kw_skill['n']} |",
        f"| 两方法均判 AI | {result_kw_skill['both_ai']} |",
        f"| 仅关键词判 AI | {result_kw_skill['keyword_only']} |",
        f"| 仅技能判 AI | {result_kw_skill['skill_only']} |",
        f"| 均判非 AI | {result_kw_skill['both_not']} |",
        f"| **重合率** | **{result_kw_skill['agreement']}** |",
        f"| **Cohen's kappa** | **{result_kw_skill['kappa']}** |",
        f"| 关键词 AI 占比 | {result_kw_skill['keyword_ai_rate']} |",
        f"| 技能 AI 占比 | {result_kw_skill['skill_ai_rate']} |",
        "",
        "### 关键词 vs AI 技能词典",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 总岗位 | {result_kw_aik['n']} |",
        f"| 两方法均判 AI | {result_kw_aik['both_ai']} |",
        f"| 仅关键词判 AI | {result_kw_aik['keyword_only']} |",
        f"| 仅 AI 技能判 AI | {result_kw_aik['skill_only']} |",
        f"| 均判非 AI | {result_kw_aik['both_not']} |",
        f"| **重合率** | **{result_kw_aik['agreement']}** |",
        f"| **Cohen's kappa** | **{result_kw_aik['kappa']}** |",
        f"| 关键词 AI 占比 | {result_kw_aik['keyword_ai_rate']} |",
        f"| AI 技能 AI 占比 | {result_kw_aik['skill_ai_rate']} |",
        "",
        "## 不一致样本（关键词 vs AI 技能，人工抽检）",
        "",
    ]

    # AI 技能法不一致样本：关键词判 AI 但无 AI 技能，或有 AI 技能但关键词未命中
    kw_flags = df["is_ai_seed"].astype(bool).reset_index(drop=True)
    ai_flags = ai_skill_flags.reset_index(drop=True)
    out = df.copy().reset_index(drop=True)
    out["ai_skill_flag"] = ai_flags
    kw_only = out[kw_flags & ~ai_flags].copy()
    kw_only["conflict_type"] = "关键词AI但无AI技能"
    ai_only = out[~kw_flags & ai_flags].copy()
    ai_only["conflict_type"] = "有AI技能但关键词未命中"
    kw_only = kw_only.sample(min(args.sample, len(kw_only)), random_state=42)
    ai_only = ai_only.sample(min(args.sample, len(ai_only)), random_state=42)
    for _, r in pd.concat([kw_only, ai_only]).iterrows():
        lines.append(
            f"- [{r['conflict_type']}] `{r['position'][:50]}` "
            f"关键词AI={r['is_ai_seed']} 命中AI技能={r['ai_skill_flag']}"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已写入: %s", report_path)


if __name__ == "__main__":
    main()
