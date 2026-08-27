"""AI 渗透率报告生成模块。

输出逐职业 × 逐年出现矩阵 CSV，以及含渗透率表格与图表的 Markdown 报告。
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("ai_penetration.report")


def build_matrix_csv(
    position_stats: pd.DataFrame,
    position_occ_map: dict[str, dict],
    ai_new: set[str],
    output_path: Path,
) -> Path:
    """构建逐职业 × 逐年出现次数矩阵 CSV。

    Args:
        position_stats: position/year/quarter/count 统计表。
        position_occ_map: position -> {occupation_name, occupation_category}。
        ai_new: AI 新职业名集合。
        output_path: 输出 CSV 路径。

    Returns:
        写入的 CSV 路径。
    """
    pos_to_occ = {
        pos: item["occupation_name"] or pos for pos, item in position_occ_map.items()
    }
    df = position_stats.copy()
    df["occupation"] = df["position"].map(pos_to_occ).fillna(df["position"])
    # occupation × year 透视
    matrix = (
        df.groupby(["occupation", "year"])["count"]
        .sum()
        .reset_index()
        .pivot(index="occupation", columns="year", values="count")
        .fillna(0)
        .astype(int)
    )
    matrix.columns = [f"year_{int(c)}" for c in matrix.columns]
    matrix["is_ai"] = matrix.index.isin(ai_new)
    matrix.to_csv(output_path, encoding="utf-8-sig")
    logger.info("矩阵已写入: %s", output_path)
    return output_path


def _plot_penetration(pen_yearly: pd.DataFrame, output_dir: Path, tag: str) -> Path:
    """绘制总体渗透率年曲线图。"""
    overall = pen_yearly.groupby("year", as_index=False)[["total_occupations", "ai_occupations"]].sum()
    overall["penetration"] = overall["ai_occupations"] / overall["total_occupations"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(overall["year"], overall["penetration"], marker="o")
    ax.set_xlabel("年份")
    ax.set_ylabel("AI 新职业渗透率")
    ax.set_title("AI 职业渗透率（按年）")
    ax.grid(True, alpha=0.3)
    path = output_dir / f"ai_penetration_{tag}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_report(
    pen_yearly: pd.DataFrame,
    pen_quarterly: pd.DataFrame,
    ai_new: set[str],
    matrix_path: Path,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """生成渗透率 Markdown 报告。

    Args:
        pen_yearly: compute_penetration 的年度结果。
        pen_quarterly: compute_quarterly_penetration 的季度结果。
        ai_new: AI 新职业名集合。
        matrix_path: 矩阵 CSV 路径。
        output_dir: 输出目录。
        timestamp: 时间戳字符串。

    Returns:
        报告 MD 路径。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = _plot_penetration(pen_yearly, output_dir, timestamp)

    lines = [
        "# AI 职业渗透率分析报告",
        "",
        f"- 分析时间：{timestamp}",
        "- 数据范围：广东省 21 市招聘数据（2014-2024）",
        "- AI 新职业判定：2022 前零出现 + 2022 后出现 ≥5 次",
        f"- AI 新职业数量：{len(ai_new)}",
        "",
        "## 1. 总体渗透率（按年）",
        "",
        f"![渗透率曲线]({chart_path.name})",
        "",
        "| 年份 | 总职业数 | AI新职业数 | 渗透率 |",
        "|---|---|---|---|",
    ]
    overall = pen_yearly.groupby("year", as_index=False)[["total_occupations", "ai_occupations"]].sum()
    overall["penetration"] = overall["ai_occupations"] / overall["total_occupations"]
    for _, r in overall.iterrows():
        lines.append(
            f"| {int(r['year'])} | {int(r['total_occupations'])} | "
            f"{int(r['ai_occupations'])} | {r['penetration']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. 职业大类分组渗透率（按年）",
            "",
            "| 年份 | 职业大类 | 总职业数 | AI新职业数 | 渗透率 |",
            "|---|---|---|---|---|",
        ]
    )
    for _, r in pen_yearly.iterrows():
        lines.append(
            f"| {int(r['year'])} | {r['category']} | {int(r['total_occupations'])} | "
            f"{int(r['ai_occupations'])} | {r['penetration']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 3. 2022 后季度渗透率",
            "",
            "| 年份 | 季度 | 职业大类 | 总职业数 | AI新职业数 | 渗透率 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for _, r in pen_quarterly.iterrows():
        lines.append(
            f"| {int(r['year'])} | {int(r['quarter'])} | {r['category']} | "
            f"{int(r['total_occupations'])} | {int(r['ai_occupations'])} | "
            f"{r['penetration']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 4. AI 新职业清单",
            "",
            "| 职业名 |",
            "|---|---|",
        ]
    )
    for occ in sorted(ai_new):
        lines.append(f"| {occ} |")

    lines.extend(
        [
            "",
            "## 5. 数据文件",
            "",
            f"- 逐职业矩阵：`{matrix_path}`",
            f"- 渗透率图：`{chart_path}`",
            "",
        ]
    )
    report_path = output_dir / f"ai_penetration_report_{timestamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已写入: %s", report_path)
    return report_path
