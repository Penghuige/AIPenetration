"""AI 渗透率关键词匹配分析模块。

用 AI 相关关键词直接匹配岗位名，计算每年的 AI 渗透率，
作为聚类方法（AI 新职业占比）的独立交叉验证视角。

两个指标：
- 岗位占比：含 AI 关键词的岗位数 / 当年总岗位数
- 职业占比：含 AI 关键词的职业数 / 当年总职业数（职业按岗位名近似）
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from config.paths import get_project_paths

logger = logging.getLogger("ai_penetration.keyword")

# AI 相关关键词集默认值（兜底；正式词典在 dicts/ai_occupation_keywords.txt）
_AI_KEYWORDS_FALLBACK: tuple[str, ...] = (
    "人工智能", "大模型", "LLM", "AIGC", "机器学习", "深度学习",
    "自动驾驶", "智能驾驶", "机器人", "机器视觉", "数据标注",
    "自然语言", "计算机视觉", "语音识别", "提示词", "多模态",
)


def _keyword_file() -> Path:
    """返回 AI 关键词词典路径。"""
    return get_project_paths().project_root / "dicts" / "ai_occupation_keywords.txt"


def _parse_keyword_weights() -> dict[str, float]:
    """解析「词|权重」格式的关键词词典。

    Returns:
        关键词 → 权重 映射；未标注权重时默认 2.0（强词）。
    """
    keyword_file = _keyword_file()
    if not keyword_file.exists():
        return {}
    weights: dict[str, float] = {}
    for line in keyword_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        term = parts[0].strip()
        # 岗位名关键词默认强权重 3（岗位名直接写明 AI 术语是强证据）
        weight = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 3.0
        weights[term] = weight
    return weights


def load_ai_keywords() -> tuple[str, ...]:
    """从 dicts/ai_occupation_keywords.txt 加载 AI 关键词（仅词名）。

    文件不存在时使用内置兜底关键词集。

    Returns:
        AI 关键词元组。
    """
    keyword_file = _keyword_file()
    if not keyword_file.exists():
        logger.warning("AI 关键词词典不存在，使用内置兜底集")
        return _AI_KEYWORDS_FALLBACK
    keywords = tuple(_parse_keyword_weights().keys())
    logger.info("加载 AI 关键词: %d 个（%s）", len(keywords), keyword_file)
    return keywords


# 模块加载时读取词典
AI_KEYWORDS: tuple[str, ...] = load_ai_keywords()
# 关键词权重 {词: 权重}
AI_KEYWORD_WEIGHTS: dict[str, float] = _parse_keyword_weights()

# 公司名特征词：括号内容含这些词时视为公司名，从匹配文本中剔除
# （公司名常含"机器人/智能"导致误报，如"电气工程师(机器人科技有限公司)"）
_COMPANY_MARKERS = (
    "公司", "集团", "有限", "股份", "实业", "科技", "贸易", "电子",
    "制造", "精密", "信息", "网络", "软件", "控股", "能源", "生物",
    "医药", "材料", "文化", "商贸", "咨询", "地产",
)
# 剔除含公司名特征的括号；不含公司特征词的括号（如"（ai中文语料）"）保留
_STRIP_COMPANY_BRACKET_RE = re.compile(
    r"[（(][^）)]*?(?:" + "|".join(_COMPANY_MARKERS) + r")[^）)]*[）)]"
)


def _keyword_matches(kw: str, upper_text: str) -> bool:
    """关键词是否命中岗位名。

    含中文的关键词用子串匹配；纯英文关键词用词边界匹配，
    避免英文子串误报（如 RAG 误匹配 beveRAGe、CNN 误匹配媒体 CNN）。

    Args:
        kw: AI 关键词。
        upper_text: 剔除公司名括号并转大写的岗位名。

    Returns:
        True 表示命中。
    """
    if re.search(r"[一-鿿]", kw):
        return kw in upper_text
    return re.search(rf"(?<![A-Za-z]){re.escape(kw)}(?![A-Za-z])", upper_text) is not None


def match_ai(position: str) -> bool:
    """岗位名是否命中任意 AI 关键词。

    匹配前剔除含公司名特征的括号（公司名常含"机器人/智能"导致误报，
    如"电气工程师(机器人科技有限公司)"），保留岗位限定括号
    （如"（ai中文语料）"）。纯英文关键词用词边界匹配。

    Args:
        position: 岗位名。

    Returns:
        True 表示该岗位与 AI 相关。
    """
    text = position or ""
    text = _STRIP_COMPANY_BRACKET_RE.sub("", text)
    upper = text.upper()
    return any(_keyword_matches(kw, upper) for kw in AI_KEYWORDS)


def compute_keyword_penetration(position_stats: pd.DataFrame) -> pd.DataFrame:
    """按关键词匹配计算每年的 AI 渗透率。

    Args:
        position_stats: position/year/quarter/count 统计表。

    Returns:
        DataFrame，列 year / total_positions / ai_positions / position_rate
        / total_jobs / ai_jobs / job_rate。
    """
    df = position_stats.copy()
    df["is_ai"] = df["position"].map(match_ai)

    rows = []
    for year, group in df.groupby("year"):
        total_positions = group["count"].sum()
        ai_positions = group[group["is_ai"]]["count"].sum()
        total_jobs = group["position"].nunique()
        ai_jobs = group[group["is_ai"]]["position"].nunique()
        rows.append({
            "year": int(year),
            "total_positions": int(total_positions),
            "ai_positions": int(ai_positions),
            "position_rate": ai_positions / total_positions if total_positions else 0.0,
            "total_jobs": int(total_jobs),
            "ai_jobs": int(ai_jobs),
            "job_rate": ai_jobs / total_jobs if total_jobs else 0.0,
        })
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    logger.info("关键词匹配完成: 命中 AI 岗位 %.1f%%", out["position_rate"].mean() * 100)
    return out


def write_keyword_report(
    pen: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """生成关键词渗透率报告 MD。

    Args:
        pen: compute_keyword_penetration 结果。
        output_dir: 输出目录。
        timestamp: 时间戳。

    Returns:
        报告路径。
    """
    lines = [
        "# AI 渗透率关键词匹配分析（交叉验证）",
        "",
        f"- 分析时间：{timestamp}",
        f"- 方法：岗位名命中 AI 关键词集（{len(AI_KEYWORDS)} 个关键词）",
        "",
        "## 每年 AI 渗透率",
        "",
        "| 年份 | 总岗位数 | AI岗位数 | 岗位渗透率 | 总职业数 | AI职业数 | 职业渗透率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in pen.iterrows():
        lines.append(
            f"| {int(r['year'])} | {int(r['total_positions'])} | {int(r['ai_positions'])} | "
            f"{r['position_rate']:.4f} | {int(r['total_jobs'])} | {int(r['ai_jobs'])} | "
            f"{r['job_rate']:.4f} |"
        )
    lines.extend([
        "",
        "## 说明",
        "",
        "- 岗位渗透率 = 含 AI 关键词的岗位数 / 当年总岗位数",
        "- 职业渗透率 = 含 AI 关键词的职业数 / 当年总职业数",
        "- 与聚类方法（AI 新职业占比）互补：关键词方法衡量 AI 相关岗位存量占比，"
        "聚类方法衡量 2022 后新出现的职业占比。",
    ])
    path = output_dir / f"ai_penetration_keyword_{timestamp}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("关键词报告已写入: %s", path)
    return path
