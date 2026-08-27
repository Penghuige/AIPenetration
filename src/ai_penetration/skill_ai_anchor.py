"""技能 AI 相关度（技能-技能共现锚点法，参考 Babina et al. 2024 / 方颖等 2026）。

与已弃用的方法二（技能-岗位共现）不同，本方法用「技能与 AI 核心锚点技能的共现」
计算每项技能的人工智能相关度 ωsAI，不依赖岗位名是否命中 AI 关键词。

公式：
    ωsAI(s) = P(岗位含锚点技能 | 岗位含技能 s) = N(s ∩ anchor) / N(s)

- 锚点技能：AI/ML/图像识别/NLP 等 AI 核心技能（须在技能词典中）
- 技能 s 越常与锚点共现，越可能是 AI 专用技能
- 岗位 AI 相关度 ωjAI = 该岗位全部技能 ωsAI 的平均值

注意：技能词典覆盖决定召回。当前通用技能词典（约 2000 项）AI 技能覆盖不足，
需与 AI 技能词表合并，后续再用 LLM 挖掘扩充。
"""
from __future__ import annotations

import logging
import re
from collections import Counter

import pandas as pd

logger = logging.getLogger("ai_penetration.skill_anchor")

# 歧义词口语语境排除：招聘文案中同形异义词（非 AI 语境）
# 如"具有深度学习能力"的"深度学习"= 深入学习（口语），非 AI 技术
# 判定：若正文存在该词的 AI 语境（后接技术后缀/非口语标记）则保留，否则剔除
_AMBIGUOUS_AI_TERMS: dict[str, re.Pattern] = {
    # 深度学习：排除"深度学习能力/精神/意识/态度/习惯/思考/领悟/钻研/学习"
    # （口语"深入学习"）；保留"深度学习模型/算法/框架/训练"等 AI 语境
    "深度学习": re.compile(r"深度学习(?!能力|精神|意识|态度|习惯|思考|领悟|钻研|学习|力)"),
    # 强化学习：排除"强化学习能力/学习"（口语"加强学习"）；保留强化学习算法等
    "强化学习": re.compile(r"强化学习(?!能力|学习)"),
}


# AI 核心锚点技能（须在合并技能词典中存在）
AI_ANCHOR_SKILLS: tuple[str, ...] = (
    "机器学习",
    "深度学习",
    "计算机视觉",
    "图像识别",
    "语音识别",
    "自然语言处理",
    "强化学习",
    "神经网络",
)


def build_skill_regex(skills: list[str]) -> re.Pattern:
    """将技能词合并为单个正则（最长词优先，一次扫描匹配全部）。

    Args:
        skills: 技能词列表。

    Returns:
        编译后的正则。
    """
    ordered = sorted({s for s in skills if s}, key=len, reverse=True)
    return re.compile("|".join(re.escape(s) for s in ordered))


def extract_skills_fast(desc: str, regex: re.Pattern) -> set[str]:
    """用合并正则快速抽取描述中的技能。

    描述截断到前 4000 字符：技能通常出现在"岗位职责/任职要求"前部，
    截断可避免超长描述触发正则回溯卡死（历史出现过 99% CPU 空转）。
    对歧义词（深度学习/强化学习）做口语语境排除：若正文中该词只以口语
    形式出现（如"具有深度学习能力"= 深入学习），则剔除该技能。

    Args:
        desc: 岗位描述。
        regex: build_skill_regex 输出。

    Returns:
        命中的技能集合（最长词优先，已排除口语歧义）。
    """
    if not desc:
        return set()
    text = desc[:4000]
    hits = set(regex.findall(text))
    for term, ai_context_re in _AMBIGUOUS_AI_TERMS.items():
        if term in hits and not ai_context_re.search(text):
            # 该词在正文中只以口语语境出现，非 AI 技术
            hits.discard(term)
    return hits


def compute_anchor_ai_scores(
    jobs_skills: list[set[str]],
    anchors: tuple[str, ...] = AI_ANCHOR_SKILLS,
    min_count: int = 20,
) -> pd.DataFrame:
    """计算每项技能的 AI 相关度 ωsAI（与锚点共现比例）。

    Args:
        jobs_skills: 每个岗位抽取出的技能集合列表。
        anchors: AI 锚点技能。
        min_count: 技能最低出现次数（低于则剔除，避免稀疏噪声）。

    Returns:
        DataFrame，列 skill / n_jobs / n_anchor / omega_ai。
    """
    anchor_set = set(anchors)
    n_skills: Counter = Counter()
    n_anchor: Counter = Counter()
    for skills in jobs_skills:
        for s in skills:
            n_skills[s] += 1
        if skills & anchor_set:  # 该岗位含锚点
            for s in skills:
                n_anchor[s] += 1
    rows = []
    for s, n in n_skills.items():
        if n < min_count:
            continue
        na = n_anchor.get(s, 0)
        rows.append({
            "skill": s,
            "n_jobs": n,
            "n_anchor": na,
            "omega_ai": na / n if n else 0.0,
        })
    df = pd.DataFrame(rows).sort_values("omega_ai", ascending=False)
    logger.info("锚点共现计算完成: %d 个技能", len(df))
    return df


def compute_anchor_ai_scores_full(
    jobs_skills: list[set[str]],
    anchors: tuple[str, ...] = AI_ANCHOR_SKILLS,
    min_count: int = 20,
) -> pd.DataFrame:
    """计算技能 AI 相关度 ωsAI（Babina 式相对似然比，显式区分非 AI 场景）。

    公式：ωsAI(s) = P(s | 含锚点) / P(s | 不含锚点)

    - P(s | 含锚点) = 含 s 且含锚点的岗位数 / 含锚点岗位总数
    - P(s | 不含锚点) = 含 s 且不含锚点的岗位数 / 不含锚点岗位总数
    - 用 +1 平滑避免除零；纯 AI 技能（几乎只与锚点共现）比值高
    - 与简化版 P(锚点|s) 的区别：完整式显式用"非锚点岗位"做分母，
      对既出现于 AI 也出现于传统场景的边缘技能判别更锐利

    Args:
        jobs_skills: 每岗位技能集合列表。
        anchors: AI 锚点技能。
        min_count: 技能最低出现次数。

    Returns:
        DataFrame，列 skill / n_jobs / n_anchor / omega_ai（相对似然比）。
    """
    anchor_set = set(anchors)
    n_anchor_jobs = sum(1 for sk in jobs_skills if sk & anchor_set)
    n_noanchor = len(jobs_skills) - n_anchor_jobs
    n_s_anchor: Counter = Counter()
    n_s_noanchor: Counter = Counter()
    for skills in jobs_skills:
        has_anchor = bool(skills & anchor_set)
        for s in skills:
            if has_anchor:
                n_s_anchor[s] += 1
            else:
                n_s_noanchor[s] += 1
    rows = []
    for s in set(n_s_anchor) | set(n_s_noanchor):
        n_a = n_s_anchor.get(s, 0)
        n_na = n_s_noanchor.get(s, 0)
        if n_a + n_na < min_count:
            continue
        p_a = n_a / n_anchor_jobs if n_anchor_jobs else 0.0
        p_na = (n_na + 1) / (n_noanchor + 1) if n_noanchor else 0.0
        ratio = p_a / p_na if p_na else 999.0
        rows.append({
            "skill": s,
            "n_jobs": n_a + n_na,
            "n_anchor": n_a,
            "omega_ai": ratio,
        })
    df = pd.DataFrame(rows).sort_values("omega_ai", ascending=False)
    logger.info("锚点共现（完整式）计算完成: %d 个技能", len(df))
    return df


def is_ai_fused(
    position: str,
    description: str,
    omega_scores: dict[str, float],
    regex: re.Pattern,
    b_threshold: float = 0.15,
    min_max_omega: float = 0.5,
) -> tuple[bool, bool, bool]:
    """融合判定：方法 A（加权）或 方法 B（锚点共现，带噪声过滤）判 AI 即 AI。

    两法互补：A 擅长抓"岗位名 AI"岗（AI产品经理/数据标注），B 擅长抓
    "描述技能 AI"岗（推荐系统/大数据ML）。取并集覆盖盲区。

    B 判定带**噪声过滤**：B-only（A 不判）岗位需更高的 ωjAI 阈值且至少一个
    强 AI 技能（max ωsAI >= min_max_omega），排除模板/词误匹配误判
    （如"薪资算法"送餐、"图像处理"美工、模板提"深度学习"的绿建岗）。

    Args:
        position: 岗位名。
        description: 岗位描述。
        omega_scores: skill -> ωsAI。
        regex: 技能合并正则。
        b_threshold: B-only 岗位的 ωjAI 判定阈值（默认 0.15，高于纯 B 的 0.10）。
        min_max_omega: B-only 岗位需至少一个技能的 ωsAI >= 此值。

    Returns:
        (方法A判定, 方法B判定, 融合判定)。
    """
    from .ai_scoring import is_ai_job

    a_ai = is_ai_job(position, description)
    hits = extract_skills_fast(description, regex)
    scored = [omega_scores[s] for s in hits if s in omega_scores]
    b_ai = False
    if scored:
        omega_j = sum(scored) / len(scored)
        b_ai = omega_j >= b_threshold and max(scored) >= min_max_omega
    return a_ai, b_ai, (a_ai or b_ai)


def job_ai_relevance_anchor(
    jobs_skills: list[set[str]],
    omega_scores: dict[str, float],
) -> pd.Series:
    """计算每个岗位的 AI 相关度 ωjAI（技能 ωsAI 平均值）。

    Args:
        jobs_skills: 每个岗位的技能集合。
        omega_scores: skill -> ωsAI 映射。

    Returns:
        每个岗位的 AI 相关度 Series。
    """
    rel = []
    for skills in jobs_skills:
        scored = [omega_scores[s] for s in skills if s in omega_scores]
        rel.append(sum(scored) / len(scored) if scored else 0.0)
    return pd.Series(rel)


def load_merged_skills(include_llm: bool = True) -> list[str]:
    """合并通用技能词典 + AI 技能词表 + LLM 挖掘技能。

    Args:
        include_llm: 是否纳入 LLM 挖掘词典（dicts/ai_skill_terms_llm.txt）。

    Returns:
        去重后的技能词列表。
    """
    from config.paths import get_project_paths

    from .skill_dictionary import load_ai_skill_terms, load_skill_names

    merged = set(load_skill_names())
    merged |= set(load_ai_skill_terms())
    if include_llm:
        llm_path = (
            get_project_paths().project_root / "dicts" / "ai_skill_terms_llm.txt"
        )
        if llm_path.exists():
            llm_terms = {
                line.strip()
                for line in llm_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            }
            merged |= llm_terms
            logger.info("纳入 LLM 挖掘词典: %d 项", len(llm_terms))
    skills = [s for s in merged if s]
    logger.info("合并技能词典: %d 项", len(skills))
    return skills
