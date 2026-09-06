"""A 级冻结词典概念识别 vs 自建词表——广深 2024 AI 率对比面板（平滑版）。

目的：公平对比「用 A 级词典（激活别名→概念）做技能识别」与现行自建词表
在 fused AI 率上的差异。两套词表在**同一抽样**上用**同一公式**
P(锚点|s) 估频数，并做**同一 Beta-Binomial 经验贝叶斯收缩**（指南 §14.2：
先验由 n>=5 技能的极大似然拟合，不收敛回退 Jeffreys）——消除首轮对比中
"低频概念 ω 抽样极端值 × max 阈值"造成的假阳性与覆盖缺口不对等问题。
A 判定、锚点词集、B 阈值（avg>=0.15 & max>=0.5）两侧一致。

流程：
1. TABLESAMPLE 抽广深 2024（默认 5%≈90 万行），单遍提取两套技能/概念
   集合，统计 (n_s, n_anchor) 频数。
2. 每套词表 fit Beta 先验（MLE）→ 平滑 ω=(n_a+α)/(n+α+β)，min_count=5。
3. ctid 8 路并行全量判定：每行算 a、b_old、b_ctier 及两版 fused，
   聚合 + 记录判定变更示例。eps 全程只读。

使用示例::

    python -X utf8 -m src.ai_penetration.compare_atier_gz2024
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from datetime import datetime

import ahocorasick
import numpy as np
import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .ai_scoring import is_ai_job
from .common import eps_conn_params, setup_logging
from .load_guangdong import GD_SHARDS
from .skill_ai_anchor import (
    AI_ANCHOR_SKILLS,
    _AMBIGUOUS_AI_TERMS,
    build_skill_regex,
    extract_skills_fast,
    load_merged_skills,
)

logger = logging.getLogger("ai_penetration.compare_atier")

YEAR = 2024
CITIES = ("广州市", "深圳市")

_ASCII_RE = re.compile(r"^[\x00-\x7f]+$")

# 商业职能岗位名（审计证实：产品语境"提到≠从事"误报集中在这些岗名——
# 其 B-only 判定需 >=2 个强概念；技术岗名单概念放行，避免误伤
# "视觉算法工程师"等真岗）。"运营"不列入：运营开发/运营算法多为真技术岗。
BIZ_POSITION_RE = re.compile(
    r"销售|售前|售后|客服|商务|市场|BD|客户经理|销售代表|销售总监|"
    r"助理|编辑|记者|储备|行政|人事|前台"
)


# ------------------------------------------------------------- A 级词表构建

def build_atier_index() -> tuple[ahocorasick.Automaton, dict[str, bool], set[str], dict[str, re.Pattern]]:
    """从 ai_dict 构建 A 级概念识别索引。

    Returns:
        (alias 自动机[小写键->(skill_id, alias)], alias->is_ascii 标记,
         锚点概念 skill_id 集合, 同形概念 skill_id->AI语境正则)。
        同形概念：别名恰为"深度学习/强化学习"等口语同形词的概念，
        命中后须正文存在 AI 语境才保留（复用现行 _AMBIGUOUS_AI_TERMS 纪律）。
    """
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT alias, skill_id FROM ai_dict.skill_aliases
            WHERE is_active='1'
        """)
        pairs = cur.fetchall()
        # 锚点概念（alias 口径，与现行词表锚点对齐）：8 锚点词作为
        # canonical 或激活别名出现的概念（如"神经网络"canonical 非该词但别名有）；
        # A 级词典完全没有的锚点词（如"强化学习"）登记告警——不静默吞掉
        anchors = list(AI_ANCHOR_SKILLS)
        anchor_ids: set[str] = set()
        cur.execute("""
            SELECT skill_id, coalesce(canonical_zh,'') FROM ai_dict.skill_concepts
            WHERE coalesce(canonical_zh,'') = ANY(%s)
        """, (anchors,))
        by_canon = cur.fetchall()
        anchor_ids |= {r[0] for r in by_canon}
        cur.execute("""
            SELECT DISTINCT skill_id, alias FROM ai_dict.skill_aliases
            WHERE alias = ANY(%s) AND is_active='1'
        """, (anchors,))
        by_alias = cur.fetchall()
        anchor_ids |= {r[0] for r in by_alias}
        covered = {r[1] for r in by_canon} | {r[1] for r in by_alias}
        missing = sorted(set(anchors) - covered)
        logger.info("锚点概念 %d 个；A 级词典缺失锚点词: %s", len(anchor_ids), missing or "无")
    finally:
        conn.close()
    # 同形概念：激活别名恰为口语同形词（深度学习/强化学习）的概念，
    # 命中后须正文出现 AI 语境才保留（现行词表管线的既有纪律迁移）
    homograph: dict[str, re.Pattern] = {}
    for term, ai_ctx in _AMBIGUOUS_AI_TERMS.items():
        for alias, sid in pairs:
            if alias == term:
                homograph[sid] = ai_ctx
    automaton = ahocorasick.Automaton()
    ascii_flags: dict[str, bool] = {}
    n = 0
    for alias, sid in pairs:
        key = alias.lower()
        if key not in ascii_flags:
            ascii_flags[key] = bool(_ASCII_RE.match(key))
        automaton.add_word(key, (sid, alias))
        n += 1
    automaton.make_automaton()
    logger.info("A 级概念索引: %d 别名键, 锚点概念 %d 个, 同形概念 %d 个",
                n, len(anchor_ids), len(homograph))
    return automaton, ascii_flags, anchor_ids, homograph


def extract_concepts(text_lower: str, automaton: ahocorasick.Automaton,
                     ascii_flags: dict[str, bool],
                     homograph: dict[str, re.Pattern] | None = None) -> set[str]:
    """A 级概念抽取：Aho 命中 + ASCII 词边界校验 + 同形词 AI 语境验证。

    Args:
        text_lower: 小写化但**保留空白**的描述文本（去空白规范化会把
            "客户开发、深度…学习" 跨词粘连成词造成假命中，审计已证实）。
        automaton: build_atier_index 的自动机。
        ascii_flags: 纯 ASCII 别名需词边界校验。
        homograph: 同形概念 id -> AI 语境正则；无 AI 语境则丢弃该概念。

    Returns:
        命中的概念 skill_id 集合。
    """
    sids: set[str] = set()
    for end, (sid, alias) in automaton.iter(text_lower):
        key = alias.lower()
        if ascii_flags.get(key):
            start = end - len(key) + 1
            before = text_lower[start - 1] if start > 0 else ""
            after = text_lower[end + 1] if end + 1 < len(text_lower) else ""
            if before.isalnum() or after.isalnum():
                continue
        sids.add(sid)
    if homograph:
        for sid, ai_ctx in homograph.items():
            if sid in sids and not ai_ctx.search(text_lower):
                sids.discard(sid)
    return sids


def collect_counts(sample_rows: list[tuple[str, str]], atier: tuple,
                   regex_old: re.Pattern) -> tuple[dict, dict]:
    """单遍抽样提取两套词表的技能/概念集合，统计 (n_s, n_anchor) 频数。

    Args:
        sample_rows: (position, description) 抽样。
        atier: build_atier_index 输出。
        regex_old: 自建词表合并正则。

    Returns:
        (old_counts, concept_counts)，均为 {item: (n_jobs, n_anchor_jobs)}。
    """
    automaton, ascii_flags, anchor_ids, homograph = atier
    old_anchor = set(AI_ANCHOR_SKILLS)
    o_n: Counter = Counter()
    o_a: Counter = Counter()
    c_n: Counter = Counter()
    c_a: Counter = Counter()
    for i, (_pos, desc) in enumerate(sample_rows):
        if i % 100000 == 0:
            logger.info("ω 频数统计进度 %d/%d", i, len(sample_rows))
        sk = extract_skills_fast(desc, regex_old)
        for s in sk:
            o_n[s] += 1
        if sk & old_anchor:
            for s in sk:
                o_a[s] += 1
        cs = extract_concepts(desc.lower(), automaton, ascii_flags, homograph)
        for c in cs:
            c_n[c] += 1
        if cs & anchor_ids:
            for c in cs:
                c_a[c] += 1
    old_counts = {s: (n, o_a.get(s, 0)) for s, n in o_n.items()}
    concept_counts = {c: (n, c_a.get(c, 0)) for c, n in c_n.items()}
    return old_counts, concept_counts


def beta_binomial_fit(counts: dict[str, tuple[int, int]], min_fit_n: int = 5,
                      ) -> tuple[float, float]:
    """Beta-Binomial 先验 (α, β) 的极大似然拟合（指南 §14.2）。

    仅用 n>=min_fit_n 的技能拟合；scipy 缺失或不收敛回退 Jeffreys (0.5, 0.5)。

    Args:
        counts: {item: (n_jobs, n_anchor_jobs)}。
        min_fit_n: 参与拟合的技能最低频数。

    Returns:
        (alpha, beta) 先验参数。
    """
    try:
        from scipy.optimize import minimize
        from scipy.special import betaln
    except ImportError:
        logger.warning("scipy 不可用，回退 Jeffreys 先验")
        return 0.5, 0.5
    obs = [(n, a) for n, a in counts.values() if n >= min_fit_n]
    if len(obs) < 50:
        return 0.5, 0.5
    ns = np.asarray([o[0] for o in obs], dtype=float)
    na = np.asarray([o[1] for o in obs], dtype=float)

    def neg_loglik(x: np.ndarray) -> float:
        alpha, beta = np.exp(x)
        return -(betaln(na + alpha, ns - na + beta).sum()
                 - betaln(alpha, beta) * len(obs))

    res = minimize(neg_loglik, x0=np.array([np.log(0.3), np.log(10.0)]),
                   method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4,
                                                  "fatol": 1e-3})
    if not res.success and not np.isfinite(res.fun):
        logger.warning("Beta-BB 先验拟合失败，回退 Jeffreys")
        return 0.5, 0.5
    alpha, beta = np.exp(res.x)
    if not (1e-4 <= alpha <= 1e4 and 1e-4 <= beta <= 1e4):
        return 0.5, 0.5
    return float(alpha), float(beta)


def smooth_omega(counts: dict[str, tuple[int, int]], alpha: float, beta: float,
                 min_count: int = 5) -> dict[str, float]:
    """Beta-Binomial 后验均值收缩：ω = (n_a + α) / (n + α + β)。

    Args:
        counts: {item: (n_jobs, n_anchor_jobs)}。
        alpha, beta: 先验参数。
        min_count: 低于该频数的 item 不给权重（指南 C 级底线 5）。

    Returns:
        {item: omega_smoothed}（n>=min_count 全部入表；低 ω 项保留，
        它们拉低岗位均值正是现行管线抑制"堆砌通用技能模板岗"的机制）。
    """
    out: dict[str, float] = {}
    for item, (n, na) in counts.items():
        if n < min_count:
            continue
        out[item] = (na + alpha) / (n + alpha + beta)
    return out


# ------------------------------------------------------------- 全量判定

_WORKER: dict = {}


def _init_worker(omega_old: dict) -> None:
    _WORKER["omega_old"] = omega_old
    _WORKER["regex_old"] = build_skill_regex(load_merged_skills(include_llm=True))
    _WORKER["atier"] = build_atier_index()


def _blocks(cur, shard: str) -> int:
    cur.execute("SELECT current_setting('block_size')::int")
    bs = int(cur.fetchone()[0])
    cur.execute("SELECT pg_relation_size(%s)", (f"public.{shard}",))
    return max(1, int(cur.fetchone()[0]) // bs)


def scan_slice(shard: str, lo: int, hi: int, omega_c: dict) -> dict:
    """ctid 切片：每行同时算 old/atier 两套 fused，聚合 + 变更示例采样。"""
    rng = random.Random(hash(shard) & 0xFFFF)
    params = eps_conn_params()
    conn = psycopg2.connect(**params)
    out = {
        "total": 0, "a": 0, "b_old": 0, "b_ctier": 0, "b_dual": 0,
        "fused_old": 0, "fused_ctier": 0, "fused_dual": 0,
        "add_examples": [], "drop_examples": [], "add_n": 0, "drop_n": 0,
        "add_dual_n": 0, "drop_dual_n": 0,
    }
    try:
        with conn.cursor() as setup:
            setup.execute("SET LOCAL work_mem = '256MB'")
            setup.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cur = conn.cursor(f"cmp_{shard}_{lo}")
        cur.itersize = 50000
        sql = f"""
            SELECT position, job_description FROM public.{shard}
            WHERE ctid >= '(%s,0)'::tid AND ctid < '(%s,0)'::tid
              AND substr(publish_time, 1, 4) = %s
              AND job_description IS NOT NULL AND job_description != ''
              AND position IS NOT NULL AND position != ''
        """
        cur.execute(sql, (int(lo), int(hi), str(YEAR)))
        omega_old = _WORKER["omega_old"]
        regex_old = _WORKER["regex_old"]
        automaton, ascii_flags, _aid, homograph = _WORKER["atier"]
        while True:
            batch = cur.fetchmany(50000)
            if not batch:
                break
            for pos, desc in batch:
                out["total"] += 1
                d = str(desc or "")
                p = str(pos or "")
                a = is_ai_job(p, d)
                sk = extract_skills_fast(d, regex_old)
                scored = [omega_old[s] for s in sk if s in omega_old]
                b_old = bool(scored) and sum(scored) / len(scored) >= 0.15 \
                    and max(scored) >= 0.5
                cs = extract_concepts(d.lower(), automaton, ascii_flags, homograph)
                scored_c = [omega_c[c] for c in cs if c in omega_c]
                b_c = bool(scored_c) and sum(scored_c) / len(scored_c) >= 0.15 \
                    and max(scored_c) >= 0.5
                # 修正规则（审计驱动）：商业职能岗名的 B-only 判定要求
                # >=2 个概念 omega>=0.5；技术岗名维持单概念（避免误伤）
                n_strong = sum(1 for w in scored_c if w >= 0.5)
                b_d = bool(scored_c) and sum(scored_c) / len(scored_c) >= 0.15 \
                    and (n_strong >= 2
                         or (n_strong >= 1 and not BIZ_POSITION_RE.search(p)))
                f_old, f_c, f_d = a or b_old, a or b_c, a or b_d
                out["a"] += a
                out["b_old"] += b_old
                out["b_ctier"] += b_c
                out["b_dual"] += b_d
                out["fused_old"] += f_old
                out["fused_ctier"] += f_c
                out["fused_dual"] += f_d
                if f_d and not f_old:
                    out["add_dual_n"] += 1
                if f_c and not f_old:
                    out["add_n"] += 1
                    if len(out["add_examples"]) < 120 and rng.random() < 0.05:
                        top = sorted(scored_c, reverse=True)[:3]
                        out["add_examples"].append(
                            {"city": shard, "position": p[:40], "top_omega": [round(t, 3) for t in top],
                             "n_concepts": len(cs), "kept_by_dual": f_d})
                elif f_old and not f_c:
                    out["drop_n"] += 1
                    if len(out["drop_examples"]) < 120 and rng.random() < 0.05:
                        out["drop_examples"].append(
                            {"city": shard, "position": p[:40],
                             "top_omega": [round(t, 3) for t in sorted(scored, reverse=True)[:3]],
                             "n_skills": len(sk)})
        cur.close()
    finally:
        conn.close()
    return out


# ------------------------------------------------------------- 入口

def main() -> None:
    """对比入口：抽样估 ω -> 并行全量判定 -> 报告。"""
    parser = argparse.ArgumentParser(description="A 级词典 vs 自建词表 AI 率对比（广深 2024，平滑版）")
    parser.add_argument("--sample-pct", type=float, default=5.0,
                        help="ω 估计抽样比例（默认 5%%≈90 万行 2024）")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--slices", type=int, default=4)
    parser.add_argument("--min-count", type=int, default=5,
                        help="入表技能最低频数（指南 C 级底线）")
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.log_dir / "compare_atier_gz2024.log")

    # ---- 阶段1：大样本同式估计 + 同式平滑（两套词表对等）
    atier = build_atier_index()
    skills_old = load_merged_skills(include_llm=True)
    regex_old = build_skill_regex(skills_old)
    params = eps_conn_params()
    conn = psycopg2.connect(**params)
    sample: list[tuple[str, str]] = []
    try:
        for city in CITIES:
            shard = GD_SHARDS[city]
            cur = conn.cursor(f"cmp_sample_{shard}")
            cur.itersize = 100000
            cur.execute(
                f"SELECT position, job_description FROM public.{shard} "
                f"TABLESAMPLE SYSTEM ({args.sample_pct}) "
                "WHERE substr(publish_time,1,4)=%s "
                "  AND job_description IS NOT NULL AND job_description != ''",
                (str(YEAR),))
            while True:
                batch = cur.fetchmany(100000)
                if not batch:
                    break
                sample.extend((str(p or ""), str(d)) for p, d in batch)
            cur.close()
    finally:
        conn.close()
    logger.info("ω 估计抽样: %d 行", len(sample))
    old_counts, concept_counts = collect_counts(sample, atier, regex_old)
    a_o, b_o = beta_binomial_fit(old_counts)
    a_c, b_c_prior = beta_binomial_fit(concept_counts)
    omega_old = smooth_omega(old_counts, a_o, b_o, args.min_count)
    omega_c = smooth_omega(concept_counts, a_c, b_c_prior, args.min_count)
    logger.info("平滑完成: 旧词表 prior=(%.3f,%.1f) 入表 %d；概念 prior=(%.3f,%.1f) 入表 %d",
                a_o, b_o, len(omega_old), a_c, b_c_prior, len(omega_c))
    # ω 表落盘（版本纪律 §4：估计产物可追溯，诊断可复用）
    stamp_w = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    (paths.report_dir / f"omega_smooth_legacy_{stamp_w}.json").write_text(
        json.dumps(omega_old, ensure_ascii=False), encoding="utf-8")
    (paths.report_dir / f"omega_smooth_atier_{stamp_w}.json").write_text(
        json.dumps(omega_c, ensure_ascii=False), encoding="utf-8")

    # ---- 阶段2：全量并行判定
    tasks: list[tuple[str, int, int]] = []
    conn = psycopg2.connect(**params)
    try:
        cur = conn.cursor()
        for city in CITIES:
            shard = GD_SHARDS[city]
            total = _blocks(cur, shard)
            step = total // args.slices + 1
            for i in range(args.slices):
                lo = i * step
                hi = 4294967295 if i == args.slices - 1 else min(total, (i + 1) * step)
                if lo < hi:
                    tasks.append((shard, lo, hi))
    finally:
        conn.close()
    logger.info("判定任务: %d 切片", len(tasks))

    from concurrent.futures import ProcessPoolExecutor
    agg = Counter()
    examples: dict[str, list] = {"add": [], "drop": []}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(omega_old,)) as pool:
        futs = [pool.submit(scan_slice, shard, lo, hi, omega_c)
                for shard, lo, hi in tasks]
        for fut in futs:
            r = fut.result()
            for k, v in r.items():
                if isinstance(v, int):
                    agg[k] += v
            examples["add"].extend(r["add_examples"])
            examples["drop"].extend(r["drop_examples"])

    # ---- 阶段3：报告
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = agg["total"] or 1
    rows = {
        "年份": YEAR, "总岗位": total,
        "A率": round(agg["a"] / total, 6),
        "旧B率(自建词表·平滑)": round(agg["b_old"] / total, 6),
        "新B率(A级概念·平滑)": round(agg["b_ctier"] / total, 6),
        "B率修正(双概念门槛)": round(agg["b_dual"] / total, 6),
        "旧fused(平滑)": round(agg["fused_old"] / total, 6),
        "新fused(单概念)": round(agg["fused_ctier"] / total, 6),
        "新fused(双概念修正)": round(agg["fused_dual"] / total, 6),
        "新增判定(add)": agg["add_n"], "新增判定(add·双概念)": agg["add_dual_n"],
        "掉出判定(drop)": agg["drop_n"],
    }
    ex_path = paths.report_dir / f"compare_atier_examples_{stamp}.csv"
    pd.DataFrame(examples["add"] + examples["drop"]).assign(
        direction=["add"] * len(examples["add"]) + ["drop"] * len(examples["drop"])
    ).to_csv(ex_path, index=False, encoding="utf-8-sig")
    md = ["# A 级词典 vs 自建词表：广深 2024 AI 率对比", "",
          "| 指标 | 值 |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in rows.items()]
    md += ["",
           f"- ω 估计样本：{len(sample):,} 行（TABLESAMPLE {args.sample_pct}%，2024 广深）",
           f"- Beta-BB 先验：旧词表 α={a_o:.3f}, β={b_o:.1f}（入表 {len(omega_old)}）；"
           f"概念表 α={a_c:.3f}, β={b_c_prior:.1f}（入表 {len(omega_c)}）",
           f"- min_count={args.min_count}；ω 平滑式 (n_a+α)/(n+α+β)（指南 §14.2）",
           f"- 概念词表：A 级激活别名→22,683 概念；锚点概念 {len(atier[2])} 个",
           "- 两侧同式估计同式平滑，A 判定/锚点/阈值完全一致，差异归因词典内容",
           "- 修正规则（人工审计后加入）：概念层同形词 AI 语境验证 + 匹配保留"
           "空白（消除跨词粘连假命中）+ 双概念门槛（B-only 需 >=2 概念 ω>=0.5，"
           "抑制产品语境单概念击穿）",
           "- 现行面板（快照 ω，无平滑）参考值：广深 2024 A 0.6222% / B 0.4155% / fused 0.7037%",
           f"- 变更示例：`{ex_path.name}`", ""]
    report = paths.report_dir / f"compare_atier_gz2024_{stamp}.md"
    report.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    logger.info("报告: %s", report)


if __name__ == "__main__":
    main()
