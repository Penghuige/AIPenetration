"""add/drop 逐岗人工审计：复现平滑判定并导出真实文本供逐条审核。

在广深 2024 的 TABLESAMPLE 样本上（默认约 20 万行），用与
compare_atier_gz2024 完全相同的估计与判定管线（同一 collect/fit/smooth
函数），标出「新方法比旧方法多判 AI（add）」与「新方法漏判（drop）」的
岗位，连同描述片段与命中的 top 概念名导出 CSV，供逐条人工审计多选取漏选。

使用示例::

    python -X utf8 -m src.ai_penetration.audit_add_drop_gz2024 --sample-rows 200000
"""
from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime

import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .ai_scoring import is_ai_job
from .common import eps_conn_params, setup_logging
from .compare_atier_gz2024 import (
    build_atier_index,
    collect_counts,
    smooth_omega,
    beta_binomial_fit,
    extract_concepts,
)
from .load_guangdong import GD_SHARDS
from .skill_ai_anchor import build_skill_regex, extract_skills_fast, load_merged_skills

logger = logging.getLogger("ai_penetration.audit_add_drop")

TECH_RE = "算法|视觉|图像|数据|挖掘|机器|智能|机器人|嵌入|软件|架构|开发|优化|训练|分析|自动化|规划|控制|感知|建模|AI|LLM|大模型|深度学习"
BIZ_RE = "销售|售前|售后|客服|商务|市场|运营|客户经理|主播|编辑|新媒体|行政|人事|财务|律师|教师|学生"
DROP_MAX = 60
ADD_PER_STRATUM = 60


def _concepts_of(desc: str, automaton, ascii_flags, omega_c: dict, names: dict,
                 homograph: dict | None = None, k: int = 3) -> str:
    """描述命中概念中 ω 前 k 的 "名称:ω" 串。"""
    cs = extract_concepts(desc.lower(), automaton, ascii_flags, homograph)
    top = sorted(((omega_c[c], c) for c in cs if c in omega_c), reverse=True)[:k]
    return " | ".join(f"{names.get(c, '?')}={w:.2f}" for w, c in top)


def main() -> None:
    """审计入口。"""
    parser = argparse.ArgumentParser(description="add/drop 人工审计导出")
    parser.add_argument("--sample-rows", type=int, default=200000)
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.log_dir / "audit_add_drop.log")

    atier = build_atier_index()
    automaton, ascii_flags, _anchors, homograph = atier
    skills_old = load_merged_skills(include_llm=True)
    regex_old = build_skill_regex(skills_old)
    conn = psycopg2.connect(**eps_conn_params())
    cur = conn.cursor()
    cur.execute("SELECT skill_id, coalesce(nullif(canonical_zh,''),canonical_en,'?') "
                "FROM ai_dict.skill_concepts")
    names = dict(cur.fetchall())

    sample: list[tuple[str, str]] = []
    target = args.sample_rows
    for city in ("广州市", "深圳市"):
        shard = GD_SHARDS[city]
        pct = min(2.0, target / 1_000_000 * 100 * 1.4)
        cur.execute(
            f"SELECT position, job_description FROM public.{shard} "
            f"TABLESAMPLE SYSTEM ({pct}) "
            "WHERE substr(publish_time,1,4)='2024' "
            "  AND job_description IS NOT NULL AND job_description != '' "
            "  AND position IS NOT NULL AND position != '' LIMIT %s",
            (target // 2,))
        sample.extend((str(p or ""), str(d)) for p, d in cur.fetchall())
    conn.close()
    logger.info("审计样本: %d 行", len(sample))

    old_counts, concept_counts = collect_counts(sample, atier, regex_old)
    a_o, b_o = beta_binomial_fit(old_counts)
    a_c, b_c = beta_binomial_fit(concept_counts)
    omega_old = smooth_omega(old_counts, a_o, b_o, args.min_count)
    omega_c = smooth_omega(concept_counts, a_c, b_c, args.min_count)

    recs_add: list[dict] = []
    recs_drop: list[dict] = []
    n_add = n_drop = n_add_dual = 0
    rng = random.Random(20260906)
    for pos, d in sample:
        a = is_ai_job(pos, d)
        sk = extract_skills_fast(d, regex_old)
        scored = [omega_old[s] for s in sk if s in omega_old]
        b_old = bool(scored) and sum(scored) / len(scored) >= 0.15 and max(scored) >= 0.5
        cs = extract_concepts(d.lower(), automaton, ascii_flags, homograph)
        scored_c = [omega_c[c] for c in cs if c in omega_c]
        b_new = bool(scored_c) and sum(scored_c) / len(scored_c) >= 0.15 and max(scored_c) >= 0.5
        b_dual = bool(scored_c) and sum(scored_c) / len(scored_c) >= 0.15 \
            and sum(1 for w in scored_c if w >= 0.5) >= 2
        f_old, f_new, f_dual = a or b_old, a or b_new, a or b_dual
        if f_dual and not f_old:
            n_add_dual += 1
        rec = None
        if f_new and not f_old:
            n_add += 1
            rec = {"kind": "add", "position": pos[:50],
                   "a_ai": bool(a), "kept_by_dual": bool(f_dual),
                   "top_concepts": _concepts_of(d, automaton, ascii_flags,
                                                omega_c, names, homograph),
                   "desc_head": d[:230].replace("\n", " ")}
        elif f_old and not f_new:
            n_drop += 1
            rec = {"kind": "drop", "position": pos[:50], "a_ai": bool(a),
                   "top_concepts": _concepts_of(d, automaton, ascii_flags,
                                                omega_c, names, homograph),
                   "old_top": " | ".join(f"{s}={w:.2f}" for s, w in
                                         sorted(((s, omega_old[s]) for s in sk if s in omega_old),
                                                key=lambda x: -x[1])[:3]),
                   "desc_head": d[:230].replace("\n", " ")}
        if rec is not None and rng.random() < 0.3:
            (recs_add if rec["kind"] == "add" else recs_drop).append(rec)

    df = pd.DataFrame(recs_add + recs_drop)
    if df.empty:
        print("无 add/drop 样本，未导出")
        return
    # 分层抽样导出：add 按技术/商业岗位名分层各取 60
    adds = df[df["kind"] == "add"]
    parts = []
    if not adds.empty:
        tech = adds[adds["position"].str.contains(TECH_RE, regex=True, na=False)]
        biz = adds[adds["position"].str.contains(BIZ_RE, regex=True, na=False)]
        oth = adds[~adds.index.isin(tech.index) & ~adds.index.isin(biz.index)]
        for name, sub in (("tech", tech), ("biz", biz), ("other", oth)):
            if not sub.empty:
                parts.append(sub.sample(min(ADD_PER_STRATUM, len(sub)),
                                        random_state=7).assign(stratum=name))
    drops = df[df["kind"] == "drop"]
    if not drops.empty:
        parts.append(drops.sample(min(DROP_MAX, len(drops)),
                                  random_state=7).assign(stratum="drop"))
    export = pd.concat(parts, ignore_index=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = paths.report_dir / f"audit_add_drop_{stamp}.csv"
    export.to_csv(out, index=False, encoding="utf-8-sig")
    total = len(sample)
    print(f"样本 {total} 行: add={n_add}({n_add/total:.3%}) "
          f"add_dual={n_add_dual}({n_add_dual/total:.3%}) drop={n_drop}({n_drop/total:.3%}) "
          f"导出 {len(export)} 条 -> {out.name}")
    logger.info("prior old=(%.2f,%.1f) n=%d | concept=(%.2f,%.1f) n=%d",
                a_o, b_o, len(omega_old), a_c, b_c, len(omega_c))


if __name__ == "__main__":
    main()
