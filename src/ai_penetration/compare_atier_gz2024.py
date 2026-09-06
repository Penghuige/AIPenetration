"""A 级冻结词典概念识别 vs 自建词表——广深 2024 AI 率对比面板。

目的：量化「用刚冻结的外部 A 级词典（激活别名→概念）做技能识别」与
现行自建词表（6,872 词 + ω 快照）在 fused AI 率上的差异。除技能识别层
与 ω 来源外，A 判定、锚点集合、B 阈值（avg>=0.15 & max>=0.5）全部一致，
差异可归因于词典方法本身。

流程：
1. ω 估计（同一抽样）：TABLESAMPLE 抽广深 2024，分别用两套词表抽取技能/
   概念集合，按简化式 P(锚点|s) 估 ω（min_count=20）。
2. 全量判定（ctid 8 路并行）：每行同时算 a_ai（不变）、b_old、b_atier、
   fused_old、fused_atier，聚合计数并记录判定变更示例。
3. 报告：两法率对比 + 变更方向分解 + 示例明细 CSV。eps 全程只读。

使用示例::

    python -X utf8 -m src.ai_penetration.compare_atier_gz2024 --sample-pct 0.5
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
import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .ai_scoring import is_ai_job
from .common import (
    DEFAULT_OMEGA_SNAPSHOT,
    eps_conn_params,
    resolve_artifact_path,
    setup_logging,
)
from .load_guangdong import GD_SHARDS
from .skill_ai_anchor import AI_ANCHOR_SKILLS, build_skill_regex, extract_skills_fast, load_merged_skills

logger = logging.getLogger("ai_penetration.compare_atier")

YEAR = 2024
CITIES = ("广州市", "深圳市")

_ASCII_RE = re.compile(r"^[\x00-\x7f]+$")


# ------------------------------------------------------------- A 级词表构建

def build_atier_index() -> tuple[ahocorasick.Automaton, dict[str, str], set[str]]:
    """从 ai_dict 构建 A 级概念识别索引。

    Returns:
        (alias 自动机[小写键->skill_id], alias->is_ascii 标记,
         锚点概念 skill_id 集合)。
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
    logger.info("A 级概念索引: %d 别名键, 锚点概念 %d 个", n, len(anchor_ids))
    return automaton, ascii_flags, anchor_ids


def extract_concepts(text_lower: str, automaton: ahocorasick.Automaton,
                     ascii_flags: dict[str, bool]) -> set[str]:
    """A 级概念抽取：Aho 命中 + ASCII 词边界校验（对齐现有英文边界惯例）。"""
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
    return sids


def estimate_omega_concepts(sample_rows: list[tuple[str, str]],
                            atier: tuple) -> tuple[dict, int]:
    """在抽样上估 A 级概念的 ω（简化式 P(锚点|概念)，min_count=20）。

    旧词表对照组直接用现行 ω 快照（阶段2 worker 加载），无需在此重估。

    Args:
        sample_rows: (position, description) 抽样。
        atier: build_atier_index 输出。

    Returns:
        ({skill_id: omega}, 参与统计的概念总数)。
    """
    automaton, ascii_flags, anchor_ids = atier
    c_skills: Counter = Counter()
    c_anchor: Counter = Counter()
    for _pos, desc in sample_rows:
        cs = extract_concepts(desc.lower(), automaton, ascii_flags)
        for c in cs:
            c_skills[c] += 1
        if cs & anchor_ids:
            for c in cs:
                c_anchor[c] += 1
    omega_c = {c: c_anchor[c] / n for c, n in c_skills.items()
               if n >= 20 and c_anchor.get(c, 0) > 0}
    return omega_c, len(c_skills)


# ------------------------------------------------------------- 全量判定

_WORKER: dict = {}


def _init_worker(omega_old_path: str) -> None:
    _WORKER["omega_old"] = json.loads(
        resolve_artifact_path(omega_old_path, artifact="ωsAI 快照").read_text(encoding="utf-8"))
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
        "total": 0, "a": 0, "b_old": 0, "b_ctier": 0,
        "fused_old": 0, "fused_ctier": 0,
        "add_examples": [], "drop_examples": [], "add_n": 0, "drop_n": 0,
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
        automaton, ascii_flags, _aid = _WORKER["atier"]
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
                cs = extract_concepts(d.lower(), automaton, ascii_flags)
                scored_c = [omega_c[c] for c in cs if c in omega_c]
                b_c = bool(scored_c) and sum(scored_c) / len(scored_c) >= 0.15 \
                    and max(scored_c) >= 0.5
                f_old, f_c = a or b_old, a or b_c
                out["a"] += a
                out["b_old"] += b_old
                out["b_ctier"] += b_c
                out["fused_old"] += f_old
                out["fused_ctier"] += f_c
                if f_c and not f_old:
                    out["add_n"] += 1
                    if len(out["add_examples"]) < 120 and rng.random() < 0.05:
                        top = sorted(scored_c, reverse=True)[:3]
                        out["add_examples"].append(
                            {"city": shard, "position": p[:40], "top_omega": [round(t, 3) for t in top],
                             "n_concepts": len(cs)})
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
    parser = argparse.ArgumentParser(description="A 级词典 vs 自建词表 AI 率对比（广深 2024）")
    parser.add_argument("--sample-pct", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--slices", type=int, default=4)
    parser.add_argument("--omega-file", type=str, default=DEFAULT_OMEGA_SNAPSHOT)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.log_dir / "compare_atier_gz2024.log")

    # ---- 阶段1：抽样估 ω（A 级概念版）；旧版对照直接用现行 ω 快照
    atier = build_atier_index()
    params = eps_conn_params()
    conn = psycopg2.connect(**params)
    sample: list[tuple[str, str]] = []
    try:
        for city in CITIES:
            shard = GD_SHARDS[city]
            cur = conn.cursor()
            cur.execute(
                f"SELECT position, job_description FROM public.{shard} "
                f"TABLESAMPLE SYSTEM ({args.sample_pct}) "
                "WHERE substr(publish_time,1,4)=%s "
                "  AND job_description IS NOT NULL AND job_description != ''",
                (str(YEAR),))
            sample.extend((str(p or ""), str(d)) for p, d in cur.fetchall())
            cur.close()
    finally:
        conn.close()
    logger.info("ω 估计抽样: %d 行", len(sample))
    omega_c, n_c = estimate_omega_concepts(sample, atier)
    logger.info("概念技能 %d 个有 ω(>=20 且与锚点共现)", n_c)

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
                             initargs=(args.omega_file,)) as pool:
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
        "旧B率(自建词表+快照)": round(agg["b_old"] / total, 6),
        "新B率(A级概念+新估ω)": round(agg["b_ctier"] / total, 6),
        "旧fused": round(agg["fused_old"] / total, 6),
        "新fused": round(agg["fused_ctier"] / total, 6),
        "新增判定(add)": agg["add_n"], "掉出判定(drop)": agg["drop_n"],
    }
    ex_path = paths.report_dir / f"compare_atier_examples_{stamp}.csv"
    pd.DataFrame(examples["add"] + examples["drop"]).assign(
        direction=["add"] * len(examples["add"]) + ["drop"] * len(examples["drop"])
    ).to_csv(ex_path, index=False, encoding="utf-8-sig")
    md = ["# A 级词典 vs 自建词表：广深 2024 AI 率对比", "",
          "| 指标 | 值 |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in rows.items()]
    md += ["", f"- ω 估计样本：{len(sample):,} 行（TABLESAMPLE {args.sample_pct}%，2024）",
           f"- 概念词表：A 级激活别名→22,683 概念；锚点概念 {len(atier[2])} 个",
           "- 两法判定阈值/锚点/A 方法完全一致，差异归因词典识别层",
           f"- 变更示例：`{ex_path.name}`", ""]
    report = paths.report_dir / f"compare_atier_gz2024_{stamp}.md"
    report.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    logger.info("报告: %s", report)


if __name__ == "__main__":
    main()
