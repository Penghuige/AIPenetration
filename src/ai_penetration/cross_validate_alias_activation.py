"""别名激活结果 × 既有 AI 率判定法 交叉验证（文本级 + 词表级审计）。

背景：激活决策依据的是「外部技能中文别名在广深语料的去重频数」，而项目
既有 AI 率方法链（方法 A 加权评分 ai_scoring、fused 锚点共现口径）是独立
构建的另一套词典/判定体系。两者对同一批岗位应给出一致的信号——本脚本抽样
广深岗位，量化两套体系的一致性，作为激活词表质量的第三方参照。

指标：
- 文本级：含新激活别名 vs 方法A/fused 判 AI 的 2×2 一致表、覆盖率与
  lift（AI 岗位中激活别名命中率 / 非 AI 岗位中命中率）、Cohen's kappa；
- 词项级：top 激活别名逐个 lift，识别疑似「模板通用词」（lift≈1 且高频）；
- 对照：既有 AI 技能词典（ai_skill_terms）中已知中文强词是否落在激活集内。

使用示例::

    python -m src.ai_penetration.cross_validate_alias_activation --per-city 10000
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime

import ahocorasick
import pandas as pd
import psycopg2

from config.paths import get_project_paths

from .ai_scoring import is_ai_job
from .common import DEFAULT_OMEGA_SNAPSHOT, eps_conn_params, resolve_artifact_path, setup_logging
from .skill_ai_anchor import build_skill_regex, is_ai_fused, load_merged_skills

logger = logging.getLogger("ai_penetration.xval_activation")


def load_latest_activation() -> tuple[set[str], int]:
    """读取最近一次激活 manifest 的候选别名集（口径与写库一致）。

    Returns:
        (alias 字符串集合, 使用的阈值)。
    """
    report_dir = get_project_paths().report_dir
    manifests = sorted(report_dir.glob("alias_activation_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(
            "无 alias_activation_manifest_*.json——先运行 zh_alias_activation（dry-run 亦可）"
        )
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    cand_path = report_dir / manifest["candidates_csv"]
    aliases: set[str] = set()
    with cand_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            aliases.add(row["alias"])
    logger.info("激活候选 %d 别名（threshold=%d, manifest=%s）",
                len(aliases), manifest["min_freq"], manifests[-1].name)
    return aliases, int(manifest["min_freq"])


def sample_jobs(per_city: int) -> pd.DataFrame:
    """TABLESAMPLE 抽取广深岗位样本（轻量，不触全表）。

    Args:
        per_city: 每城市目标行数。

    Returns:
        DataFrame[city, platform, position, job_description]。
    """
    from .load_guangdong import GD_SHARDS

    conn = psycopg2.connect(**eps_conn_params())
    frames = []
    try:
        for city in ("广州市", "深圳市"):
            shard = GD_SHARDS[city]
            pct = min(1.0, max(0.02, per_city / 40_000_000 * 100 * 1.2))
            cur = conn.cursor()
            cur.execute(
                f"SELECT position, job_description FROM public.{shard} "
                f"TABLESAMPLE SYSTEM ({pct}) "
                "WHERE job_description IS NOT NULL AND job_description != '' "
                "  AND position IS NOT NULL AND position != '' "
                "LIMIT %s",
                (per_city,),
            )
            rows = cur.fetchall()
            frames.append(pd.DataFrame(rows, columns=["position", "job_description"]))
            frames[-1]["city"] = city
            logger.info("%s 抽样 %d 行（pct=%.3f%%）", city, len(rows), pct)
    finally:
        conn.close()
    return pd.concat(frames, ignore_index=True)


def build_activation_automaton(aliases: set[str]) -> ahocorasick.Automaton:
    """为激活别名构建小写 Aho 自动机（匹配小写化描述文本）。"""
    automaton = ahocorasick.Automaton()
    for alias in aliases:
        automaton.add_word(alias.lower(), alias)
    automaton.make_automaton()
    return automaton


def cohen_kappa(a_both: int, a_only: int, b_only: int, both_not: int) -> float:
    """二分类一致性 Cohen's kappa。"""
    n = a_both + a_only + b_only + both_not
    if n == 0:
        return 0.0
    po = (a_both + both_not) / n
    pe = (((a_both + a_only) * (a_both + b_only))
          + ((b_only + both_not) * (a_only + both_not))) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def agreement_row(df: pd.DataFrame, hit_col: str) -> dict:
    """计算 命中激活别名 × AI判定 的一致表与 lift/kappa。

    Args:
        df: 含 hit_col(bool) 与 ai 判定列的样本表。
        hit_col: 激活别名命中标记列名。

    Returns:
        {ai, not_ai, both, only_hit, only_ai, neither, coverage_ai,
         coverage_not_ai, lift, kappa}。
    """
    out: dict[str, dict] = {}
    for ai_col in ("ai_a", "ai_fused"):
        both = int((df[hit_col] & df[ai_col]).sum())
        only_hit = int((df[hit_col] & ~df[ai_col]).sum())
        only_ai = int((~df[hit_col] & df[ai_col]).sum())
        neither = int((~df[hit_col] & ~df[ai_col]).sum())
        cov_ai = both / max(1, both + only_ai)
        cov_not = only_hit / max(1, only_hit + neither)
        out[ai_col] = {
            "both": both, "only_hit": only_hit, "only_ai": only_ai,
            "neither": neither, "coverage_ai": round(cov_ai, 4),
            "coverage_not_ai": round(cov_not, 4),
            "lift": round(cov_ai / cov_not, 2) if cov_not else float("inf"),
            "kappa": round(cohen_kappa(both, only_hit, only_ai, neither), 4),
        }
    return out


def per_alias_lift(df: pd.DataFrame, alias_col: str, top: int) -> pd.DataFrame:
    """top 激活别名逐个的 AI 样本命中 lift（explode 后分组统计）。"""
    tmp = df.explode(alias_col, ignore_index=True)
    tmp = tmp[tmp[alias_col].notna()]
    stats = tmp.groupby(alias_col).agg(
        n=("ai_a", "size"), ai_n=("ai_a", "sum"),
    ).reset_index()
    stats["ai_rate_in_hit"] = stats["ai_n"] / stats["n"]
    base = float(df["ai_a"].mean())
    stats["lift"] = stats["ai_rate_in_hit"] / base if base else 0.0
    return stats.sort_values("n", ascending=False).head(top)


def known_ai_terms_coverage(activated: set[str]) -> dict:
    """对照：既有方法A AI 技能词典中的中文词在激活集/样本频数的覆盖。

    Returns:
        {总数, 在激活集, 在候选但未激活, 不在候选}。
    """
    from .skill_dictionary import load_ai_skill_weights

    weights = load_ai_skill_weights()
    zh_terms = [t for t in weights if len(t) >= 2 and any("一" <= c <= "鿿" for c in t)]
    in_act = [t for t in zh_terms if t in activated]
    return {
        "known_zh_terms": len(zh_terms),
        "in_activated": len(in_act),
        "rate": round(len(in_act) / len(zh_terms), 3) if zh_terms else 0,
        "examples_in": in_act[:15],
    }


def main() -> None:
    """交叉验证入口。"""
    parser = argparse.ArgumentParser(description="激活别名 × AI 率方法交叉验证")
    parser.add_argument("--per-city", type=int, default=10000)
    parser.add_argument("--omega-file", type=str, default=DEFAULT_OMEGA_SNAPSHOT)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.log_dir / "xval_alias_activation.log")

    activated, threshold = load_latest_activation()
    df = sample_jobs(args.per_city)
    logger.info("样本 %d 行，开始判定", len(df))

    omega_path = resolve_artifact_path(args.omega_file, artifact="ωsAI 分数快照")
    omega_scores = json.loads(omega_path.read_text(encoding="utf-8"))
    fused_regex = build_skill_regex(load_merged_skills(include_llm=True))

    act_autom = build_activation_automaton(activated)

    hits_list: list[list[str]] = []
    ai_a_list: list[bool] = []
    ai_f_list: list[bool] = []
    for position, desc in zip(df["position"], df["job_description"], strict=True):
        d = str(desc)
        low = d.lower()
        hits_list.append(sorted({v for _e, v in act_autom.iter(low)}))
        ai_a_list.append(is_ai_job(str(position), d))
        ai_f_list.append(is_ai_fused(str(position), d, omega_scores, fused_regex)[2])
    df["activated_hits"] = hits_list
    df["has_activated"] = [len(h) > 0 for h in hits_list]
    df["ai_a"] = ai_a_list
    df["ai_fused"] = ai_f_list

    agree = agreement_row(df, "has_activated")
    top_lift = per_alias_lift(df, "activated_hits", 60)
    coverage = known_ai_terms_coverage(activated)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = paths.report_dir / f"xval_alias_activation_detail_{stamp}.csv"
    df[["city", "position", "has_activated", "activated_hits", "ai_a", "ai_fused"]].to_csv(
        out_csv, index=False, encoding="utf-8-sig")
    lift_csv = paths.report_dir / f"xval_alias_top_lift_{stamp}.csv"
    top_lift.to_csv(lift_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# 别名激活 × AI 率方法 交叉验证报告",
        "",
        f"- 时间：{datetime.now():%Y-%m-%d %H:%M}；样本：广深 {len(df)} 行（TABLESAMPLE）",
        f"- 激活集：{len(activated)} 别名（threshold freq>={threshold}）",
        "- AI 率参照：方法A is_ai_job（dicts/ai_skill_terms.txt 加权）"
        f" + fused（ω={omega_path.name}）",
        "",
        "## 一、文本级一致性（含激活别名 vs AI 判定）",
        "",
        "| 参照口径 | both | 仅激活命中 | 仅AI判定 | 均无 | AI样本覆盖 | 非AI命中 | lift | kappa |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ai_col, r in agree.items():
        lines.append(
            f"| {ai_col} | {r['both']} | {r['only_hit']} | {r['only_ai']} | "
            f"{r['neither']} | {r['coverage_ai']:.3f} | {r['coverage_not_ai']:.3f} | "
            f"{r['lift']:.2f} | {r['kappa']:.3f} |"
        )
    lines += [
        "",
        "解读：lift 高（≫1）= 激活别名命中强烈偏向 AI 岗位，激活集与既有",
        "AI 率体系方向一致；kappa 为绝对一致度（体系不同不会到 1，>0.3 即强相关）。",
        "非AI命中（only_hit）是扩展价值所在：这些是既有 AI 词典漏掉、",
        "外部技能中文别名新捕获的岗位，top_lift 表逐项审计。",
        "",
        "## 二、与既有 AI 技能词典的覆盖对照",
        "",
        f"- 已知中文 AI 词 {coverage['known_zh_terms']} 个，其中已激活 "
        f"{coverage['in_activated']} 个（{coverage['rate']:.1%}）",
        f"- 例子：{('、'.join(coverage['examples_in'])) or '无'}",
        "",
        "## 三、产出文件",
        "",
        f"- 样本明细：`{out_csv.name}`",
        f"- top 词项 lift：`{lift_csv.name}`",
    ]
    report = paths.report_dir / f"xval_alias_activation_{stamp}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告: %s", report)
    print(json.dumps({"agree": agree, "coverage": {k: v for k, v in coverage.items() if k != 'examples_in'}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
