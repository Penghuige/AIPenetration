"""v2d 词表治理（2026-09-09）：legacy 层按指南 §8-§10 通道化 + tautological 剔除。

背景（docs/10 缺陷登记 F1 与合规对照）：自建 6,872 词当年为补 A 级盲区直接
进生产词表，未过"频数验证→阈值激活→QC"通道；193 个 tautological 键（技能
文本自身命中 main 锚点 ⇒ ω≡1.00，恒等式非证据）正是绕过闸门的产物。

治理规则（本模块实现，产 v1.2 激活层）：
1. **taut 剔除（两层）**：技能键 × main 锚点正则逐键判定；A 级 36 键不动词典、
   仅标记 score_excluded（得分层排除，§12 锚点与 §7 概念保留）；
2. **legacy 频数门**：主样本 pooled 计数（skill_ai_counts main/pooled 的
   n_skill，去重岗位口径——与 A 级激活所用的词典语料不同，QC 报告如实标注）
   ≥ MIN_FREQ 才激活；未过者降 D 级候选；
3. 同形守卫词条随表披露（发布痕迹，修"守卫无 artifact"半合规项）。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.lexicon_v2d          # 产治理表+QC
    python -X utf8 -m src.ai_penetration.panel_v2.lexicon_v2d --check  # 只打印计数
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from .anchors import _COMPILED
from .export_release import _meta
from .lexicon import LEGACY_PREFIX, build_union_lexicon

logger = logging.getLogger("ai_penetration.panel_v2.lexicon_v2d")

MIN_FREQ = 100            # 与 A 级别名激活阈值同值（接收方裁量，同先例）
LEXICON_VERSION = "bilingual_a_frozen_v1.1+legacy_union_v1.2"


def tautological_keys(lex) -> dict[str, list[str]]:
    """技能键文本自身被 main 锚点命中 ⇒ ω≡1 构造性同义反复。

    Args:
        lex: 已构建 UnionLexicon。

    Returns:
        {match_key: [命中锚点组,...]}（键序无关，组排序稳定）。
    """
    pats = _COMPILED["main"]
    out: dict[str, list[str]] = {}
    for key in lex.keys_map:
        groups = sorted({g for g, _t, _r, p in pats if p.search(key)})
        if groups:
            out[key] = groups
    return out


def governance_frame(lex, taut: dict[str, list[str]],
                     pooled_n: dict[str, int]) -> pd.DataFrame:
    """legacy 词逐词治理表（纯函数，可离线测）。

    Args:
        lex: union 词表（用 keys_map/homograph）。
        taut: ``tautological_keys`` 输出。
        pooled_n: skill_id → 主样本 pooled 频数（缺失记 0）。

    Returns:
        DataFrame[term(match_key), skill_id, tier, pooled_freq,
        taut_anchor_groups, homograph_guard, disposition]，
        disposition ∈ {activated_keep, removed_taut, removed_lowfreq}
        （低优先级：taut 判定先于频数）。
    """
    rows = []
    for key, sid in lex.keys_map.items():
        tier = "legacy" if sid.startswith(LEGACY_PREFIX) else "atier"
        tg = taut.get(key, [])
        if tg:
            disp = "score_excluded_taut_atier" if tier == "atier" else "removed_taut"
        elif tier == "legacy":
            n = int(pooled_n.get(sid, 0))
            disp = "removed_lowfreq" if n < MIN_FREQ else "activated_keep"
        else:
            disp = "atier_keep"
        rows.append({
            "term": key, "skill_id": sid, "tier": tier,
            "pooled_freq": int(pooled_n.get(sid, 0)),
            "taut_anchor_groups": "|".join(tg),
            "homograph_guard": int(sid in lex.homograph),
            "disposition": disp,
        })
    return pd.DataFrame(rows).sort_values(
        ["disposition", "tier", "term"], kind="stable").reset_index(drop=True)


def _pooled_freq(rel: Path) -> dict[str, int]:
    """从发布 counts（main×pooled）取 skill_id→n_skill（去重岗位频数）。"""
    counts = pq.read_table(rel / "skill_ai_counts.parquet",
                           columns=["skill_code", "anchor_version",
                                    "window_type", "n_skill"]).to_pandas()
    cm = counts[(counts.anchor_version == "main")
                & (counts.window_type == "pooled")]
    vocab = json.loads((get_project_paths().output_dir / "panel_v2" / "pass2"
                        / "skill_vocab.json").read_text(encoding="utf-8"))
    code2sid = {v: k for k, v in vocab.items()}
    return {code2sid[c]: int(n) for c, n in
            zip(cm.skill_code.to_numpy(), cm.n_skill.to_numpy())
            if c in code2sid}


def run(check_only: bool = False) -> pd.DataFrame:
    """生产路径重建词表→治理→落盘（词典件+QC 报告+排除清单）。"""
    paths = get_project_paths()
    from ..skill_ai_anchor import load_merged_skills
    from .lexicon import _load_atier_aliases
    lex = build_union_lexicon(legacy_terms=load_merged_skills(include_llm=True),
                              aliases=_load_atier_aliases())
    taut = tautological_keys(lex)
    pooled = _pooled_freq(paths.output_dir / "release" / "panel_v2")
    frame = governance_frame(lex, taut, pooled)
    n = frame.disposition.value_counts()
    logger.info("v2d 治理: 键 %d | legacy 保留 %d / taut 剔除 %d / 低频降D %d"
                " | A级得分排除 %d", len(frame),
                int(n.get("activated_keep", 0)), int(n.get("removed_taut", 0)),
                int(n.get("removed_lowfreq", 0)),
                int(n.get("score_excluded_taut_atier", 0)))
    if check_only:
        print(frame.disposition.value_counts().to_string())
        return frame
    dic = paths.output_dir / "dictionary"
    dic.mkdir(parents=True, exist_ok=True)
    lg = frame[frame.tier == "legacy"]
    keep = lg[lg.disposition == "activated_keep"]
    csv = dic / "skill_legacy_activated_v1.2.csv"
    keep.to_csv(csv, index=False, encoding="utf-8-sig")
    _meta(csv, run_id=datetime.now().strftime("%Y%m%d_v2d_gov"),
          primary_key="skill_id",
          source_files="panel_v2/lexicon_v2d.py(§8通道化)",
          dictionary_version=LEXICON_VERSION)
    excl = frame[frame.disposition.isin(
        ["removed_taut", "score_excluded_taut_atier"])]
    ecsv = dic / "v2d_taut_excluded_skill_ids.csv"
    excl.to_csv(ecsv, index=False, encoding="utf-8-sig")
    _meta(ecsv, run_id=datetime.now().strftime("%Y%m%d_v2d_gov"),
          primary_key="skill_id",
          source_files="panel_v2/lexicon_v2d.py(taut 枚举完备)",
          dictionary_version=LEXICON_VERSION)
    stamp = dic / "v2d_lexicon_version.json"
    stamp.write_text(json.dumps(
        {"lexicon_version": LEXICON_VERSION,
         "min_freq": MIN_FREQ, "keys_total": len(frame),
         "counts": n.to_dict()}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    if not check_only:
        rep = paths.report_dir / (
            f"legacy_governance_qc_{datetime.now():%Y%m%d_%H%M%S}.md")
        top_keep = keep.nlargest(15, "pooled_freq")[["term", "pooled_freq"]]
        top_drop = lg[lg.disposition == "removed_lowfreq"].nlargest(
            15, "pooled_freq")[["term", "pooled_freq"]]
        rep.write_text("\n".join([
            "# legacy 词 §8 通道化 QC（v1.2，2026-09-09）", "",
            f"- 词表键 {len(frame):,}（A级 {int((frame.tier=='atier').sum()):,}"
            f" / legacy {len(lg):,}）；频数门 MIN_FREQ={MIN_FREQ}"
            "（主样本 pooled 去重岗位口径；A 级激活当时用词典语料 1.01 亿行，"
            "口径差异如实记录）",
            f"- legacy：保留 {len(keep):,} / taut 剔除 "
            f"{int((lg.disposition=='removed_taut').sum())} / 低频降D "
            f"{int((lg.disposition=='removed_lowfreq').sum())}",
            f"- A 级 taut（仅得分排除，不动词典）："
            f"{int(n.get('score_excluded_taut_atier', 0))}",
            f"- 同形守卫词条：{int(frame.homograph_guard.sum())} 键"
            "（发布痕迹见本表 homograph_guard 列）",
            "", "## 保留 legacy 词频 top15", top_keep.to_markdown(index=False),
            "", "## 低频降D 中频最高 15（边界人工复核候选）",
            top_drop.to_markdown(index=False), "",
            "## 边界人工复核指引",
            "removed_lowfreq 中 pooled_freq 接近 100 者可申请豁免；"
            "activated_keep 中含营销热词特征（工具名/产品名）者由 F2 讨论，"
            "本轮不处置（带位置规避）。",
        ]), encoding="utf-8")
        logger.info("QC 报告: %s", rep)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="v2d legacy 词 §8 通道化治理")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_lexicon_v2d.log")
    run(check_only=args.check)


if __name__ == "__main__":
    main()
