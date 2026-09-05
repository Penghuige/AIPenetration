"""中文/混合别名激活执行（交接冻结前置·步骤3）。

依据 ai_dict.zh_alias_freq（去重频数）与 ai_dict.alias_ambiguity（歧义表），
把「freq_total >= 阈值 且 不在歧义表」的未激活 zh/mixed 别名置为 is_active='1'
并记录 activation_reason；零命中/低命中/歧义别名维持非激活（指南 §7.6.4、
§11.1.3：一个表面命中不得同时扩展为多个 skill_id）。

激活仅涉及 zh/mixed；en 来源别名的 is_active 不在本脚本管辖内。
同名多技能（n_skills>1）的别名一律不激活。

默认 dry-run 只输出候选清单；--apply 才写库（CLAUDE.md §27 惯例）。

使用示例::

    python -m src.ai_penetration.zh_alias_activation --min-freq 100            # 预览
    python -m src.ai_penetration.zh_alias_activation --min-freq 100 --apply    # 执行
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import psycopg2

from config.paths import get_project_paths

from .common import eps_conn_params, setup_logging

logger = logging.getLogger("ai_penetration.alias_activation")

# 激活依据标记（写入 activation_reason，可回溯）
RUN_DATE = datetime.now().strftime("%Y%m%d")


def load_candidates(min_freq: int) -> list[tuple[str, str, int]]:
    """查询满足频数阈值且无歧义的未激活 zh/mixed 别名。

    Args:
        min_freq: 去重频数下限（distinct platform×text）。

    Returns:
        [(alias, alias_id, freq_total)]，按 freq 降序。
    """
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT f.alias, f.alias_id, f.freq_total
            FROM ai_dict.zh_alias_freq f
            WHERE f.freq_total >= %s
              AND NOT EXISTS (
                  SELECT 1 FROM ai_dict.alias_ambiguity a
                  WHERE a.alias = f.alias
              )
            ORDER BY f.freq_total DESC
        """, (min_freq,))
        rows = [(r[0], r[1], int(r[2])) for r in cur.fetchall()]
        return rows
    finally:
        conn.close()


def apply_activation(aliases: list[str], reason: str) -> int:
    """按 alias 字符串激活 skill_aliases 行（歧义排除后同名唯一）。

    Args:
        aliases: 通过频数+歧义检查的 alias 字符串列表。
        reason: 写入 activation_reason 的审计标记。

    Returns:
        实际更新的行数。
    """
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("SET LOCAL work_mem = '256MB'")
        updated = 0
        page = 1000
        for i in range(0, len(aliases), page):
            cur.execute(
                """
                UPDATE ai_dict.skill_aliases
                SET is_active = '1', activation_reason = %s
                WHERE alias = ANY(%s)
                  AND is_active = '0' AND language IN ('zh','mixed')
                """,
                (reason, aliases[i : i + page]),
            )
            updated += cur.rowcount
        conn.commit()
        return updated
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def write_manifest(candidates: list[tuple[str, str, int]], min_freq: int,
                   applied: bool, updated: int, out_dir: Path) -> Path:
    """输出运行清单（候选 CSV + manifest JSON），供 QC 报告引用。

    Args:
        candidates: load_candidates 结果。
        min_freq: 使用的阈值。
        applied: 是否已实际写库。
        updated: 更新行数（dry-run 为 0）。
        out_dir: 输出目录。

    Returns:
        manifest JSON 路径。
    """
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cand_path = out_dir / f"alias_activation_candidates_{stamp}.csv"
    with cand_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["alias", "alias_id", "freq_total"])
        writer.writerows(candidates)
    manifest = {
        "run_id": stamp, "min_freq": min_freq, "applied": applied,
        "candidates": len(candidates), "rows_updated": updated,
        "reason": f"freq_ge_{min_freq}_no_ambiguity_gzsz_{RUN_DATE}",
        "candidates_csv": cand_path.name,
    }
    mpath = out_dir / f"alias_activation_manifest_{stamp}.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return mpath


def main() -> None:
    """激活入口。"""
    parser = argparse.ArgumentParser(description="中文/混合别名阈值激活")
    parser.add_argument("--min-freq", type=int, default=100,
                        help="去重频数下限（需与冻结 QC 报告记录一致）")
    parser.add_argument("--apply", action="store_true",
                        help="实际写库；默认仅 dry-run 预览")
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.log_dir / "zh_alias_activation.log")

    candidates = load_candidates(args.min_freq)
    logger.info("候选（freq>=%d 且无歧义）: %d 个别名", args.min_freq, len(candidates))
    if not candidates:
        logger.error("无候选别名——检查 zh_alias_freq 是否已按去重口径重算")
        sys.exit(2)

    reason = f"freq_ge_{args.min_freq}_no_ambiguity_gzsz_{RUN_DATE}"
    updated = 0
    if args.apply:
        updated = apply_activation([a for a, _i, _f in candidates], reason)
        logger.info("已激活 %d 行（threshold=%d）", updated, args.min_freq)
    else:
        logger.info("dry-run：未写库。top10: %s",
                  [(a, f) for a, _i, f in candidates[:10]])

    mpath = write_manifest(candidates, args.min_freq, args.apply, updated,
                           paths.report_dir)
    logger.info("清单: %s", mpath)
    print(f"{'已执行' if args.apply else 'DRY-RUN'}: 候选 {len(candidates)}，更新 {updated} 行")


if __name__ == "__main__":
    main()
