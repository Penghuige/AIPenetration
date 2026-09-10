"""panel_v2 M3-c：技能 AI 共现率与平滑（指南 §14，9 组权重）。

对每个 (anchor_version × window_type × period) 单元：
- 原始共现率 ai_rate_raw = n_ai_cooccur / n_skill（整数相除，§13.6.7；64 位浮点）；
- Beta-Binomial 经验贝叶斯平滑（§14.2）：单元内 n_skill>=5 技能极大似然拟合
  (alpha, beta)，失败/异常回退 Jeffreys(0.5,0.5)；
  ai_rate_smoothed = (c+alpha)/(n+alpha+beta)；**不覆盖原始权重**；
- 全部技能保留（§13.4），rare_lt10/20/50 标记 + confidence_tier（A 级概念
  属性，legacy 为 NULL）；
- 保存 alpha/beta/拟合状态（§14.2.5）。

产物：skill_ai_relevance.parquet（§14.4 字段全集）。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2  # noqa: F401  (eps 只读：tier 查询)

from config.paths import get_project_paths

from ..common import eps_conn_params, setup_logging

logger = logging.getLogger("ai_penetration.panel_v2.relevance")

MIN_FIT_N = 5       # §14.2.2
MIN_RATE_N = 1      # 全部技能都算共现率（§13.4 不删除）


def beta_binomial_fit_unit(n: np.ndarray, c: np.ndarray) -> tuple[float, float, str]:
    """单单元 Beta-Binomial 先验 MLE（边际似然）。

    Args:
        n: 技能分母（仅 n>=MIN_FIT_N 参与）。
        c: 技能分子。

    Returns:
        (alpha, beta, status)，status 区分回退原因（§14.2.5）：
        fitted / jeffreys:no_scipy / jeffreys:few_skills /
        jeffreys:not_converged / jeffreys:unstable / jeffreys:error。
    """
    try:
        from scipy.optimize import minimize
        from scipy.special import betaln
    except ImportError:
        return 0.5, 0.5, "jeffreys:no_scipy"
    if n.size < 50:
        return 0.5, 0.5, "jeffreys:few_skills"

    def nll(x: np.ndarray) -> float:
        a, b = np.exp(x)
        return -(betaln(c + a, n - c + b).sum() - n.size * betaln(a, b))

    try:
        res = minimize(nll, x0=np.array([np.log(0.3), np.log(8.0)]),
                       method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-3})
        if not res.success:  # §14.2.4 优化不收敛回退
            return 0.5, 0.5, "jeffreys:not_converged"
        a, b = np.exp(res.x)
        if not (1e-4 <= a <= 1e4 and 1e-4 <= b <= 1e4):
            return 0.5, 0.5, "jeffreys:unstable"
        return float(a), float(b), "fitted"
    except Exception:  # noqa: BLE001
        return 0.5, 0.5, "jeffreys:error"


def _load_tier_map(vocab_path: Path) -> np.ndarray:
    """skill_code → confidence_tier（A 级概念属性；legacy/未命中的 None）。"""
    import json
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))  # {skill_id: code}
    n = max(vocab.values()) + 1
    tiers = np.full(n, "", dtype=object)
    uuids = [s for s in vocab if not s.startswith("legacy:")]
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        for i in range(0, len(uuids), 5000):
            chunk = uuids[i:i + 5000]
            cur.execute("SELECT skill_id, coalesce(confidence_tier,'') "
                        "FROM ai_dict.skill_concepts WHERE skill_id = ANY(%s)",
                        (chunk,))
            for sid, t in cur.fetchall():
                tiers[vocab[sid]] = t
    finally:
        conn.close()
    return tiers


def compute_relevance(counts: pd.DataFrame, tier_map: np.ndarray) -> pd.DataFrame:
    """§14.1 全部 9 组（3 锚点 × 3 窗口口径）raw + smoothed。"""
    out = []
    for (ver, win, year), g in counts.groupby(
            ["anchor_version", "window_type", "year"], sort=False):
        n = g["n_skill"].to_numpy(np.float64)
        c = g["n_ai_cooccur"].to_numpy(np.float64)
        raw = c / n
        fit_m = n >= MIN_FIT_N
        alpha, beta, status = beta_binomial_fit_unit(n[fit_m], c[fit_m])
        smoothed = (c + alpha) / (n + alpha + beta)
        sc = g["skill_code"].to_numpy()
        out.append(pd.DataFrame({
            "skill_code": sc,
            "skill_id": pd.NA,  # 解码在导出层做（vocab 反查）
            "year": year if win != "pooled" else g["year"].iloc[0],
            "window_type": win,
            "window_start": g["window_start"].iloc[0],
            "window_end": g["window_end"].iloc[0],
            "anchor_version": ver,
            "n_skill": g["n_skill"].to_numpy(),
            "n_ai_cooccur": g["n_ai_cooccur"].to_numpy(),
            "ai_rate_raw": raw,
            "ai_rate_smoothed": smoothed,
            "rare_lt10": n < 10, "rare_lt20": n < 20, "rare_lt50": n < 50,
            "alpha": alpha, "beta": beta, "smoothing_status": status,
        }))
    rel = pd.concat(out, ignore_index=True)
    rel["confidence_tier"] = tier_map[rel["skill_code"].to_numpy()]
    return rel


def decode_skill_ids(rel: pd.DataFrame, vocab_path: Path) -> pd.DataFrame:
    """skill_code → skill_id 解码（I4：发布件内部自含可解释）。"""
    import json
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    code_to_sid = {c: s for s, c in vocab.items()}
    rel = rel.drop(columns=["skill_id"], errors="ignore").copy()
    rel.insert(1, "skill_id",
               pd.Series(code_to_sid).reindex(rel.skill_code).to_numpy())
    return rel


def main() -> None:
    ap = argparse.ArgumentParser(description="panel_v2 §14 共现率与平滑")
    ap.add_argument("--rel", default="panel_v2", help="release 子目录名")
    ap.add_argument("--pass2", default="pass2", help="pass2 目录名（词表来源）")
    args = ap.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_relevance.log")
    rel_dir = paths.output_dir / "release" / args.rel
    counts = pd.read_parquet(rel_dir / "skill_ai_counts.parquet")
    pass2 = paths.output_dir / "panel_v2" / args.pass2
    tier_map = _load_tier_map(pass2 / "skill_vocab.json")
    rel = decode_skill_ids(compute_relevance(counts, tier_map),
                           pass2 / "skill_vocab.json")
    rel.to_parquet(rel_dir / "skill_ai_relevance.parquet", index=False)
    st = rel.groupby("smoothing_status").size().to_dict()
    logger.info("skill_ai_relevance: %d 行，拟合状态 %s", len(rel), st)
    print(f"共现率完成: {len(rel):,} 行；smoothing_status={st}")


if __name__ == "__main__":
    main()
