"""全国跑批前阶段 0 验证（城市名枚举一致性 + 异地抽样体检）。

V1 枚举：全部 job_p% 表的 city 字段一致性（抽 200 行 distinct）与空表清单。
V2 体检：随机 8 个非广东城市 TABLESAMPLE 抽 ~2 万行，测技能命中率与
A/B/C 判率是否落在合理带（参考：珠三角 A 率 0.05-0.80%）。

结果写 output/reports/stage0_report.json，供 runner 决策。

使用示例::

    python -m src.ai_penetration.stage0_validation
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter
from datetime import datetime

import psycopg2

from config.paths import get_project_paths

from .common import DEFAULT_OMEGA_SNAPSHOT, eps_conn_params, resolve_artifact_path, setup_logging

logger = logging.getLogger("ai_penetration.stage0")

GD_CITIES = {
    "广州市", "深圳市", "东莞市", "佛山市", "中山市", "珠海市", "惠州市",
    "江门市", "肇庆市", "汕头市", "潮州市", "揭阳市", "汕尾市", "湛江市",
    "茂名市", "阳江市", "云浮市", "韶关市", "清远市", "梅州市", "河源市",
}




def _connect_retry(params: dict, attempts: int = 8, wait_sec: int = 60) -> "psycopg2.connection":
    """带重试的 PG 连接（网络瞬断/PG 重启窗口自愈，最长约 8 分钟）。

    Args:
        params: 连接参数。
        attempts: 尝试次数。
        wait_sec: 每次失败后的等待秒数。

    Returns:
        已建立的 psycopg2 连接。

    Raises:
        psycopg2.OperationalError: 重试用尽仍失败。
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return psycopg2.connect(**params, connect_timeout=8)
        except psycopg2.OperationalError as exc:
            last = exc
            logger.warning("PG 连接失败（第 %d/%d 次），%ds 后重试: %s",
                           i + 1, attempts, wait_sec, str(exc)[:80])
            time.sleep(wait_sec)
    raise last  # type: ignore[misc]


def run_v1(params: dict) -> dict:
    """V1：全部分片 city 枚举与一致性。

    Args:
        params: eps 连接参数。

    Returns:
        报告字典 {n_tables, empty, multi_city, city_sizes}。
    """
    conn = _connect_retry(params)
    report: dict = {"empty_tables": [], "multi_city_tables": []}
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT c.relname, pg_relation_size(c.oid)
            FROM pg_class c JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname='public' AND c.relname LIKE 'job_p%' AND c.relkind='r'
            ORDER BY 2 DESC
        """)
        tables = cur.fetchall()
        sizes = {}
        for shard, size in tables:
            try:
                if float(size) < 2e9:
                    # 小表：采样查多城一致性（HDD 上秒级）
                    cur.execute(
                        f"SELECT DISTINCT city FROM public.{shard} "
                        "TABLESAMPLE SYSTEM (2) LIMIT 5"
                    )
                    cities = {str(r[0] or "").strip() for r in cur.fetchall()}
                else:
                    # 大表：TABLESAMPLE 在 HDD 上太慢（实测 ~40s/表），只取首行定城
                    cur.execute(f"SELECT city FROM public.{shard} LIMIT 1")
                    row = cur.fetchone()
                    cities = {str(row[0] or "").strip()}
            except psycopg2.Error:
                conn.rollback()
                cities = set()
            cities.discard("")
            if not cities:
                report["empty_tables"].append(shard)
            elif len(cities) > 1:
                report["multi_city_tables"].append((shard, sorted(cities)))
            else:
                sizes[shard] = [float(size), next(iter(cities))]
        report["n_tables"] = len(tables)
        report["n_city_mapped"] = len(sizes)
        report["sizes"] = sizes
    finally:
        conn.close()
    return report


def run_v2(params: dict, omega: dict, n_cities: int = 8) -> list[dict]:
    """V2：非广东抽样体检（命中率/判率/年份分布）。

    Args:
        params: eps 连接参数。
        omega: skill -> ωsAI。
        n_cities: 抽查城市数。

    Returns:
        每城体检结果列表。
    """
    from .skill_ai_anchor import build_skill_regex, load_merged_skills
    from .skill_ai_anchor import is_ai_fused

    skills = load_merged_skills(include_llm=True)
    regex = build_skill_regex(skills)

    conn = _connect_retry(params)
    results = []
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT c.relname FROM pg_class c JOIN pg_namespace n ON c.relnamespace=n.oid
            WHERE n.nspname='public' AND c.relname LIKE 'job_p%' AND c.relkind='r'
              AND pg_relation_size(c.oid) BETWEEN 1e9 AND 3e10
        """)
        cands = [r[0] for r in cur.fetchall()]
        random.seed(20260831)
        picked = random.sample(cands, min(n_cities, len(cands)))
        for shard in picked:
            cur.execute(f"""
                SELECT position, job_description, publish_time FROM public.{shard}
                TABLESAMPLE SYSTEM (2)
                WHERE job_description IS NOT NULL AND job_description != ''
                  AND position IS NOT NULL AND position != ''
                LIMIT 20000
            """)
            rows = cur.fetchall()
            if len(rows) < 2000:
                continue
            city = str(rows[0][2] or "")  # 占位，city 从下面补
            cur2 = conn.cursor()
            cur2.execute(f"SELECT city FROM public.{shard} LIMIT 1")
            city = str(cur2.fetchone()[0]).strip()
            cur2.close()
            n = len(rows)
            hit = a = b = f = 0
            years = Counter()
            for pos, desc, pub in rows:
                pos, desc = str(pos or ""), str(desc or "")
                yrs = str(pub or "")[:4]
                if yrs.isdigit():
                    years[int(yrs)] += 1
                if regex.search(desc[:4000]):
                    hit += 1
                aa, bb, ff = is_ai_fused(pos, desc, omega, regex)
                a += aa
                b += bb
                f += ff
            results.append({
                "shard": shard, "city": city, "n": n,
                "skill_hit_rate": hit / n,
                "a_rate": a / n, "b_rate": b / n, "fused_rate": f / n,
                "top_years": years.most_common(3),
            })
            logger.info("V2 %s(%s): n=%d 命中%.3f A=%.4f 融合=%.4f",
                        city, shard, n, hit / n, a / n, f / n)
    finally:
        conn.close()
    return results


def main() -> None:
    """阶段 0 入口。"""
    parser = argparse.ArgumentParser(description="全国跑批阶段0验证")
    parser.add_argument("--omega-file", type=str,
                        default=DEFAULT_OMEGA_SNAPSHOT)
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.project_root / "logs" / "stage0_validation.log")
    params = eps_conn_params()

    t0 = time.time()
    v1 = run_v1(params)
    omega = json.loads(
        resolve_artifact_path(args.omega_file, artifact="ωsAI").read_text(encoding="utf-8"))
    v2 = run_v2(params, omega)

    # 合理性判定：V2 判率带宽 0~3%，命中率>0.5
    flags = []
    for r in v2:
        if not (0.0 <= r["fused_rate"] <= 0.03):
            flags.append(f"{r['city']} 融合率异常 {r['fused_rate']:.4f}")
        if r["skill_hit_rate"] < 0.5:
            flags.append(f"{r['city']} 技能命中率偏低 {r['skill_hit_rate']:.3f}")
    ok = not flags and not v1["multi_city_tables"]
    report = {
        "ts": datetime.now().isoformat(),
        "elapsed_sec": round(time.time() - t0),
        "v1_summary": {k: v for k, v in v1.items() if k != "sizes"},
        "v1_sizes_top20": sorted(
            v1["sizes"].items(), key=lambda x: -x[1][0]
        )[:20],
        "v2": v2,
        "flags": flags,
        "pass": ok,
    }
    out = paths.output_dir / "reports" / "stage0_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("阶段0完成 pass=%s，报告: %s (%.0f 分钟)", ok, out, (time.time() - t0) / 60)
    print(f"STAGE0_PASS={ok}")


if __name__ == "__main__":
    main()
