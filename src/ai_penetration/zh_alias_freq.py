"""中文/混合别名岗位描述频数计算（交接包冻结前置步骤，去重口径版）。

对 ai_dict.skill_aliases 中未激活的 zh/mixed 别名，在**广深语料**上计算
「命中的不同规范化岗位描述数」，并输出歧义检查表。这是交接说明（00_交接
说明.md 第二节）规定的中文化别名冻结前置步骤（频数 + 歧义）。

口径（对齐指南 §6.2.0/§10.3.2）：
- 频数 = COUNT(DISTINCT (platform, text_hash))——字典发现语料按平台内文本
  去重，同文本跨平台各计一次；不受相同文本重复发布影响。
- 规范化文本 = 小写 + 删除全部空白字符；text_hash 取 blake2b 6 字节(48bit)。
- 去重键 key = (platform_id << 48) | text_hash，platform_id 用固定字符串
  字典（未知串散列到高位段），8 ctid 切片并行扫描 + 主进程 numpy 全局去重，
  eps 全程只读。

输出：
- ai_dict.zh_alias_freq(alias_id, alias, language, freq_total)  去重频数
- ai_dict.alias_ambiguity(...)                                  一对多/过短歧义标记

使用示例::

    python -m src.ai_penetration.zh_alias_freq --force
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from hashlib import blake2b
from pathlib import Path

import ahocorasick
import numpy as np
import psycopg2
from concurrent.futures import ProcessPoolExecutor, as_completed

from config.paths import get_project_paths

from .common import eps_connect, eps_conn_params, setup_logging
from .load_guangdong import GD_SHARDS

logger = logging.getLogger("ai_penetration.zh_alias_freq")

# 语料城市：交接口径下的字典发现语料（用户 2026-09-06 决策：仅广深）
FREQ_CITIES = ("广州市", "深圳市")

_WS_RE = re.compile(r"\s+")

# 平台固定字典（2026-09-06 采样确认的基础值；未列入的字符串散列到 10~1023）
_PLATFORM_FIXED: dict[str, int] = {
    "其他": 0, "58同城": 1, "51Job": 2, "职友集": 3, "猎聘网": 4,
    "智联招聘": 5, "BOSS直聘": 6, "拉勾网": 7, "百姓网": 8, "9120": 9,
}


def platform_id(platform: str) -> int:
    """平台字符串 → 10bit 编号（固定字典优先，未知串散列到高位段）。"""
    pid = _PLATFORM_FIXED.get((platform or "").strip())
    if pid is not None:
        return pid
    return 10 + (int.from_bytes(
        blake2b((platform or "").strip().encode("utf-8"), digest_size=2).digest(), "big"
    ) % 1014)


def normalize_desc(desc: str) -> str:
    """规范化岗位描述：小写 + 删除全部空白（交接口径的规范化文本定义）。"""
    return _WS_RE.sub("", (desc or "").lower())


def text_key(desc: str, platform: str) -> int:
    """(platform, 规范化描述) → 56bit 去重键（pid<<48 | blake2b-48）。"""
    h48 = int.from_bytes(
        blake2b(normalize_desc(desc).encode("utf-8"), digest_size=6).digest(), "big"
    )
    return (platform_id(platform) << 48) | h48


def load_zh_mixed_aliases() -> dict[str, tuple[str, str]]:
    """从 PG 加载未激活的 zh/mixed 别名 {alias: (alias_id, language)}。

    同名别名取首个 alias_id（一对多映射另行歧义标记）。

    Returns:
        alias -> (alias_id, language)。
    """
    conn = eps_connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT alias, min(alias_id), min(language)
            FROM ai_dict.skill_aliases
            WHERE is_active='0' AND language IN ('zh','mixed')
              AND length(trim(alias)) >= 2
            GROUP BY alias
        """)
        rows = cur.fetchall()
        logger.info("加载 zh/mixed 未激活别名(去重): %d 条", len(rows))
        return {r[0]: (r[1], r[2]) for r in rows}
    finally:
        conn.close()


def build_automaton(alias_map: dict[str, tuple[str, str]]):
    """构建 Aho-Corasick 自动机（键统一小写，匹配小写规范化语料）。

    Args:
        alias_map: alias -> (alias_id, language) 映射。

    Note:
        仅大小写不同的别名会冲突，后写入者覆盖先写入者；
        同形异名的 alias_id 一对多歧义由 alias_ambiguity 表另行标记。
    """
    automaton = ahocorasick.Automaton()
    for alias, (aid, _lang) in alias_map.items():
        automaton.add_word(alias.lower(), (aid, alias))
    automaton.make_automaton()
    return automaton


# ---------------------------------------------------------------- 并行扫描

_WORKER_STATE: dict = {}


def _init_worker(alias_map: dict[str, tuple[str, str]]) -> None:
    """进程初始化：构建本进程自动机与 alias 索引表。"""
    aliases = list(alias_map.items())
    _WORKER_STATE["autom"] = build_automaton(alias_map)
    # aid_index: alias_id -> uint32 序号（用于紧凑数组）
    _WORKER_STATE["aid_index"] = {aid: i for i, (a, (aid, _)) in enumerate(aliases)}
    _WORKER_STATE["n_alias"] = len(aliases)


def _table_blocks(cur, shard: str) -> int:
    """ctid 切片用的总块数 = pg_relation_size / block_size（不信任 relpages）。"""
    cur.execute("SELECT current_setting('block_size')::int")
    block_size = int(cur.fetchone()[0])
    cur.execute("SELECT pg_relation_size(%s)", (f"public.{shard}",))
    size = int(cur.fetchone()[0])
    return max(1, size // block_size)


def scan_slice(table: str, b_start: int, b_end: int, tmp_dir: str) -> dict:
    """扫描一个 ctid 切片：行级规范化→局部去重→Aho→(aid,key)对落盘。

    Args:
        table: public 分片表名。
        b_start: 起始块（含）。
        b_end: 结束块（不含）。

    Returns:
        {rows, local_new, pairs, keys_file, aids_file, task}。
    """
    autom = _WORKER_STATE["autom"]
    aid_index = _WORKER_STATE["aid_index"]
    n_alias = _WORKER_STATE["n_alias"]
    task = f"{table}_{b_start}_{b_end}"
    keys_file = Path(tmp_dir) / f"{task}.keys.bin"
    aids_file = Path(tmp_dir) / f"{task}.aids.bin"
    if keys_file.exists() and aids_file.exists():
        logger.info("切片 %s 已有中间文件，跳过（断点复用）", task)
        meta = json.loads((Path(tmp_dir) / f"{task}.json").read_text(encoding="utf-8"))
        return meta

    params = eps_conn_params()
    conn = psycopg2.connect(**params)
    rows = 0
    local_seen: set[int] = set()
    buckets: list[list[int]] = [[] for _ in range(n_alias)]
    try:
        with conn.cursor() as setup:
            setup.execute("SET LOCAL work_mem = '256MB'")
            # worker 自身并行，关闭 PG 额外并行避免超订
            setup.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cur = conn.cursor(f"freq_{task}")
        cur.itersize = 50000
        sql = f"""
            SELECT platform, job_description FROM public.{table}
            WHERE ctid >= '(%s,0)'::tid AND ctid < '(%s,0)'::tid
              AND job_description IS NOT NULL AND job_description != ''
              AND position IS NOT NULL AND position != ''
        """
        cur.execute(sql, (int(b_start), int(b_end)))
        while True:
            batch = cur.fetchmany(100000)
            if not batch:
                break
            for platform, desc in batch:
                rows += 1
                norm = normalize_desc(desc)
                h48 = int.from_bytes(
                    blake2b(norm.encode("utf-8"), digest_size=6).digest(), "big"
                )
                key = (platform_id(platform) << 48) | h48
                if key in local_seen:
                    continue
                local_seen.add(key)
                for _end, (aid, _alias) in autom.iter(norm):
                    buckets[aid_index[aid]].append(key)
            logger.info("切片 %s 进度: rows=%d new_text=%d", task, rows, len(local_seen))
        cur.close()
    finally:
        conn.close()

    keys_np: list[int] = []
    aids_np: list[int] = []
    for idx, bucket in enumerate(buckets):
        if bucket:
            keys_np.extend(bucket)
            aids_np.extend([idx] * len(bucket))
    keys_arr = np.asarray(keys_np, dtype=np.uint64)
    aids_arr = np.asarray(aids_np, dtype=np.uint32)
    del keys_np, aids_np, buckets, local_seen
    keys_arr.tofile(keys_file)
    aids_arr.tofile(aids_file)
    meta = {
        "task": task, "rows": rows, "local_new": int(len(np.unique(keys_arr)))
        if keys_arr.size else 0,
        "pairs": int(keys_arr.size),
        "keys_file": str(keys_file), "aids_file": str(aids_file),
    }
    (Path(tmp_dir) / f"{task}.json").write_text(
        json.dumps(meta), encoding="utf-8")
    return meta


def aggregate_counts(metas: list[dict], tmp_dir: Path | None,
                     n_alias: int = 0) -> np.ndarray:
    """合并各切片 (aid,key) 对，numpy 全局去重后按 aid 计数。

    Args:
        metas: 各切片 scan_slice 返回值。
        tmp_dir: 中间文件目录（聚合完成后统一清理）。
        n_alias: 别名总数（bincount minlength，保证数组覆盖全部别名）。

    Returns:
        uint64 数组 freq[alias序号] = distinct (platform,text) 命中数。
    """
    keys = np.concatenate([np.fromfile(m["keys_file"], dtype=np.uint64)
                           for m in metas if m["pairs"] > 0]) if metas else np.empty(0, np.uint64)
    aids = np.concatenate([np.fromfile(m["aids_file"], dtype=np.uint32)
                           for m in metas if m["pairs"] > 0]) if metas else np.empty(0, np.uint32)
    n_pairs = keys.size
    logger.info("全局合并: %d 对 (aid,key)，排序去重中…", n_pairs)
    order = np.lexsort((keys, aids))
    keys_s = keys[order]
    aids_s = aids[order]
    del keys, aids, order
    change = np.ones(keys_s.size, dtype=bool)
    change[1:] = (keys_s[1:] != keys_s[:-1]) | (aids_s[1:] != aids_s[:-1])
    uniq_aids = aids_s[change]
    freq = np.bincount(uniq_aids.astype(np.int64), minlength=n_alias).astype(np.uint64)
    n_uniq = int(uniq_aids.size)
    del keys_s, aids_s, change, uniq_aids
    logger.info("去重完成: distinct (aid,text) 对 %d（重复率 %.2f%%）",
                n_uniq, 100.0 * (1 - n_uniq / max(1, n_pairs)))
    # 中间文件统一清理（删除容错：Windows 偶发占用）
    for m in metas:
        for f in (m["keys_file"], m["aids_file"],
                  str(Path(tmp_dir) / f"{m['task']}.json")):
            try:
                Path(f).unlink()
            except OSError:
                logger.warning("中间文件清理失败（稍后手动删 %s）", f)
    return freq


# ---------------------------------------------------------------- 写库与报告

def _existing_freq_rows(cur) -> int | None:
    """返回 ai_dict.zh_alias_freq 现有行数；表不存在返回 None（防误重跑用）。"""
    cur.execute("SELECT to_regclass('ai_dict.zh_alias_freq')")
    if cur.fetchone()[0] is None:
        return None
    cur.execute("SELECT count(*) FROM ai_dict.zh_alias_freq")
    return cur.fetchone()[0]


def write_freq_table(conn_params: dict, alias_rows: list[tuple[str, str, str]],
                     freq: np.ndarray) -> None:
    """写入 ai_dict.zh_alias_freq（去重口径整表重建）。

    Args:
        conn_params: eps 连接参数。
        alias_rows: [(alias_id, alias, language)]，与 freq 数组同序。
        freq: 每个别名的 distinct (platform,text) 命中数。
    """
    conn = psycopg2.connect(**conn_params)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS ai_dict.zh_alias_freq")
        cur.execute("""
            CREATE TABLE ai_dict.zh_alias_freq (
                alias_id text PRIMARY KEY,
                alias text,
                language text,
                freq_total bigint
            )
        """)
        rows = [(aid, alias, lang, int(freq[i]))
                for i, (aid, alias, lang) in enumerate(alias_rows)]
        from psycopg2.extras import execute_values
        execute_values(cur,
            "INSERT INTO ai_dict.zh_alias_freq (alias_id, alias, language, freq_total) VALUES %s",
            rows, page_size=5000)
        conn.commit()
        logger.info("zh_alias_freq 写入: %d 条（freq>0: %d）",
                    len(rows), int((freq > 0).sum()))
    finally:
        conn.close()


def write_ambiguity_table() -> None:
    """生成歧义检查表：一对多映射、过短词、跨技能同名。"""
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS ai_dict.alias_ambiguity")
        cur.execute("""
            CREATE TABLE ai_dict.alias_ambiguity AS
            SELECT s.alias, s.alias_normalized, count(DISTINCT s.skill_id) AS n_skills,
                   bool_or(length(trim(s.alias)) <= 2) AS too_short
            FROM ai_dict.skill_aliases s
            WHERE s.is_active='0' AND s.language IN ('zh','mixed')
            GROUP BY s.alias, s.alias_normalized
            HAVING count(DISTINCT s.skill_id) > 1 OR bool_or(length(trim(s.alias)) <= 2)
        """)
        cur.execute("SELECT count(*) FROM ai_dict.alias_ambiguity")
        n = cur.fetchone()[0]
        conn.commit()
        logger.info("alias_ambiguity 歧义条目: %d", n)
    finally:
        conn.close()


def distribution_report(alias_rows: list[tuple[str, str, str]],
                        freq: np.ndarray, out_dir: Path) -> Path:
    """输出频数分布报告（阈值决策依据），返回 CSV 路径。

    Args:
        alias_rows: [(alias_id, alias, language)]。
        freq: 各别名去重频数。
        out_dir: 输出目录（report_dir）。

    Returns:
        明细 CSV 路径。
    """
    import pandas as pd

    df = pd.DataFrame(alias_rows, columns=["alias_id", "alias", "language"])
    df["freq_total"] = freq[: len(df)]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"zh_alias_freq_distribution_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    df.sort_values("freq_total", ascending=False).to_csv(
        path, index=False, encoding="utf-8-sig")
    buckets = {"0": 0, "1-4": 0, "5-9": 0, "10-19": 0, "20-99": 0, "100-999": 0, ">=1000": 0}
    f = df["freq_total"].to_numpy()
    buckets["0"] = int((f == 0).sum())
    buckets["1-4"] = int(((f >= 1) & (f <= 4)).sum())
    buckets["5-9"] = int(((f >= 5) & (f <= 9)).sum())
    buckets["10-19"] = int(((f >= 10) & (f <= 19)).sum())
    buckets["20-99"] = int(((f >= 20) & (f <= 99)).sum())
    buckets["100-999"] = int(((f >= 100) & (f <= 999)).sum())
    buckets[">=1000"] = int((f >= 1000).sum())
    logger.info("频数分布: %s", buckets)
    print("\n频数分布（阈值决策依据）:")
    for k, v in buckets.items():
        print(f"  freq {k:>8}: {v:>6} 个别名")
    print(f"明细 CSV: {path}")
    return path


def main() -> None:
    """中文别名频数计算入口（广深去重语料，8 路 ctid 并行）。"""
    parser = argparse.ArgumentParser(description="中文/混合别名频数计算（去重口径）")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--slices", type=int, default=4, help="每表 ctid 切片数")
    parser.add_argument("--force", action="store_true",
                        help="已有频数表时也强制 DROP 重算（默认拒绝，防止误重跑丢历史结果）")
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.log_dir / "zh_alias_freq.log")
    tmp_dir = paths.output_dir / "freq_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not args.force:
        conn = eps_connect()
        try:
            existing = _existing_freq_rows(conn.cursor())
        finally:
            conn.close()
        if existing is not None:
            logger.error(
                "ai_dict.zh_alias_freq 已存在（%d 行），重算会 DROP 覆盖历史结果；"
                "确需重算请加 --force", existing,
            )
            raise SystemExit(2)

    alias_map = load_zh_mixed_aliases()
    alias_rows = [(aid, alias, lang) for alias, (aid, lang) in alias_map.items()]
    logger.info("自动机候选别名 %d 条（zh/mixed 去重后）", len(alias_rows))

    # 任务规划：每表按块数均分 slices 个切片
    tasks: list[tuple[str, int, int]] = []
    conn = eps_connect()
    try:
        cur = conn.cursor()
        for city in FREQ_CITIES:
            shard = GD_SHARDS[city]
            total_blocks = _table_blocks(cur, shard)
            step = total_blocks // args.slices + 1
            for i in range(args.slices):
                s = i * step
                # 末切片上界取无穷：pg_relation_size 换算外可能仍有追加行
                e = 4294967295 if i == args.slices - 1 else min(total_blocks, (i + 1) * step)
                if s < e:
                    tasks.append((shard, s, e))
    finally:
        conn.close()
    logger.info("扫描任务: %d 切片 / %d workers", len(tasks), args.workers)

    metas: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(alias_map,)) as pool:
        futs = {pool.submit(scan_slice, t, s, e, str(tmp_dir)): (t, s, e)
                for t, s, e in tasks}
        for fut in as_completed(futs):
            meta = fut.result()
            metas.append(meta)
            logger.info("切片完成 %s: rows=%s pairs=%s",
                        meta["task"], meta["rows"], meta["pairs"])

    total_rows = sum(m["rows"] for m in metas)
    total_pairs = sum(m["pairs"] for m in metas)
    logger.info("扫描完成: rows=%d pairs=%d，进入全局去重", total_rows, total_pairs)
    freq = aggregate_counts(metas, tmp_dir, n_alias=len(alias_rows))

    write_freq_table(eps_conn_params(), alias_rows, freq)
    write_ambiguity_table()
    distribution_report(alias_rows, freq, paths.report_dir)
    print("\n完成。zh_alias_freq 已按去重口径重建（广深 2014-2024 语料）。")


if __name__ == "__main__":
    main()
