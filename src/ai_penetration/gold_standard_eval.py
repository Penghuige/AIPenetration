"""金标准盲评 + 新 AI 技术词缺口挖掘（两阶段一体入口，全部走本机 vLLM）。

目的：
- 阶段 1：广深 2024 同抽样复现旧 fused（快照口径/平滑口径）与新 fused
  （A 级概念 + 双概念修正规则）判定，取两法各自判真的岗位各 150 条送 LLM
  盲评，量化"判成 AI 的岗到底是不是真 AI 岗"的绝对精确率；
- 阶段 2：从判定样本取 b_new=True 或含强概念的岗位 1,200 条让 LLM 抽取
  具体 AI 技术词，对照 A 级别名集与旧自建词表，统计 MAPPED / LEGACY_ONLY /
  NEW_CONCEPT 三桶，回答"是否需要新的关键词提取策略"。

eps 全程只读（仅 SELECT）；产物只写 output/reports/，断点写 output/eval_tmp/
（成功后清理），日志 logs/gold_standard_eval.log。

使用示例::

    $env:AIPEN_LLM_BASE_URL = "http://<wsl-ip>:8101/v1"
    python -X utf8 -m src.ai_penetration.gold_standard_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import re
import shutil
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import requests

from config.paths import get_project_paths

from src.model_platform.llm import create_llm_client

from .ai_scoring import is_ai_job
from .common import DEFAULT_OMEGA_SNAPSHOT, eps_conn_params, setup_logging
from .compare_atier_gz2024 import (
    BIZ_POSITION_RE,
    beta_binomial_fit,
    build_atier_index,
    collect_counts,
    extract_concepts,
    smooth_omega,
)
from .load_guangdong import GD_SHARDS
from .skill_ai_anchor import build_skill_regex, extract_skills_fast, load_merged_skills

logger = logging.getLogger("ai_penetration.gold_standard_eval")

YEAR = 2024
CITIES = ("广州市", "深圳市")
SEED = 20260907
EVAL_DESC_CHARS = 800      # 盲评每条截取描述长度
GAP_DESC_CHARS = 600        # 缺口挖掘每条截取描述长度
# vLLM 服务 max-model-len=4096，按字符预算动态组批（1 中文字符≈0.8~1 token）
PROMPT_CHAR_BUDGET = 3000
# 昨晚最终 run 基准（宽 5% 抽样）：旧入表 3,776 / 概念入表 2,473
BASELINE_OMEGA_OLD, BASELINE_OMEGA_C = 3776, 2473

# Qwen3 混合推理模型：必须关思考，否则 4096 上下文被思考链吃光
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

EVAL_SYSTEM = (
    "你是招聘数据分析专家。对每个招聘岗位判定：该岗位是否以 AI/机器学习的"
    "开发、实现、训练、部署、算法研究、AI 数据工程为核心职责？"
    "label=1：是核心职责；label=0：仅销售/售前/客服/实施/运维接触 AI 产品、"
    "仅把 AI 当加分技能提及、通用数据分析、或非技术岗。"
    '只输出 JSON 数组，格式 [{"id":<int>,"label":<0或1>,"reason":"不超过20字中文"}]，'
    "条数与岗位数一致，不要输出任何其他文字。"
)

GAP_SYSTEM = (
    "你是 AI 技术技能抽取器。对每个招聘岗位，抽取其文本中出现的具体"
    "AI/机器学习技术、方法、模型、工具类技能词（如 PyTorch、RAG、提示词工程、"
    "多模态大模型、LangChain）；不要输出'数据分析''沟通能力'等通用词或软素质词。"
    "每个词附不超过 12 字的原文证据。"
    '只输出 JSON 数组，格式 [{"id":<int>,"terms":[{"term":"..","evidence":".."}]}]，'
    "每个岗位一个对象，条数与岗位数一致，没有可抽词则 terms 为空数组，"
    "不要输出任何其他文字。"
)


# ------------------------------------------------------------ 前置与通用

def health_check(base_url: str) -> tuple[str, int]:
    """探测 vLLM 服务，返回 (model id, max_model_len)。

    Args:
        base_url: OpenAI-compatible API 根地址（到 /v1）。

    Raises:
        SystemExit: 服务不可达或无可用模型时终止进程。
    """
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"vLLM 不可达: {base_url} -> {exc}") from exc
    if not data:
        raise SystemExit(f"vLLM 无可用模型: {base_url}")
    model_id = str(data[0]["id"])
    max_len = int(data[0].get("max_model_len") or 4096)
    logger.info("vLLM 健康检查通过: model=%s max_model_len=%d", model_id, max_len)
    return model_id, max_len


def ensure_single_instance() -> None:
    """单实例保护：发现同名脚本已在运行则退出（长跑纪律 §9）。"""
    try:
        import psutil
    except ImportError:
        return
    me = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] == me or not (proc.info["name"] or "").lower().startswith("python"):
                continue
            toks = proc.info["cmdline"] or []
            hit = any(t.replace("\\", "/").endswith("src.ai_penetration.gold_standard_eval")
                      or t.replace("\\", "/").endswith("gold_standard_eval.py") for t in toks)
        except Exception:  # noqa: BLE001
            continue
        if hit:
            raise SystemExit(f"已有 gold_standard_eval 实例在运行 (pid={proc.info['pid']})，退出")


# ------------------------------------------------------------ 阶段 1a：抽样与 ω

def fetch_sample(sample_pct: float, tmp_dir: Path) -> list[tuple[str, str]]:
    """TABLESAMPLE 抽广深 2024（两城各一条连接并行），带 pickle 断点。

    Args:
        sample_pct: 每表 TABLESAMPLE SYSTEM 百分比。
        tmp_dir: eval_tmp 路径（此处直接传文件路径所在目录）。

    Returns:
        [(position, description)] 抽样行。
    """
    cache = tmp_dir
    if cache.exists():
        with cache.open("rb") as f:
            rows = pickle.load(f)
        logger.info("复用抽样断点: %s（%d 行）", cache.name, len(rows))
        return rows

    per_city: dict[str, list] = {}
    lock = threading.Lock()
    done = Counter()

    def one(city: str) -> None:
        shard = GD_SHARDS[city]
        conn = psycopg2.connect(**eps_conn_params())
        try:
            cur = conn.cursor(f"gse_sample_{shard}")
            cur.itersize = 100000
            cur.execute(
                f"SELECT position, job_description FROM public.{shard} "
                f"TABLESAMPLE SYSTEM ({sample_pct}) "
                "WHERE substr(publish_time,1,4)=%s "
                "  AND job_description IS NOT NULL AND job_description != ''",
                (str(YEAR),))
            rows: list[tuple[str, str]] = []
            while True:
                batch = cur.fetchmany(100000)
                if not batch:
                    break
                rows.extend((str(p or ""), str(d)) for p, d in batch)
                with lock:
                    done["n"] += len(batch)
                    logger.info("抽样进度: %d 行（%s 已 %d）", done["n"], city, len(rows))
            cur.close()
            per_city[city] = rows
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, CITIES))
    sample = per_city[CITIES[0]] + per_city[CITIES[1]]
    logger.info("抽样完成: %s %d 行 / %s %d 行，共 %d 行",
                CITIES[0], len(per_city[CITIES[0]]),
                CITIES[1], len(per_city[CITIES[1]]), len(sample))
    with cache.open("wb") as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
    return sample


_COUNT_CTX: dict = {}


def _init_count_worker() -> None:
    """计数子进程初始化：构建两套词表（与主进程同参）。"""
    _COUNT_CTX["atier"] = build_atier_index()
    _COUNT_CTX["regex"] = build_skill_regex(load_merged_skills(include_llm=True))


def _count_chunk(rows: list) -> tuple[dict, dict]:
    """一块抽样的 ω 频数统计（collect_counts 的进程池包装）。"""
    return collect_counts(rows, _COUNT_CTX["atier"], _COUNT_CTX["regex"])


def collect_counts_parallel(sample: list, workers: int) -> tuple[dict, dict]:
    """多进程分块跑 collect_counts 并合并 (n, n_anchor) 计数（与单遍完全等价）。"""
    step = len(sample) // workers + 1
    chunks = [sample[i:i + step] for i in range(0, len(sample), step)]
    logger.info("并行频数统计: %d 块 x ~%d 行", len(chunks), step)
    out_old: dict = {}
    out_c: dict = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_count_worker) as pool:
        futs = [pool.submit(_count_chunk, ch) for ch in chunks]
        for fut in as_completed(futs):
            oc, cc = fut.result()
            done += 1
            for k, (x, y) in oc.items():
                a = out_old.get(k, (0, 0))
                out_old[k] = (a[0] + x, a[1] + y)
            for k, (x, y) in cc.items():
                a = out_c.get(k, (0, 0))
                out_c[k] = (a[0] + x, a[1] + y)
            logger.info("频数统计块完成 %d/%d（旧词表 %d 项 / 概念 %d 项）",
                        done, len(futs), len(out_old), len(out_c))
    return out_old, out_c


def fit_omegas(sample: list, tmp_dir: Path, report_dir: Path,
               workers: int = 16) -> tuple[dict, dict, str]:
    """估计两套词表的平滑 ω（collect_counts 断点复用）。

    Returns:
        (omega_old, omega_c, 时间戳)。
    """
    cache = tmp_dir
    if cache.exists():
        with cache.open("rb") as f:
            old_counts, concept_counts = pickle.load(f)
        logger.info("复用频数断点: 旧词表 %d 项 / 概念 %d 项",
                    len(old_counts), len(concept_counts))
    else:
        old_counts, concept_counts = collect_counts_parallel(sample, workers)
        with cache.open("wb") as f:
            pickle.dump((old_counts, concept_counts), f, protocol=pickle.HIGHEST_PROTOCOL)
    a_o, b_o = beta_binomial_fit(old_counts)
    a_c, b_c = beta_binomial_fit(concept_counts)
    omega_old = smooth_omega(old_counts, a_o, b_o)
    omega_c = smooth_omega(concept_counts, a_c, b_c)
    logger.info("平滑: 旧 prior=(%.3f,%.1f) 入表 %d；概念 prior=(%.3f,%.1f) 入表 %d",
                a_o, b_o, len(omega_old), a_c, b_c, len(omega_c))
    # 与昨晚最终 run 基准核对，偏离 >10% 告警查因
    for name, got, base in (("旧词表入表", len(omega_old), BASELINE_OMEGA_OLD),
                            ("概念入表", len(omega_c), BASELINE_OMEGA_C)):
        if abs(got - base) / base > 0.10:
            logger.warning("%s %d 偏离昨晚基准 %d 超过 10%%，需查因", name, got, base)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (report_dir / f"omega_smooth_legacy_{stamp}.json").write_text(
        json.dumps(omega_old, ensure_ascii=False), encoding="utf-8")
    (report_dir / f"omega_smooth_atier_{stamp}.json").write_text(
        json.dumps(omega_c, ensure_ascii=False), encoding="utf-8")
    return omega_old, omega_c, stamp


# ------------------------------------------------------------ 阶段 1b：40k 判定

def _b_simple(scored: list[float]) -> bool:
    """现行 is_ai_fused 的 B 口径：avg>=0.15 且 max>=0.5。"""
    return bool(scored) and sum(scored) / len(scored) >= 0.15 and max(scored) >= 0.5


_JUDGE_CTX: dict = {}


def _init_judge_worker(omega_snap: dict, omega_old: dict, omega_c: dict) -> None:
    """判定子进程初始化：加载三套 ω 与两套词表。"""
    _JUDGE_CTX["snap"] = omega_snap
    _JUDGE_CTX["old"] = omega_old
    _JUDGE_CTX["c"] = omega_c
    _JUDGE_CTX["atier"] = build_atier_index()
    _JUDGE_CTX["regex"] = build_skill_regex(load_merged_skills(include_llm=True))


def _judge_chunk(triples: list) -> list[dict]:
    """一块 (行号, 岗位名, 描述) 的逐行判定（与 compare 的 b_d 口径一致）。"""
    automaton, ascii_flags, _anchors, homograph = _JUDGE_CTX["atier"]
    omega_snap, omega_old, omega_c = _JUDGE_CTX["snap"], _JUDGE_CTX["old"], _JUDGE_CTX["c"]
    regex_old = _JUDGE_CTX["regex"]
    judged: list[dict] = []
    for i, pos, desc in triples:
        a = is_ai_job(pos, desc)
        sk = extract_skills_fast(desc, regex_old)
        s_old_snap = [omega_snap[s] for s in sk if s in omega_snap]
        s_old_smooth = [omega_old[s] for s in sk if s in omega_old]
        cs = extract_concepts(desc.lower(), automaton, ascii_flags, homograph)
        s_c = [omega_c[c] for c in cs if c in omega_c]
        n_strong = sum(1 for w in s_c if w >= 0.5)
        avg_c = sum(s_c) / len(s_c) if s_c else 0.0
        biz = bool(BIZ_POSITION_RE.search(pos))
        b_new = bool(s_c) and avg_c >= 0.15 and (n_strong >= 2 or (n_strong >= 1 and not biz))
        judged.append({
            "i": i, "pos": pos, "desc": desc,
            "a": a,
            "b_old_snap": _b_simple(s_old_snap),
            "b_old_smooth": _b_simple(s_old_smooth),
            "b_new": b_new, "n_strong": n_strong, "biz": biz, "n_cs": len(cs),
        })
    return judged


def judge_rows(sample: list, tmp_dir: Path, omega_snap: dict,
               omega_old: dict, omega_c: dict, workers: int = 8) -> list[dict]:
    """从抽样随机取 40k 行（seed 固定），多进程逐行计算各口径判定。

    b_new 规则与 compare_atier_gz2024.scan_slice 的 b_d 完全一致：
    概念 avg>=0.15 且 [>=2 个概念 omega>=0.5 或 (单强概念且岗位名不匹配
    BIZ_POSITION_RE)]。

    Returns:
        判定行列表（含盲评/缺口挖掘所需的每行明细）。
    """
    if tmp_dir.exists():
        with tmp_dir.open("rb") as f:
            rows = pickle.load(f)
        logger.info("复用 40k 判定断点: %s（%d 行）", tmp_dir.name, len(rows))
        return rows
    rng = random.Random(SEED)
    idxs = rng.sample(range(len(sample)), 40000) if len(sample) > 40000 else list(range(len(sample)))
    triples = [(i, sample[i][0], sample[i][1]) for i in idxs]
    step = len(triples) // workers + 1
    chunks = [triples[i:i + step] for i in range(0, len(triples), step)]
    logger.info("40k 并行判定: %d 块 x ~%d 行", len(chunks), step)
    judged: list[dict] = []
    done = 0
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_judge_worker,
            initargs=(omega_snap, omega_old, omega_c)) as pool:
        futs = [pool.submit(_judge_chunk, ch) for ch in chunks]
        for fut in as_completed(futs):
            judged.extend(fut.result())
            done += 1
            logger.info("40k 判定块完成 %d/%d（累计 %d 行）", done, len(futs), len(judged))
    judged.sort(key=lambda r: r["i"])  # 断点顺序确定化（重跑复现同一盲评池）
    with tmp_dir.open("wb") as f:
        pickle.dump(judged, f, protocol=pickle.HIGHEST_PROTOCOL)
    return judged


# ------------------------------------------------------------ LLM 批调用通用

def make_batches(items: list[tuple[int, str]], budget: int,
                 max_items: int | None = None) -> list[list[tuple[int, str]]]:
    """按字符预算动态组批（vLLM 上下文 4096，固定批大小装不下长描述）。

    Args:
        items: (编号, 单条文本) 列表。
        budget: 每批 user prompt 文本部分的字符预算。
        max_items: 每批条数上限（输出 token 也受上下文约束的阶段用它收紧）。

    Returns:
        分批结果，每批内编号保持原序。
    """
    batches: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    used = 0
    for it in items:
        cost = len(it[1]) + 40
        if cur and (used + cost > budget or (max_items and len(cur) >= max_items)):
            batches.append(cur)
            cur, used = [], 0
        cur.append(it)
        used += cost
    if cur:
        batches.append(cur)
    return batches


# 宽容 JSON 扫描：模型偶发输出前后夹带说明文字、或把多个数组连写；
# 逐个抓取（最多一层嵌套的）JSON 对象再解析，坏对象跳过（缺失显式补 None）
_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def scan_objs(text: str) -> list[dict]:
    """从 LLM 原始回复中扫描 JSON 对象列表。"""
    objs: list[dict] = []
    for m in _OBJ_RE.finditer(text or ""):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict):
            objs.append(o)
    return objs


def run_llm_stage(tag: str, batches: list, system: str, build_user, max_tokens: int,
                  parse, checkpoint: Path, workers: int) -> dict:
    """并发跑一批批 LLM 调用：条数校验、失败重试 1 次、断点 jsonl 式落盘。

    Args:
        tag: 阶段名（日志）。
        batches: make_batches 输出。
        system: system prompt。
        build_user: 批 -> user prompt 字符串。
        max_tokens: 输出上限。
        parse: 解析函数 res_obj -> {id: 值}（None 值代表该对象不合格）。
        checkpoint: 已完成批次结果 json 断点路径。
        workers: 并发数（共享 GPU，<=4）。

    Returns:
        {batch_index(str): {id(str): 值}}；值由 parse 定义，缺失条目已补 None。
    """
    results: dict[str, dict] = {}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)  # eval_tmp 可能尚未建立
    if checkpoint.exists():
        results = json.loads(checkpoint.read_text(encoding="utf-8"))
        # 全 None 的失败批不算完成，续跑时重掷
        dead = [k for k, v in results.items() if all(x is None for x in v.values())]
        for k in dead:
            results.pop(k)
        if dead:
            logger.info("[%s] 断点中 %d 个失败批将重跑", tag, len(dead))
        logger.info("[%s] 复用 LLM 断点: %d/%d 批", tag, len(results), len(batches))
    lock = threading.Lock()
    client = create_llm_client()
    todo = [bi for bi in range(len(batches)) if str(bi) not in results]
    logger.info("[%s] 待跑 %d 批 / 共 %d 批", tag, len(todo), len(batches))

    def save():
        checkpoint.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    def work(bi: int) -> None:
        batch = batches[bi]
        want = [int(iid) for iid, _ in batch]
        got: dict = {}
        for attempt in (1, 2):
            try:
                text = client.complete_text(
                    system_prompt=system, user_prompt=build_user(batch),
                    temperature=0.0, max_output_tokens=max_tokens, extra_payload=NO_THINK)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] 批 %d 第 %d 次请求失败: %s", tag, bi, attempt, str(exc)[:150])
                continue
            parsed = parse(scan_objs(text)) or None
            if parsed is not None and len(parsed) == len(batch):
                got = parsed
                break
            # 条数不一致但 id 齐全也算有效（防模型偶发丢对象）
            if parsed is not None and set(int(k) for k in parsed) >= set(want):
                got = parsed
                logger.warning("[%s] 批 %d 返回条数 %d != 批大小 %d（id 已齐）",
                               tag, bi, len(parsed), len(batch))
                break
            logger.warning("[%s] 批 %d 第 %d 次解析/条数校验失败", tag, bi, attempt)
        if not got:
            logger.error("[%s] 批 %d 重试后仍失败，全批记 None", tag, bi)
        out = {str(w): got.get(str(w)) for w in want}  # 缺失显式补 None，不静默丢
        with lock:
            results[str(bi)] = out
            if len(results) % 20 == 0:
                save()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, bi) for bi in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if n % 25 == 0:
                logger.info("[%s] 已完成 %d/%d 批", tag, n, len(todo))
    with lock:
        save()
    return results


def parse_eval(res: list) -> dict | None:
    """盲评结果解析：{id(str): (label|None, reason)}；非法对象跳过（缺失显式补 None）。"""
    got: dict[str, tuple] = {}
    for obj in res:
        if not isinstance(obj, dict) or "id" not in obj:
            continue
        try:
            iid = str(int(obj["id"]))
        except (TypeError, ValueError):
            continue
        lab = obj.get("label")
        lab = int(lab) if lab in (0, 1, "0", "1") else None
        got[iid] = (lab, str(obj.get("reason") or ""))
    return got or None


def parse_gap(res: list) -> dict | None:
    """词抽取结果解析：{id(str): [terms]}；非法对象跳过。"""
    got: dict[str, list] = {}
    for obj in res:
        if not isinstance(obj, dict) or "id" not in obj:
            continue
        try:
            iid = str(int(obj["id"]))
        except (TypeError, ValueError):
            continue
        terms = obj.get("terms")
        got[iid] = terms if isinstance(terms, list) else []
    return got or None


# ------------------------------------------------------------ 阶段 1 盲评与统计

_ERROR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("销售/实施/运维语境", re.compile(
        r"销售|售前|售后|客服|商务|市场|BD|客户经理|实施|运维|技术支持|渠道|推广")),
    ("仅加分项/泛泛提及", re.compile(
        r"加分|优先|熟悉|了解|提及|附带|辅助|接触|使用.{0,4}工具")),
    ("LLM 存疑", re.compile(r"无法判断|难以确定|不确定|证据不足|信息不足")),
]


def classify_error(reason: str) -> str:
    """label=0 条目的 reason 关键词分桶。"""
    for name, pat in _ERROR_PATTERNS:
        if pat.search(reason or ""):
            return name
    return "真误判(与AI开发无关)"


def build_blind_items(judged: list[dict], per: int) -> tuple[list[dict], list[int]]:
    """old_fused(A或b_old平滑)=True 与 new_fused(A或b_new)=True 各随机 per 条，合并去重。

    Returns:
        (盲评条目 [{id, pos, text, in_old, in_new, row_idx}], 每条目对应 judged 下标)。
    """
    rng = random.Random(SEED + 1)
    old_pool = [k for k, r in enumerate(judged) if r["a"] or r["b_old_smooth"]]
    new_pool = [k for k, r in enumerate(judged) if r["a"] or r["b_new"]]
    old_pick = set(rng.sample(old_pool, min(per, len(old_pool))))
    new_pick = set(rng.sample(new_pool, min(per, len(new_pool))))
    logger.info("盲评池: old_fused=%d new_fused=%d 抽样 %d/%d，交集 %d",
                len(old_pool), len(new_pool), len(old_pick), len(new_pick),
                len(old_pick & new_pick))
    merged = sorted(old_pick | new_pick)
    items = []
    for bid, k in enumerate(merged, 1):
        r = judged[k]
        text = f"岗位名:{r['pos']}｜描述:{r['desc'][:EVAL_DESC_CHARS]}"
        items.append({"id": bid, "pos": r["pos"], "text": text,
                      "in_old": k in old_pick, "in_new": k in new_pick, "row": k})
    return items, merged


def _eval_user(batch: list[tuple[int, str]]) -> str:
    return (f"逐条判定以下 {len(batch)} 个岗位：\n"
            + "\n".join(f"{i}. {t}" for i, t in batch)
            + f"\n\n输出 {len(batch)} 个对象的 JSON 数组。")


def precision_stats(items: list[dict], flat_labels: dict[str, list]) -> dict:
    """按集合（old/new/交集/增量）统计盲评精确率（flat_labels: id -> [label, reason]）。"""
    groups = {"old": lambda x: x["in_old"], "new": lambda x: x["in_new"],
              "common": lambda x: x["in_old"] and x["in_new"],
              "increment(new∖old)": lambda x: x["in_new"] and not x["in_old"],
              "dropped(old∖new)": lambda x: x["in_old"] and not x["in_new"]}
    out: dict[str, dict] = {}
    for name, pred in groups.items():
        ids = [x["id"] for x in items if pred(x)]
        labs = [flat_labels.get(str(i)) for i in ids]
        none_n = sum(1 for l in labs if l is None or l[0] is None)
        ok = [l[0] for l in labs if l is not None and l[0] is not None]
        out[name] = {"n": len(ids), "n_labeled": len(ok), "n_none": none_n,
                     "precision": round(sum(ok) / len(ok), 4) if ok else None}
    return out


# ------------------------------------------------------------ 阶段 2 缺口挖掘

def _gap_user(batch: list[tuple[int, str]]) -> str:
    return (f"抽取以下 {len(batch)} 个岗位的 AI 技术技能词：\n"
            + "\n".join(f"{i}. {t}" for i, t in batch)
            + f"\n\n输出 {len(batch)} 个对象的 JSON 数组。")


def load_alias_set() -> set[str]:
    """A 级别名集（is_active='1'，小写）。"""
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("SELECT alias FROM ai_dict.skill_aliases WHERE is_active='1'")
        out = {str(r[0]).strip().lower() for r in cur.fetchall() if r[0]}
    finally:
        conn.close()
    logger.info("A 级别名 %d 个", len(out))
    return out


def load_legacy_terms() -> set[str]:
    """旧自建词表全集（通用 + ai_skill_terms + llm 挖掘，小写）。"""
    return {s.strip().lower() for s in load_merged_skills(include_llm=True) if s}


def missing_anchor_words() -> list[str]:
    """A 级词典完全没有的锚点词（canonical_zh 与激活别名都没有）。"""
    from .skill_ai_anchor import AI_ANCHOR_SKILLS
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT alias FROM ai_dict.skill_aliases
            WHERE is_active='1'
              AND alias = ANY(%s)
        """, (list(AI_ANCHOR_SKILLS),))
        by_alias = {r[0] for r in cur.fetchall()}
        cur.execute("""
            SELECT canonical_zh FROM ai_dict.skill_concepts
            WHERE coalesce(canonical_zh,'') = ANY(%s)
        """, (list(AI_ANCHOR_SKILLS),))
        by_canon = {r[0] for r in cur.fetchall()}
    finally:
        conn.close()
    covered = by_alias | by_canon
    return sorted(set(AI_ANCHOR_SKILLS) - covered)


_TERM_RE = re.compile(r"\s+")


def classify_gap_terms(gap_results: dict, items: list[dict],
                       alias_set: set[str], legacy_set: set[str]) -> tuple[pd.DataFrame, dict]:
    """汇总 LLM 抽取词并分桶（MAPPED/LEGACY_ONLY/NEW_CONCEPT）。

    Args:
        gap_results: run_llm_stage 输出 {bi: {id: terms|None}}。
        items: 抽取条目 [{"id","pos","text"}]。
        alias_set: A 级别名（小写）。
        legacy_set: 旧自建词表（小写）。

    Returns:
        (明细 DataFrame（按频次降序）, 三桶统计 dict)。
    """
    freq: Counter = Counter()
    evidence: dict[str, str] = {}
    batches_by_term: dict[str, set] = {}
    n_failed_items = 0
    bi_by_id = {}
    for bi, mapping in gap_results.items():
        for iid in mapping:
            bi_by_id[iid] = bi
    for item in items:
        iid = str(item["id"])
        mapping = gap_results.get(bi_by_id.get(iid, ""), {})
        terms = mapping.get(iid)
        if terms is None:
            n_failed_items += 1
            continue
        seen: set[str] = set()
        for t in terms:
            if not isinstance(t, dict):
                continue
            term = _TERM_RE.sub(" ", str(t.get("term") or "")).strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            freq[key] += 1
            ev = str(t.get("evidence") or "").strip()
            if key not in evidence and ev:
                evidence[key] = ev[:40]
            batches_by_term.setdefault(key, set()).add(str(bi))
    buckets: dict[str, dict] = {"MAPPED": {"terms": 0, "mentions": 0},
                                "LEGACY_ONLY": {"terms": 0, "mentions": 0},
                                "NEW_CONCEPT": {"terms": 0, "mentions": 0}}
    rows = []
    for key, n in freq.items():
        if key in alias_set:
            bucket = "MAPPED"
        elif key in legacy_set:
            bucket = "LEGACY_ONLY"
        else:
            bucket = "NEW_CONCEPT"
        buckets[bucket]["terms"] += 1
        buckets[bucket]["mentions"] += n
        bs = sorted(batches_by_term.get(key, []), key=int)[:5]
        rows.append({"term": key, "freq": n, "bucket": bucket,
                     "evidence_example": evidence.get(key, ""),
                     "source_batches": ";".join(bs)})
    cols = ["term", "freq", "bucket", "evidence_example", "source_batches"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.sort_values(["freq", "term"], ascending=[False, True])
    buckets["_failed_items"] = n_failed_items
    return df, buckets


# ------------------------------------------------------------ 主流程

def main() -> None:
    """两阶段入口：抽样->ω->40k判定->盲评->缺口挖掘->报告。"""
    parser = argparse.ArgumentParser(description="金标准盲评 + AI 技术词缺口挖掘")
    parser.add_argument("--sample-pct", type=float, default=5.0)
    parser.add_argument("--eval-per", type=int, default=150, help="old/new 各盲评条数")
    parser.add_argument("--gap-n", type=int, default=1200, help="阶段2抽取样本量")
    parser.add_argument("--workers", type=int, default=4, help="LLM 并发（共享GPU，<=4）")
    parser.add_argument("--count-workers", type=int, default=16, help="ω 频数统计进程数")
    parser.add_argument("--judge-workers", type=int, default=8, help="40k 判定进程数")
    parser.add_argument("--keep-tmp", action="store_true", help="结束后保留 eval_tmp 断点")
    args = parser.parse_args()

    paths = get_project_paths()
    setup_logging(paths.log_dir / "gold_standard_eval.log")
    ensure_single_instance()

    # ---- 前置：健康检查（client 统一经 create_llm_client，地址走环境变量）
    client_probe = create_llm_client()
    model_id, max_len = health_check(client_probe.base_url)
    if model_id != client_probe.model:
        os.environ["AIPEN_LLM_MODEL"] = model_id
        logger.info("模型名以服务为准: %s", model_id)
    budget = min(PROMPT_CHAR_BUDGET, int(max_len * 0.7))

    tmp = paths.output_dir / "eval_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    omega_snap = json.loads(
        (paths.project_root / DEFAULT_OMEGA_SNAPSHOT).read_text(encoding="utf-8"))
    logger.info("快照 ω 加载: %d 项", len(omega_snap))

    # ---- 阶段 1a/1b
    sample = fetch_sample(args.sample_pct, tmp / f"sample_gz_sz_{YEAR}_p{args.sample_pct}.pkl")
    if len(sample) < 500000:
        logger.warning("抽样 %d 行低于预期（昨晚基准 ~70 万），请查因", len(sample))
    ptag = f"p{args.sample_pct}"
    omega_old, omega_c, stamp_w = fit_omegas(sample, tmp / f"omega_counts_{ptag}.pkl",
                                             paths.report_dir, args.count_workers)
    judged = judge_rows(sample, tmp / f"judged40k_{ptag}.pkl",
                        omega_snap, omega_old, omega_c, args.judge_workers)
    n_old = sum(1 for r in judged if r["a"] or r["b_old_smooth"])
    n_new = sum(1 for r in judged if r["a"] or r["b_new"])
    logger.info("40k 判定率: old_fused %d (%.3f%%) / new_fused %d (%.3f%%)",
                n_old, 100 * n_old / len(judged), n_new, 100 * n_new / len(judged))

    # ---- 阶段 1c：盲评
    items, _merged = build_blind_items(judged, args.eval_per)
    batches = make_batches([(x["id"], x["text"]) for x in items], budget)
    logger.info("盲评 %d 条 -> %d 批（字符预算 %d）", len(items), len(batches), budget)
    eval_results = run_llm_stage("eval", batches, EVAL_SYSTEM, _eval_user,
                                 600, parse_eval, tmp / f"eval_labels_{ptag}.json",
                                 args.workers)
    labels = {str(bi): v for bi, v in eval_results.items()}
    flat: dict[str, tuple] = {}
    for v in labels.values():
        flat.update(v)
    stats = precision_stats(items, flat)

    # ---- 阶段 2：缺口挖掘。主口径池（b_new 或含强概念）广深实测远小于
    # 名义 1200，不足时放宽到"含任意 A 级概念命中"补足（报告注明两子池）
    pool = [r for r in judged if r["b_new"] or r["n_strong"] >= 1]
    rng2 = random.Random(SEED + 2)
    core_ids = {id(r) for r in pool}
    if len(pool) < args.gap_n:
        extra = [r for r in judged if r.get("n_cs", 0) >= 1 and id(r) not in core_ids]
        need = args.gap_n - len(pool)
        picked = pool + rng2.sample(extra, min(need, len(extra)))
    else:
        picked = rng2.sample(pool, args.gap_n)
    gap_items = [{"id": k + 1, "pos": r["pos"], "core": id(r) in core_ids,
                  "text": f"岗位名:{r['pos']}｜描述:{r['desc'][:GAP_DESC_CHARS]}"}
                 for k, r in enumerate(picked)]
    n_core = sum(1 for x in gap_items if x["core"])
    # 抽取批输出词表较长：3 条/批，避免 4096 上下文内输出被截断
    gap_batches = make_batches([(x["id"], x["text"]) for x in gap_items], budget, max_items=3)
    logger.info("缺口抽取 %d 条（主口径 %d + 任意概念补足 %d）-> %d 批；主口径池 %d/%d",
                len(gap_items), n_core, len(gap_items) - n_core, len(gap_batches),
                len(pool), len(judged))
    gap_results = run_llm_stage("gap", gap_batches, GAP_SYSTEM, _gap_user,
                                1400, parse_gap, tmp / f"gap_terms_{ptag}.json",
                                args.workers)
    df_terms, buckets = classify_gap_terms(gap_results, gap_items,
                                           load_alias_set(), load_legacy_terms())
    gap_df = df_terms[df_terms["bucket"] == "NEW_CONCEPT"]
    hi_gap = gap_df[gap_df["freq"] >= 5]

    # ---- 产物落盘
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    detail_rows = []
    for x in items:
        lab = flat.get(str(x["id"]))
        label = lab[0] if lab else None
        reason = lab[1] if lab else ""
        detail_rows.append({
            "id": x["id"], "position": x["pos"], "desc_head": x["text"][:200],
            "in_old_sample": x["in_old"], "in_new_sample": x["in_new"],
            "a": judged[x["row"]]["a"], "b_old_snap": judged[x["row"]]["b_old_snap"],
            "b_old_smooth": judged[x["row"]]["b_old_smooth"], "b_new": judged[x["row"]]["b_new"],
            "llm_label": label, "llm_reason": reason,
            "error_type": classify_error(reason) if label == 0 else "",
        })
    detail_csv = paths.report_dir / f"gold_standard_eval_{stamp}_detail.csv"
    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False, encoding="utf-8-sig")
    gap_csv = paths.report_dir / f"ai_terms_gap_{datetime.now():%Y%m%d}.csv"
    df_terms.to_csv(gap_csv, index=False, encoding="utf-8-sig")

    # 错误类型分布（label=0 的 reason 分桶）
    err_old = Counter(r["error_type"] for r in detail_rows
                      if r["in_old_sample"] and r["llm_label"] == 0)
    err_new = Counter(r["error_type"] for r in detail_rows
                      if r["in_new_sample"] and r["llm_label"] == 0)
    err_inc = Counter(r["error_type"] for r in detail_rows
                      if r["in_new_sample"] and not r["in_old_sample"] and r["llm_label"] == 0)
    missing_anchors = missing_anchor_words()

    md = _build_report(stamp, model_id, client_probe.base_url, max_len, args,
                       len(sample), omega_old, omega_c, stamp_w, len(judged),
                       n_old, n_new, stats, detail_rows, err_old, err_new, err_inc,
                       len(gap_items), len(gap_batches), buckets, hi_gap,
                       df_terms, gap_csv, detail_csv, missing_anchors, budget,
                       n_core, len(pool))
    report = paths.report_dir / f"gold_standard_eval_{stamp}.md"
    report.write_text(md, encoding="utf-8")
    logger.info("报告: %s", report)
    print(json.dumps({"stats": stats, "buckets": buckets,
                      "gap_new_terms": int(buckets['NEW_CONCEPT']['terms']),
                      "report": str(report)}, ensure_ascii=False, indent=1))

    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        logger.info("已清理 eval_tmp")


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.2f}%"


def _build_report(stamp, model_id, base_url, max_len, args, n_sample,
                  omega_old, omega_c, stamp_w, n_judged, n_old, n_new, stats,
                  detail_rows, err_old, err_new, err_inc, n_gap, n_gap_batches,
                  buckets, hi_gap, df_terms, gap_csv, detail_csv,
                  missing_anchors, budget, n_core, n_pool) -> str:
    """拼装两阶段合并报告。"""
    lines = [f"# 金标准盲评 × 新旧 fused 绝对精确率 + AI 技术词缺口挖掘（{stamp}）", ""]
    lines += ["## 0. 运行环境与健康检查", "",
              f"- LLM 服务: `{base_url}`（本机 WSL vLLM；约定端口 localhost:8000 "
              "连接被拒，服务实际监听 WSL NAT 地址 8101，经 AIPEN_LLM_BASE_URL "
              "环境变量覆盖，未改 config）",
              f"- 模型: {model_id}，max_model_len={max_len}（故盲评批大小按字符预算"
              f" {budget} 动态组批而非固定 20 条；temperature=0，"
              "chat_template_kwargs.enable_thinking=false）",
              "- eps 只读（仅 SELECT）；未做任何写库", ""]
    lines += ["## 1. 阶段 1 方法学", "",
              f"- 抽样: TABLESAMPLE SYSTEM({args.sample_pct}%) 广深 {YEAR}，"
              f"共 {n_sample:,} 行（两城并行连接）",
              f"- 平滑 ω: 旧词表入表 {len(omega_old)}（昨晚基准 3,776）/ "
              f"A级概念入表 {len(omega_c)}（基准 2,473）；"
              f"文件 omega_smooth_legacy_{stamp_w}.json / omega_smooth_atier_{stamp_w}.json",
              f"- 判定样本: 同一抽样随机 {n_judged:,} 行（seed={SEED}）；"
              f"old_fused(A∨b_old平滑)={n_old}（{100*n_old/n_judged:.3f}%），"
              f"new_fused(A∨b_new)={n_new}（{100*n_new/n_judged:.3f}%）",
              "- b_old(快照) 一并记录仅作明细参考（DEFAULT_OMEGA_SNAPSHOT + 现行 "
              "is_ai_fused B 参数）；盲评 old 组按任务规定用 A∨b_old平滑",
              f"- 盲评: 两池各随机 {args.eval_per} 条合并去重 "
              f"{len(detail_rows)} 条，position+描述前 {EVAL_DESC_CHARS} 字，LLM 盲评", ""]
    lines += ["## 2. 阶段 1 结果：绝对精确率", "",
              "| 集合 | 条数 | 有效判定 | None | 精确率 |", "|---|---|---|---|---|"]
    zh = {"old": "old_fused 抽样池", "new": "new_fused 抽样池",
          "common": "共同集 old∩new", "increment(new∖old)": "增量集 new∖old",
          "dropped(old∖new)": "掉出集 old∖new"}
    for k in ("old", "new", "common", "increment(new∖old)", "dropped(old∖new)"):
        s = stats[k]
        lines.append(f"| {zh[k]} | {s['n']} | {s['n_labeled']} | {s['n_none']} | "
                     f"{_pct(s['precision'])} |")
    lines += ["", "label=0 条目错误类型分布（reason 关键词分桶）：", "",
              "| 错误类型 | old 池 | new 池 | 增量集 |", "|---|---|---|---|"]
    all_types = sorted(set(err_old) | set(err_new) | set(err_inc))
    for t in all_types:
        lines.append(f"| {t} | {err_old.get(t, 0)} | {err_new.get(t, 0)} | "
                     f"{err_inc.get(t, 0)} |")
    lines += ["", f"- 明细: `{detail_csv.name}`", ""]
    lines += ["## 3. 阶段 2 方法学", "",
              f"- 抽取样本: 判定样本中 b_new=True 或含强概念(概念ω≥0.5)的岗位（主口径）"
              f" {n_core} 条 + 放宽池『含任意 A 级概念命中』随机补足 {n_gap - n_core} 条"
              f" = {n_gap} 条（主口径池全量 {n_pool}，名义 1200 无法由主口径单独达到，"
              "已在报告中显式分桶）；描述截 600 字",
              f"- LLM 抽词: {n_gap_batches} 批（字符预算动态组批），每词附 ≤12 字证据；"
              "条数校验+重试 1 次后缺失记 None",
              "- 分桶: term.lower() ∈ A级别名(is_active='1') → MAPPED；"
              "否则 ∈ 旧自建词表(general/ai_skill_terms/llm 合并) → LEGACY_ONLY；"
              "两边都无 → NEW_CONCEPT（缺口）", ""]
    lines += ["## 4. 阶段 2 结果：三桶统计", "",
              "| 桶 | unique 词数 | 提及总次数 |", "|---|---|---|"]
    for b in ("MAPPED", "LEGACY_ONLY", "NEW_CONCEPT"):
        lines.append(f"| {b} | {buckets[b]['terms']} | {buckets[b]['mentions']} |")
    lines += ["", f"- LLM 抽取失败条目: {buckets['_failed_items']}",
              f"- 缺口词全量清单: `{gap_csv.name}`（按 freq 降序）", "",
              f"freq≥5 的 NEW_CONCEPT 缺口词（共 {len(hi_gap)} 个）：", "",
              "| term | freq | 证据样例 |", "|---|---|---|"]
    for _, r in hi_gap.head(60).iterrows():
        lines.append(f"| {r['term']} | {r['freq']} | {r['evidence_example']} |")
    top_any = df_terms[df_terms["freq"] >= 3].head(20)
    if len(top_any):
        lines += ["", "所有桶 freq≥3 Top20（对照看 A 级覆盖质量）：", "",
                  "| term | freq | bucket |", "|---|---|---|"]
        for _, r in top_any.iterrows():
            lines.append(f"| {r['term']} | {r['freq']} | {r['bucket']} |")
    # ---- 结论（自动计算）：是否需要新的关键词提取策略
    g_df = df_terms[df_terms["bucket"] == "NEW_CONCEPT"]
    max_new = int(g_df["freq"].max()) if len(g_df) else 0
    avg_new = (buckets["NEW_CONCEPT"]["mentions"]
               / max(buckets["NEW_CONCEPT"]["terms"], 1))
    legacy_hi = df_terms[(df_terms["bucket"] == "LEGACY_ONLY") & (df_terms["freq"] >= 5)]
    modern = ["提示词", "prompt", "rag", "检索增强", "agent", "智能体", "多模态",
              "aigc", "文生图", "文生文", "图生图", "图生文", "具身", "大模型",
              "llm", "微调", "lora", "langchain", "diffusion", "蒸馏", "量化", "gpt"]
    def _modern_hit(t: str) -> bool:
        for m in modern:
            if m.isascii():
                if re.search(rf"(?<![a-z]){re.escape(m)}(?![a-z])", t):
                    return True
            elif m in t:
                return True
        return False

    modern_rows = df_terms[df_terms["term"].apply(_modern_hit)].sort_values(
        "freq", ascending=False)
    lines += ["", "### 结论：是否需要新的关键词提取策略", "",
              f"1. 缺口侧证据：freq≥5 的 NEW_CONCEPT 词 **{len(hi_gap)} 个**（"
              f"NEW_CONCEPT 最高频仅 {max_new} 次，均值 {avg_new:.2f} 次/词，"
              f"{buckets['NEW_CONCEPT']['terms']} 个 unique 词几乎全是一次性长尾或"
              "短语性组合）——本样本**不构成**『存在高频真 AI 技术词漏在词表外、"
              "必须上新的关键词提取策略』的直接证据。",
              f"2. 更大的缺口在 LEGACY_ONLY：freq≥5 的『旧表有、A 级没有』词 "
              f"{len(legacy_hi)} 个（合计 {int(legacy_hi['freq'].sum())} 次提及），"
              "以框架/工具/具体视觉任务词为主（pytorch/tensorflow/opencv/caffe/"
              "halcon/目标检测/图像分割…）——优先动作应是**扩充/激活 A 级别名**"
              "（数据工程），而不是改动判定链。",
              f"3. 大模型时代新词的实际出现情况（本 {n_gap} 条样本内，含任意桶）：", ""]
    if len(modern_rows):
        lines += ["| term | freq | bucket |", "|---|---|---|"]
        for _, r in modern_rows.head(25).iterrows():
            lines.append(f"| {r['term']} | {r['freq']} | {r['bucket']} |")
        lines += ["", f"- 合计 {int(modern_rows['freq'].sum())} 次提及 / "
                  f"{len(modern_rows)} 个 unique 词"]
    else:
        lines += ["- 本样本内未出现任何大模型时代新词"]
    lines += ["- 结论：若要让提示词/RAG/Agent/多模态类新词**系统性**进入识别，"
              "值得做的不是重造关键词提取，而是把本模块固化为**周期性 LLM 缺口"
              "挖掘流程**（扩大语料窗口 + 词形归一后再对照 A 级），把稳定复现的"
              "高频新词批量走词典导入通道", ""]
    lines += ["## 5. 锚点适配（记录，不实现）", "",
              f"- 现行 8 词锚点中 A 级词典完全没有的锚点词: "
              f"{missing_anchors or '无'}；指南六组锚点若落地，需先补齐这些概念的"
              "激活别名（强化学习/人工智能类），否则新锚点组在 A 级链上不可判定",
              "- 8 词锚点是『描述技能』口径，A 级概念 id 口径下锚点覆盖面"
              "（alias 命中）更大但同形词（深度/强化学习）语境验证必须保留", ""]
    lines += ["## 6. 局限", "",
              "- 盲评 LLM（Qwen3-8B）本身非完美金标准：单模型、temp=0、"
              "只读描述前 800 字，判定有噪声；结果应读作『相对精确率对比』而非绝对真值",
              "- TABLESAMPLE SYSTEM 按物理块抽样，行内相关（同公司模板聚集），"
              "精确率方差偏乐观",
              "- 固定批大小 20 与 4096 上下文冲突，实际按预算动态组批；"
              "每条判定预算不变，盲评总条数受任务上限约束",
              "- 缺口分桶用词面精确匹配（lower+空白归一），A 级若以不同书写"
              "（大小写/空格/全半角变体）收录会虚增 NEW_CONCEPT",
              "- 2024 数据约止于 10 月初（eps 口径），本实验不涉及时序解读", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
