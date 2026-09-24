"""指南 §8.7/§9：正式候选发现抽取器。

输入由 discovery_formal 选出的样本 CSV；长文本用固定 tokenizer 在句末/项目符号
边界附近分块，目标 <=5000 tokens、相邻块约200字符重叠。Qwen 只抽原文连续
surface；程序再次校验局部跨度并换算成全文跨度。失败记录保留，不从分母消失。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from config.paths import get_project_paths, load_config_yaml
from src.model_platform.llm import create_llm_client, extract_json_from_response

from ..common import setup_logging

_BOUNDARY_RE = re.compile(r"[\n。！？；;!?]+")
STATUS = {"pending", "success", "empty", "schema_failed", "timeout", "runtime_failed"}

logger = logging.getLogger(__name__)


def _load_existing_records(raw_path: Path) -> list[dict]:
    """读取断点 raw 文件为记录列表，容忍末尾撕裂行。

    强杀进程可能在 flush 中途留下半行 JSON（单条记录含完整 LLM 原文，可达
    数十 KB，flush 不保证单行原子落盘）。末尾撕裂行丢弃并告警；**非末尾**
    行损坏说明文件深层损坏，fail-closed 报出行号——静默跳过中间记录会导致
    该 chunk 被当作"已发出未完成"永久 pending。

    Args:
        raw_path: 断点 jsonl 文件。

    Returns:
        按文件顺序的全部完整记录。
    """
    if not raw_path.exists():
        return []
    try:
        text = raw_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = raw_path.read_bytes().decode("utf-8", errors="replace")
        logger.warning("断点文件含非法 UTF-8（疑似撕裂尾行），按 replace 解码")
    lines = text.splitlines()
    out: list[dict] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                logger.warning("忽略断点文件末尾撕裂行（上次进程被强杀）")
                break
            raise RuntimeError(
                f"断点 raw 文件第 {i + 1} 行损坏（非末行），请人工检查: "
                f"{raw_path}"
            ) from None
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fixed_config() -> tuple[dict, object]:
    cfg = load_config_yaml("model_config_v1.yaml")
    runtime = load_config_yaml("model_runtime.yaml")
    actual_model = str(runtime.get("llm", {}).get("model", "")).strip()
    expected_model = str(cfg.get("model", {}).get("repository", "")).strip()
    if actual_model != expected_model:
        raise RuntimeError(
            f"model_runtime 当前模型 {actual_model!r} != "
            f"冻结 production 模型 {expected_model!r}"
        )
    needed = [
        ("model", "revision"), ("model", "tokenizer_version"),
        ("model", "tokenizer_path"), ("runtime", "framework_version"),
        ("runtime", "cuda_version"), ("runtime", "pytorch_version"),
    ]
    bad = [
        f"{s}.{k}" for s, k in needed
        if str(cfg.get(s, {}).get(k, "")).strip() in {"", "TO_BE_CONFIRMED"}
    ]
    if bad:
        raise RuntimeError("正式 discovery 环境未冻结: " + ", ".join(bad))
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("正式长文本分块需要 transformers tokenizer") from exc
    tok = AutoTokenizer.from_pretrained(
        str(cfg["model"]["tokenizer_path"]),
        revision=str(cfg["model"]["revision"]),
        local_files_only=True,
        trust_remote_code=True,
    )
    return cfg, tok


def _tokens(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def chunk_text(text: str, tok, max_tokens: int = 5000,
               overlap_chars: int = 200) -> list[tuple[int, int, str]]:
    """按 tokenizer 真 token 数分块，尽量在句末/项目符号边界截断。"""
    if _tokens(tok, text) <= max_tokens:
        return [(0, len(text), text)]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        lo, hi = start + 1, n
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if _tokens(tok, text[start:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        end = best
        if end < n:
            search_lo = start + max(1, int((end - start) * 0.65))
            boundaries = [m.end() for m in _BOUNDARY_RE.finditer(
                text, search_lo, end
            )]
            if boundaries:
                candidate = boundaries[-1]
                if candidate > start and _tokens(tok, text[start:candidate]) <= max_tokens:
                    end = candidate
        if end <= start:
            raise RuntimeError("无法推进文本分块")
        chunks.append((start, end, text[start:end]))
        if end >= n:
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def _cache_key(text_hash: str, cfg: dict, prompt_sha: str,
               schema_sha: str) -> str:
    inference = json.dumps(
        {"generation": cfg["generation"], "text": cfg["text"]},
        sort_keys=True, ensure_ascii=False,
    )
    payload = "|".join([
        str(text_hash),
        str(cfg["model"]["revision"]),
        prompt_sha,
        schema_sha,
        _sha_text(inference),
    ])
    return _sha_text(payload)


def _locate_surface(
    chunk: str,
    surface: str,
    evidence: str,
) -> tuple[int, int, int] | None:
    """§8.11 确定性跨度回退：唯一命中→证据句→首次位置。"""
    if not surface:
        return None
    starts = []
    pos = chunk.find(surface)
    while pos >= 0:
        starts.append(pos)
        pos = chunk.find(surface, pos + 1)
    if not starts:
        return None
    multiple = int(len(starts) > 1)
    if len(starts) == 1:
        start = starts[0]
        return start, start + len(surface), multiple
    ev = str(evidence or "")
    ev_start = chunk.find(ev) if ev else -1
    if ev_start >= 0:
        ev_end = ev_start + len(ev)
        inside = [
            s for s in starts
            if ev_start <= s and s + len(surface) <= ev_end
        ]
        if inside:
            start = inside[0]
            return start, start + len(surface), multiple
    start = starts[0]
    return start, start + len(surface), multiple


def _validate(
    parsed,
    jid: str,
    chunk: str,
    offset: int,
    allowed_types: set[str],
) -> tuple[str, list[dict], list[dict]]:
    if not isinstance(parsed, dict):
        return "schema_failed", [], []
    # 不校验 job_id 是否逐字回显：与 model_benchmark 同因——schema 对 job_id
    # 只要求 type=string、minLength=1，逐字复制 32 位十六进制串是额外的业务
    # 校验且是 LLM 已知弱点，模型会幻觉出重复片段。抽取正确性已由 surface
    # 在 chunk 内可定位（否则判 schema_failed）独立保证。
    skills = parsed.get("skills")
    if not isinstance(skills, list):
        return "schema_failed", [], []
    out = []
    dropped: list[dict] = []
    for rec in skills:
        if not isinstance(rec, dict):
            return "schema_failed", [], []
        required = {
            "surface", "canonical_suggestion", "skill_type",
            "evidence", "start", "end", "existing_skill_id",
        }
        if not required.issubset(rec):
            return "schema_failed", [], []
        surface = str(rec.get("surface", ""))
        canonical = str(rec.get("canonical_suggestion", ""))
        stype = str(rec.get("skill_type", ""))
        evidence = str(rec.get("evidence", ""))
        if not surface or not canonical or not evidence or stype not in allowed_types:
            return "schema_failed", [], []
        existing = rec.get("existing_skill_id")
        if existing is not None and not isinstance(existing, str):
            return "schema_failed", [], []
        span_source = "model"
        multiple = 0
        try:
            start = int(rec["start"])
            end = int(rec["end"])
        except (TypeError, ValueError):
            start = end = -1
        if not (
            0 <= start < end <= len(chunk)
            and chunk[start:end] == surface
        ):
            located = _locate_surface(chunk, surface, evidence)
            if located is None:
                # §8.11 第 7 条：surface 无法在模型输入文本中找到时判为**该条**
                # 跨度失败。原实现 return "schema_failed", [] 会丢弃整个 chunk
                # 的全部技能——按实测 2.4% 的非逐字率，含 n 个技能的 chunk 整块
                # 失败的概率是 1-0.976^n（n=30 时约 52%），会丢掉近半数据。
                # 改为只丢弃该条，并把丢弃内容随记录留存，使模型不逐字这一
                # 事实留在数据里而不是被静默吞掉。
                dropped.append({"surface": surface, "skill_type": stype})
                continue
            start, end, multiple = located
            span_source = "deterministic_fallback"
        out.append({
            "surface": surface,
            "canonical_suggestion": canonical,
            "skill_type": stype,
            "evidence": evidence,
            "start": offset + start,
            "end": offset + end,
            "existing_skill_id": existing,
            "span_source": span_source,
            "multiple_span_flag": multiple,
            # 本字段恒为 True：能走到这里就说明 surface 在 chunk 内可逐字
            # 定位（不可定位的已在上方丢弃并记入 nonverbatim_dropped）。
            # 显式写出是为了让下游可以依赖这个契约，而不是依赖注释。
            "verbatim": True,
        })
    return ("empty" if not out else "success"), out, dropped


def _extract_one(
    client,
    prompt: str,
    cfg: dict,
    schema: dict,
    allowed_types: set,
    common: dict,
    chunk: str,
    offset: int,
    retry_invalid: int,
) -> dict:
    """单个 chunk 的抽取调用（含 invalid JSON/schema 重试）。

    并发化改造：原实现嵌在 for row / for chunk 双层循环里串行调用，
    网络等待期间 GPU 空转（实测单请求约 11.3s）。抽成独立函数后由线程池
    并发执行；重试条件、状态判定与落盘字段均与原实现逐字一致。

    Args:
        client: LLM 客户端。
        prompt: system prompt 文本。
        cfg: 冻结的 model_config_v1 内容。
        schema: 受控 JSON schema。
        allowed_types: 允许的 skill_type 枚举。
        common: 该 chunk 的公共字段（含 record_key/job_id/定位信息）。
        chunk: 该 chunk 的正文。
        offset: chunk 在原文中的起始位置（供 _validate 回查 span）。
        retry_invalid: schema 失败时的额外重试次数。

    Returns:
        待落盘的 final 记录（含 status/skills/raw/error/attempt）。
    """
    jid = str(common["job_id"])
    user = json.dumps(
        {"job_id": jid, "text": chunk},
        ensure_ascii=False,
    )
    final = None
    # 修复重试用的重复惩罚。temperature=0 下重跑是确定性的、只会复现同样
    # 结果；实测约 2% 的样本会陷入「token 之间无限输出空白」的退化循环
    # （不是内容过长——给多大预算就烧多大），靠该参数才能打破循环。
    # 正常样本首次尝试即通过，不会进入修复分支，因此不受它影响。
    repair_penalty = float(
        cfg["generation"].get("repair_repetition_penalty", 1.0)
    )
    # 两档输出预算。实测（60 条真实语料、8 并发）：
    #   8192  -> 11.2 条/分钟    12288 -> 8.9 条/分钟（慢 20%）
    #   两档的输出中位同为 750 token、截断数同为 2/60
    # 即：12288 对占多数(96.7%)的普通样本毫无收益，只为极少数技能极密的样本
    # 准备；而后者在 8192 下同样截断，说明它们需要的超过 12288。故普通调用走
    # base（快 20%），只有已失败的修复重试才用满 config 的 max_output_tokens
    # ——既拿到速度，也不放弃对密集样本的补救机会。
    # 不直接改 config 的 max_output_tokens：那会使技术基准与 prerun 已冻结并
    # 通过验收的取值失配。
    base_out = int(
        cfg["generation"].get("extract_base_output_tokens", 8192)
    )
    for attempt in range(1, retry_invalid + 2):
        status = "runtime_failed"
        skills: list[dict] = []
        dropped: list[dict] = []
        raw = ""
        error = ""
        extra_payload = {
            "seed": int(cfg["generation"]["seed"]),
            "chat_template_kwargs": {"enable_thinking": False},
            "guided_json": schema,
        }
        if attempt > 1 and repair_penalty > 1.0:
            extra_payload["repetition_penalty"] = repair_penalty
        try:
            raw = client.complete_text(
                system_prompt=prompt,
                user_prompt=user,
                temperature=float(
                    cfg["generation"]["temperature"]
                ),
                max_output_tokens=(
                    int(cfg["generation"]["max_output_tokens"])
                    if attempt > 1
                    else base_out
                ),
                extra_payload=extra_payload,
            )
            try:
                parsed = extract_json_from_response(raw)
            except ValueError as exc:
                status = "schema_failed"
                error = str(exc)
            else:
                status, skills, dropped = _validate(
                    parsed, jid, chunk, offset, allowed_types
                )
                if status == "schema_failed":
                    error = "schema_or_span_validation_failed"
        except TimeoutError as exc:
            status, error = "timeout", str(exc)
        except Exception as exc:
            status = "runtime_failed"
            error = f"{type(exc).__name__}: {exc}"

        final = {
            **common,
            "status": status,
            "skills": skills,
            "raw": raw,
            "error": error,
            "attempt": attempt,
            # 模型给出但无法在 chunk 中逐字定位、因而被丢弃的条目。留在这里是
            # 为了不让「模型不逐字」这一事实随丢弃一起消失：下游若要评估抽取
            # 质量或复算非逐字率，可直接统计本字段。
            "nonverbatim_dropped": dropped,
            "nonverbatim_dropped_count": len(dropped),
        }
        # 指南只要求 invalid JSON/schema 重试；运行/超时保留原状态。
        if status != "schema_failed" or attempt > retry_invalid:
            break
    if final is None:
        raise RuntimeError("候选抽取未产生最终状态")
    return final


def run(sample_csv: Path) -> tuple[Path, Path]:
    cfg, tok = _fixed_config()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_discovery_extract.log")
    logger.info("正式候选发现抽取开始: sample=%s", sample_csv)
    prompt_path = paths.config_dir / "skill_extraction_prompt_v1.md"
    schema_path = paths.config_dir / "skill_extraction_schema_v1.json"
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    schema_sha = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    allowed_types = set(
        schema["properties"]["skills"]["items"]["properties"]
        ["skill_type"]["enum"]
    )
    retry_invalid = int(cfg["generation"].get("retry_invalid_json", 1))

    sample = pd.read_csv(sample_csv)
    required = {
        "job_id", "year", "text_hash", "description",
        "anchor_main", "discovery_round",
    }
    missing = required - set(sample.columns)
    if missing:
        raise ValueError(
            "formal sample 缺列: " + ", ".join(sorted(missing))
        )
    if sample.job_id.duplicated().any():
        raise ValueError("formal discovery sample job_id 重复")

    out_dir = paths.output_dir / "llm_review" / "formal_discovery_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "chunk_results.jsonl"
    mentions_path = out_dir / "mentions.parquet"
    candidate_surface_path = (
        paths.output_dir / "dictionary" / "skill_candidate_surface.parquet"
    )
    manifest_path = out_dir / "extraction_manifest.json"

    latest: dict[str, dict] = {}
    content_cache: dict[str, dict] = {}
    for rec in _load_existing_records(raw_path):
        record_key = str(rec.get("record_key", ""))
        if record_key:
            latest[record_key] = rec
        chunk_key = str(rec.get("chunk_cache_key", ""))
        if (
            chunk_key
            and rec.get("status") in {"success", "empty"}
        ):
            content_cache[chunk_key] = rec
    logger.info(
        "断点载入: 唯一 record %s 条，可复用 chunk 缓存 %s 条",
        len(latest), len(content_cache),
    )

    client = create_llm_client()
    expected: list[str] = []

    # 阶段 1（顺序）：按原顺序遍历，落 pending、处理缓存命中，并把需要
    # 真实调用的 chunk 收集为任务。pending 仍在发请求前写入并 flush，
    # 因此断点语义不变：崩溃后能看出哪些 chunk 已发出、哪些未开始。
    tasks: list[tuple] = []
    with raw_path.open("a", encoding="utf-8") as fh:
        for row in sample.itertuples(index=False):
            jid = str(row.job_id)
            text = str(row.description)
            base_key = _cache_key(
                str(row.text_hash), cfg, prompt_sha, schema_sha
            )
            chunks = chunk_text(
                text, tok,
                int(cfg["text"]["chunk_target_tokens"]),
                int(cfg["text"]["chunk_overlap_chars"]),
            )
            for chunk_id, (offset, end_global, chunk) in enumerate(chunks):
                chunk_sha = _sha_text(chunk)
                chunk_cache_key = _sha_text(
                    f"{base_key}|{chunk_id}|{chunk_sha}"
                )
                record_key = _sha_text(
                    f"{jid}|{chunk_cache_key}"
                )
                expected.append(record_key)
                existing = latest.get(record_key)
                if existing and existing.get("status") in (
                    STATUS - {"pending"}
                ):
                    continue

                common = {
                    "record_key": record_key,
                    "cache_key": base_key,
                    "chunk_cache_key": chunk_cache_key,
                    "chunk_input_sha256": chunk_sha,
                    "chunk_id": chunk_id,
                    "job_id": jid,
                    "year": int(row.year),
                    "text_hash": str(row.text_hash),
                    "anchor_main": int(row.anchor_main),
                    "discovery_round": int(row.discovery_round),
                    "chunk_start": offset,
                    "chunk_end": end_global,
                }
                pending = {
                    **common, "status": "pending", "skills": [],
                    "raw": "", "error": "", "attempt": 0,
                }
                fh.write(
                    json.dumps(pending, ensure_ascii=False) + "\n"
                )
                fh.flush()
                latest[record_key] = pending

                cached = content_cache.get(chunk_cache_key)
                if cached is not None:
                    final = {
                        **common,
                        "status": cached["status"],
                        "skills": cached.get("skills", []),
                        "raw": cached.get("raw", ""),
                        "error": cached.get("error", ""),
                        "attempt": 0,
                        "cache_reused_from_job": cached.get("job_id"),
                    }
                    fh.write(
                        json.dumps(final, ensure_ascii=False) + "\n"
                    )
                    fh.flush()
                    latest[record_key] = final
                    continue

                tasks.append(
                    (record_key, chunk_cache_key, common, chunk, offset)
                )

    # 阶段 2（并发 + 顺序落盘）：网络等待是主要瓶颈（实测单请求约 11.3s），
    # 串行时 GPU 空转。pool.map 按 tasks 顺序产出结果，边取边写即与串行版
    # 记录顺序一致，同时每条立即 flush：长跑可观察进度，中断后已完成部分
    # 仍留在 raw 文件里，不必重跑。
    #
    # 并发上限由服务端 --max-num-seqs 决定（AIPEN_EXTRACT_WORKERS 可覆盖）。
    # 差异仅在于：批次内新增的跨 chunk 缓存复用不再发生（结果由固定 seed +
    # temperature=0 决定，内容等价）。
    workers = max(1, int(os.getenv("AIPEN_EXTRACT_WORKERS", "8")))
    logger.info(
        "阶段 2 开始: 待调用 chunk %s，并发 %s（缓存命中 %s 条已在阶段 1 落盘）",
        len(tasks), workers,
        len(latest) - len(tasks) if len(latest) >= len(tasks) else 0,
    )
    t_phase2 = time.time()
    done_in_phase2 = 0
    with raw_path.open("a", encoding="utf-8") as fh:
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            for (record_key, chunk_cache_key, _common, _chunk, _off), final in (
                zip(
                    tasks,
                    pool.map(
                        lambda t: _extract_one(
                            client, prompt, cfg, schema, allowed_types,
                            t[2], t[3], t[4], retry_invalid,
                        ),
                        tasks,
                    ),
                )
            ):
                fh.write(
                    json.dumps(final, ensure_ascii=False) + "\n"
                )
                fh.flush()
                latest[record_key] = final
                if final["status"] in {"success", "empty"}:
                    content_cache[chunk_cache_key] = final
                done_in_phase2 += 1
                if done_in_phase2 % 100 == 0 or done_in_phase2 == len(tasks):
                    elapsed = time.time() - t_phase2
                    logger.info(
                        "抽取进度: %s/%s（%.1f%%），平均 %.1fs/条，已用 %.1f 分钟",
                        done_in_phase2, len(tasks),
                        done_in_phase2 * 100.0 / max(len(tasks), 1),
                        elapsed / max(done_in_phase2, 1), elapsed / 60,
                    )
        except BaseException:
            # 消费循环中断（写盘失败/Ctrl+C）时取消排队任务：with 退出的
            # shutdown(wait=True) 默认不取消已排队 future，会把整个队列跑完
            # 却不再落盘（prerun 规模实测可假死约 2 小时）。
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    logger.info("阶段 2 完成，用时 %.1f 分钟", (time.time() - t_phase2) / 60)

    records = []
    for key in expected:
        rec = latest.get(key)
        if rec is None:
            rec = {"record_key": key, "status": "pending", "skills": []}
        records.append(rec)

    mention_rows = []
    seen_mentions: set[tuple] = set()
    for rec in records:
        if rec.get("status") not in {"success", "empty"}:
            continue
        for sk in rec.get("skills", []):
            key = (
                str(rec["job_id"]), int(sk["start"]),
                int(sk["end"]), sk["surface"],
            )
            if key in seen_mentions:
                continue
            seen_mentions.add(key)
            mention_rows.append({
                "job_id": rec["job_id"],
                "year": rec["year"],
                "text_hash": rec["text_hash"],
                "anchor_main": rec["anchor_main"],
                "discovery_round": rec["discovery_round"],
                **sk,
                "span_valid": 1,
            })
    mention_cols = [
        "job_id", "year", "text_hash", "anchor_main", "discovery_round",
        "surface", "canonical_suggestion", "skill_type", "evidence",
        "start", "end", "existing_skill_id", "span_source",
        "multiple_span_flag", "span_valid",
    ]
    mentions = pd.DataFrame(mention_rows, columns=mention_cols)
    mentions.to_parquet(
        mentions_path, index=False, compression="zstd"
    )
    candidate_surface_path.parent.mkdir(parents=True, exist_ok=True)
    mentions.to_parquet(
        candidate_surface_path, index=False, compression="zstd"
    )

    status_counts = pd.Series(
        [r.get("status", "pending") for r in records]
    ).value_counts().to_dict()
    failures = sum(
        v for k, v in status_counts.items()
        if k not in {"success", "empty"}
    )
    total = len(records)
    manifest = {
        "status": (
            "formal_pass"
            if total and failures / total <= 0.01
            and status_counts.get("pending", 0) == 0
            else "failed"
        ),
        "sample_sha256": hashlib.sha256(
            sample_csv.read_bytes()
        ).hexdigest(),
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
        "model_revision": cfg["model"]["revision"],
        "tokenizer_version": cfg["model"]["tokenizer_version"],
        "tokenizer_path": cfg["model"]["tokenizer_path"],
        "inference_config_sha256": _sha_text(json.dumps(
            {"generation": cfg["generation"], "text": cfg["text"]},
            sort_keys=True, ensure_ascii=False,
        )),
        "total_chunks": total,
        "status_counts": status_counts,
        "failure_rate": failures / max(total, 1),
        "workers": workers,
        "raw_sha256": hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest(),
        "mentions_sha256": hashlib.sha256(
            mentions_path.read_bytes()
        ).hexdigest(),
        "skill_candidate_surface_path": str(candidate_surface_path),
        "skill_candidate_surface_sha256": hashlib.sha256(
            candidate_surface_path.read_bytes()
        ).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if manifest["status"] != "formal_pass":
        # SystemExit 不打印 traceback：不留日志的话进程会"无声消失"。
        logger.error(
            "抽取未达 formal_pass: failure_rate=%.4f status_counts=%s",
            manifest["failure_rate"], status_counts,
        )
        raise SystemExit(2)
    logger.info("抽取完成并 formal_pass: manifest=%s", manifest_path)
    return mentions_path, manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(description="指南 §8.7/§9 正式技能发现抽取")
    ap.add_argument("--sample", type=Path, required=True)
    args = ap.parse_args()
    run(args.sample)


if __name__ == "__main__":
    main()
