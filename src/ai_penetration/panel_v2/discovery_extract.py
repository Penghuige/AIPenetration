"""指南 §8.7/§9：正式候选发现抽取器。

输入由 discovery_formal 选出的样本 CSV；长文本用固定 tokenizer 在句末/项目符号
边界附近分块，目标 <=5000 tokens、相邻块约200字符重叠。Qwen 只抽原文连续
surface；程序再次校验局部跨度并换算成全文跨度。失败记录保留，不从分母消失。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from config.paths import get_project_paths, load_config_yaml
from src.model_platform.llm import create_llm_client, extract_json_from_response

_BOUNDARY_RE = re.compile(r"[\n。！？；;!?]+")
STATUS = {"pending", "success", "empty", "schema_failed", "timeout", "runtime_failed"}


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fixed_config() -> tuple[dict, object]:
    cfg = load_config_yaml("model_config_v1.yaml")
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
) -> tuple[str, list[dict]]:
    if not isinstance(parsed, dict) or str(parsed.get("job_id")) != jid:
        return "schema_failed", []
    skills = parsed.get("skills")
    if not isinstance(skills, list):
        return "schema_failed", []
    out = []
    for rec in skills:
        if not isinstance(rec, dict):
            return "schema_failed", []
        required = {
            "surface", "canonical_suggestion", "skill_type",
            "evidence", "start", "end", "existing_skill_id",
        }
        if not required.issubset(rec):
            return "schema_failed", []
        surface = str(rec.get("surface", ""))
        canonical = str(rec.get("canonical_suggestion", ""))
        stype = str(rec.get("skill_type", ""))
        evidence = str(rec.get("evidence", ""))
        if not surface or not canonical or not evidence or stype not in allowed_types:
            return "schema_failed", []
        existing = rec.get("existing_skill_id")
        if existing is not None and not isinstance(existing, str):
            return "schema_failed", []
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
                return "schema_failed", []
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
        })
    return ("empty" if not out else "success"), out


def run(sample_csv: Path) -> tuple[Path, Path]:
    cfg, tok = _fixed_config()
    paths = get_project_paths()
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
    manifest_path = out_dir / "extraction_manifest.json"

    latest: dict[str, dict] = {}
    content_cache: dict[str, dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            record_key = str(rec.get("record_key", ""))
            if record_key:
                latest[record_key] = rec
            chunk_key = str(rec.get("chunk_cache_key", ""))
            if (
                chunk_key
                and rec.get("status") in {"success", "empty"}
            ):
                content_cache[chunk_key] = rec

    client = create_llm_client()
    expected: list[str] = []
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

                user = json.dumps(
                    {"job_id": jid, "text": chunk},
                    ensure_ascii=False,
                )
                final = None
                for attempt in range(1, retry_invalid + 2):
                    status = "runtime_failed"
                    skills: list[dict] = []
                    raw = ""
                    error = ""
                    try:
                        raw = client.complete_text(
                            system_prompt=prompt,
                            user_prompt=user,
                            temperature=float(
                                cfg["generation"]["temperature"]
                            ),
                            max_output_tokens=int(
                                cfg["generation"]["max_output_tokens"]
                            ),
                            extra_payload={
                                "seed": int(cfg["generation"]["seed"]),
                                "chat_template_kwargs": {
                                    "enable_thinking": False
                                },
                                "guided_json": schema,
                            },
                        )
                        try:
                            parsed = extract_json_from_response(raw)
                        except ValueError as exc:
                            status = "schema_failed"
                            error = str(exc)
                        else:
                            status, skills = _validate(
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
                    }
                    # 指南只要求 invalid JSON/schema 重试；运行/超时保留原状态。
                    if status != "schema_failed" or attempt > retry_invalid:
                        break
                if final is None:
                    raise RuntimeError("候选抽取未产生最终状态")
                fh.write(
                    json.dumps(final, ensure_ascii=False) + "\n"
                )
                fh.flush()
                latest[record_key] = final
                if final["status"] in {"success", "empty"}:
                    content_cache[chunk_cache_key] = final

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
        "raw_sha256": hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest(),
        "mentions_sha256": hashlib.sha256(
            mentions_path.read_bytes()
        ).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if manifest["status"] != "formal_pass":
        raise SystemExit(2)
    return mentions_path, manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(description="指南 §8.7/§9 正式技能发现抽取")
    ap.add_argument("--sample", type=Path, required=True)
    args = ap.parse_args()
    run(args.sample)


if __name__ == "__main__":
    main()
