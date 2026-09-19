"""指南 §8：RTX 4090 Qwen 候选比较、技术基准与 1 万条预运行。

本模块只验证技术一致性，不用 benchmark 直接决定技能/岗位指标。
正式顺序：
1. 至少两个不同模型规模或量化配置各跑 technical（1000 条、重复 2 次）；
2. select 生成 model_selection_manifest_v1.json，并把选中配置冻结到
   config/model_config_v1.yaml；
3. 选中生产配置跑 prerun（10000 条）；
4. 两道 manifest 均 formal_pass 后候选发现才允许继续。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from config.paths import get_project_paths, load_config_yaml
from src.model_platform.llm import create_llm_client, extract_json_from_response

THRESHOLDS = {
    "json_schema_valid_rate": 0.995,
    "direct_span_valid_rate": 0.99,
    "hallucinated_surface_rate_max": 0.01,
    "failure_rate_max": 0.01,
    "repeat_exact_rate": 0.95,
}
_REQUIRED_CONFIG = [
    ("model", "repository"), ("model", "revision"),
    ("model", "quantization"), ("model", "tokenizer_version"),
    ("model", "tokenizer_path"),
    ("runtime", "framework"), ("runtime", "framework_version"),
    ("runtime", "cuda_version"), ("runtime", "pytorch_version"),
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cfg(config_file: Path | None) -> tuple[dict, Path]:
    paths = get_project_paths()
    if config_file is None:
        path = paths.config_dir / "model_config_v1.yaml"
        cfg = load_config_yaml("model_config_v1.yaml")
    else:
        path = config_file.resolve()
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bad = []
    for sec, key in _REQUIRED_CONFIG:
        value = str(cfg.get(sec, {}).get(key, "")).strip()
        if not value or value == "TO_BE_CONFIRMED":
            bad.append(f"{sec}.{key}")
    if bad:
        raise RuntimeError(
            "模型环境尚未冻结: " + ", ".join(bad)
        )
    generation = cfg.get("generation", {})
    expected = {
        "do_sample": False, "temperature": 0.0, "top_p": 1.0,
        "seed": 20260822, "max_model_len": 8192,
        "max_input_tokens": 6000, "max_output_tokens": 2000,
        "guided_json": True, "retry_invalid_json": 1,
    }
    mismatch = {
        k: (generation.get(k), v)
        for k, v in expected.items()
        if generation.get(k) != v
    }
    if mismatch:
        raise RuntimeError(
            "技术基准配置偏离指南 §8.6: "
            + json.dumps(mismatch, ensure_ascii=False)
        )
    return cfg, path


def _assert_runtime_matches(cfg: dict) -> None:
    runtime = load_config_yaml("model_runtime.yaml")
    actual = str(runtime.get("llm", {}).get("model", "")).strip()
    expected = str(cfg["model"]["repository"]).strip()
    if actual != expected:
        raise RuntimeError(
            f"model_runtime.yaml 当前模型 {actual!r} != benchmark 配置 {expected!r}"
        )


def _gpu_start() -> tuple[object, str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("正式 RTX 4090 benchmark 需要 PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("正式 benchmark 必须在 CUDA GPU 上运行")
    name = str(torch.cuda.get_device_name(0))
    if "4090" not in name.upper():
        raise RuntimeError(
            f"指南 §8.2 要求 RTX 4090，当前设备为 {name!r}"
        )
    torch.cuda.reset_peak_memory_stats(0)
    return torch, name


def _load_samples(path: Path, minimum: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if not str(row.get("job_id", "")) or not str(row.get("text", "")):
            raise ValueError("benchmark sample 每行必须有 job_id/text")
    if len(rows) < minimum:
        raise ValueError(f"样本不足: {len(rows)} < {minimum}")
    return rows[:minimum]


def _base_aliases() -> set[str]:
    paths = get_project_paths()
    p = (
        paths.output_dir / "dictionary"
        / "skill_alias_active_bilingual_a_frozen_v1.1.csv"
    )
    if not p.exists():
        raise RuntimeError(
            "缺少冻结 A 级激活别名，无法计算 §8.3 基础词典重合率"
        )
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    return {
        str(x).strip().casefold()
        for x in df.alias.dropna()
        if str(x).strip()
    }


def _validate(
    parsed,
    expected_job_id: str,
    text: str,
    allowed_types: set[str],
) -> tuple[bool, int, int, int, int, tuple, list[str]]:
    if not isinstance(parsed, dict):
        return False, 0, 0, 0, 0, (), []
    if str(parsed.get("job_id")) != expected_job_id:
        return False, 0, 0, 0, 0, (), []
    skills = parsed.get("skills")
    if not isinstance(skills, list):
        return False, 0, 0, 0, 0, (), []

    direct_span_ok = hallucinated = evidence_ok = total = 0
    canonical = []
    surfaces = []
    required = {
        "surface", "canonical_suggestion", "skill_type",
        "evidence", "start", "end", "existing_skill_id",
    }
    for skill in skills:
        if not isinstance(skill, dict) or not required.issubset(skill):
            return False, direct_span_ok, total, hallucinated, evidence_ok, (), []
        surface = str(skill.get("surface", ""))
        canonical_suggestion = str(skill.get("canonical_suggestion", ""))
        stype = str(skill.get("skill_type", ""))
        evidence = str(skill.get("evidence", ""))
        if (
            not surface or not canonical_suggestion or not evidence
            or stype not in allowed_types
        ):
            return False, direct_span_ok, total, hallucinated, evidence_ok, (), []
        existing = skill.get("existing_skill_id")
        if existing is not None and not isinstance(existing, str):
            return False, direct_span_ok, total, hallucinated, evidence_ok, (), []
        total += 1
        try:
            start, end = int(skill["start"]), int(skill["end"])
        except (TypeError, ValueError):
            start = end = -1
        if 0 <= start < end <= len(text) and text[start:end] == surface:
            direct_span_ok += 1
        if surface not in text:
            hallucinated += 1
        if evidence in text:
            evidence_ok += 1
        surfaces.append(surface.casefold())
        canonical.append((surface, canonical_suggestion, stype))
    return (
        True, direct_span_ok, total, hallucinated, evidence_ok,
        tuple(sorted(set(canonical))), surfaces,
    )


def _output_names(candidate_id: str, phase: str) -> tuple[Path, Path, Path]:
    paths = get_project_paths()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id)
    if not safe:
        raise ValueError("candidate_id 为空")
    raw = paths.output_dir / "model_benchmark" / f"{safe}_{phase}_raw.jsonl"
    if phase == "prerun" and safe == "production":
        manifest = paths.report_dir / "model_benchmark_prerun_manifest_v1.json"
        report = paths.report_dir / "model_benchmark_prerun_report.md"
    else:
        manifest = (
            paths.report_dir
            / f"model_benchmark_{safe}_{phase}_manifest_v1.json"
        )
        report = (
            paths.report_dir
            / f"model_benchmark_{safe}_{phase}_report.md"
        )
    return raw, manifest, report


def run(
    sample_file: Path,
    phase: str,
    candidate_id: str,
    config_file: Path | None = None,
) -> Path:
    cfg, cfg_path = _load_cfg(config_file)
    _assert_runtime_matches(cfg)
    minimum = 1000 if phase == "technical" else 10000
    repeats = 2 if phase == "technical" else 1
    samples = _load_samples(sample_file, minimum)
    paths = get_project_paths()
    prompt_path = paths.config_dir / "skill_extraction_prompt_v1.md"
    schema_path = paths.config_dir / "skill_extraction_schema_v1.json"
    system = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    allowed_types = set(
        schema["properties"]["skills"]["items"]["properties"]
        ["skill_type"]["enum"]
    )
    base_aliases = _base_aliases()
    client = create_llm_client()
    torch, gpu_name = _gpu_start()

    raw_path, manifest_path, report_path = _output_names(
        candidate_id, phase
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    schema_ok = direct_span_ok = spans = hallucinated = evidence_ok = 0
    failures = retries = 0
    repeated: dict[str, list[tuple]] = {}
    first_pass_surfaces: list[str] = []
    first_pass_new: list[str] = []
    inference_seconds = 0.0
    retry_invalid = int(cfg["generation"]["retry_invalid_json"])

    with raw_path.open("w", encoding="utf-8") as fh:
        for rep in range(repeats):
            for row in samples:
                jid, text = str(row["job_id"]), str(row["text"])
                user = json.dumps(
                    {"job_id": jid, "text": text}, ensure_ascii=False
                )
                final = None
                for attempt in range(1, retry_invalid + 2):
                    raw = ""
                    error = ""
                    t0 = time.perf_counter()
                    try:
                        raw = client.complete_text(
                            system_prompt=system,
                            user_prompt=user,
                            temperature=float(
                                cfg["generation"]["temperature"]
                            ),
                            max_output_tokens=int(
                                cfg["generation"]["max_output_tokens"]
                            ),
                            extra_payload={
                                "seed": int(cfg["generation"]["seed"]),
                                "top_p": float(cfg["generation"]["top_p"]),
                                "chat_template_kwargs": {
                                    "enable_thinking": False
                                },
                                "guided_json": schema,
                            },
                        )
                        try:
                            parsed = extract_json_from_response(raw)
                        except ValueError as exc:
                            valid = False
                            error = str(exc)
                            details = (0, 0, 0, 0, (), [])
                        else:
                            (
                                valid, sok, stotal, hall, eok,
                                skillset, surfaces,
                            ) = _validate(
                                parsed, jid, text, allowed_types
                            )
                            details = (
                                sok, stotal, hall, eok,
                                skillset, surfaces,
                            )
                            if not valid:
                                error = "schema_validation_failed"
                    except TimeoutError as exc:
                        valid = False
                        error = "timeout: " + str(exc)
                        details = (0, 0, 0, 0, (), [])
                    except Exception as exc:
                        valid = False
                        error = f"{type(exc).__name__}: {exc}"
                        details = (0, 0, 0, 0, (), [])
                    finally:
                        inference_seconds += time.perf_counter() - t0
                    if valid or attempt > retry_invalid:
                        final = (valid, raw, error, details, attempt)
                        break
                    retries += 1

                if final is None:
                    raise RuntimeError("benchmark 未产生最终调用状态")
                valid, raw, error, details, attempts = final
                sok, stotal, hall, eok, skillset, surfaces = details
                if valid:
                    schema_ok += 1
                    direct_span_ok += sok
                    spans += stotal
                    hallucinated += hall
                    evidence_ok += eok
                    repeated.setdefault(jid, []).append(skillset)
                    if rep == 0:
                        first_pass_surfaces.extend(surfaces)
                        first_pass_new.extend(
                            s for s in surfaces if s not in base_aliases
                        )
                else:
                    failures += 1
                fh.write(json.dumps({
                    "job_id": jid,
                    "repeat": rep,
                    "attempts": attempts,
                    "schema_valid": bool(valid),
                    "raw": raw,
                    "error": error,
                }, ensure_ascii=False) + "\n")

    calls = len(samples) * repeats
    repeat_exact = 1.0
    if repeats > 1:
        repeat_exact = sum(
            1 for vals in repeated.values()
            if len(vals) == repeats and len(set(vals)) == 1
        ) / len(samples)
    overlap_n = sum(1 for s in first_pass_surfaces if s in base_aliases)
    unique_new = len(set(first_pass_new))
    metrics = {
        "json_schema_valid_rate": schema_ok / calls,
        "evidence_backfill_rate": evidence_ok / max(spans, 1),
        "direct_span_valid_rate": direct_span_ok / max(spans, 1),
        "hallucinated_surface_rate": hallucinated / max(spans, 1),
        "repeat_exact_rate": repeat_exact,
        "base_dictionary_overlap_rate": (
            overlap_n / max(len(first_pass_surfaces), 1)
        ),
        "candidates_per_1000_jobs": (
            len(first_pass_surfaces) / len(samples) * 1000.0
        ),
        "new_candidate_duplicate_rate": (
            1.0 - unique_new / max(len(first_pass_new), 1)
        ),
        "inference_seconds_per_call": inference_seconds / calls,
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated(0)) / (1024 ** 2)
        ),
        "failure_rate": failures / calls,
        "retry_rate": retries / calls,
    }
    passed = (
        metrics["json_schema_valid_rate"]
            >= THRESHOLDS["json_schema_valid_rate"]
        and metrics["direct_span_valid_rate"]
            >= THRESHOLDS["direct_span_valid_rate"]
        and metrics["hallucinated_surface_rate"]
            <= THRESHOLDS["hallucinated_surface_rate_max"]
        and metrics["failure_rate"]
            <= THRESHOLDS["failure_rate_max"]
        and (
            phase != "technical"
            or metrics["repeat_exact_rate"]
                >= THRESHOLDS["repeat_exact_rate"]
        )
    )
    model_signature = "|".join([
        str(cfg["model"]["repository"]),
        str(cfg["model"]["revision"]),
        str(cfg["model"]["quantization"]),
    ])
    manifest = {
        "candidate_id": candidate_id,
        "phase": phase,
        "status": "formal_pass" if passed else "failed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(samples),
        "repeats": repeats,
        "gpu_name": gpu_name,
        "model_signature": model_signature,
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "sample_sha256": _sha(sample_file),
        "prompt_sha256": _sha(prompt_path),
        "schema_sha256": _sha(schema_path),
        "raw_output_sha256": _sha(raw_path),
        "config_path": str(cfg_path),
        "config_sha256": _sha(cfg_path),
        "model_config": cfg,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Qwen 技术基准",
        "",
        f"- candidate_id: {candidate_id}",
        f"- phase: {phase}",
        f"- status: {manifest['status']}",
        f"- samples: {len(samples)}",
        f"- GPU: {gpu_name}",
        "",
        "## 指标",
    ]
    lines.extend(f"- {k}: {v}" for k, v in metrics.items())
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(2)
    return manifest_path


def _comparable_config(cfg: dict) -> dict:
    return {
        "model": {
            k: cfg["model"].get(k)
            for k in (
                "repository", "revision", "quantization",
                "tokenizer_version", "tokenizer_path",
            )
        },
        "runtime": {
            k: cfg["runtime"].get(k)
            for k in (
                "framework", "framework_version",
                "cuda_version", "pytorch_version",
            )
        },
        "generation": cfg["generation"],
        "text": cfg["text"],
        "protocol": cfg["protocol"],
    }


def select(
    manifests: list[Path],
    selected_candidate: str,
) -> Path:
    if len(manifests) < 2:
        raise ValueError("§8.2 至少需要两个候选 technical manifest")
    payloads = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in manifests
    ]
    for p, m in zip(manifests, payloads):
        if m.get("phase") != "technical":
            raise ValueError(f"{p} 不是 technical benchmark")
        if m.get("status") != "formal_pass":
            raise ValueError(f"{p} 未通过 technical benchmark")
    signatures = {str(m.get("model_signature", "")) for m in payloads}
    if len(signatures) < 2:
        raise ValueError(
            "§8.2 要求至少两个不同模型规模或量化版本；当前模型签名不足 2"
        )
    ids = [str(m.get("candidate_id", "")) for m in payloads]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id 重复")
    if selected_candidate not in ids:
        raise ValueError("selected_candidate 不在候选 manifest 中")

    selected = payloads[ids.index(selected_candidate)]
    production, prod_path = _load_cfg(None)
    if _comparable_config(production) != _comparable_config(
        selected["model_config"]
    ):
        raise ValueError(
            "选中候选的实际 benchmark 配置与 config/model_config_v1.yaml 不一致"
        )

    out = get_project_paths().report_dir / "model_selection_manifest_v1.json"
    manifest = {
        "status": "formal_pass",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_candidate": selected_candidate,
        "selected_model_signature": selected["model_signature"],
        "production_config_sha256": _sha(prod_path),
        "candidate_manifests": [
            {
                "path": str(p),
                "sha256": _sha(p),
                "candidate_id": m["candidate_id"],
                "model_signature": m["model_signature"],
                "metrics": m["metrics"],
            }
            for p, m in zip(manifests, payloads)
        ],
    }
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = get_project_paths().report_dir / "model_benchmark_report.md"
    lines = [
        "# Qwen 模型候选技术基准与生产选择",
        "",
        f"- selected_candidate: {selected_candidate}",
        f"- selected_model_signature: {selected['model_signature']}",
        f"- production_config_sha256: {manifest['production_config_sha256']}",
        "",
        "## 候选比较",
        "",
    ]
    for item in manifest["candidate_manifests"]:
        lines += [
            f"### {item['candidate_id']}",
            f"- model_signature: {item['model_signature']}",
            *[
                f"- {k}: {v}"
                for k, v in item["metrics"].items()
            ],
            "",
        ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="指南 §8 Qwen benchmark/selection")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sample-file", type=Path, required=True)
    r.add_argument("--phase", choices=["technical", "prerun"], required=True)
    r.add_argument("--candidate-id", required=True)
    r.add_argument("--config-file", type=Path)
    s = sub.add_parser("select")
    s.add_argument("--manifests", type=Path, nargs="+", required=True)
    s.add_argument("--selected-candidate", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(
            args.sample_file, args.phase,
            args.candidate_id, args.config_file,
        )
    else:
        select(args.manifests, args.selected_candidate)


if __name__ == "__main__":
    main()
