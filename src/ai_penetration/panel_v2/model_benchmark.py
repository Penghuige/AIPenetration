"""指南 §8：Qwen 正式技术基准与 1 万条预运行。

本脚本不参与指标公式。只有真实调用本机固定模型并达到指南门槛后，
才写 formal_pass manifest，供 v2i 发布前置门读取。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from config.paths import get_project_paths, load_config_yaml
from src.model_platform.llm import create_llm_client, extract_json_from_response

THRESHOLDS = {
    "json_valid_rate": 0.995,
    "span_valid_rate": 0.99,
    "hallucinated_surface_rate_max": 0.01,
    "failure_rate_max": 0.01,
    "repeat_exact_rate": 0.95,
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_fixed_config() -> dict:
    cfg = load_config_yaml("model_config_v1.yaml")
    required = [
        ("model", "repository"), ("model", "revision"),
        ("model", "quantization"), ("model", "tokenizer_version"),
        ("runtime", "framework"), ("runtime", "framework_version"),
        ("runtime", "cuda_version"), ("runtime", "pytorch_version"),
    ]
    bad = []
    for sec, key in required:
        value = str(cfg.get(sec, {}).get(key, "")).strip()
        if not value or value == "TO_BE_CONFIRMED":
            bad.append(f"{sec}.{key}")
    if bad:
        raise RuntimeError(
            "正式模型环境尚未冻结，先填写 config/model_config_v1.yaml: "
            + ", ".join(bad)
        )
    return cfg


def _load_samples(path: Path, minimum: int) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if not str(row.get("job_id", "")) or not str(row.get("text", "")):
            raise ValueError("benchmark sample 每行必须有 job_id/text")
    if len(rows) < minimum:
        raise ValueError(f"样本不足: {len(rows)} < {minimum}")
    return rows


def _validate(parsed, expected_job_id: str, text: str) -> tuple[bool, int, int, int, tuple]:
    if not isinstance(parsed, dict):
        return False, 0, 0, 0, ()
    if str(parsed.get("job_id")) != expected_job_id:
        return False, 0, 0, 0, ()
    skills = parsed.get("skills")
    if not isinstance(skills, list):
        return False, 0, 0, 0, ()
    span_ok = hallucinated = total = 0
    canonical = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        total += 1
        surface = str(skill.get("surface", ""))
        try:
            start, end = int(skill["start"]), int(skill["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start < end <= len(text) and text[start:end] == surface:
            span_ok += 1
        if not surface or surface not in text:
            hallucinated += 1
        canonical.append((
            surface,
            str(skill.get("canonical_suggestion", "")),
            str(skill.get("skill_type", "")),
        ))
    return True, span_ok, total, hallucinated, tuple(sorted(set(canonical)))


def run(sample_file: Path, phase: str) -> Path:
    cfg = _require_fixed_config()
    minimum = 1000 if phase == "technical" else 10000
    repeats = 2 if phase == "technical" else 1
    samples = _load_samples(sample_file, minimum)[:minimum]
    paths = get_project_paths()
    prompt_path = paths.config_dir / "skill_extraction_prompt_v1.md"
    schema_path = paths.config_dir / "skill_extraction_schema_v1.json"
    system = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    client = create_llm_client()

    out_dir = paths.output_dir / "model_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{phase}_raw.jsonl"

    parsed_ok = span_ok = spans = hallucinated = failures = 0
    repeated: dict[str, list[tuple]] = {}
    with raw_path.open("w", encoding="utf-8") as fh:
        for rep in range(repeats):
            for row in samples:
                jid, text = str(row["job_id"]), str(row["text"])
                user = json.dumps({"job_id": jid, "text": text}, ensure_ascii=False)
                try:
                    raw = client.complete_text(
                        system_prompt=system,
                        user_prompt=user,
                        temperature=0.0,
                        max_output_tokens=int(cfg["generation"]["max_output_tokens"]),
                        extra_payload={
                            "seed": int(cfg["generation"]["seed"]),
                            "chat_template_kwargs": {"enable_thinking": False},
                            "guided_json": schema,
                        },
                    )
                    parsed = extract_json_from_response(raw)
                    valid, sok, stotal, hall, skillset = _validate(
                        parsed, jid, text
                    )
                    parsed_ok += int(valid)
                    span_ok += sok
                    spans += stotal
                    hallucinated += hall
                    repeated.setdefault(jid, []).append(skillset)
                    rec = {"job_id": jid, "repeat": rep, "raw": raw}
                except Exception as exc:
                    failures += 1
                    rec = {
                        "job_id": jid, "repeat": rep,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    calls = len(samples) * repeats
    exact = 1.0
    if repeats > 1:
        exact = sum(
            1 for vals in repeated.values()
            if len(vals) == repeats and len(set(vals)) == 1
        ) / len(samples)
    metrics = {
        "json_valid_rate": parsed_ok / calls,
        "span_valid_rate": span_ok / max(spans, 1),
        "hallucinated_surface_rate": hallucinated / max(spans, 1),
        "failure_rate": failures / calls,
        "repeat_exact_rate": exact,
    }
    passed = (
        metrics["json_valid_rate"] >= THRESHOLDS["json_valid_rate"]
        and metrics["span_valid_rate"] >= THRESHOLDS["span_valid_rate"]
        and metrics["hallucinated_surface_rate"]
            <= THRESHOLDS["hallucinated_surface_rate_max"]
        and metrics["failure_rate"] <= THRESHOLDS["failure_rate_max"]
        and (phase != "technical"
             or metrics["repeat_exact_rate"] >= THRESHOLDS["repeat_exact_rate"])
    )
    manifest = {
        "phase": phase,
        "status": "formal_pass" if passed else "failed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(samples),
        "repeats": repeats,
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "sample_sha256": _sha(sample_file),
        "prompt_sha256": _sha(prompt_path),
        "schema_sha256": _sha(schema_path),
        "raw_output_sha256": _sha(raw_path),
        "model_config": cfg,
    }
    manifest_path = paths.report_dir / f"model_benchmark_{phase}_manifest_v1.json"
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = paths.report_dir / f"model_benchmark_{phase}_report.md"
    report.write_text(
        "# Qwen 技术基准\n\n"
        + f"- phase: {phase}\n- status: {manifest['status']}\n"
        + f"- samples: {len(samples)}\n"
        + "\n".join(f"- {k}: {v:.4%}" for k, v in metrics.items())
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit(2)
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(description="指南 §8 Qwen 正式基准")
    ap.add_argument("--sample-file", type=Path, required=True)
    ap.add_argument("--phase", choices=["technical", "prerun"], required=True)
    args = ap.parse_args()
    run(args.sample_file, args.phase)


if __name__ == "__main__":
    main()
