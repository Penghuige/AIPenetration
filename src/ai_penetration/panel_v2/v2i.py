"""v2i：原始交接合规重跑代。

修复范围只落实原始交接已经明确的实现约束，不改变主计算公式：
- 正式技能空间 = A/B/C，D 仅候选；
- 多义激活别名按 primary_skill_id 解析；
- B/C 同义词先映射已有概念，新概念使用稳定 UUIDv5；
- 正式匹配 longest-match，并保存 surface/start/end 证据；
- pass2 只扫描 dedup 选中的 canonical 描述；
- 主岗位标识仍为 main + annual + raw + strictly > 0.05。

历史 v2h 保留不覆盖。新流程：
  lexicon_llm t1/t2/merge
  -> scan --out-tag _handoff_scan
  -> v2i

产物：output/release/panel_v2i/（正式发布件 + metadata + run_manifest.json）。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.paths import get_project_paths

from ..common import setup_logging
from . import quality, scoring
from .counts import compute_counts
from .anchors import anchor_dictionary_rows
from .governance import (
    materialize_formal_dictionary,
    materialize_source_skill_records,
    normalize_term,
)
from .lexicon import _load_atier_alias_records
from .export_release import _meta
from .relevance import compute_relevance, decode_skill_ids
from .reproducibility import sha256_file, write_run_manifest

logger = logging.getLogger("ai_penetration.panel_v2.v2i")


LEX_VERSION = "bilingual_a_frozen_v1.1+governed_v1.5"


def require_handoff_manifests(paths) -> list[Path]:
    """正式 v2i 发布前必须有真实执行产生的上游合规凭证。"""
    required = [
        paths.output_dir / "data_audit" / "source_db_manifest_v1.json",
        paths.report_dir / "data_audit" / "data_audit_manifest_v1.json",
        paths.output_dir / "dictionary" / "external_translation_completion_manifest_v1.json",
        paths.report_dir / "model_benchmark_technical_manifest_v1.json",
        paths.report_dir / "model_benchmark_prerun_manifest_v1.json",
        paths.output_dir / "dictionary" / "formal_discovery_manifest_v1.json",
        paths.output_dir / "llm_review" / "formal_discovery_v1" / "extraction_manifest.json",
        paths.output_dir / "dictionary" / "formal_discovery_review_manifest_v1.json",
        paths.output_dir / "dictionary" / "formal_discovery_review_manifest_v1.json",
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"缺少原始交接必需 manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = (
            "complete"
            if path.name in {
                "formal_discovery_review_manifest_v1.json",
                "formal_discovery_review_manifest_v1.json",
            }
            else "formal_pass"
        )
        if payload.get("status") != expected:
            raise RuntimeError(
                f"上游 manifest 未通过: {path.name} status={payload.get('status')!r}"
            )
    return required


def filter_longs(rel2: Path, codes: np.ndarray,
                 grade_by_sid: dict[str, str]) -> None:
    """按最终 B/C/D 分级过滤并补齐指南 §11.3 的正式长表字段。"""
    target = rel2 / "job_skill_long.parquet"
    tmp = rel2 / "_job_skill_long_filtered.parquet"
    src = pq.ParquetFile(target)
    required = {
        "job_id", "year", "skill_code", "skill_id", "surface_form",
        "match_start", "match_end", "mention_count", "match_method",
        "ambiguity_flag", "span_verified",
    }
    missing = required - set(src.schema_arrow.names)
    if missing:
        src.close()
        raise RuntimeError(
            "pass2 仍是旧版无证据长表，必须按 handoff-compliant scan 重扫: "
            + ", ".join(sorted(missing))
        )
    keep = drop = 0
    writer = None
    try:
        for rb in src.iter_batches(batch_size=2_000_000):
            t = pa.Table.from_batches([rb])
            sc = t["skill_code"].to_numpy()
            m = ~np.isin(sc, codes)
            keep += int(m.sum())
            drop += int((~m).sum())
            t = t.filter(pa.array(m))
            if len(t) == 0:
                continue
            names = [
                "start" if n == "match_start" else
                "end" if n == "match_end" else n
                for n in t.column_names
            ]
            t = t.rename_columns(names)
            sids = t["skill_id"].to_pylist()
            tiers = [grade_by_sid.get(str(sid), "A") for sid in sids]
            t = t.append_column("confidence_tier", pa.array(tiers, pa.string()))
            t = t.append_column(
                "dictionary_version",
                pa.array([LEX_VERSION] * len(t), pa.string()),
            )
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
            writer.write_table(t)
    finally:
        src.close()
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("D 级过滤后 job_skill_long 为空，拒绝发布")
    import gc
    import time
    gc.collect()
    for attempt in range(5):
        try:
            target.unlink(missing_ok=True)
            tmp.rename(target)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(3)
    logger.info("longs 过滤/证据补齐: 保留 %d，剔除 D 级 %d", keep, drop)

def main() -> None:
    ap = argparse.ArgumentParser(description="v2i 原始交接合规全链重算")
    ap.add_argument("--run-id", default="20260919_v2i")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-release", action="store_true")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        raise SystemExit("--run-id 仅允许字母数字、点、下划线、连字符")
    paths = get_project_paths()
    setup_logging(paths.log_dir / "panel_v2_v2i.log")
    t0 = datetime.now()
    handoff_manifests = require_handoff_manifests(paths)
    release_root = paths.output_dir / "release"
    rel_src = release_root / "panel_v2"
    frozen_concepts = (
        paths.output_dir / "dictionary"
        / "skill_concept_bilingual_a_frozen_v1.1.csv"
    )
    frozen_aliases = (
        paths.output_dir / "dictionary"
        / "skill_alias_active_bilingual_a_frozen_v1.1.csv"
    )
    rel2 = release_root / "panel_v2_handoff_scan"  # 新扫描输入
    target = release_root / "panel_v2i"
    rel3 = release_root / f".panel_v2i_{args.run_id}.tmp"

    scan_inputs = ("job_anchor_flag.parquet", "job_skill_long.parquet",
                   "job_firm.parquet", "job_text_clean.parquet")
    preflight_files = [rel2 / f for f in scan_inputs]
    preflight_files += [
        frozen_concepts,
        frozen_aliases,
        paths.output_dir / "dictionary" / "skill_governed_ABCD_v4.csv",
        paths.output_dir / "panel_v2" / "pass2_handoff_scan" / "skill_vocab.json",
    ]
    missing_preflight = [str(p) for p in preflight_files if not p.exists()]
    if missing_preflight:
        raise RuntimeError(
            "v2i 前置产物缺失:\n" + "\n".join(missing_preflight)
        )
    if args.dry_run:
        print("v2i dry-run 前置检查通过；未写任何发布文件")
        return

    if target.exists() and not args.force_release:
        raise SystemExit(
            f"正式发布目录已存在: {target}；默认拒绝覆盖。"
            "如确需替换请显式 --force-release"
        )
    if rel3.exists() and not args.resume:
        raise SystemExit(
            f"staging 已存在: {rel3}；使用 --resume 继续，或人工清理后重跑"
        )
    rel3.mkdir(parents=True, exist_ok=True)
    if (target.exists() and (target / "quality_stats.json").exists()
            and not (rel3 / "quality_stats.json").exists()):
        shutil.copy2(target / "quality_stats.json", rel3 / "quality_stats.json")

    # 1) 输入件从 handoff scan 移入 staging
    for f in scan_inputs:
        shutil.copy2(rel2 / f, rel3 / f)

    # 2) v1.3 D 级过滤
    gcsv = paths.output_dir / "dictionary" / "skill_governed_ABCD_v4.csv"
    governance_manifest = paths.output_dir / "dictionary" / "formal_discovery_review_manifest_v1.json"
    if not governance_manifest.exists():
        raise RuntimeError("缺少 formal discovery review provenance manifest")
    grade = pd.read_csv(gcsv, encoding="utf-8-sig")
    discovery_manifest = json.loads(
        (paths.output_dir / "dictionary" / "formal_discovery_manifest_v1.json")
        .read_text(encoding="utf-8")
    )
    if discovery_manifest.get("governance_sha256") != sha256_file(gcsv):
        raise RuntimeError(
            "formal discovery manifest 绑定的治理表与当前 "
            "skill_governed_ABCD_v4.csv 不一致"
        )
    pass2b = paths.output_dir / "panel_v2" / "pass2_handoff_scan"
    vocab_path = pass2b / "skill_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    formal_grade = grade[grade.final_grade.isin(["A", "B", "C"])].copy()
    if formal_grade.final_skill_id.isna().any():
        raise RuntimeError("治理表 A/B/C 存在空 final_skill_id")
    grade_by_sid = dict(zip(
        formal_grade.final_skill_id.astype(str),
        formal_grade.final_grade.astype(str),
    ))
    # handoff-compliant scan 已只加载 A/B/C；此处不再“先扫 D 后删除”。
    filter_longs(rel3, np.array([], np.int32), grade_by_sid)

    # 3) counts（新 flag）→ relevance
    counts = compute_counts(rel3)
    counts.to_parquet(rel3 / "skill_ai_counts.parquet", index=False)
    logger.info("skill_ai_counts: %d 行", len(counts))
    n_vocab = max(vocab.values()) + 1
    tier_map = np.full(n_vocab, "A", dtype=object)
    for sid, code in vocab.items():
        tier_map[int(code)] = grade_by_sid.get(str(sid), "A")
    rel_df = decode_skill_ids(
        compute_relevance(counts, tier_map), vocab_path
    )
    rel_df.to_parquet(rel3 / "skill_ai_relevance.parquet", index=False)
    logger.info("skill_ai_relevance: %d 行", len(rel_df))

    # 4) scoring
    scoring.run(rel3)

    # 5) 装配：§18 正式词典必须与实际 matcher 同一 A/B/C 概念集合。
    base_concepts = pd.read_csv(frozen_concepts, encoding="utf-8-sig", dtype=str)
    pending = base_concepts.translation_status.fillna("").astype(str).isin(
        ["", "pending_codex_zh"]
    )
    if pending.any():
        raise RuntimeError(
            f"A级基础词典仍有 {int(pending.sum())} 个未完成中文化概念，拒绝 v2i"
        )
    base_aliases = pd.read_csv(frozen_aliases, encoding="utf-8-sig", dtype=str)
    resolved = {
        normalize_term(r.alias): r.skill_id
        for r in _load_atier_alias_records()
    }
    base_aliases["_norm"] = base_aliases.alias.astype(str).map(normalize_term)
    keep_base = [
        resolved.get(key) == str(sid)
        for key, sid in zip(base_aliases["_norm"], base_aliases.skill_id)
    ]
    base_aliases = base_aliases.loc[keep_base].drop(columns=["_norm"]).reset_index(drop=True)
    concepts, aliases, d_candidates = materialize_formal_dictionary(
        base_concepts, base_aliases, grade, LEX_VERSION
    )
    source_records = materialize_source_skill_records(
        base_concepts, LEX_VERSION
    )
    concepts.to_parquet(rel3 / "skill_concept_v1.parquet", index=False)
    aliases.to_parquet(rel3 / "skill_alias_v1.parquet", index=False)
    d_candidates.to_parquet(rel3 / "skill_candidate_d_v1.parquet", index=False)
    source_records.to_parquet(
        rel3 / "source_skill_record_v1.parquet", index=False
    )
    pd.DataFrame(anchor_dictionary_rows()).to_csv(
        rel3 / "ai_anchor_dictionary_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shutil.copy2(gcsv, rel3 / gcsv.name)
    shutil.copy2(governance_manifest, rel3 / governance_manifest.name)

    # 6) 正式词典、长表、得分全部到位后再跑 §17 质量门。
    quality.run(rel3)

    single = rel3 / "job_ai_score.parquet"
    files = sorted((rel3 / "job_ai_score").glob("*.parquet"))
    if not files:
        raise RuntimeError("job_ai_score 分区为空，拒绝装配 v2i release")
    with pq.ParquetWriter(single, pq.read_schema(files[0]),
                          compression="zstd") as w:
        for f in files:
            for rb in pq.ParquetFile(f).iter_batches(batch_size=2_000_000):
                w.write_table(pa.Table.from_batches([rb]))

    specs = [
        ("skill_concept_v1.parquet", "skill_id", "governance.py(A/B/C formal)"),
        ("skill_alias_v1.parquet", "alias_id", "governance.py(A/B/C aliases)"),
        ("skill_candidate_d_v1.parquet", "term", "governance.py(D only)"),
        ("source_skill_record_v1.parquet",
         "source_name+source_skill_id+internal_skill_id",
         "governance.py(external source crosswalk)"),
        ("skill_governed_ABCD_v4.csv", "term",
         "panel_v2/discovery_review.py(v3 legacy + formal discovery)"),
        ("formal_discovery_review_manifest_v1.json", "-",
         "panel_v2/discovery_review.py(provenance)"),
        ("ai_anchor_dictionary_v1.csv", "anchor_version+keyword",
         "panel_v2/anchors.py(规则 20260909_b)"),
        ("job_anchor_flag.parquet", "job_id", "panel_v2/scan.py(canonical+terms)"),
        ("job_skill_long.parquet", "job_id+skill_id",
         "panel_v2/scan.py(longest-match+span evidence)"),
        ("job_firm.parquet", "job_id", "panel_v2/scan.py"),
        ("job_text_clean.parquet", "job_id",
         "panel_v2/scan.py(raw/clean/match/full stable id)"),
        ("job_text_clean.parquet", "job_id",
         "panel_v2/scan.py(raw/clean/match §6.1)"),
        ("skill_ai_counts.parquet", "skill+ver+win+year", "panel_v2/counts.py"),
        ("skill_ai_relevance.parquet", "skill+ver+win+year", "panel_v2/relevance.py"),
        ("job_ai_score.parquet", "job+ver+win+stype", "panel_v2/scoring.py"),
        ("job_ai_classification.parquet", "job_id", "panel_v2/scoring.py"),
        ("job_ai_score_loo.parquet", "job_id", "panel_v2/scoring.py(§14.3)"),
        ("quality_control_report.md", "-", "panel_v2/quality.py"),
    ]
    missing = [name for name, _, _ in specs if not (rel3 / name).exists()]
    if missing:
        raise RuntimeError(
            "指南 §18 必需发布件缺失，拒绝生成正式 v2i release: "
            + ", ".join(missing)
        )

    for name, pk, src in specs:
        _meta(
            rel3 / name, args.run_id, pk, src,
            anchor_version="main,cn_paper,babina",
            dictionary_version=LEX_VERSION,
        )

    # 7) 指南 §4.2：把本次正式运行的代码/配置、输入、输出绑定成一个总账。
    manifest_inputs = [rel2 / f for f in scan_inputs]
    manifest_inputs += [frozen_concepts, frozen_aliases, gcsv,
                        governance_manifest, vocab_path]
    manifest_inputs += handoff_manifests
    manifest_outputs = [rel3 / name for name, _, _ in specs]
    # staging manifest 先验证所有文件可哈希/计数。
    write_run_manifest(
        rel3,
        run_id=args.run_id,
        started_at=t0,
        input_paths=manifest_inputs,
        output_paths=manifest_outputs,
        dictionary_version=LEX_VERSION,
        anchor_version="main,cn_paper,babina@20260909_b",
    )

    backup = None
    if target.exists():
        backup = release_root / (
            "panel_v2i.previous_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        target.replace(backup)
    try:
        rel3.replace(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise

    final_outputs = [target / name for name, _, _ in specs]
    manifest = write_run_manifest(
        target,
        run_id=args.run_id,
        started_at=t0,
        input_paths=manifest_inputs,
        output_paths=final_outputs,
        dictionary_version=LEX_VERSION,
        anchor_version="main,cn_paper,babina@20260909_b",
    )
    (target / "RELEASE_COMPLETE").write_text(
        args.run_id + "\n", encoding="utf-8"
    )
    print(f"panel_v2i 发布: {len(specs)} 件 + metadata + {manifest.name} @ {target}，用时 "
          f"{(datetime.now() - t0).total_seconds() / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
