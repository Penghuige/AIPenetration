"""正式 discovery 候选聚合、Top-10 概念检索与 A/B/C/D 入级。

输入 discovery_extract.py 的 mentions.parquet。先按全局 DISTINCT(text_hash)
计算 df_unique_description；精确别名唯一命中直接映射。其余候选使用确定性
字符 bigram 检索 Top-10 概念，再由固定 Qwen 只返回 MATCH_EXISTING /
NEW_CONCEPT / AMBIGUOUS / NOT_SKILL。最终产出候选审计表与 v4 治理表。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd

from config.paths import get_project_paths, load_config_yaml
from src.model_platform.llm import create_llm_client, extract_json_from_response
from .governance import stable_bc_skill_id

REVIEW_VERSION = "formal_discovery_review_v2_full_corpus"
RETRIEVAL_VERSION = "char_bigram_jaccard_top10_v1"
ALLOWED_TYPES = {
    "programming_language", "method_algorithm", "software_tool",
    "platform_framework_library", "data_database", "hardware_equipment",
    "domain_knowledge", "business_management", "general_work_skill",
    "soft_skill", "other_skill",
}


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).lower().strip()


def _bigrams(text: str) -> set[str]:
    x = _norm(text)
    if len(x) < 2:
        return {x} if x else set()
    return {x[i:i + 2] for i in range(len(x) - 1)}


def aggregate_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "job_id", "year", "text_hash", "anchor_main", "discovery_round",
        "surface", "canonical_suggestion", "skill_type", "evidence", "span_valid",
    }
    missing = required - set(mentions.columns)
    if missing:
        raise ValueError("mentions 缺列: " + ", ".join(sorted(missing)))
    if not mentions.span_valid.astype(bool).all():
        raise ValueError("mentions 含无效跨度")

    rows = []
    for term, g in mentions.groupby(mentions.surface.astype(str).map(_norm)):
        texts = g.drop_duplicates("text_hash")
        df = int(texts.text_hash.nunique())
        anchor_df = int(texts.loc[texts.anchor_main == 1, "text_hash"].nunique())
        suggestions = [
            str(x) for x in g.canonical_suggestion.dropna().astype(str)
            if str(x).strip()
        ]
        types = [
            str(x) for x in g.skill_type.dropna().astype(str)
            if str(x) in ALLOWED_TYPES
        ]
        examples = []
        for rec in g.head(5).itertuples():
            examples.append({
                "surface": str(rec.surface),
                "evidence": str(rec.evidence),
                "job_id": str(rec.job_id),
            })
        rows.append({
            "term": term,
            "df_unique_description": df,
            "candidate_anchor_cooc": anchor_df / df if df else 0.0,
            "first_year": int(g.year.min()),
            "source_round": int(g.discovery_round.min()),
            "evidence_count": int(len(g)),
            "span_valid": True,
            "canonical_suggestion": (
                pd.Series(suggestions).value_counts().index[0]
                if suggestions else term
            ),
            "skill_type_suggestion": (
                pd.Series(types).value_counts().index[0]
                if types else "other_skill"
            ),
            "evidence_examples": json.dumps(examples, ensure_ascii=False),
        })
    return pd.DataFrame(rows).sort_values(
        ["df_unique_description", "term"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def _load_registry(paths) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, str]]:
    concepts = pd.read_csv(
        paths.output_dir / "dictionary"
        / "skill_concept_bilingual_a_frozen_v1.1.csv",
        encoding="utf-8-sig",
        dtype=str,
    )
    aliases = pd.read_csv(
        paths.output_dir / "dictionary"
        / "skill_alias_active_bilingual_a_frozen_v1.1.csv",
        encoding="utf-8-sig",
        dtype=str,
    )
    base = concepts.set_index("skill_id", drop=False)
    records = []
    exact: dict[str, set[str]] = {}
    tier: dict[str, str] = {}
    for row in aliases.itertuples():
        sid = str(row.skill_id)
        key = _norm(row.alias)
        exact.setdefault(key, set()).add(sid)
        tier[sid] = "A"
        c = base.loc[sid] if sid in base.index else None
        records.append({
            "alias": key,
            "skill_id": sid,
            "tier": "A",
            "canonical_zh": "" if c is None else str(c.get("canonical_zh", "") or ""),
            "canonical_en": "" if c is None else str(c.get("canonical_en", "") or ""),
            "definition": "" if c is None else str(
                c.get("definition_en", c.get("description_en", "")) or ""
            ),
            "category": "" if c is None else str(c.get("skill_category", "") or ""),
        })

    v3_path = paths.output_dir / "dictionary" / "skill_legacy_graded_BCD_v3.csv"
    v3 = pd.read_csv(v3_path, encoding="utf-8-sig")
    for row in v3[v3.final_grade.isin(["A", "B", "C"])].itertuples():
        sid = str(row.final_skill_id)
        key = _norm(row.term)
        exact.setdefault(key, set()).add(sid)
        old = tier.get(sid)
        rtier = str(row.final_grade)
        tier[sid] = old or rtier
        records.append({
            "alias": key,
            "skill_id": sid,
            "tier": tier[sid],
            "canonical_zh": str(row.term),
            "canonical_en": "",
            "definition": "",
            "category": str(getattr(row, "t2_cat", "") or ""),
        })
    return pd.DataFrame(records), exact, tier


def _registry_hash(registry: pd.DataFrame) -> str:
    cols = [
        "alias", "skill_id", "tier", "canonical_zh",
        "canonical_en", "definition", "category",
    ]
    stable = registry[cols].fillna("").astype(str).sort_values(
        ["skill_id", "alias", "canonical_zh"], kind="stable"
    )
    payload = stable.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Retriever:
    def __init__(self, registry: pd.DataFrame):
        self.registry = registry.reset_index(drop=True)
        self.alias_bigrams = [_bigrams(x) for x in self.registry.alias]
        self.index: dict[str, list[int]] = {}
        for i, bgs in enumerate(self.alias_bigrams):
            for bg in bgs:
                self.index.setdefault(bg, []).append(i)

    def top10(self, term: str) -> list[dict]:
        qb = _bigrams(term)
        candidate_idx: set[int] = set()
        for bg in qb:
            candidate_idx.update(self.index.get(bg, ()))
        best: dict[str, tuple[float, int]] = {}
        for i in candidate_idx:
            rb = self.alias_bigrams[i]
            union = len(qb | rb)
            sim = len(qb & rb) / union if union else 0.0
            sid = str(self.registry.iloc[i].skill_id)
            prev = best.get(sid)
            if prev is None or (sim, -i) > (prev[0], -prev[1]):
                best[sid] = (sim, i)
        ranked = sorted(
            best.items(),
            key=lambda kv: (
                -kv[1][0],
                str(self.registry.iloc[kv[1][1]].canonical_zh),
                kv[0],
            ),
        )[:10]
        out = []
        for sid, (sim, i) in ranked:
            r = self.registry.iloc[i]
            out.append({
                "skill_id": sid,
                "tier": str(r.tier),
                "canonical_zh": str(r.canonical_zh),
                "canonical_en": str(r.canonical_en),
                "definition": str(r.definition)[:300],
                "category": str(r.category),
                "similarity": round(float(sim), 6),
            })
        return out


SYSTEM = (
    "你是技能概念归一化评审。给定招聘语料候选技能及至多10个现有概念。"
    "只能判断同一概念，不可把上下位、主题相关或共同使用关系合并。"
    "decision 只能为 MATCH_EXISTING、NEW_CONCEPT、AMBIGUOUS、NOT_SKILL。"
    "MATCH_EXISTING 时 index 必须为候选列表下标；其余 index=null。"
    "skill_type 必须来自项目受控枚举；new_tech 表示是否属于新兴软件/模型/框架/方法。"
    "只输出一个 JSON 对象。"
)


def review(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = get_project_paths()
    cfg = load_config_yaml("model_config_v1.yaml")
    revision = str(cfg.get("model", {}).get("revision", "")).strip()
    if revision in {"", "TO_BE_CONFIRMED"}:
        raise RuntimeError("model_config_v1 尚未冻结 revision")
    registry, exact, tiers = _load_registry(paths)
    registry_sha = _registry_hash(registry)
    retriever = Retriever(registry)
    client = create_llm_client()
    prompt_sha = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()
    cache_path = (
        paths.output_dir / "llm_review" / "formal_discovery_v1"
        / "candidate_review.jsonl"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = str(rec.get("cache_key", ""))
            if key:
                cached[key] = rec

    results = []
    with cache_path.open("a", encoding="utf-8") as fh:
        for row in candidates.itertuples(index=False):
            term = _norm(row.term)
            exact_ids = exact.get(term, set())
            if len(exact_ids) == 1:
                sid = next(iter(exact_ids))
                rec = {
                    "term": term, "decision": "MATCH_EXISTING_EXACT",
                    "final_skill_id": sid, "existing_tier": tiers.get(sid, "A"),
                    "skill_type": str(row.skill_type_suggestion),
                    "new_tech": False, "ambiguous": False, "top10": [],
                    "raw_output": "", "cache_key": "",
                }
            elif len(exact_ids) > 1:
                rec = {
                    "term": term, "decision": "AMBIGUOUS_EXACT",
                    "final_skill_id": "", "existing_tier": "",
                    "skill_type": str(row.skill_type_suggestion),
                    "new_tech": False, "ambiguous": True, "top10": [],
                    "raw_output": "", "cache_key": "",
                }
            else:
                top = retriever.top10(term)
                payload = {
                    "term": term,
                    "canonical_suggestion": row.canonical_suggestion,
                    "skill_type_suggestion": row.skill_type_suggestion,
                    "evidence_examples": json.loads(row.evidence_examples),
                    "candidates": top,
                }
                payload_text = json.dumps(
                    payload, ensure_ascii=False, sort_keys=True
                )
                cache_key = hashlib.sha256(
                    "|".join([
                        term, revision, prompt_sha, registry_sha,
                        RETRIEVAL_VERSION,
                        hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
                    ]).encode("utf-8")
                ).hexdigest()
                if cache_key in cached:
                    rec = cached[cache_key]
                else:
                    try:
                        raw = client.complete_text(
                            system_prompt=SYSTEM,
                            user_prompt=payload_text,
                            temperature=float(cfg["generation"]["temperature"]),
                            max_output_tokens=500,
                            extra_payload={
                                "seed": int(cfg["generation"]["seed"]),
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                        )
                        res = extract_json_from_response(raw)
                    except Exception as exc:
                        raise RuntimeError(
                            f"候选语义评审失败 term={term!r}: {exc}"
                        ) from exc
                    if not isinstance(res, dict):
                        raise RuntimeError(
                            f"候选语义评审非对象 term={term!r}"
                        )
                    decision = str(res.get("decision", ""))
                    if decision not in {
                        "MATCH_EXISTING", "NEW_CONCEPT",
                        "AMBIGUOUS", "NOT_SKILL",
                    }:
                        raise RuntimeError(
                            f"未知 decision term={term!r}: {decision!r}"
                        )
                    stype = str(
                        res.get("skill_type", row.skill_type_suggestion)
                    )
                    if stype not in ALLOWED_TYPES:
                        raise RuntimeError(
                            f"未知 skill_type term={term!r}: {stype!r}"
                        )
                    sid = ""
                    existing_tier = ""
                    if decision == "MATCH_EXISTING":
                        try:
                            idx = int(res["index"])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise RuntimeError(
                                f"MATCH_EXISTING 缺 index: {term!r}"
                            ) from exc
                        if idx < 0 or idx >= len(top):
                            raise RuntimeError(
                                f"MATCH_EXISTING index 越界: {term!r}"
                            )
                        sid = str(top[idx]["skill_id"])
                        existing_tier = str(top[idx]["tier"])
                    rec = {
                        "cache_key": cache_key,
                        "term": term,
                        "decision": decision,
                        "final_skill_id": sid,
                        "existing_tier": existing_tier,
                        "skill_type": stype,
                        "new_tech": bool(res.get("new_tech", False)),
                        "ambiguous": bool(
                            res.get("ambiguous", decision == "AMBIGUOUS")
                        ),
                        "top10": top,
                        "raw_output": raw,
                        "registry_sha256": registry_sha,
                        "review_prompt_sha256": prompt_sha,
                        "retrieval_version": RETRIEVAL_VERSION,
                    }
                    fh.write(
                        json.dumps(rec, ensure_ascii=False) + "\n"
                    )
                    fh.flush()
            results.append(rec)

    review_df = pd.DataFrame(results)
    audit = candidates.merge(
        review_df, on="term", how="left", validate="one_to_one"
    )
    grades = []
    final_ids = []
    for row in audit.itertuples():
        decision = str(row.decision)
        df = int(row.df_unique_description)
        if decision.startswith("MATCH_EXISTING"):
            grade = str(row.existing_tier or "A")
            sid = str(row.final_skill_id)
        elif decision in {"AMBIGUOUS", "AMBIGUOUS_EXACT", "NOT_SKILL"}:
            grade, sid = "D", ""
        elif decision == "NEW_CONCEPT":
            if df >= 100:
                grade = "B"
            elif 10 <= df < 100:
                grade = "C"
            elif df >= 5 and (
                float(row.candidate_anchor_cooc) >= 0.50
                or bool(row.new_tech)
            ):
                grade = "C"
            else:
                grade = "D"
            sid = (
                stable_bc_skill_id(str(row.term), int(row.first_year))
                if grade in {"B", "C"} else ""
            )
        else:
            raise RuntimeError(f"不可识别决策: {decision}")
        grades.append(grade)
        final_ids.append(sid)
    audit["final_grade"] = grades
    audit["final_skill_id"] = final_ids
    audit["span_valid"] = True
    return audit, review_df


def prepare_candidates(mentions_path: Path) -> tuple[Path, Path]:
    """先聚合 Qwen 表面形式，供全量 legacy_freq 确定性扫描。"""
    paths = get_project_paths()
    mentions = pd.read_parquet(mentions_path)
    candidates = aggregate_mentions(mentions)
    out_dir = paths.output_dir / "dictionary"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "formal_discovery_candidates_v1.csv"
    terms_path = out_dir / "formal_discovery_terms_v1.txt"
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    terms_path.write_text(
        "\n".join(candidates.term.astype(str)) + "\n",
        encoding="utf-8",
    )
    return candidate_path, terms_path


def write_outputs(
    mentions_path: Path,
    full_freq_path: Path,
) -> tuple[Path, Path]:
    """用全量招聘语料频数重新定级，不使用抽样 mentions 次数替代 §10.3.2。"""
    paths = get_project_paths()
    mentions = pd.read_parquet(mentions_path)
    candidates = aggregate_mentions(mentions)
    candidates = candidates.rename(columns={
        "df_unique_description": "sample_df_unique_description",
        "candidate_anchor_cooc": "sample_candidate_anchor_cooc",
    })

    full = pd.read_csv(full_freq_path, encoding="utf-8-sig")
    required_freq = {
        "match_key", "df_unique_text",
        "main_anchor_unique_text", "candidate_anchor_cooc",
    }
    missing = required_freq - set(full.columns)
    if missing:
        raise ValueError(
            "全量候选频数文件缺列: " + ", ".join(sorted(missing))
        )
    full = full.rename(columns={
        "match_key": "term",
        "df_unique_text": "df_unique_description",
    })
    full["term"] = full.term.astype(str).map(_norm)
    if full.term.duplicated().any():
        raise ValueError("全量候选频数 term 不唯一")
    candidates = candidates.merge(
        full[[
            "term", "df_unique_description",
            "main_anchor_unique_text", "candidate_anchor_cooc",
        ]],
        on="term",
        how="left",
        validate="one_to_one",
    )
    if candidates.df_unique_description.isna().any():
        missing_terms = candidates.loc[
            candidates.df_unique_description.isna(), "term"
        ].astype(str).head(10).tolist()
        raise RuntimeError(
            "候选未经过全量 §10.3.2 扫描: " + ", ".join(missing_terms)
        )

    audit, _review = review(candidates)
    out_dir = paths.output_dir / "dictionary"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "formal_discovery_candidate_audit_v1.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    v3_path = out_dir / "skill_legacy_graded_BCD_v3.csv"
    v3 = pd.read_csv(v3_path, encoding="utf-8-sig")
    known = {_norm(x) for x in v3.term.astype(str)}
    new_rows = []
    for row in audit.itertuples():
        if _norm(row.term) in known:
            continue
        new_rows.append({
            "term": _norm(row.term),
            "source_skill_id": "",
            "final_skill_id": str(row.final_skill_id),
            "df_freq": int(row.df_unique_description),
            "cand_cooc": float(row.candidate_anchor_cooc),
            "first_year": int(row.first_year),
            "taut": 0,
            "v2e_grade": "",
            "t1_cat": "",
            "t2_skill": str(row.decision) != "NOT_SKILL",
            "t2_cat": str(row.skill_type),
            "t2_new_tech": bool(row.new_tech),
            "t2_relation": "",
            "t2_ambig": bool(row.ambiguous),
            "mapping_action": str(row.decision),
            "final_grade": str(row.final_grade),
            "demote_reason": "",
            "source": "formal_discovery_v1",
        })
    base = v3.copy()
    if "source" not in base.columns:
        base["source"] = "legacy_governance_v3"
    v4 = pd.concat(
        [base, pd.DataFrame(new_rows)], ignore_index=True, sort=False
    )
    if v4.term.astype(str).map(_norm).duplicated().any():
        dup = v4.loc[
            v4.term.astype(str).map(_norm).duplicated(), "term"
        ].head().tolist()
        raise RuntimeError(f"v4 治理表 term 重复: {dup}")
    formal = v4[v4.final_grade.isin(["A", "B", "C"])]
    if formal.final_skill_id.isna().any() or (
        formal.final_skill_id.astype(str).str.len() == 0
    ).any():
        raise RuntimeError("v4 A/B/C 存在空 final_skill_id")

    v4_path = out_dir / "skill_governed_ABCD_v4.csv"
    v4.to_csv(v4_path, index=False, encoding="utf-8-sig")
    review_cache = (
        paths.output_dir / "llm_review" / "formal_discovery_v1"
        / "candidate_review.jsonl"
    )
    manifest = {
        "status": "complete",
        "review_version": REVIEW_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "mentions_sha256": hashlib.sha256(
            mentions_path.read_bytes()
        ).hexdigest(),
        "full_freq_sha256": hashlib.sha256(
            full_freq_path.read_bytes()
        ).hexdigest(),
        "candidate_audit_sha256": hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
        "governance_sha256": hashlib.sha256(
            v4_path.read_bytes()
        ).hexdigest(),
        "review_cache_sha256": (
            hashlib.sha256(review_cache.read_bytes()).hexdigest()
            if review_cache.exists() else None
        ),
        "review_prompt_sha256": hashlib.sha256(
            SYSTEM.encode("utf-8")
        ).hexdigest(),
        "n_candidates": len(audit),
        "n_new_terms": len(new_rows),
        "grade_counts": audit.final_grade.value_counts().to_dict(),
    }
    (out_dir / "formal_discovery_review_manifest_v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return audit_path, v4_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="正式候选聚合、全量频数接入与概念治理"
    )
    ap.add_argument("--mentions", type=Path, required=True)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--full-freq", type=Path)
    args = ap.parse_args()
    if args.prepare_only:
        prepare_candidates(args.mentions)
        return
    if args.full_freq is None:
        raise SystemExit(
            "正式分级必须提供 --full-freq；先 --prepare-only 生成 terms，"
            "再用 legacy_freq --terms-file 对全量语料扫描"
        )
    write_outputs(args.mentions, args.full_freq)


if __name__ == "__main__":
    main()
