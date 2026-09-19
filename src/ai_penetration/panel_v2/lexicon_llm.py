"""词表治理 LLM 语义复核（本机 vLLM Qwen3-8B；指南 §10.3.3/§9.3 通道）。

补齐 v2e 披露的两处"确定性代理"：
- T1 非技能停用表（§10.3.1 B 级条件之一"不在非技能停用表中"——指南只点名
  未给内容，本轮建成 v1；作用于新词入级与 A 级敏感性分析，不回删 A 级冻结件）；
- T2 legacy 全量语义复核（is_skill / 类型 / 新词性 / 与 A 级 10 候选映射），
  输出喂 §10.3.1 重分级（vLLM 替代 first_year≥2019 与 cooc 救援的语义判断面）。

通用框架：按字符预算组批（服务 max-model-len=4096）、Qwen3 关思考、
JSONL 断点续跑（按 item id 去重）、解析失败单条重试一轮后标 parse_error。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.lexicon_llm t1
    python -X utf8 -m src.ai_penetration.panel_v2.lexicon_llm t2 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config.paths import get_project_paths

from ..common import setup_logging
from src.model_platform.llm import create_llm_client

logger = logging.getLogger("ai_penetration.panel_v2.lexicon_llm")

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
BUDGET_CHARS = 2600          # 请求字符预算（留 reply+模板余量）
_LOCK = threading.Lock()

SYS_T1 = (
    "你是中文招聘语料技能词典治理员。对每个词条判断其类别：\n"
    "skill=具体可定位的技术/工具/方法/专业能力；\n"
    "soft_trait=软素质/通用能力套话（如 责任心、沟通能力、抗压、团队合作、学习能力）；\n"
    "edu_req=学历/专业/证书要求；exp_req=工作年限/经验要求；benefit=福利；"
    "job_title=岗位名称；company=公司或品牌名；task_fragment=任务片段或口号；"
    "goods_service=商品/服务名而非技能；other。\n"
    "注意：像 ai技术、大数据开发 这类含技术指称的复合词是 skill；裸泛词"
    "（开发、分析、优化、安全、营销 等单独成词且无技术指称）判 task_fragment。"
    '只输出 JSON 数组 [{"i":<int>,"c":"<类别>"}]，条数与输入一致，无其他文字。')

SYS_T2 = (
    "你是技能词典编纂评审。给你一个候选技能词条（招聘语料挖掘），以及基础词典中"
    "最相近的至多 10 个已有词条。判断：\n"
    "is_skill: 是否是明确的技能（技术/工具/方法/模型/工程实践）true/false；\n"
    "cat: skill 时给出类型 method|tool|software|model|framework|data|language|other；\n"
    "new_tech: 是否 2019 年后兴起的新软件/新模型/新框架/新方法 true/false；\n"
    "rel: 与哪个已有词条是同一概念→输出其序号 0-9；都不是新造词→-1；"
    "证据不足→-2(AMBIGUOUS)；非技能→-3；\n"
    "ambiguity: 是否存在明显跨领域同形风险（如裸 ai、ml 缩写、产品名撞名）true/false。\n"
    "只输出 JSON 数组 [{\"i\":<int>,\"s\":true|false,\"c\":\"...\",\"n\":true|false,"
    "\"r\":<int>,\"a\":true|false}]，条数一致，无其他文字。")


def _client():
    return create_llm_client()


def _norm_key(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).lower()


def batch_review(items: list[dict], system: str, build_user, out: Path,
                 workers: int = 3, budget: int = 0) -> int:
    """通用批量评审（断点续跑）。

    Args:
        items: [{"id": int, ...}]，build_user 用整条。
        system: 系统提示。
        build_user: callable(list[item]) -> str（含每条 id 标注）。
        out: JSONL 结果路径（存在则按已完成 id 续跑）。
        workers: 并发（GPU 共享纪律 ≤4）。
        budget: 组批字符预算（0=默认 BUDGET_CHARS；长文本任务传更小值）。

    Returns:
        本次新完成条数。
    """
    done: set[int] = set()
    if out.exists():
        for ln in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(ln)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [it for it in items if it["id"] not in done]
    logger.info("评审 %s: 总 %d，已完成 %d，待跑 %d", out.name, len(items),
                len(done), len(todo))
    # 组批
    budget = budget or BUDGET_CHARS
    batches: list[list[dict]] = []
    cur: list[dict] = []
    n = 0
    for it in todo:
        cost = len(json.dumps(it, ensure_ascii=False))
        if cur and n + cost > budget:
            batches.append(cur)
            cur, n = [], 0
        cur.append(it)
        n += cost
    if cur:
        batches.append(cur)
    fh = out.open("a", encoding="utf-8")
    cnt = [0]

    def run(batch: list[dict]) -> None:
        pending = batch
        for attempt in (1, 2):
            got: dict[int, dict] = {}
            try:
                parsed = _client().complete_json(
                    system_prompt=system, user_prompt=build_user(pending),
                    temperature=0.0, max_output_tokens=1200,
                    extra_payload=NO_THINK)
                if isinstance(parsed, dict):  # 容错单对象
                    parsed = [parsed]
                got = {int(p.get("i", -999)): p for p in parsed
                       if isinstance(p, dict)}
            except (ValueError, RuntimeError) as e:
                logger.warning("批解析失败(attempt %d): %s", attempt, e)
            hit = [it for it in pending if it["id"] in got]
            miss = [it for it in pending if it["id"] not in got]
            with _LOCK:
                for it in hit:
                    fh.write(json.dumps({"id": it["id"], "res": got[it["id"]]},
                                        ensure_ascii=False) + "\n")
                    cnt[0] += 1
                fh.flush()
            if not miss:
                return
            if attempt == 2:  # 两轮皆缺：仅缺口标错（已写条目不重复落）
                with _LOCK:
                    for it in miss:
                        fh.write(json.dumps(
                            {"id": it["id"], "res": {"error": "batch_fail"}},
                            ensure_ascii=False) + "\n")
                        cnt[0] += 1
                    fh.flush()
                return
            pending = miss

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, batches))
    fh.close()
    logger.info("本次完成 %d 条", cnt[0])
    return cnt[0]


# ------------------------------------------------------------------ T1

def candidates_t1() -> list[dict]:
    """T1 候选面：legacy 全键 + A 级高频别名 TOP 8000 + 8 个时代缺词。"""
    import pandas as pd
    paths = get_project_paths()
    dic = paths.output_dir / "dictionary"
    rel = paths.output_dir / "release" / "panel_v2"
    out: list[dict] = []
    lg = pd.read_csv(dic / "legacy_df_freq_v1.csv", encoding="utf-8-sig")
    out += [{"term": t, "tier": "legacy", "id": i}
            for i, t in enumerate(lg.match_key)]
    base = len(out)
    counts = pd.read_parquet(rel / "skill_ai_counts.parquet")
    cm = counts[(counts.anchor_version == "main")
                & (counts.window_type == "pooled")]
    vocab = json.loads((paths.output_dir / "panel_v2" / "pass2" / "skill_vocab.json")
                       .read_text(encoding="utf-8"))
    code2sid = {v: k for k, v in vocab.items()}
    ali = pd.read_parquet(rel / "skill_alias_v1.parquet")
    sid_keys = ali.groupby("skill_id").alias.min()  # 代表别名（确定性）
    hot = cm.nlargest(8000, "n_skill")
    rows = []
    for c, fq in zip(hot.skill_code, hot.n_skill):
        sid = code2sid.get(c)
        if sid is None or str(sid).startswith("legacy:"):
            continue
        key = _norm_key(str(sid_keys.get(sid, "")))
        if key and len(key) >= 2:
            rows.append({"term": key, "tier": "atier",
                         "id": base + len(rows), "freq": int(fq)})
    out += rows
    for j, t in enumerate(["ollama", "dify", "coze", "deepseek", "sora",
                           "mcp", "gpt4", "multimodal", "aigc 内容", "提示词工程"]):
        out.append({"term": t, "tier": "era_missing", "id": 900000 + j})
    return out


def run_t1(workers: int = 3) -> None:
    paths = get_project_paths()
    items = candidates_t1()
    out = paths.output_dir / "llm_review" / "t1_stopword.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    def user(batch: list[dict]) -> str:
        return "\n".join(f'{it["id"]}. {it["term"]}' for it in batch)
    batch_review(items, SYS_T1, user, out, workers=workers)


# ------------------------------------------------------------------ T2

def _bigram_index(aliases: list[str]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for i, a in enumerate(aliases):
        for bg in {_norm_key(a)[k:k + 2] for k in range(len(a) - 1)}:
            if len(bg) == 2:
                idx.setdefault(bg, []).append(i)
    return idx


def candidates_t2() -> list[dict]:
    """legacy 全键 + 确定性 top-10 已有词条（§10.3.4 检索段）。"""
    import pandas as pd
    paths = get_project_paths()
    lg = pd.read_csv(paths.output_dir / "dictionary" / "legacy_df_freq_v1.csv",
                     encoding="utf-8-sig")
    ali = pd.read_parquet(paths.output_dir / "release" / "panel_v2"
                          / "skill_alias_v1.parquet")
    aliases = sorted({_norm_key(a) for a in ali.alias.astype(str)
                      if len(_norm_key(str(a))) >= 2})
    aidx = _bigram_index(aliases)
    out = []
    for i, key in enumerate(lg.match_key):
        key = str(key)
        bgs = {_norm_key(key)[k:k + 2] for k in range(len(key) - 1)}
        score: dict[int, int] = {}
        for bg in bgs:
            for j in aidx.get(bg, ()):
                score[j] = score.get(j, 0) + 1
        top = [aliases[j] for j, _ in
               sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]
        out.append({"id": i, "term": key,
                    "df": int(lg.df_unique_text.iloc[i]),
                    "cand": top})
    return out


def run_t2(limit: int = 0, workers: int = 3) -> None:
    paths = get_project_paths()
    items = candidates_t2()
    if limit:
        items = items[:limit]
    out = paths.output_dir / "llm_review" / "t2_legacy_review.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    def user(batch: list[dict]) -> str:
        lines = []
        for it in batch:
            cands = "；".join(f"{j}:{t}" for j, t in enumerate(it["cand"])) or "无"
            lines.append(f'{it["id"]}. 词条「{it["term"]}」'
                         f'(语料频次{it["df"]}) 已有相近: {cands}')
        return "\n".join(lines)
    batch_review(items, SYS_T2, user, out, workers=workers)


# ------------------------------------------------------------------ 合并


# T1 类别中视为非技能的集合（cert/iso/data_standard/software 从技能宽定义保留）
NON_SKILL_CATS = {"task_fragment", "soft_trait", "edu_req", "exp_req",
                  "benefit", "job_title", "company", "goods_service", "other"}


def _atier_alias_to_sid() -> dict[str, str]:
    """返回已按 primary_skill_id 消歧的 A 级别名键 → skill_id。"""
    from .lexicon import _load_atier_alias_records
    out: dict[str, str] = {}
    for rec in _load_atier_alias_records():
        key = _norm_key(rec.alias)
        if key in out and out[key] != rec.skill_id:
            raise RuntimeError(
                f"A级别名消歧后仍多概念: {key!r} -> {out[key]!r}/{rec.skill_id!r}"
            )
        out[key] = rec.skill_id
    return out


def merge_final() -> None:
    """把 T1/T2 真正落实到概念映射，生成 handoff-compliant v3 治理表。

    与旧 v2 的关键区别：
    - T2 r=0..9 的 MATCH_EXISTING 不再被忽略，而是映射回 A 级 skill_id；
    - NEW_CONCEPT 的 B/C 技能按规范名+首次发现年份生成稳定 UUIDv5；
    - AMBIGUOUS / 非技能 / 非技能停用表命中均降 D；
    - 产物显式保存 final_skill_id / mapping_action / T2 类型与首次年份。
    """
    import pandas as pd
    from .governance import stable_bc_skill_id

    paths = get_project_paths()
    rd = paths.output_dir / "llm_review"
    dic = paths.output_dir / "dictionary"

    def load_jsonl(name: str) -> list[dict]:
        path = rd / name
        if not path.exists():
            raise RuntimeError(f"缺少治理评审结果: {path}")
        return [
            json.loads(ln)
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    t1_items = candidates_t1()
    t2_items = candidates_t2()
    id2t1 = {it["id"]: it for it in t1_items}
    id2t2 = {it["id"]: it for it in t2_items}
    t2_item_by_term = {str(it["term"]): it for it in t2_items}
    alias_to_sid = _atier_alias_to_sid()

    t1_cat: dict[str, str] = {}
    t1_rows: list[dict] = []
    covered: set[int] = set()
    for rec in load_jsonl("t1_stopword.jsonl"):
        iid = int(rec["id"])
        it = id2t1.get(iid)
        if it is None:
            raise RuntimeError("T1 结果含重建外 id（词表漂移）")
        covered.add(iid)
        res = rec.get("res") or {}
        if "error" in res:
            raise RuntimeError(f"T1 存在失败记录 id={iid}: {res}")
        cat = res.get("c")
        if cat is None:
            raise RuntimeError(f"T1 缺类别 id={iid}")
        t1_cat[str(it["term"])] = str(cat)
        t1_rows.append({
            "term": it["term"], "tier": it["tier"], "category": cat
        })
    if covered != set(id2t1):
        raise RuntimeError(f"T1 覆盖不齐 {len(covered)}/{len(id2t1)}")

    t2_by_term: dict[str, dict] = {}
    covered_t2: set[int] = set()
    for rec in load_jsonl("t2_legacy_review.jsonl"):
        iid = int(rec["id"])
        it = id2t2.get(iid)
        if it is None:
            raise RuntimeError("T2 结果含重建外 id（候选列表漂移）")
        res = rec.get("res") or {}
        if "error" in res:
            raise RuntimeError(f"T2 存在失败记录 id={iid}: {res}")
        required = {"s", "c", "n", "r", "a"}
        if not required.issubset(res):
            raise RuntimeError(
                f"T2 结果缺字段 id={iid}: {sorted(required - set(res))}"
            )
        t2_by_term[str(it["term"])] = res
        covered_t2.add(iid)
    if covered_t2 != set(id2t2):
        raise RuntimeError(f"T2 覆盖不齐 {len(covered_t2)}/{len(id2t2)}")

    g = pd.read_csv(
        dic / "skill_legacy_graded_BCD_v1.csv", encoding="utf-8-sig"
    )
    required_v1 = {"term", "skill_id", "df_freq", "grade", "first_year"}
    if not required_v1.issubset(g.columns):
        raise RuntimeError(
            "v1 分级表缺少生成稳定概念ID所需字段: "
            + ", ".join(sorted(required_v1 - set(g.columns)))
        )

    rows = []
    for _, row in g.iterrows():
        term = str(row.term)
        r2 = t2_by_term.get(term)
        item = t2_item_by_term.get(term)
        if r2 is None or item is None:
            raise RuntimeError(f"T2 未覆盖 legacy 词: {term!r}")
        c1 = t1_cat.get(term)
        final = str(row.grade)
        reasons: list[str] = []
        mapping_action = ""
        final_skill_id = ""
        relation = int(r2["r"])

        if r2.get("s") is False:
            reasons.append("t2_not_skill")
        if c1 in NON_SKILL_CATS:
            reasons.append(f"t1_{c1}")
        if relation == -3:
            reasons.append("t2_not_skill_relation")
        if relation == -2:
            reasons.append("t2_ambiguous")
        if r2.get("a") is True and int(row.tautological) == 1:
            reasons.append("ambig_taut")

        if reasons or final == "D":
            final = "D"
            mapping_action = "REJECT_D"
        elif relation >= 0:
            cands = list(item.get("cand") or [])
            if relation >= len(cands):
                raise RuntimeError(
                    f"T2 选择越界 term={term!r}: r={relation}, candidates={len(cands)}"
                )
            alias = _norm_key(cands[relation])
            final_skill_id = alias_to_sid.get(alias, "")
            if not final_skill_id:
                raise RuntimeError(
                    f"T2 MATCH_EXISTING 无法解析 A 级 skill_id: {term!r} -> {alias!r}"
                )
            final = "A"
            mapping_action = "MATCH_EXISTING"
        elif relation == -1:
            if final not in {"B", "C"}:
                raise RuntimeError(
                    f"NEW_CONCEPT 但基础分级不是 B/C: {term!r} grade={final!r}"
                )
            final_skill_id = stable_bc_skill_id(term, int(row.first_year))
            mapping_action = "NEW_CONCEPT"
        else:
            raise RuntimeError(f"未知 T2 relation: term={term!r}, r={relation}")

        rows.append({
            "term": term,
            "source_skill_id": str(row.skill_id),
            "final_skill_id": final_skill_id,
            "df_freq": int(row.df_freq),
            "cand_cooc": row.get("cand_cooc", pd.NA),
            "first_year": int(row.first_year),
            "taut": int(row.tautological),
            "v2e_grade": str(row.grade),
            "t1_cat": c1,
            "t2_skill": r2.get("s"),
            "t2_cat": r2.get("c"),
            "t2_new_tech": r2.get("n"),
            "t2_relation": relation,
            "t2_ambig": r2.get("a"),
            "mapping_action": mapping_action,
            "final_grade": final,
            "demote_reason": "|".join(reasons),
        })

    out = pd.DataFrame(rows).sort_values(
        ["final_grade", "term"], kind="stable"
    ).reset_index(drop=True)
    formal = out[out.final_grade.isin(["A", "B", "C"])]
    if (formal.final_skill_id.astype(str).str.len() == 0).any():
        raise RuntimeError("A/B/C 存在空 final_skill_id")
    out.to_csv(
        dic / "skill_legacy_graded_BCD_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fc = out.final_grade.value_counts()
    logger.info(
        "v1.4 handoff治理: A映射 %d / B %d / C %d / D %d；"
        "MATCH_EXISTING=%d / NEW_CONCEPT=%d",
        int(fc.get("A", 0)), int(fc.get("B", 0)),
        int(fc.get("C", 0)), int(fc.get("D", 0)),
        int((out.mapping_action == "MATCH_EXISTING").sum()),
        int((out.mapping_action == "NEW_CONCEPT").sum()),
    )

    sb = pd.DataFrame(t1_rows)
    sb = sb[sb.category.isin(NON_SKILL_CATS)].drop_duplicates("term")
    sb = sb.sort_values(["tier", "category", "term"], kind="stable")
    sb.to_csv(
        dic / "non_skill_stopword_v1.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="词表治理 LLM 语义复核")
    ap.add_argument("task", choices=["t1", "t2", "merge"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / f"lexicon_llm_{args.task}.log")
    if args.workers > 4:
        raise SystemExit("GPU 共享纪律：并发 ≤4")
    if args.task == "t1":
        run_t1(args.workers)
    elif args.task == "t2":
        run_t2(args.limit, args.workers)
    else:
        merge_final()


if __name__ == "__main__":
    main()
