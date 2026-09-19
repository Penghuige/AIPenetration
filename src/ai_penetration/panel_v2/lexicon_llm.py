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
    # 候选顺序仍由规范化激活别名字符串决定，保持与既有 T2 JSONL 的
    # index 语义兼容；同时绑定其正式 primary_skill_id，供 merge 真正归一化。
    from .lexicon import _load_atier_aliases
    resolved_aliases = _load_atier_aliases()
    alias_to_sid: dict[str, str] = {}
    for alias, sid in resolved_aliases:
        key = _norm_key(alias)
        if len(key) < 2:
            continue
        if key in alias_to_sid and alias_to_sid[key] != sid:
            raise RuntimeError(
                f"A 级规范化别名 {key!r} 仍映射多个 primary skill"
            )
        alias_to_sid[key] = sid
    aliases = sorted(alias_to_sid)
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
        out.append({
            "id": i,
            "term": key,
            "df": int(lg.df_unique_text.iloc[i]),
            "cand": top,
            "cand_skill_ids": [alias_to_sid[t] for t in top],
        })
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


def apply_semantic_review_to_grade(
    grade: "pd.DataFrame",
    *,
    t1_cat: dict[str, str],
    t2_by_term: dict[str, dict],
    t2_items_by_term: dict[str, dict],
) -> "pd.DataFrame":
    """把 §10.3.3 语义复核真正落实到 A/B/C/D 与概念映射。

    T2 判非技能、T1 非技能类别、AMBIGUOUS 映射均降 D；仅靠
    first_year>=2019 代理进入的低频 C 必须由 T2 new_tech=true 确认；
    T2 明确映射到既有 A 级候选时，最终 grade=A 并复用正式 skill_id。
    """
    import pandas as pd

    rows: list[dict] = []
    for _, row in grade.iterrows():
        rec = row.to_dict()
        term = str(row.term)
        r2 = t2_by_term.get(term, {})
        item = t2_items_by_term.get(term, {})
        c1 = t1_cat.get(term)
        rel_idx = r2.get("r")
        mapped_sid: str | None = None
        if isinstance(rel_idx, int) and rel_idx >= 0:
            cand_sids = list(item.get("cand_skill_ids") or [])
            if rel_idx >= len(cand_sids):
                raise RuntimeError(
                    f"T2 映射索引越界 term={term!r} r={rel_idx} "
                    f"candidates={len(cand_sids)}"
                )
            mapped_sid = str(cand_sids[rel_idx])

        final = str(row.grade)
        reasons: list[str] = []
        if r2.get("s") is not True:
            reasons.append("t2_skill_not_confirmed")
        valid_types = {
            "method", "tool", "software", "model",
            "framework", "data", "language", "other",
        }
        if r2.get("s") is True and r2.get("c") not in valid_types:
            reasons.append("t2_type_missing_or_invalid")
        if c1 in NON_SKILL_CATS:
            reasons.append(f"t1_{c1}")
        if rel_idx == -2:
            reasons.append("t2_ambiguous_mapping")
        taut = int(rec.get("tautological", rec.get("taut", 0)) or 0)
        if r2.get("a") is True and taut == 1:
            reasons.append("ambig_taut")

        df_freq = int(rec.get("df_freq", 0) or 0)
        cand_cooc = float(rec.get("cand_cooc", 0.0) or 0.0)
        proxy_low_c = (
            str(row.grade) == "C"
            and df_freq < 10
            and cand_cooc < 0.50
        )
        if proxy_low_c and r2.get("n") is not True:
            reasons.append("c_new_tech_not_confirmed")

        if reasons:
            final = "D"
            mapped_sid = None
        elif mapped_sid:
            final = "A"

        rec.update({
            "v2e_grade": str(row.grade),
            "t1_cat": c1,
            "t2_skill": r2.get("s"),
            "t2_type": r2.get("c"),
            "t2_new_tech": r2.get("n"),
            "t2_ambig": r2.get("a"),
            "t2_relation": rel_idx,
            "mapped_existing_skill_id": mapped_sid,
            "final_grade": final,
            "demote_reason": "|".join(reasons),
            "formal_skill_id": mapped_sid or str(row.skill_id),
        })
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(
        ["final_grade", "term"], kind="stable"
    ).reset_index(drop=True)

def merge_final() -> None:
    """T1∧T2 信号合并进 v2e 分级 → 终版词表 v1.3（只出不进）。

    降级规则（相对 v2e grade，任一命中即 D）：
    ① T2 is_skill=false；② T1 类别 ∈ 非技能集；③ 真歧义 ∧ 同义反复
    （裸锚点词如 ai：既被 T2 标 ambiguity 又在 taut 枚举——v2e 里它们靠 df
    入了 B，正是 S6 带 46% 误判元凶，按用户"最准确结合"裁决清除）。
    id→term 用 candidates_t1()/candidates_t2() 确定性重建（与评审运行同
    输入同序）；评审 jsonl 与重建词表条数不一致即硬失败（防漂移）。
    输出：skill_legacy_graded_BCD_v2.csv（final_grade/demote_reason 列）+
    non_skill_stopword_v1.csv（B 表：T1 非技能类别全清单，含 A 级热词面）。
    """
    import pandas as pd
    paths = get_project_paths()
    rd = paths.output_dir / "llm_review"
    dic = paths.output_dir / "dictionary"

    def load_jsonl(name: str) -> list[dict]:
        return [json.loads(ln) for ln in
                (rd / name).read_text(encoding="utf-8").splitlines()]

    t1_items = candidates_t1()
    t2_items = candidates_t2()
    id2t1 = {it["id"]: it for it in t1_items}
    id2t2 = {it["id"]: it for it in t2_items}
    t1_cat: dict[str, str] = {}
    t1_rows: list[dict] = []
    covered: set[int] = set()
    for r in load_jsonl("t1_stopword.jsonl"):
        iid = int(r["id"])
        it = id2t1.get(iid)
        if it is None:
            raise RuntimeError("T1 结果含重建外 id（词表漂移？）")
        covered.add(iid)
        res = r.get("res") or {}
        cat = res.get("c")
        t1_cat[str(it["term"])] = str(cat)  # 跨层重复词同判，后写无害
        t1_rows.append({"term": it["term"], "tier": it["tier"],
                        "category": cat})
    if covered != set(id2t1):
        raise RuntimeError(f"T1 覆盖不齐 {len(covered)}/{len(id2t1)}")
    t2_by_term: dict[str, dict] = {}
    for r in load_jsonl("t2_legacy_review.jsonl"):
        it = id2t2.get(int(r["id"]))
        if it is None:
            raise RuntimeError("T2 结果含重建外 id")
        t2_by_term[str(it["term"])] = r.get("res") or {}
    if len(t2_by_term) != len(t2_items):
        raise RuntimeError("T2 覆盖不齐")

    g = pd.read_csv(
        dic / "skill_legacy_graded_BCD_v1.csv", encoding="utf-8-sig"
    )
    t2_items_by_term = {str(it["term"]): it for it in t2_items}
    out = apply_semantic_review_to_grade(
        g,
        t1_cat=t1_cat,
        t2_by_term=t2_by_term,
        t2_items_by_term=t2_items_by_term,
    )
    out.to_csv(dic / "skill_legacy_graded_BCD_v2.csv", index=False,
               encoding="utf-8-sig")
    fc = out.final_grade.value_counts()
    vc = out.v2e_grade.value_counts()
    dem = out[out.demote_reason != ""]
    logger.info(
        "v1.3: A映射 %d / B %d / C %d / D %d（v2e 为 B %d/C %d/D %d；"
        "降级 %d 词，样例 %s）",
        int(fc.get("A", 0)), int(fc.get("B", 0)), int(fc.get("C", 0)),
        int(fc.get("D", 0)), int(vc.get("B", 0)), int(vc.get("C", 0)),
        int(vc.get("D", 0)), len(dem), dem.term.head(20).tolist(),
    )
    # B 表：T1 全部非技能判定（legacy + A 级热词面；A 级本体不回删）
    sb = pd.DataFrame(t1_rows)
    sb = sb[sb.category.isin(NON_SKILL_CATS)].drop_duplicates("term")
    sb = sb.sort_values(["tier", "category", "term"], kind="stable")
    sb.to_csv(dic / "non_skill_stopword_v1.csv", index=False,
              encoding="utf-8-sig")
    logger.info("停用表 v1: %d 词条（legacy %d / atier %d / era %d），"
                "A 级不回删，表供入级闸门与敏感性分析", len(sb),
                int((sb.tier == "legacy").sum()),
                int((sb.tier == "atier").sum()),
                int((sb.tier == "era_missing").sum()))


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
