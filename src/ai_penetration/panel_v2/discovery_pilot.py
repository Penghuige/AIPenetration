"""§8-§9 B/C 级中文技能发现试点（v2e 改进项④的管道首跑）。

指南发现通道浓缩：Qwen 抽技能候选（§9.3 跨度可定位校验，未定位即失败记录）
→ 归一化聚合 → 与现有正式词表确定性比对（净新词）→ 样本频次排序 →
top 候选过 §10 语义评审（lexicon_llm SYS_T2）→ 候选池 CSV + 试点报告。
本机 vLLM Qwen3-8B（用户授权 2026-09-10；GPU 并发 ≤4 纪律）。

试点规模默认 20,000 条去重文本（2023-2024，TABLESAMPLE REPEATABLE 可复现）。
候选池为**待办清单**（需全量语料 df + 正式分级才可入词典 v1.3），本模块
不改动任何现行发布物。

用法::
    python -X utf8 -m src.ai_penetration.panel_v2.discovery_pilot [--n 20000]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import psycopg2

from config.paths import get_project_paths

from ..common import eps_conn_params, setup_logging
from .anchors import normalize_desc
from .dedup import SHARDS
from .lexicon import build_union_lexicon
from .lexicon_llm import SYS_T2, batch_review

logger = logging.getLogger("ai_penetration.panel_v2.discovery_pilot")

ADMIT = ("AND job_description IS NOT NULL "
         "AND length(trim(job_description)) >= 10 "
         "AND position IS NOT NULL AND position != ''")
TEXT_CHARS = 600          # 每条进 prompt 的描述截断
BATCH_BUDGET = 2800       # 组批字符预算（4096 ctx 含 system/输出留量）

SYS_EXTRACT = (
    "你是招聘文本技能抽取器。对每条岗位描述，抽取其中**明确出现**的具体技能"
    "（技术、工具、框架、模型、方法、专业能力），每个词必须原样摘自原文"
    "（≤14字），不得根据常识补写原文没有的词。跳过软素质（沟通能力、责任心）、"
    "学历经验福利、岗位名与公司名。通用裸词（开发、分析、办公）不抽，但带技术"
    "指称的复合词要抽（如 后端开发、数据可视化、目标检测）。"
    '只输出 JSON 数组 [{"id":<int>,"terms":[{"t":"词","e":"≤10字原文片段"}]}]，'
    "每个岗位一个对象，无技能则 terms 为空数组，条数与输入一致。")

_SP_RE = re.compile(r"[\s，。；、．,.;:：!！?？()（）\[\]【】/\\|\-—<>「」\"'`~@#$%^&*+=]+")


def _strip_all(s: str) -> str:
    return _SP_RE.sub("", s)


def fetch_sample(n: int, seed: int = 20260910) -> list[str]:
    """广深 2023/2024 随机去重文本 n 条（match 态；可复现）。"""
    conn = psycopg2.connect(**eps_conn_params())
    cur = conn.cursor()
    cur.execute("SET statement_timeout=0")
    seen: set[str] = set()
    out: list[str] = []
    per = int(n * 1.35)
    for _city, shard, _cid in SHARDS:
        cur.execute(
            f"SELECT position, job_description FROM public.{shard} "
            f"TABLESAMPLE SYSTEM (2.0) REPEATABLE ({seed}) "
            "WHERE substr(publish_time,1,4) IN ('2023','2024') "
            f"{ADMIT} LIMIT %s", (per,))
        for pos, desc in cur.fetchall():
            m = normalize_desc(str(desc))[:1200]
            if len(m) < 30 or m in seen:
                continue
            seen.add(m)
            out.append(f"{normalize_desc(str(pos))}｜{m}")
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    conn.close()
    logger.info("试点样本 %d 条（去重后）", len(out))
    return out


def ensure_samples(rd: Path, n: int) -> list[str]:
    """样本落盘复用（重跑/续跑必须同序，id 才对得上）。"""
    f = rd / "t4_samples.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    s = fetch_sample(n)
    rd.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    return s


def run(n: int) -> None:
    paths = get_project_paths()
    rd = paths.output_dir / "llm_review"
    rd.mkdir(parents=True, exist_ok=True)
    samples = ensure_samples(rd, n)
    items = [{"id": i, "text": t[:TEXT_CHARS]} for i, t in enumerate(samples)]
    out = rd / "t4_extract.jsonl"

    def user(batch: list[dict]) -> str:
        return "\n".join(f'{it["id"]}.<{it["text"]}>' for it in batch)
    batch_review(items, SYS_EXTRACT, user, out, workers=3,
                 budget=BATCH_BUDGET)

    # 聚合 + 跨度校验（§9.3：词条或证据必须在原文可定位，否则记失败）
    cand: dict[str, int] = {}
    ok = fail_span = 0
    done_ids: set[int] = set()
    for ln in out.read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        done_ids.add(int(rec["id"]))
        res = rec.get("res") or {}
        terms = res.get("terms") if isinstance(res, dict) else None
        if not isinstance(terms, list):
            continue
        src = _strip_all(samples[int(rec["id"])])
        for e in terms:
            if not isinstance(e, dict):
                continue
            w = unicodedata.normalize("NFKC", str(e.get("t", ""))).lower().strip()
            ev = _strip_all(str(e.get("e", "")))
            if not (2 <= len(w) <= 14):
                continue
            if _strip_all(w) in src or (ev and ev in src):
                cand[w] = cand.get(w, 0) + 1
                ok += 1
            else:
                fail_span += 1
    logger.info("跨度校验：有效 %d 提及 / 定位失败(疑似补写) %d；覆盖文本 %d/%d",
                ok, fail_span, len(done_ids), len(items))

    # 现有正式词表比对（净新候选）——生产同源 union 键面
    from ..skill_ai_anchor import load_merged_skills
    from .lexicon import _load_atier_aliases
    lex = build_union_lexicon(legacy_terms=load_merged_skills(include_llm=True),
                              aliases=_load_atier_aliases())
    known = set(lex.keys_map)
    net = {k: v for k, v in cand.items() if k not in known}
    logger.info("候选面 %d，净新 %d", len(cand), len(net))

    import pandas as pd
    top = sorted(net.items(), key=lambda kv: (-kv[1], kv[0]))[:200]
    cdf = pd.DataFrame(top, columns=["term", "sample_df"])
    # top200 语义评审（§10.3.3：确定性 top10 相近词 + 模型判同概念/新/非技）
    gitems = [{"id": 800_000 + i, "term": t, "df": int(c),
               "cand": []} for i, (t, c) in enumerate(top)]
    gout = rd / "t4_grade.jsonl"
    def guser(batch: list[dict]) -> str:
        return "\n".join(f'{it["id"]}. 词条「{it["term"]}」'
                         f'(试点频次{it["df"]}) 已有相近: 无预检索'
                         for it in batch)
    batch_review(gitems, SYS_T2, guser, gout, workers=3)
    grade: dict[int, dict] = {}
    for ln in gout.read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        grade[int(rec["id"]) - 800_000] = rec.get("res") or {}
    rows = []
    for i, (t, c) in enumerate(top):
        g = grade.get(i, {})
        rows.append({"term": t, "sample_df": c,
                     "is_skill": g.get("s"), "category": g.get("c"),
                     "new_tech": g.get("n"), "ambiguity": g.get("a")})
    csv = paths.output_dir / "dictionary" / "discovery_candidates_v1.csv"
    pd.DataFrame(rows).to_csv(csv, index=False, encoding="utf-8-sig")
    n_skill = sum(1 for r in rows if r["is_skill"])
    rep = paths.report_dir / (
        f"discovery_pilot_{datetime.now():%Y%m%d_%H%M}.md")
    rep.write_text("\n".join([
        "# §8-9 B/C 技能发现试点报告（2026-09-10）", "",
        f"- 样本：2023-2024 去重文本 {len(samples):,} 条（seed=20260910，可复现）",
        f"- 抽取提及：有效 {ok:,} / 跨度定位失败 {fail_span:,}"
        f"（失败率 {fail_span/max(ok+fail_span,1):.1%}，§9.3 幻觉防线）",
        f"- 候选面 {len(cand):,}，其中净新（不在 A∪legacy 词表）{len(net):,}",
        f"- top200 送审：is_skill 通过 {n_skill}/200；结果件 "
        f"`dictionary/discovery_candidates_v1.csv`",
        "- **状态**：候选池（待全量语料 df 与 §10.3.1 正式分级后方可入 v1.3；"
        "本试点不改任何发布物）",
        "", "## top30 净新候选",
        pd.DataFrame(rows[:30]).to_markdown(index=False),
    ]), encoding="utf-8")
    print(f"试点完成：净新候选 {len(net)}，top200 通过 {n_skill}，报告 {rep}")


def main() -> None:
    ap = argparse.ArgumentParser(description="B/C 技能发现试点")
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    paths = get_project_paths()
    setup_logging(paths.log_dir / "discovery_pilot.log")
    run(args.n)


if __name__ == "__main__":
    main()
