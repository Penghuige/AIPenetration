"""union 技能词表：A 级冻结概念（主键空间）∪ 自建技术词（legacy 合成空间）。

背景（2026-09-07 缺口挖掘实证）：A 级词典缺 pytorch/tensorflow/opencv/halcon/
强化学习 等 AI 技术词，自建 LLM 挖掘表恰能补位——v2 词表取并集而非替换。

- A 级条目：ai_dict.skill_aliases is_active='1'，lower(NFKC(alias)) → skill_id(uuid)；
- legacy 条目：v1 合并技能词典中**未被 A 级别名覆盖**的词，skill_id 合成为
  ``legacy:<term>``，不占用 uuid 空间、不回写词典库；
- 同形保护：别名恰为口语同形词（深度学习/强化学习）的条目命中后仍需
  AI 语境校验（复用现行 _AMBIGUOUS_AI_TERMS 纪律）；
- 纯 ASCII 键做字母数字边界校验（对齐 anchors §12.6 与现行英文边界惯例）。

匹配输入一律 match 态文本（anchors.normalize_desc / text_clean.to_match）。
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

import ahocorasick
import psycopg2

from ..common import eps_conn_params
from ..skill_ai_anchor import _AMBIGUOUS_AI_TERMS
from ..skill_ai_anchor import load_merged_skills

logger = logging.getLogger("ai_penetration.panel_v2.lexicon")

_ASCII_RE = re.compile(r"^[\x00-\x7f]+$")
LEGACY_PREFIX = "legacy:"


def _is_ascii_alnum(ch: str) -> bool:
    """ASCII 字母数字判定（§12.6.4 边界语义；中文不是 ASCII alnum）。"""
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9")


@dataclass(frozen=True)
class SkillMatch:
    """一次岗位内唯一技能命中；跨度基于 job_description_match。"""

    skill_id: str
    surface_form: str
    start: int
    end: int
    mention_count: int
    match_key: str
    ambiguity_flag: int = 0


@dataclass
class UnionLexicon:
    """union 词表匹配器（A 级概念 + legacy 合成词共用一个自动机）。"""

    automaton: ahocorasick.Automaton = field(default=None)  # type: ignore[assignment]
    ascii_keys: frozenset[str] = frozenset()
    homograph: dict[str, re.Pattern] = field(default_factory=dict)
    ambiguity_keys: frozenset[str] = frozenset()
    keys_map: dict[str, str] = field(default_factory=dict)  # match键→skill_id
    n_atier: int = 0
    n_legacy: int = 0
    n_concepts: int = 0
    overlap_terms: tuple[str, ...] = ()

    def extract_matches(self, match_text: str) -> list[SkillMatch]:
        """按指南 §11.1.1 返回最长优先、岗位内 skill_id 唯一的命中证据。

        不同概念发生完全嵌套时保留较长跨度；部分交叠但互不包含时均保留。
        同一 skill_id 多次出现只输出首次位置，同时记录 mention_count。
        """
        raw: list[tuple[int, int, str, str]] = []
        for end_inclusive, (sid, key) in self.automaton.iter(match_text):
            start = end_inclusive - len(key) + 1
            end = end_inclusive + 1
            if key in self.ascii_keys:
                before = match_text[start - 1] if start > 0 else ""
                after = match_text[end] if end < len(match_text) else ""
                if _is_ascii_alnum(before) or _is_ascii_alnum(after):
                    continue
            if sid in self.homograph and not self.homograph[sid].search(match_text):
                continue
            raw.append((start, end, sid, key))

        # 长跨度先占位；相同长度按起点、skill_id、key 确定性排序。
        accepted: list[tuple[int, int, str, str]] = []
        for hit in sorted(
            set(raw),
            key=lambda x: (-(x[1] - x[0]), x[0], x[1], x[2], x[3]),
        ):
            start, end, _sid, _key = hit
            if any(
                start >= a0 and end <= a1 and (start, end) != (a0, a1)
                for a0, a1, _as, _ak in accepted
            ):
                continue
            accepted.append(hit)

        by_sid: dict[str, list[tuple[int, int, str, str]]] = {}
        for hit in accepted:
            by_sid.setdefault(hit[2], []).append(hit)

        out: list[SkillMatch] = []
        for sid, hits in by_sid.items():
            first = min(hits, key=lambda x: (x[0], -(x[1] - x[0]), x[3]))
            start, end, _sid, key = first
            surface = match_text[start:end]
            if surface != key:
                raise RuntimeError(
                    f"词典跨度回填失败: sid={sid} key={key!r} surface={surface!r}"
                )
            out.append(SkillMatch(
                skill_id=sid,
                surface_form=surface,
                start=start,
                end=end,
                mention_count=len(hits),
                match_key=key,
                ambiguity_flag=int(
                    key in self.ambiguity_keys or sid in self.homograph
                ),
            ))
        return sorted(out, key=lambda x: (x.start, x.end, x.skill_id))

    def extract(self, match_text: str) -> set[str]:
        """兼容旧调用：返回按 §11 最长匹配规则去重后的 skill_id 集合。"""
        return {hit.skill_id for hit in self.extract_matches(match_text)}


def _resolve_active_alias_rows(
    rows: list[tuple[str, str, str | None]]
) -> list[tuple[str, str]]:
    """按 §7.4.4/§11.1.3 用 primary_skill_id 解析激活同形别名。

    多概念激活别名若没有唯一 primary_skill_id 属词典不变量错误，禁止再以
    min(skill_id) 猜测主概念。
    """
    grouped: dict[str, list[tuple[str, str | None]]] = {}
    for alias, sid, primary in rows:
        grouped.setdefault(str(alias), []).append((str(sid), primary or None))

    resolved: list[tuple[str, str]] = []
    for alias in sorted(grouped):
        values = grouped[alias]
        sids = {sid for sid, _ in values}
        primaries = {str(p) for _, p in values if p}
        if len(primaries) > 1:
            raise RuntimeError(
                f"激活别名 {alias!r} 存在多个 primary_skill_id: {sorted(primaries)}"
            )
        if primaries:
            sid = next(iter(primaries))
        elif len(sids) == 1:
            sid = next(iter(sids))
        else:
            raise RuntimeError(
                f"激活别名 {alias!r} 同时映射多个 skill_id 且未指定 primary_skill_id"
            )
        resolved.append((alias, sid))
    return resolved


def _load_atier_aliases() -> list[tuple[str, str]]:
    """读取 A 级激活别名，并严格应用词典 primary_skill_id 语义。"""
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT alias, skill_id, nullif(primary_skill_id, '')
            FROM ai_dict.skill_aliases
            WHERE is_active='1' AND alias IS NOT NULL AND length(trim(alias))>=2
            ORDER BY alias, skill_id
        """)
        return _resolve_active_alias_rows(cur.fetchall())
    finally:
        conn.close()



def load_frozen_atier_aliases(
    path,
) -> tuple[list[tuple[str, str]], frozenset[str]]:
    """从 A 级冻结 alias CSV 读取 matcher 输入与歧义键，不依赖 mutable DB。"""
    from pathlib import Path

    import pandas as pd

    frame = pd.read_csv(Path(path), encoding="utf-8-sig")
    required = {
        "alias", "skill_id", "is_active", "primary_skill_id",
        "ambiguity_flag",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            "A 级冻结 alias 文件缺字段: " + ", ".join(sorted(missing))
        )
    active = frame[frame.is_active.astype(str) == "1"].copy()
    rows = [
        (
            str(r.alias),
            str(r.skill_id),
            None if pd.isna(r.primary_skill_id) else str(r.primary_skill_id),
        )
        for _, r in active.iterrows()
    ]
    resolved = _resolve_active_alias_rows(rows)
    resolved_map = dict(resolved)

    ambiguity: set[str] = set()
    for _, row in active.iterrows():
        alias = str(row.alias)
        if resolved_map.get(alias) != str(row.skill_id):
            continue
        raw = str(row.ambiguity_flag).strip().lower()
        if raw in {"1", "true", "yes"}:
            ambiguity.add(unicodedata.normalize("NFKC", alias).lower())
    return resolved, frozenset(ambiguity)


def build_union_lexicon(
    include_legacy: bool = True,
    legacy_terms: list[str] | None = None,
    aliases: list[tuple[str, str]] | None = None,
    legacy_skill_ids: dict[str, str] | None = None,
    ambiguous_alias_keys: frozenset[str] | set[str] | None = None,
) -> UnionLexicon:
    """构建 union 词表匹配器。

    Args:
        include_legacy: 是否并入自建技术词（config lexicon.legacy_terms）。
        legacy_terms: 自建词表；None 时加载现行合并词典（6,872 词）。
        aliases: A 级 (alias, skill_id) 对；None 时从 ai_dict 读取（注入参数
            供离线单测）。
        legacy_skill_ids: 可选的规范化 legacy key → 最终 formal_skill_id。
            用于把经 §10.3.3 确认的别名直接映射回既有 A 级概念。
        ambiguous_alias_keys: 冻结词典中 ambiguity_flag=1 的规范化表面键。

    Returns:
        UnionLexicon。
    """
    if aliases is None:
        aliases = _load_atier_aliases()
    keys: dict[str, str] = {}  # match_key -> skill_id（同键先入优先：A 级 uuid）
    homograph: dict[str, re.Pattern] = {}
    for alias, sid in aliases:
        key = unicodedata.normalize("NFKC", alias).lower()
        if key in keys and keys[key] != sid:
            raise RuntimeError(
                f"激活别名规范化键 {key!r} 映射多个概念: {keys[key]} vs {sid}"
            )
        keys[key] = sid
        if alias in _AMBIGUOUS_AI_TERMS:
            homograph[sid] = _AMBIGUOUS_AI_TERMS[alias]

    ambiguity_keys = set(ambiguous_alias_keys or ())
    n_atier = len(keys)
    n_legacy = 0
    overlap: list[str] = []
    if include_legacy:
        terms = legacy_terms if legacy_terms is not None \
            else load_merged_skills(include_llm=True)
        for term in terms:
            key = unicodedata.normalize("NFKC", term).lower()
            if len(key) < 2:
                continue
            target_sid = (
                legacy_skill_ids.get(key)
                if legacy_skill_ids is not None
                else None
            )
            if key in keys:
                if target_sid and keys[key] != target_sid:
                    raise RuntimeError(
                        f"legacy 规范键 {key!r} 要求映射 {target_sid}，"
                        f"但 A 级已映射 {keys[key]}"
                    )
                overlap.append(term)
                continue
            sid = target_sid or (LEGACY_PREFIX + key)
            keys[key] = sid
            if term in _AMBIGUOUS_AI_TERMS:
                homograph[sid] = _AMBIGUOUS_AI_TERMS[term]
                ambiguity_keys.add(key)
            if sid.startswith(LEGACY_PREFIX):
                n_legacy += 1

    automaton = ahocorasick.Automaton()
    for key, sid in keys.items():
        automaton.add_word(key, (sid, key))
    automaton.make_automaton()
    ascii_keys = frozenset(k for k in keys if _ASCII_RE.match(k))
    logger.info("union 词表: A级键 %d + legacy %d（重叠 %d）→ 键 %d，同形保护 %d",
                n_atier, n_legacy, len(overlap), len(keys), len(homograph))
    return UnionLexicon(
        automaton=automaton, ascii_keys=ascii_keys, homograph=homograph,
        ambiguity_keys=frozenset(ambiguity_keys),
        keys_map=keys, n_atier=n_atier, n_legacy=n_legacy,
        n_concepts=len(set(keys.values())), overlap_terms=tuple(overlap),
    )


def load_formal_legacy_spec(grade_path) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """读取最终 §10 分级，返回正式 legacy 表面、概念映射与 tier。

    final_grade=D 的候选不进入正式 matcher；A 表示 T2 已映射到既有 A 概念，
    B/C 保持独立 formal_skill_id。返回映射键均为 NFKC+lower。
    """
    from pathlib import Path

    import pandas as pd

    frame = pd.read_csv(Path(grade_path), encoding="utf-8-sig")
    required = {"term", "final_grade", "formal_skill_id", "t2_relation"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            "最终分级文件缺字段: " + ", ".join(sorted(missing))
        )
    formal = frame[frame.final_grade.isin(["A", "B", "C"])].copy()
    terms: list[str] = []
    sid_by_key: dict[str, str] = {}
    tier_by_sid: dict[str, str] = {}
    for _, row in formal.iterrows():
        term = str(row.term)
        key = unicodedata.normalize("NFKC", term).lower()
        raw_sid = row.get("formal_skill_id")
        if pd.isna(raw_sid) or not str(raw_sid).strip():
            raise RuntimeError(f"正式词条 {term!r} 缺 formal_skill_id")
        sid = str(raw_sid).strip()
        if key in sid_by_key and sid_by_key[key] != sid:
            raise RuntimeError(
                f"正式 legacy 键 {key!r} 映射多个概念: "
                f"{sid_by_key[key]} vs {sid}"
            )
        sid_by_key[key] = sid
        terms.append(term)
        if str(row.final_grade) in {"B", "C"}:
            tier_by_sid[sid] = str(row.final_grade)
    return terms, sid_by_key, tier_by_sid

def anchor_concept_ids(lex_terms: set[str]) -> dict[str, bool]:
    """（辅助）标记 legacy 空间供导出统计；A 级概念覆盖判断在导出层做。

    Args:
        lex_terms: skill_id 集合。

    Returns:
        {skill_id: is_legacy}。
    """
    return {sid: sid.startswith(LEGACY_PREFIX) for sid in lex_terms}
