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
_SHORT_SKILL_KEYS = {"r"}
_SPECIAL_ASCII_KEYS = {"c++", "c#", ".net", "r", "go"}


@dataclass(frozen=True)
class AliasRecord:
    """正式激活别名的一条可审计记录。"""

    alias: str
    skill_id: str
    primary_skill_id: str = ""
    ambiguity_flag: int = 0


@dataclass(frozen=True)
class SkillMatch:
    """岗位—技能唯一匹配证据（指南 §11.1/§11.3）。"""

    skill_id: str
    surface_form: str
    start: int
    end: int
    mention_count: int
    match_method: str
    ambiguity_flag: int
    covered_candidate_count: int = 0
    covered_candidates: str = "[]"


def _norm_key(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).lower()


def resolve_active_alias_records(rows: list[AliasRecord]) -> list[AliasRecord]:
    """按 §11.1.3 把激活歧义别名解析到唯一主概念。"""
    grouped: dict[str, list[AliasRecord]] = {}
    for row in rows:
        grouped.setdefault(_norm_key(row.alias), []).append(row)
    resolved: list[AliasRecord] = []
    for key in sorted(grouped):
        group = grouped[key]
        skill_ids = {r.skill_id for r in group}
        primaries = {r.primary_skill_id for r in group if r.primary_skill_id}
        if len(skill_ids) == 1:
            target = next(iter(skill_ids))
        elif len(primaries) == 1 and next(iter(primaries)) in skill_ids:
            target = next(iter(primaries))
        else:
            raise RuntimeError(
                "激活别名存在多概念但无唯一 primary_skill_id: "
                f"{key!r} -> skills={sorted(skill_ids)} primaries={sorted(primaries)}"
            )
        candidates = [r for r in group if r.skill_id == target]
        chosen = sorted(candidates, key=lambda r: (r.alias, r.skill_id))[0]
        resolved.append(AliasRecord(
            alias=chosen.alias,
            skill_id=target,
            primary_skill_id=target if len(skill_ids) > 1 else chosen.primary_skill_id,
            ambiguity_flag=max(int(r.ambiguity_flag) for r in group),
        ))
    return resolved

def _is_ascii_alnum(ch: str) -> bool:
    """ASCII 字母数字判定（§12.6.4 边界语义；中文不是 ASCII alnum）。"""
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9")


def _boundary_ok(key: str, text: str, start: int, end: int) -> bool:
    """§11.1.2 ASCII 边界；特殊技能显式处理而非一刀切过滤。"""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if key in {"c++", "c#"}:
        # 允许 C++17 / C#8 这类版本后缀数字，但拒绝嵌入英文单词。
        return (not _is_ascii_alnum(before)) and not (
            ("a" <= after.lower() <= "z")
        )
    # .NET / R / Go 与一般英文术语均采用 ASCII 字母数字边界。
    return not _is_ascii_alnum(before) and not _is_ascii_alnum(after)


@dataclass
class UnionLexicon:
    """union 词表匹配器（A 级概念 + legacy 合成词共用一个自动机）。"""

    automaton: ahocorasick.Automaton = field(default=None)  # type: ignore[assignment]
    ascii_keys: frozenset[str] = frozenset()
    homograph: dict[str, re.Pattern] = field(default_factory=dict)
    keys_map: dict[str, str] = field(default_factory=dict)  # match键→skill_id
    ambiguous_keys: frozenset[str] = frozenset()
    n_atier: int = 0
    n_legacy: int = 0
    n_concepts: int = 0
    overlap_terms: tuple[str, ...] = ()

    def extract_matches(self, match_text: str) -> list[SkillMatch]:
        """执行 §11.1.1 longest-match，并保留可回填的匹配证据。"""
        candidates: list[tuple[int, int, str, str]] = []
        for end_inclusive, (sid, key) in self.automaton.iter(match_text):
            start = end_inclusive - len(key) + 1
            end = end_inclusive + 1
            if key in self.ascii_keys and not _boundary_ok(
                key, match_text, start, end
            ):
                continue
            ctx = self.homograph.get(sid)
            if ctx is not None and not ctx.search(match_text):
                continue
            candidates.append((start, end, sid, key))

        # 指南 §11.1.1：先按跨度长短排序，任何与已接收较长跨度重叠的
        # 候选均不再进入主匹配；并列使用确定性次序。
        accepted: list[tuple[int, int, str, str]] = []
        covered: dict[
            tuple[int, int, str, str],
            list[tuple[int, int, str, str]],
        ] = {}
        for cand in sorted(
            candidates, key=lambda x: (-(x[1] - x[0]), x[0], x[2], x[3])
        ):
            start, end, _sid, _key = cand
            overlaps = [
                a for a in accepted
                if not (end <= a[0] or start >= a[1])
            ]
            if overlaps:
                # accepted 按“更长优先”进入，首个 owner 即确定性覆盖者。
                covered.setdefault(overlaps[0], []).append(cand)
                continue
            accepted.append(cand)

        # 同一 skill_id 多次出现只保留首次跨度，同时记录 mention_count。
        by_sid: dict[str, list[tuple[int, int, str, str]]] = {}
        for item in accepted:
            by_sid.setdefault(item[2], []).append(item)
        out: list[SkillMatch] = []
        for sid, mentions in by_sid.items():
            mentions.sort(key=lambda x: (x[0], -(x[1] - x[0]), x[3]))
            start, end, _sid, key = mentions[0]
            surface = match_text[start:end]
            if surface != key:
                raise RuntimeError(
                    f"匹配跨度无法回填: key={key!r} surface={surface!r} "
                    f"span=({start},{end})"
                )
            covered_rows = []
            for mention in mentions:
                for c0, c1, csid, ckey in covered.get(mention, []):
                    covered_rows.append({
                        "skill_id": csid,
                        "surface_form": ckey,
                        "start": c0,
                        "end": c1,
                    })
            import json
            out.append(SkillMatch(
                skill_id=sid, surface_form=surface, start=start, end=end,
                mention_count=len(mentions), match_method="aho_longest",
                ambiguity_flag=1 if key in self.ambiguous_keys else 0,
                covered_candidate_count=len(covered_rows),
                covered_candidates=json.dumps(
                    covered_rows, ensure_ascii=False, separators=(",", ":")
                ),
            ))
        return sorted(out, key=lambda m: (m.start, m.end, m.skill_id))

    def extract(self, match_text: str) -> set[str]:
        """兼容旧调用方：返回 longest-match 后岗位内唯一 skill_id 集合。"""
        return {m.skill_id for m in self.extract_matches(match_text)}

def _load_atier_alias_records() -> list[AliasRecord]:
    """读 A 级激活别名，并按 primary_skill_id 解析多概念同形词。"""
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT alias, skill_id, coalesce(primary_skill_id, ''),
                   coalesce(ambiguity_flag, '0')
            FROM ai_dict.skill_aliases
            WHERE is_active='1' AND alias IS NOT NULL
              AND (length(trim(alias))>=2 OR lower(trim(alias))='r')
            ORDER BY alias, skill_id
        """)
        rows = [
            AliasRecord(
                alias=str(alias), skill_id=str(skill_id),
                primary_skill_id=str(primary or ""),
                ambiguity_flag=1 if str(ambiguity) == "1" else 0,
            )
            for alias, skill_id, primary, ambiguity in cur.fetchall()
        ]
        return resolve_active_alias_records(rows)
    finally:
        conn.close()


def _load_atier_aliases() -> list[tuple[str, str]]:
    """兼容旧调用方的 (alias, skill_id) 视图；不再使用 min(skill_id) 裁决。"""
    return [(r.alias, r.skill_id) for r in _load_atier_alias_records()]


def build_union_lexicon(
    include_legacy: bool = True,
    legacy_terms: list[str] | None = None,
    aliases: list[tuple[str, str] | AliasRecord] | None = None,
    legacy_id_map: dict[str, str] | None = None,
    legacy_ambiguous_keys: set[str] | None = None,
) -> UnionLexicon:
    """构建 union 词表匹配器。

    Args:
        include_legacy: 是否并入自建技术词（config lexicon.legacy_terms）。
        legacy_terms: 自建词表；None 时加载现行合并词典（6,872 词）。
        aliases: A 级 (alias, skill_id) 对；None 时从 ai_dict 读取（注入参数
            供离线单测）。

    Returns:
        UnionLexicon。
    """
    if aliases is None:
        aliases = _load_atier_alias_records()
    keys: dict[str, str] = {}  # match_key -> skill_id（同键先入优先：A 级 uuid）
    ambiguous_keys: set[str] = set()
    homograph: dict[str, re.Pattern] = {}
    for item in aliases:
        if isinstance(item, AliasRecord):
            alias, sid = item.alias, item.skill_id
            ambiguity_flag = int(item.ambiguity_flag)
        else:
            alias, sid = item
            ambiguity_flag = 0
        key = _norm_key(alias)
        if key in keys and keys[key] != sid:
            raise RuntimeError(
                f"规范化激活别名仍映射多个概念: {key!r} -> {keys[key]!r}/{sid!r}"
            )
        keys.setdefault(key, sid)
        if ambiguity_flag:
            ambiguous_keys.add(key)
        if alias in _AMBIGUOUS_AI_TERMS:
            homograph[sid] = _AMBIGUOUS_AI_TERMS[alias]

    n_atier = len(keys)
    n_legacy = 0
    overlap: list[str] = []
    if include_legacy:
        terms = legacy_terms if legacy_terms is not None \
            else load_merged_skills(include_llm=True)
        for term in terms:
            key = _norm_key(term)
            if len(key) < 2 and key not in _SHORT_SKILL_KEYS:
                continue
            if legacy_id_map is not None:
                # handoff-compliant 正式扫描只允许治理表中的 A/B/C。
                target_sid = legacy_id_map.get(key)
                if not target_sid:
                    continue
            else:
                target_sid = LEGACY_PREFIX + key
            if key in keys:
                overlap.append(term)
                if legacy_id_map is not None and keys[key] != target_sid:
                    raise RuntimeError(
                        f"治理映射与 A 级同形键冲突: {key!r} -> "
                        f"{keys[key]!r}/{target_sid!r}"
                    )
                continue
            keys[key] = target_sid
            if legacy_ambiguous_keys and key in legacy_ambiguous_keys:
                ambiguous_keys.add(key)
            if term in _AMBIGUOUS_AI_TERMS:
                homograph[keys[key]] = _AMBIGUOUS_AI_TERMS[term]
                ambiguous_keys.add(key)
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
        keys_map=keys, ambiguous_keys=frozenset(ambiguous_keys),
        n_atier=n_atier, n_legacy=n_legacy,
        n_concepts=len(keys), overlap_terms=tuple(overlap),
    )


def anchor_concept_ids(lex_terms: set[str]) -> dict[str, bool]:
    """（辅助）标记 legacy 空间供导出统计；A 级概念覆盖判断在导出层做。

    Args:
        lex_terms: skill_id 集合。

    Returns:
        {skill_id: is_legacy}。
    """
    return {sid: sid.startswith(LEGACY_PREFIX) for sid in lex_terms}
