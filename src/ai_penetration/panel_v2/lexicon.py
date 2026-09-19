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

    def extract(self, match_text: str) -> set[str]:
        """从 match 态文本抽取技能 id 集合（岗位内去重）。

        Args:
            match_text: normalize_desc/NFKC+lower 后的描述文本。

        Returns:
            skill_id 集合（uuid 或 legacy:<term>）。
        """
        hits: set[str] = set()
        for end, (sid, key) in self.automaton.iter(match_text):
            if key in self.ascii_keys:
                start = end - len(key) + 1
                before = match_text[start - 1] if start > 0 else ""
                after = match_text[end + 1] if end + 1 < len(match_text) else ""
                if _is_ascii_alnum(before) or _is_ascii_alnum(after):
                    continue
            hits.add(sid)
        for sid, ctx in self.homograph.items():
            if sid in hits and not ctx.search(match_text):
                hits.discard(sid)
        return hits


def _load_atier_aliases() -> list[tuple[str, str]]:
    """读 A 级激活别名 (alias, skill_id)。

    确定性保证（B3 审计修复）：同一 alias 多 skill_id 时取 min(skill_id)，
    按 alias 排序返回——first-wins 结果跨进程/跨重跑稳定（实证 36 个
    碰撞键如 abap/ansible/cobol，若不定序会造成 skill_code 跨分片错位）。
    """
    conn = psycopg2.connect(**eps_conn_params())
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT alias, min(skill_id) AS skill_id FROM ai_dict.skill_aliases
            WHERE is_active='1' AND alias IS NOT NULL AND length(trim(alias))>=2
            GROUP BY alias ORDER BY alias
        """)
        return [(r[0], r[1]) for r in cur.fetchall()]
    finally:
        conn.close()


def build_union_lexicon(
    include_legacy: bool = True,
    legacy_terms: list[str] | None = None,
    aliases: list[tuple[str, str]] | None = None,
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
        aliases = _load_atier_aliases()
    keys: dict[str, str] = {}  # match_key -> skill_id（同键先入优先：A 级 uuid）
    homograph: dict[str, re.Pattern] = {}
    for alias, sid in aliases:
        key = unicodedata.normalize("NFKC", alias).lower()
        if key not in keys:
            keys[key] = sid
        if alias in _AMBIGUOUS_AI_TERMS:
            homograph[sid] = _AMBIGUOUS_AI_TERMS[alias]

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
            if key in keys:
                overlap.append(term)
                continue  # A 级已覆盖（同形键），概念空间优先
            keys[key] = LEGACY_PREFIX + key
            if term in _AMBIGUOUS_AI_TERMS:
                homograph[keys[key]] = _AMBIGUOUS_AI_TERMS[term]
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
        keys_map=keys, n_atier=n_atier, n_legacy=n_legacy,
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
