"""岗位描述三态文本（指南 §6.1/§6.1.1）：raw / clean / match + text_hash。

清洗原则（§6.1）：只删展示性噪声，保留技术技能全部字符信息。
允许：HTML 实体与标签还原、控制字符删除、Unicode NFKC（全角转半角）、
连续空白合并、重复标点压缩、平台模板页脚（可配置正则表）删除。
禁止（§6.1.2）：删标点/加号/井号/小数点/数字、词干化、分词替代原文、
过滤 C++/C#/.NET 等短技能词、删括号内英文缩写。

match 态 = anchors.normalize_desc（NFKC+小写+空白折叠，复用不另起炉灶）。
text_hash = blake2b(match, 8B) int64，供 §6.2.0 文本唯一化与 §6.2.1 分组。
"""
from __future__ import annotations

import html as _html
import html.parser
import re
import unicodedata
from hashlib import blake2b

from .panel_v2.anchors import normalize_desc  # match 态唯一实现

# 控制字符（保留 \n \t \r 供标点压缩处理）
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# HTML 标签（保守：只删闭合良好的标签对/自闭合，保留 < 的数学用法如 "a<b"）
_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]{0,80})?/?>")
# 重复标点压缩：连续 2 个以上同类标点压为 1 个；长串空白点线（模板分隔行）整行删
_REPEAT_PUNCT_RE = re.compile(r"([，。、；：．.！!？?；;])\s*(?:[，。、；：．.！!？?；;]\s*)+")
_TEMPLATE_LINE_RE = re.compile(
    r"^(?:[-—=＝*_#\s　·．.。]{4,}|(职位介绍|岗位职责|任职要求|工作内容|岗位要求)\s*[:：]?\s*)$",
    re.M,
)


class _TextOnlyParser(html.parser.HTMLParser):
    """HTML 实体还原 + 标签剥离（保留标签内文本）。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_description(raw: str, template_lines: tuple[str, ...] = ()) -> str:
    """raw → clean 态：可读、去展示噪声，保留技能字符信息（§6.1）。

    Args:
        raw: 原始 job_description。
        template_lines: 平台模板页脚/页首精确行（去重扫描阶段统计生成，
            配置注入；空元组时仅删结构性模板行）。

    Returns:
        clean 文本。
    """
    if not raw:
        return ""
    text = _CTRL_RE.sub(" ", str(raw))
    if "<" in text and ">" in text:  # 可能含 HTML：实体还原+剥标签
        try:
            parser = _TextOnlyParser()
            parser.feed(_TAG_RE.sub(" ", text))
            text = "".join(parser.parts)
        except Exception:  # noqa: BLE001 - 解析失败退回实体还原
            text = _html.unescape(_TAG_RE.sub(" ", text))
    # 结构性模板行与配置的精确模板行
    text = _TEMPLATE_LINE_RE.sub("\n", text)
    for tpl in template_lines:
        text = text.replace(tpl, "\n")
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"[ \t　]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_match(clean_text: str) -> str:
    """clean → match 态（词典与锚点匹配唯一用此态，§6.1.1）。"""
    return normalize_desc(clean_text)


def match_from_raw(raw: str, template_lines: tuple[str, ...] = ()) -> str:
    """一步到位 raw → match（扫描热路径用，避免中间串传递）。"""
    return to_match(clean_description(raw, template_lines))


def text_hash(match_text: str) -> int:
    """match 文本哈希（63bit int64 安全，§6.2.0 DISTINCT(source_platform, text_hash)）。

    哈希前删除全部空白：排版差异（换行/空格数）不影响"完全相同描述"判定，
    与锚点/词典匹配用的带空格 match 文本分工不同（match 态本身不删空白）。
    """
    compact = re.sub(r"\s+", "", match_text)
    return int.from_bytes(
        blake2b(compact.encode("utf-8"), digest_size=8).digest(), "big",
        signed=False,
    ) & 0x7FFF_FFFF_FFFF_FFFF  # 转 63bit 保证 int64 有符号安全


def normalize_position(raw: str) -> str:
    """岗位名规范化（§6.2.1.2 去重组键用，非职业标准化）。

    保守规则：NFKC + 小写 + 删空白/括号/数字编号（"软件工程师（HS0287）"
    与"软件工程师"归一）；不删字母内容。
    """
    norm = unicodedata.normalize("NFKC", str(raw or "")).lower()
    return re.sub(r"[\s()（）\[\]【】{}0-9#\-_]+", "", norm)
