"""zh_alias_freq 去重口径纯函数与 numpy 聚合管线的离线测试。

口径按指南 §10.3.2：freq_total = COUNT(DISTINCT normalized_text_hash)，
平台不计入频数键——同一规范化岗位描述跨平台重复发布只计一次。
"""
from __future__ import annotations

import numpy as np

from src.ai_penetration.zh_alias_freq import (
    aggregate_counts,
    build_automaton,
    normalize_desc,
    text_key,
)


def test_normalize_desc_folds_whitespace_and_lowercases():
    """raw→match 链：小写化 + 连续空白折叠为单空格（match 态本身不删空白）。"""
    assert normalize_desc("  深度 Learning\t\n AI ") == "深度 learning ai"


def test_text_key_same_text_ignores_layout_difference():
    k1 = text_key("负责 深度学习\n模型", "51Job")
    k2 = text_key("负责深度学习模型", "51Job")
    assert k1 == k2


def test_text_key_ignores_platform():
    """§10.3.2：平台不计入频数键，同文本跨平台视为同一文档。"""
    assert text_key("相同文本", "51Job") == text_key("相同文本", "BOSS直聘")
    assert text_key("相同文本") == text_key("相同文本", "51Job")


def test_text_key_is_63bit_safe():
    """text_hash 取 blake2b 8B 后压到 63bit，保证 int64 有符号安全。"""
    for desc in ("相同文本", "深度学习工程师", "AI 训练师"):
        key = text_key(desc, "51Job")
        assert 0 <= key < (1 << 63)


def test_automaton_matches_mixed_case_alias_on_lowered_text():
    """回归：含大写字母的 mixed 别名必须能在小写规范化语料上命中。"""
    autom = build_automaton({"ChatGPT提示词": ("a1", "mixed"), "AI训练师": ("a2", "zh")})
    text = normalize_desc("公司招聘 AI训练师，负责 ChatGPT提示词优化")
    hits = {aid for _e, (aid, _alias) in autom.iter_long(text)}
    assert hits == {"a1", "a2"}


def test_aggregate_counts_dedups_across_slices(tmp_path):
    """全局去重：同 (aid,key) 跨切片重复只计一次；n_alias 保证覆盖无命中别名。"""
    # 切片1: aid0-{111,222}, aid1-{111}（111 在同切片内重复一次）
    k1 = np.array([111, 222, 111], dtype=np.uint64)
    a1 = np.array([0, 0, 1], dtype=np.uint32)
    # 切片2: aid0-{222(跨片重复),333}, aid1-{111(跨片重复)}
    k2 = np.array([222, 333, 111], dtype=np.uint64)
    a2 = np.array([0, 0, 1], dtype=np.uint32)
    metas = []
    for idx, (ks, aa) in enumerate([(k1, a1), (k2, a2)]):
        kf = tmp_path / f"s{idx}.keys.bin"
        af = tmp_path / f"s{idx}.aids.bin"
        ks.tofile(kf)
        aa.tofile(af)
        metas.append({"task": f"s{idx}", "rows": 0, "local_new": 0,
                      "pairs": int(ks.size),
                      "keys_file": str(kf), "aids_file": str(af)})
    freq = aggregate_counts(metas, tmp_path, n_alias=3)
    assert freq.tolist() == [3, 1, 0]  # aid0:{111,222,333}, aid1:{111}, aid2:无
    # 中间文件已清理
    assert list(tmp_path.glob("*.bin")) == []


def test_aggregate_counts_empty():
    freq = aggregate_counts([], None)
    assert freq.size == 0
