"""counts 三窗口计数与 §13.7 不变量的离线单测（小 fixture，无 DB）。"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.ai_penetration.panel_v2.counts import compute_counts, verify_counts


def _fixture(tmp_path):
    """3 岗位 2 年份：s0 全部与锚点共现，s1 只在非锚点岗位，s2 跨年。"""
    flags = pa.table({
        "job_id": pa.array([1, 2, 3], pa.int64()),
        "year": pa.array([2014, 2014, 2015], pa.int32()),
        "anchor_main": pa.array([1, 0, 1], pa.int8()),
        "anchor_cn_paper": pa.array([1, 0, 0], pa.int8()),
        "anchor_babina": pa.array([1, 0, 1], pa.int8()),
        "groups_main_bits": pa.array([1, 0, 3], pa.int16()),
    })
    longs = pa.table({
        "job_id": pa.array([1, 1, 2, 3], pa.int64()),
        "year": pa.array([2014, 2014, 2014, 2015], pa.int32()),
        "skill_code": pa.array([0, 2, 1, 2], pa.int32()),
    })
    pq.write_table(flags, tmp_path / "job_anchor_flag.parquet")
    pq.write_table(longs, tmp_path / "job_skill_long.parquet")
    return tmp_path


def test_annual_pooled_roll3_counts(tmp_path):
    counts = compute_counts(_fixture(tmp_path))
    sel = lambda v, w, sc, y: counts[(counts.anchor_version == v)
                                     & (counts.window_type == w)
                                     & (counts.skill_code == sc)
                                     & (counts.year == y)]
    # main：skill0 n=1 cooc=1；skill1 只在非锚点岗 n=1 cooc=0
    r = sel("main", "annual", 0, 2014)
    assert (r.n_skill.item(), r.n_ai_cooccur.item()) == (1, 1)
    r = sel("main", "annual", 1, 2014)
    assert (r.n_skill.item(), r.n_ai_cooccur.item()) == (1, 0)
    # pooled skill2：2014 一次 + 2015 一次 = n2 cooc2
    r = sel("main", "pooled", 2, -1)
    assert (r.n_skill.item(), r.n_ai_cooccur.item()) == (2, 2)
    # roll3 2014 窗口 {2014,2015}：skill2 计 2（含 2015 行），边界正确
    r = sel("main", "roll3_centered", 2, 2014)
    assert (r.n_skill.item(), r.n_ai_cooccur.item()) == (2, 2)
    assert int(r.window_start.item()) == 2014 and int(r.window_end.item()) == 2015
    # cn_paper：skill0 的 cooc=1/n=1；babina skill2 2015 锚点=1 ✓
    r = sel("babina", "annual", 2, 2015)
    assert (r.n_skill.item(), r.n_ai_cooccur.item()) == (1, 1)


def test_invariants_reject_bad_counts(tmp_path):
    counts = compute_counts(_fixture(tmp_path))
    broken = counts.copy()
    broken.loc[broken.index[0], "n_ai_cooccur"] = broken.loc[
        broken.index[0], "n_skill"] + 5
    with pytest.raises(RuntimeError):
        verify_counts(broken)



def test_pair_year_must_match_job_year(tmp_path):
    rel = _fixture(tmp_path)
    t = pq.read_table(rel / "job_skill_long.parquet").to_pandas()
    t.loc[t.index[0], "year"] = 2015
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False),
                   rel / "job_skill_long.parquet")
    with pytest.raises(RuntimeError, match="year"):
        compute_counts(rel)
