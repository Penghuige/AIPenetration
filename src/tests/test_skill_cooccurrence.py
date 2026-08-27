"""岗位 AI 技能共现度纯函数测试。"""
from src.ai_penetration.skill_cooccurrence import (
    compute_skill_ai_scores,
    compute_job_ai_relevance,
)


def test_skill_ai_scores_ratio_raw():
    # 技能A 出现在 80/100 AI 岗位，但 20/1000 全部岗位 → 强 AI 相关
    # 技能B 出现在 10/100 AI 岗位，400/1000 全部岗位 → 非 AI
    # normalize=False 保留原始比率（>1 表示偏向 AI）
    scores = compute_skill_ai_scores(
        skill_counts_ai={"A": 80, "B": 10},
        skill_counts_all={"A": 20, "B": 400},
        n_ai_jobs=100,
        n_all_jobs=1000,
        normalize=False,
    )
    assert scores["A"] > 1.0
    assert scores["B"] < 1.0
    assert scores["A"] > scores["B"]


def test_skill_ai_scores_normalized_in_unit_range():
    # 默认归一化：分数应在 [0,1)，且强 AI 技能仍高于非 AI
    scores = compute_skill_ai_scores(
        skill_counts_ai={"A": 80, "B": 10},
        skill_counts_all={"A": 20, "B": 400},
        n_ai_jobs=100,
        n_all_jobs=1000,
    )
    assert 0.0 <= scores["A"] < 1.0
    assert 0.0 <= scores["B"] < 1.0
    assert scores["A"] > scores["B"]


def test_skill_ai_scores_normalize_compresses_outlier():
    # 离群技能（极高比率）归一化后不应接近无穷
    scores = compute_skill_ai_scores(
        skill_counts_ai={"outlier": 100},
        skill_counts_all={"outlier": 1},
        n_ai_jobs=100,
        n_all_jobs=1000,
        min_count=1,
    )
    assert scores["outlier"] < 1.0


def test_skill_ai_scores_min_count_smoothing():
    # 出现次数过少的技能（< min_count）得分被压低
    scores = compute_skill_ai_scores(
        skill_counts_ai={"rare": 2},
        skill_counts_all={"rare": 2},
        n_ai_jobs=100,
        n_all_jobs=1000,
        min_count=5,
    )
    assert scores["rare"] <= 1.0


def test_job_ai_relevance_mean():
    scores = {"A": 2.0, "B": 0.5}
    rel = compute_job_ai_relevance(["A", "B"], scores)
    assert abs(rel - 1.25) < 1e-9


def test_job_ai_relevance_no_skills():
    assert compute_job_ai_relevance([], {"A": 2.0}) == 0.0


def test_load_ai_keyword_set_contains_core():
    from src.ai_penetration.skill_dictionary import load_ai_keyword_set

    s = load_ai_keyword_set()
    assert "大模型" in s
    assert "机器学习" in s


def test_extract_skills_matches_substrings():
    from src.ai_penetration.skill_data import _extract_skills

    desc = "熟悉Java和Spring Boot，会Python数据分析"
    skills = _extract_skills(desc, ["Java", "Spring Boot", "Python", "数据分析", "C++"])
    assert "Java" in skills
    assert "Spring Boot" in skills
    assert "Python" in skills
    assert "C++" not in skills


def test_extract_skills_short_english_boundary():
    from src.ai_penetration.skill_data import _extract_skills

    # 短英文技能名需词边界，避免 "R" 匹配到 "Spring" 或 "R" 在单词内
    desc = "掌握Spring框架和R语言"
    skills = _extract_skills(desc, ["R", "C", "Go", "Java"])
    assert "R" in skills          # R语言，独立词，应命中
    assert "C" not in skills      # 无独立 C，不应命中
    assert "Go" not in skills     # 无 Go
    assert "Java" not in skills   # 描述无 Java

    desc2 = "精通C语言编程"
    skills2 = _extract_skills(desc2, ["C", "R", "Go"])
    assert "C" in skills2


def test_extract_ai_skills_matches_ai_terms():
    from src.ai_penetration.skill_data import _extract_ai_skills

    # AI 技能应命中，工艺/设备词不命中
    desc = "负责深度学习模型训练，使用PyTorch和CUDA"
    ai = _extract_ai_skills(desc, ["深度学习", "PyTorch", "CUDA", "皮套键盘工艺", "深低温操作"])
    assert "深度学习" in ai
    assert "PyTorch" in ai
    assert "CUDA" in ai
    assert "皮套键盘工艺" not in ai
    assert "深低温操作" not in ai

    # 纯工艺描述不应命中 AI 技能
    desc2 = "从事皮套键盘生产工艺，深低温操作"
    ai2 = _extract_ai_skills(desc2, ["深度学习", "PyTorch", "CUDA", "皮套键盘工艺"])
    assert ai2 == []
