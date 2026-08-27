# AI 渗透率分析：数据库调用速查

> 更新：2026-08-24  
> 数据库：`eps`（独立 PostgreSQL，2014-2024 广东招聘数据）

## 1. 怎么连

```python
from src.ai_penetration.load_guangdong import get_eps_engine
engine = get_eps_engine()   # 已指向 eps 库，直接用
```

## 2. 库里有什么（3 类表，按城市分片）

| 表 | 是什么 | 干什么用 |
|---|---|---|
| `job_p0387` 等 | **岗位表**：每条招聘（岗位名、描述、发布时间、公司名、recruit_id） | AI 判定、技能抽取、渗透率计数的数据源 |
| `ent_p0387` 等 | **企业表**：企业信息（company_id、credit_code、**industry_code 行业**） | 补行业维度、公司级聚合 |
| `comp_p0387` 等 | **待遇表**：薪资/福利字段 | 暂未使用 |

- 城市→分片表：`load_guangdong.GD_SHARDS`（21 市）
- 广州 `job_p0387`、深圳 `job_p0389`（核心分析对象）

## 3. 关键字段

- **job**：`position`（岗位名）、`job_description`（描述）、`publish_time`（'YYYY-MM-DD'）、`recruit_id`
- **ent**：`recruit_id`（关联 job）、`industry_code`（GB/T 行业编码）、`credit_code`/`company_id`（标准化企业标识）

## 4. 怎么取数据

```python
# 取某年岗位（判定 AI + 抽技能用）
SELECT position, job_description FROM job_p0387 WHERE substr(publish_time,1,4)='2024'

# 岗位关联行业（ent 有重复 recruit_id，须 LATERAL 去重）
SELECT j.position, e.industry_code FROM job_p0387 j
LEFT JOIN LATERAL (SELECT industry_code FROM ent_p0387 WHERE recruit_id=j.recruit_id LIMIT 1) e ON true
WHERE substr(j.publish_time,1,4)='2024'

# 岗位关联公司（公司级 AIRatio 用）
SELECT j.position, e.company_id, e.company_name FROM job_p0387 j
LEFT JOIN LATERAL (SELECT company_id, company_name FROM ent_p0387 WHERE recruit_id=j.recruit_id LIMIT 1) e ON true
WHERE substr(j.publish_time,1,4)='2024'

# 行业编码归类到大类：classify_industry(code) -> "C22造纸和纸制品业"
```

## 5. 现成调用

| 想要什么 | 调用 |
|---|---|
| AI 判定岗位（方法 A） | `ai_scoring.is_ai_job(position, desc)` |
| AI 判定岗位（融合 A/B-过滤） | `skill_ai_anchor.is_ai_fused(position, desc, omega, regex)` |
| 抽岗位技能 | `skill_ai_anchor.extract_skills_fast(desc, regex)` |
| 全量渗透率（方法 A/B） | `stream_penetration` / `anchor_penetration` |
| 融合渗透率（按年抽样） | `fusion_sampling.py` |
| 公司级 AIRatio | `company_airatio.py --method a/fused` |
| 行业大类映射 | `industry_classification.classify_industry` |
| 技能词典 | `dicts/ai_skill_terms.txt`（方法A）、`dicts/ai_skill_terms_llm.txt`（方法B） |
