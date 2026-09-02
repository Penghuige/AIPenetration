# AI 渗透率分析：数据库调用速查

> 更新：2026-09-01  
> 数据库：`eps`（独立 PostgreSQL，2014-2024 广东招聘数据）  
> 结果表：`Employ26` 库 `public` schema（见第 6 节，由 `import_penetration_results.py` 导入维护）

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
| 融合城市面板（21 市全量，多进程） | `fused_cities.py --workers 24 --scan-workers 8` |
| 融合行业渗透率（A/B/C 三口径，多进程） | `fused_industry.py --year 2024 --workers 24 --scan-workers 8` |
| 公司级 AIRatio | `company_airatio.py --method a/fused` |
| 行业大类映射 | `industry_classification.classify_industry` |
| 技能词典 | `dicts/ai_skill_terms.txt`（方法A）、`dicts/ai_skill_terms_llm.txt`（方法B） |

## 6. 结果表（Employ26 库 public schema）

分析产出已入库（幂等导入：`python -m src.ai_penetration.import_penetration_results`，TRUNCATE 后重写）：

| 表 | 内容 | 粒度 | 行数 |
|---|---|---|---|
| `public.ai_penetration_fused_cities` | 21 市融合面板：A/B/C 三口径计数与率 + A-only/B-only/both 分解 | city × year（2014-2024） | 226 |
| `public.ai_penetration_fused_industry` | 广深行业三口径（GB/T 大类；导入时剔除 total<500 小样本） | city × year × industry（2024） | 153 |

字段：`total / a_jobs / b_jobs / fused_jobs [ / ab_both_jobs / a_only_jobs / b_only_jobs] / a_rate / b_rate / fused_rate`。

查询示例（注意 PG `round` 需 numeric cast）：

```sql
-- 21 市 2024 融合率排名
SELECT city, total, fused_jobs, round(100*fused_rate::numeric, 2) AS pct
FROM public.ai_penetration_fused_cities WHERE year = 2024
ORDER BY fused_rate DESC;

-- 行业 Top8（跨广深合并，样本≥500）
SELECT industry, sum(total)::bigint AS n,
       round(100.0*sum(fused_jobs)/sum(total), 2) AS fused_pct
FROM public.ai_penetration_fused_industry
GROUP BY industry HAVING sum(total) >= 500
ORDER BY fused_pct DESC LIMIT 8;
```

数据锚点（与 CSV 一致）：广州 2024 total=5,939,763 fused=26,337；深圳 2024 total=7,414,265 fused=67,632。
