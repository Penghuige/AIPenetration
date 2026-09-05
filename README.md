# AIPenetration

AI 岗位探索 / 公司级 AI 投入强度（AIRatio）分析工具包。

从招聘语料中判定 AI 相关岗位并计算多口径渗透率：关键词加权评分、技能-锚点共现
（ωsAI/ωjAI）、融合口径，并下钻到公司-年度层面的 AIRatio。本项目是 2026-08-27
从 `Employ26` 仓库 `src/ai_penetration` 独立出来的工作副本，已不依赖原仓库。

## 目录结构

```
config/
  paths.py            # 路径与 PG 连接配置（显式参数 > AIPEN_PG_* 环境变量 > database.yaml > 默认值）
  database.yaml       # eps 数据库连接参数
  model_runtime.yaml  # LLM OpenAI-compatible API 配置
src/
  model_platform/
    llm.py            # 统一 LLM client（create_llm_client / LLMClient 协议）
  ai_penetration/     # 主包
    common.py         # setup_logging / eps_conn_params / eps_connect / DEFAULT_OMEGA_SNAPSHOT
    load_guangdong.py # eps 连接 engine + 广东 21 市分片映射 GD_SHARDS
    keyword_penetration.py / per_city_keyword.py        # 方法一：岗位名关键词
    ai_scoring.py     # 方法A：岗位名+描述三层加权评分（is_ai_job）
    skill_data.py / skill_dictionary.py / skill_cooccurrence.py   # 技能数据与词典
    skill_ai_anchor.py / anchor_penetration.py / anchor_industry.py  # 方法B：锚点共现 ωsAI/ωjAI
    fusion_sampling.py                                  # 方法A∪B 融合口径抽样估计
    company_airatio.py                                    # 公司-年度 AIRatio（--method a|fused）
    stream_penetration.py / penetration_detail.py / per_city_combined.py  # 全量流式与明细
    standardize_jobs.py / llm_skill_mining.py / llm_alias_generation.py / import_exchange_dict.py / zh_alias_freq.py  # LLM 工具链与词典治理
    report.py / cross_validate.py / sample_check.py     # 报告、交叉验证、人工抽检
    compute_penetration.py / compute_skill_penetration.py / combined_penetration.py / industry_classification.py
  tests/              # pytest 单测（离线可跑）
dicts/                # AI 关键词/技能词典（含权重分层标注）
docs/                 # 分析方法与数据库手册
output/reports/       # 运行产物（报告 CSV/MD、omega 快照）
logs/                 # 运行日志
```

## 快速开始

1. **环境**: Python 3.10+，依赖 pandas / SQLAlchemy / psycopg2 / requests /
   pyahocorasick / matplotlib，可访问 PostgreSQL 与 OpenAI-compatible API（LLM 步骤才需要）

2. **配置连接** — 编辑 `config/database.yaml` 填入 eps 库的 host/port/user/password：
   ```yaml
   database:
     host: localhost
     port: 5432
     dbname: eps            # 源库（只读为主）
     user: postgres
     password: "<你的密码>"
     results_db: ai_pen_results   # 结果库：分析产出表写这里，不污染 eps
   ```
   也可用环境变量 `AIPEN_PG_*`（源库）/ `AIPEN_RESULTS_*`（结果库）覆盖。
   结果导入脚本 `import_penetration_results.py` 首次运行加 `--create-db` 建库。
   LLM API 用 `AIPEN_LLM_BASE_URL/MODEL/API_KEY` 或编辑 `config/model_runtime.yaml`。

3. **运行**（均从项目根目录执行）:
   ```bash
   # 各市关键词渗透率
   python -m src.ai_penetration.per_city_keyword --min-freq 10

   # 公司级 AIRatio（融合口径，广深 2024）
   python -m src.ai_penetration.company_airatio --method fused --cities "广州市,深圳市" --min-jobs 10

   # 入口聚合菜单
   python -m src.ai_penetration --help
   ```
   全量流式脚本支持断点续跑（`output/ai_penetration/*_cp*.json`），中断后重跑即可跳过已完成分片。

## 判定方法速览

| 口径 | 实现 | 说明 |
|---|---|---|
| 方法一 关键词 | `keyword_penetration` | 岗位名命中 AI 关键词 |
| 方法A 加权评分 | `ai_scoring.is_ai_job` | 岗位名+描述得分 >= 3；含 AI设计/口语同形词排除 |
| 方法B 锚点共现 | `skill_ai_anchor` / `anchor_penetration` | 技能 ωsAI 均值 ωjAI >= 阈值，纯数据驱动 |
| fused | `company_airatio --method fused` | 方法A 或 B(过滤) 并集 |

写报告时请注明所用 `--method` 与 omega 快照时间戳。

## 测试

```bash
pytest src/tests/ -q          # 27 个离线单测，不需要数据库和 LLM
python -m compileall -q src config
```

## 快照来源与已知修复记录

源仓库: `Employ26` 分支 `feat/bge-none-improvement-phase1`（2026-08-27 复制）。
复制后完成了独立化改造与一轮代码审查修复：

1. 新建 `config/` 与 `src/model_platform/`，移除对 Employ26 的全部导入依赖
2. 修复 `per_city_keyword` 调用签名错误导致的入口不可用（原 TypeError 被 except 吞掉误报为 PG 连接失败）
3. 修复 `zh_alias_freq` 语料 lower() 后别名大小写不匹配导致活跃别名词频记零的问题
4. `company_airatio` checkpoint 按判定方法区分命名并在文件内记录 method，
   避免 a/fused 断点混用污染 AIRatio；断点改为报告写成功后再清理
5. `anchor_penetration` 断点恢复行补算 `ai_rate`（原先 NaN）
6. 方法A 补齐口语同形词排除（`_AI_CONTEXT_REQUIRED`），与方法B 口径对齐
7. `standardize_jobs` 批量返回长度校验，杜绝截断静默丢岗
8. 大表扫描改用 psycopg2 服务端命名游标分批读取，消除整结果集内存缓冲
9. 各入口的 ωsAI 快照默认值统一为 `common.DEFAULT_OMEGA_SNAPSHOT`（单一版本）
10. 连接层收口：URL 一律经 `pg_sqlalchemy_url()` 生成并对凭据做编码，
    消除 13 处硬编码 dbname 与手拼密码字符串；12 份重复 setup_logging 收拢到 `common.py`

有待完善：硬编码年份范围 `range(2014, 2025)` 可参数化；GB/T 行业映射建议外置到 `dicts/`；
LLM 标准化的 `max_output_tokens=1024` 对大批次偏小，建议随 batch_size 自适应。
