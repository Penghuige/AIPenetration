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

## 外部技能词典冻结管线（exchange 交接·第二节四步）

```bash
# 1. 频数计算（广深语料，分级/激活口径 COUNT(DISTINCT 规范化text_hash)）
python -X utf8 -m src.ai_penetration.zh_alias_freq --force

# 2-3. 阈值激活（先 dry-run 看候选，确认后 --apply）
python -m src.ai_penetration.zh_alias_activation --min-freq 100
python -m src.ai_penetration.zh_alias_activation --min-freq 100 --apply

# 4. 交叉验证（用既有 AI 率方法审计激活质量）
python -m src.ai_penetration.cross_validate_alias_activation --per-city 10000

# 5. 导出 A 级冻结词典 + QC 报告（不变量自检失败会阻断）
python -m src.ai_penetration.freeze_external_dictionary
```

## 判定方法速览

| 口径 | 实现 | 说明 |
|---|---|---|
| 方法一 关键词 | `keyword_penetration` | 岗位名命中 AI 关键词 |
| 方法A 加权评分 | `ai_scoring.is_ai_job` | 岗位名+描述得分 >= 3；含 AI设计/口语同形词排除 |
| 方法B 锚点共现 | `skill_ai_anchor` / `anchor_penetration` | 技能 ωsAI 均值 ωjAI >= 阈值，纯数据驱动 |
| fused | `company_airatio --method fused` | 方法A 或 B(过滤) 并集 |

写报告时请注明所用 `--method` 与 omega 快照时间戳。

## 原始交接合规重跑（v2i）

历史 v2h 只作为旧结果保留；不要覆盖或把旧结果重新标成 v2i。PR #4 的
v2i 是一次整体迁移，正式发布要求“上游真实执行 manifest → v4 治理 →
handoff scan → v2i”全部闭环。下面命令**不应在 GitHub Actions 中自动跑**，
长任务在项目机器本地执行。

### 0. 先冻结人工可核定配置

1. `config/model_config_v1.yaml`：模型 revision、量化、tokenizer、vLLM、
   CUDA、PyTorch 等不得保留 `TO_BE_CONFIRMED`。
2. `config/discovery_strata_v1.yaml`：必须从
   `data_field_dictionary.xlsx` 核定行业来源/字段、企业规模来源/字段，并冻结
   `tech_flag.position_regex`；代码不会自行猜测。

### 1. 源数据、翻译与 A 级冻结

```bash
python -m src.ai_penetration.panel_v2.source_audit \
  --snapshot-id <固定数据库快照/备份ID>

python -m src.ai_penetration.translation_completion \
  --results-dir <170批result目录> \
  --qc-dir <170批QC目录> \
  --input-jsonl <完整中文化输入.jsonl> \
  --batch-manifest <批次清单> \
  --prompt-file <中文化prompt> \
  --schema-file <中文化schema>

python -m src.ai_penetration.zh_alias_freq
python -m src.ai_penetration.zh_alias_activation --apply
python -m src.ai_penetration.freeze_external_dictionary
```

冻结器必须生成三个 §7.6.7 Parquet，translation verifier 必须生成
`external_translation_log_v1.jsonl`；缺任意一件时 v2i 拒绝发布。

### 2. 主样本去重与当前全量 legacy 频数

```bash
python -m src.ai_penetration.panel_v2.dedup

python -m src.ai_penetration.panel_v2.legacy_freq \
  --tag legacy_df_freq_v1
```

`legacy_df_freq_v1.csv` 的 B/C 证据来自全量规范化文本：
`df_unique_text / main_anchor_unique_text / candidate_anchor_cooc / first_year`。
v3 不继承旧 proxy grade。

### 3. 至少两个 Qwen 候选做技术基准，再冻结生产模型

每个候选实际运行前，`config/model_runtime.yaml` 必须指向对应正在服务的模型。

```bash
python -m src.ai_penetration.panel_v2.model_benchmark run \
  --phase technical --candidate-id qwen_candidate_1 \
  --config-file <candidate1.yaml> \
  --sample-file <1000条技术基准.jsonl>

python -m src.ai_penetration.panel_v2.model_benchmark run \
  --phase technical --candidate-id qwen_candidate_2 \
  --config-file <candidate2.yaml> \
  --sample-file <同一1000条技术基准.jsonl>
```

选定候选后，把**该候选同一份配置**写入 `config/model_config_v1.yaml`，
并把 runtime 指向同一模型，再执行：

```bash
python -m src.ai_penetration.panel_v2.model_benchmark select \
  --manifests <candidate1_manifest.json> <candidate2_manifest.json> \
  --selected-candidate <被选candidate_id>

python -m src.ai_penetration.panel_v2.model_benchmark run \
  --phase prerun --candidate-id production \
  --sample-file <10000条预运行.jsonl>
```

### 4. 用冻结 production Qwen 生成 pre-discovery v3，再构建 frame

```bash
python -m src.ai_penetration.panel_v2.lexicon_llm t1
python -m src.ai_penetration.panel_v2.lexicon_llm t2
python -m src.ai_penetration.panel_v2.lexicon_llm merge
```

T1/T2 会强制检查 `model_runtime.yaml` 与冻结的
`model_config_v1.yaml` 是同一个 production 模型，并生成
`skill_legacy_governance_manifest_v3.json`。

随后构建正式 discovery corpus / frame：

```bash
python -m src.ai_penetration.panel_v2.discovery_frame

python -m src.ai_penetration.panel_v2.discovery_formal baseline \
  --frame output/dictionary/discovery_frame_v1.parquet \
  --out output/dictionary/discovery_selected_r0.csv
```

`dictionary_discovery_corpus_v1.parquet` 严格按
`DISTINCT(source_platform,text_hash)` 唯一化，并保存代表记录、原始出现次数、
企业数、年份范围和来源岗位 ID 列表。

### 5. discovery 迭代：抽取 → 全量频数 → 动态概念归一 → 指标

对当前**累计 selected CSV**运行抽取；缓存会跳过已经成功的文本：

```bash
python -m src.ai_penetration.panel_v2.discovery_extract \
  --sample output/dictionary/discovery_selected_r<N>.csv

python -m src.ai_penetration.panel_v2.discovery_review \
  --mentions output/llm_review/formal_discovery_v1/mentions.parquet \
  --prepare-only

python -m src.ai_penetration.panel_v2.legacy_freq \
  --terms-file output/dictionary/formal_discovery_terms_v1.txt \
  --tag formal_discovery_full_freq_v1

python -m src.ai_penetration.panel_v2.discovery_review \
  --mentions output/llm_review/formal_discovery_v1/mentions.parquet \
  --full-freq output/dictionary/formal_discovery_full_freq_v1.csv

python -m src.ai_penetration.panel_v2.discovery_formal metrics \
  --selected output/dictionary/discovery_selected_r<N>.csv \
  --candidate-audit output/dictionary/formal_discovery_candidate_audit_v1.csv \
  --out output/dictionary/discovery_round_metrics.csv
```

如果尚未满足连续两个**增量轮次**新增 B/C 新概念均少于 5，则：

```bash
python -m src.ai_penetration.panel_v2.discovery_formal round \
  --frame output/dictionary/discovery_frame_v1.parquet \
  --selected output/dictionary/discovery_selected_r<N>.csv \
  --round <N+1> \
  --out output/dictionary/discovery_selected_r<N+1>.csv
```

然后重复本节。覆盖率增益只作为监测；如已计算，可用
`discovery_formal metrics --coverage <coverage.csv>` 合并，不参与准入或停止。

饱和后：

```bash
python -m src.ai_penetration.panel_v2.discovery_formal finalize \
  --frame output/dictionary/discovery_frame_v1.parquet \
  --frame-manifest output/dictionary/discovery_frame_manifest_v1.json \
  --selected output/dictionary/discovery_selected_r<N>.csv \
  --metrics output/dictionary/discovery_round_metrics.csv \
  --candidate-audit output/dictionary/formal_discovery_candidate_audit_v1.csv \
  --governance output/dictionary/skill_governed_ABCD_v4.csv
```

### 6. v4 正式匹配与 v2i 发布

```bash
# 默认就是 _handoff_scan；显式写出仅为可读性
python -m src.ai_penetration.panel_v2.scan --out-tag _handoff_scan

python -m src.ai_penetration.panel_v2.v2i --dry-run
python -m src.ai_penetration.panel_v2.v2i \
  --run-id YYYYMMDD_HHMM_v2i
```

`v2i --dry-run` 会验证所有上游 manifest、生产模型一致性、discovery frame、
v4 治理表以及 handoff scan 哈希闭环。正式 release 采用 staging → 质量门 →
manifest → 原子提升；默认拒绝覆盖已有 `panel_v2i`。


## 测试

```bash
pytest src/tests/ -q
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
