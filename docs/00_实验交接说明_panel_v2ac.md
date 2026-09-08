# AI 岗位渗透率面板 v2（广深）实验交接说明

实验版本：`panel_v2a-c`（run_id `20260908_v2ac`，duplicate_version `main_v2a_20260908c`）
日期：2026-08-27 立项，2026-09-08 定稿（Asia/Shanghai）
仓库：`D:\PythonProjects\AIPenetration`（独立项目，不依赖 Employ26；git 全链可回溯）

本文档面向后续研究人员与代码代理，记录最终结果所用方法与最终数字；实验过程与排障记录见 `docs/交接对照检查单.md` 与 git 历史。

---

## 一、当前完成状态

- 技能词典冻结四步全部完成，发布 **A 级正式冻结版 v1.1**（`bilingual_a_frozen_v1.1`）：
  概念 22,683 / 激活别名 110,970（较候选版净激活 zh/mixed 别名 3,503 个）
- 面板 v2 按《岗位AI技能共现率执行指南》§6/§12–§18 全链跑通并过全部质量门：
  **22,957,171 个去重岗位（广深 2014–2024）× 三套锚点 × 三种时间窗口 × 双得分类型**
- 2022 年数据缺陷（blank 描述污染）已完成治理并纳入准入规则
- 全国 392 城扩展：管线就绪，未执行（待运行窗口）

## 二、最终方法（定稿口径）

### 2.1 数据准入与去重（主样本）
1. 岗位级并行扫描（ctid 切片 ≤8 流，HDD 纪律），文本三态清洗（raw/clean/match，NFKC+小写+空白折叠）
2. 准入过滤：描述 trim 后有效长度 ≥10（v2a-c 新增）、position/publish_time 非空、年份 `^\d{4}` 白名单
3. 去重主规则（指南 §6.2.1）：(platform, recruit_id) 唯一记录 → 30 天链式改**组首锚定 30 天桶**保证组内两两 ≤30 天；分组键 =（company_id, 规范化岗位名 hash, city, match 文本 blake2b-63 hash, 自然年）；canonical 选择序 = 完整度→描述长度→日期→rid 字典序；规则1跨城折叠与全部组映射落库可审计

### 2.2 技能识别（union 词表）
- **A 级冻结概念词典**（110,970 激活别名 → 22,683 概念，pyahocorasick 全别名匹配 + ASCII 词边界 + 同形词 AI 语境校验）
- ∪ **自建技术词表**（6,872 词中未被 A 级别名覆盖者，合成 `legacy:<term>` 命名空间；补 ESCO/O*NET 缺失的 pytorch/tensorflow/opencv 等技术工具粒度）
- 岗位内技能去重，别名→概念归一后一词一权重

### 2.3 AI 锚点（三套，指南 §12 逐字）
- `anchor_main` 六组：AI / ML / NLP / CV（计算机视觉+图像识别）/ LLM（大语言模型系，禁裸"大模型"）/ TRANS（Transformer+模型/架构限定，禁裸词）
- `anchor_cn_paper`：AI/ML/NLP/图像识别四组；`anchor_babina`：AI/ML/NLP/计算机视觉四组
- 仅在岗位描述上匹配（岗位名不进任何锚点版本）；命中组以 bitmask 存明细

### 2.4 技能权重（9 组 × 双轨，指南 §13–14）
- 共现率 ωsAI = P(含锚点 | 含技能)，对每个（锚点版本×窗口×年份）单元计算
- 窗口：annual / pooled（年度整数计数精确加总）/ roll3_centered（居中三年滚动，首末年两年窗）
- 平滑：Beta-Binomial 经验贝叶斯，单元内 n≥5 技能 MLE 拟合先验，不收敛回退 Jeffreys(0.5,0.5)；原始权重与平滑权重并存，主结果用 raw
- 低频技能不删除，打 rare_lt10/20/50 标记
- 产物 493,173 行（技能×9 组），拟合成功率 99.86%

### 2.5 岗位得分与判定（指南 §15–16）
- `aijob_<锚点>_<窗口>_<得分类型>_<阈值>`：得分 = 岗位去重技能共现率简单平均，**严格大于**阈值
- 三阈值并存 0.05 / 0.10 / 0.15；主指标 = `aijob_main_annual_raw_005`
- 零技能岗位：保留记录、得分缺失、三标识为 0、`zero_skill_override=1`；技能连接覆盖率=1 硬约束（实测 min=1.0）
- 企业留一（§14.3）：main×三窗口×raw 三列，随发布输出

## 三、最终结果

### 3.1 总体 AI 率（22,957,171 去重岗位，2014–2024 合计）

| 阈值（main annual raw） | AI 率 |
|---|---|
| **>0.05（主指标）** | **10.81%** |
| >0.10 | 3.92% |
| >0.15 | 1.98% |

### 3.2 年度趋势（>0.05，main annual raw）

| 年 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|
| AI率 | 4.2% | 5.9% | 10.1% | 15.3% | 16.0% | **16.6%** | 15.3% | 10.8% | 13.6% |

形态：2019–2021 疫情在线化峰值 → 2023 谷 → 2024 回升；曲线连续。2020 年前样本占比小（早期年份见 3.4），趋势解读建议自 2016 起。

### 3.3 锚点敏感性（>0.05，annual / pooled / roll3）

| 锚点 | annual | pooled | roll3 |
|---|---|---|---|
| main | 10.81% | 10.49% | 10.76% |
| cn_paper | 10.42% | 10.08% | 10.39% |
| babina | 10.63% | 10.31% | 10.60% |

三套锚点差异 <0.4pp；三窗口差异 <0.5pp。窗口间转换矩阵：annual×pooled 双向错判 1.7%/1.5%，annual×roll3 0.7%/0.6%（见 `transition_matrix_main_raw005.json`）。

### 3.4 质量门（§17.6 十项阻断全部通过）

岗位 22,957,171 = 分类表 = flag = master 逐级守恒；技能对 203,028,979；(job_id, skill) 唯一；分子≤分母；权重/得分∈[0,1]；年度加总=pooled；roll3=年度窗口和（独立重算）；覆盖率=1；阈值单调；重复命中经 keep-first 仲裁 72,266 例。
警告（披露不阻断）：6,139 个低频技能原始权重为 0/1（§13.4 预期行为）；2014/2015 样本 <2k 仅作存在性标记。

## 四、使用限制与数据特性

1. **范围**：广东省 21 市中广州市、深圳市（用户决策 2026-09-06）；全国扩展未跑
2. **eps 源数据止于 2024 年 10 月初**（Q4 桶小；2024 年度值略低估，季度/半年度解读需标注）；**无 2025 数据**（与交接指南口径 2014–2025 的差异，待交接方确认）
3. **2022 年**：数据源存在 18.3% blank 描述，v2a-c 已按有效长度 ≥10 准入治理；该年仅保留高质量样本（864k），时序分析建议降权或剔除
4. **口径性质**：本方法测量的是**技能暴露度**（岗位描述含 AI 关联技能），非"核心职责为 AI 开发"的岗位占比。与 v1 fused 面板（0.7% 量级，非去重分母+强证据阈值 avgω≥0.15&maxω≥0.5）**不可直接比较**；文献可比口径建议 >0.15
5. **绝对精度参考**（LLM 盲评，非金标准）：文本法在"核心职责"严格判据下精确率约 53-61%，错误集中于"提到≠从事"型岗位——口径属性，非管线缺陷
6. **A 级词典已知盲区**：「强化学习」「人工智能」标准概念缺失（由 union 词表 legacy 层承接）；别名激活阈值 freq≥100 为接收方裁量（依据已录 QC 报告）

## 五、产出物

### 5.1 发布包（`output/release/panel_v2/`，§18 十三件 + 同名 `.metadata.json` 12 字段）
skill_concept_v1 / skill_alias_v1 / skill_candidate_d_v1 / ai_anchor_dictionary_v1.csv /
job_anchor_flag(22.96M) / job_skill_long(2.03亿) / job_firm / skill_ai_counts(493k) /
skill_ai_relevance(493k) / job_ai_score(4.13亿长表, 3.14GB) / job_ai_classification(宽表含54标识) /
job_ai_score_loo / quality_control_report.md
（治理前版本归档于 `panel_v2_pre_blankfix/`）

### 5.2 结果库（PostgreSQL `ai_pen_results`，与 eps 源库物理隔离）
`job_master_gzsz`（22.96M，duplicate_group_id/records_collapsed/platform_count 等 §6.3 字段齐备）、
`dup_group_map_gzsz`（37.2M 组↔原始行全映射）、v1 遗留三面板表不受影响

### 5.3 词典冻结件（`output/dictionary/`）
`skill_concept_bilingual_a_frozen_v1.1.csv`（22,683，SHA-256 见 QC）
`skill_alias_active_bilingual_a_frozen_v1.1.csv`（110,970，同上）
QC 报告与激活 manifest 在 `output/reports/freeze_qc_*` / `alias_activation_manifest_*`

### 5.4 代码（均带单测，`python -m pytest src/tests -q` 全绿）
`src/ai_penetration/panel_v2/`：anchors / lexicon / dedup / scan / counts / relevance / scoring / quality / export_release
`src/ai_penetration/text_clean.py`、`zh_alias_freq.py`（去重口径）、`zh_alias_activation.py`、`freeze_external_dictionary.py`、`gold_standard_eval.py`、`compare_atier_gz2024.py`
配置：`config/database.yaml`（源库/结果库）、`config/panel_v2.yaml`（全部口径参数）

## 六、复现路径

```bash
# 前提：config/database.yaml 配好 eps 连接（或 AIPEN_PG_* 环境变量）
# 词典冻结链（结果落 ai_dict schema，eps 写入唯一合法目标）
python -X utf8 -m src.ai_penetration.zh_alias_freq --force            # ~40min
python -m src.ai_penetration.zh_alias_activation --min-freq 100 --apply
python -m src.ai_penetration.freeze_external_dictionary

# 面板 v2 全链（串行，任一守恒失败即停）
python -X utf8 -m src.ai_penetration.panel_v2.dedup  --workers 8 --slices 4   # ~3h
python -X utf8 -m src.ai_penetration.panel_v2.scan   --workers 8 --slices 4   # ~1.5h
python -X utf8 -m src.ai_penetration.panel_v2.counts
python -X utf8 -m src.ai_penetration.panel_v2.relevance
python -X utf8 -m src.ai_penetration.panel_v2.scoring
python -X utf8 -m src.ai_penetration.panel_v2.quality
python -X utf8 -m src.ai_penetration.panel_v2.export_release --run-id <id> --force-release
```
断点：dedup/scan 均按切片落盘可续跑；eps 全程只读；所有写库仅发生在 ai_dict（词典治理）与 ai_pen_results（结果）。

## 七、与 exchange 交接包的工作分界

- **已完成**：交接 §2 冻结前置四步、§6 清洗去重、§12–§18 指标管线（广深范围、字典发现语料含 platform×text_hash 口径）
- **未做（阶段二遗留）**：全国 392 城面板、企业留一面板级汇总、B/C/D 级中文技能发现（§8–§10 Qwen 合规化，现有 gold_standard_eval 缺口挖掘为其雏形）、§18 之外的滞后信息窗口口径
- **待反馈交接方**：2025 数据不可得、eps 2022 年 blank 描述缺陷、A 级词典「强化学习/人工智能」概念缺失、激活阈值 100 的确认
- **交接包纪律遵守**：170 批翻译未重跑；冻结版词典不回写候选版文件；版本命名无语义后缀（§4.1）

## 八、稳健性与归因附录（2026-09-09 补强，均见 output/reports/ 同名文件）

### 8.1 口径归因矩阵（2024 同抽样四格，`caliber_matrix_*.md`）
v1 规则×全量 0.712%（与 v1 面板发布值自洽）→ v1 规则×去重 0.820%（去重效应 +15%）
→ v2>0.15×去重 2.34%（判据+词表 ×2.85）→ v2>0.05 13.58%（阈值 ×5.81）。
**结论：与 v1 的 ~19 倍差距中，去重+词表贡献约 ×3.3，其余 ×5.8 纯为阈值放宽。**

### 8.2 金标准分层判定（`gold_standard_verdict_2024.md`，192 份逐条人工判读）
>0.30 带真值率 **92%**；>0.05 对 v1 的增量带真值率仅 **4%**（同名概念暴露噪声）；
0.03-0.05 带 0%（cutoff 无召回损失）；负类带 0% 假阳性。
**主指标建议 >0.15（净 AI 岗占比 2024≈1.6-2.1%，Babina 同数量级）；
>0.05 更名"AI 技能暴露率"使用。**

### 8.3 平台分层与构成调整（`composition_adjusted_curve.csv`）
平台样本构成年度波动 2-3 个数量级（小样本平台-年有强选择偏差）；
固定权重调整后曲线形态不变（2019-21 峰 13.8-15.6% → 2023 谷 9.3% → 2024 14.0%）。
2014/2015（<600 行）与 **2022 建议时序分析剔除**——2022 缺陷定位为
**51Job 单平台 blank 事故（当年 blank 率 93.7%）**，治理后该年仍缺此平台。

### 8.4 企业留一（§14.3 面板级，发布列 `job_ai_score_loo.parquet`）
annual 主指标判定翻转仅 **0.082%**（10.79% vs 10.81%，暴露度非企业模板自推）；
pooled 翻转 3.4%（全期权重下企业贡献占比高，解读以 annual 为准）。

### 8.5 词典质检
36 个规范化别名碰撞键复核（12 同名变体无害；24 真歧义缩写 min(sid) 确定性裁决，
低频长尾影响极小）；**发现疑似源词典 bug：别名"gps"的规范概念为"地理信息系统"
（应为 GIS）**——与"强化学习/人工智能缺失、2025 数据不可得、51Job 2022 blank"
并列反馈数据提供方。

## 九、遗留事项（优先级序）

1. 主指标切换 >0.15 的发布重算（改 panel_v2.yaml 阈值语义标注，分类表已含三列，零重跑）
2. 全国 392 城 v2 重跑（约 48h HDD 窗口，代码零改动）
3. 0.15-0.30 得分带补一批金标准样本（当前外推 70-90%，实测更稳）
4. B/C 级技能发现按 §8 合规化（扩词典演进通道）
5. S4 型"AI 数据管线岗"（标注/测试）的归类决策（纳入或显式排除，影响主指标 ±0.3pp）
