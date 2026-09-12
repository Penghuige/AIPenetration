# 岗位 AI 技能共现率执行指南

版本：v1.5  
日期：2026-08-23  
项目目录：`D:\ai渗透率`

## 1. 指南目的与研究边界

本指南用于指导2014—2025年招聘广告的技能词典构建和岗位 AI 属性测算。完整测量链条为：

```text
招聘岗位描述
    → 标准化技能概念词典
    → 岗位—技能匹配结果
    → 技能 AI 共现率
    → 岗位 AI 相关度
    → 是否为 AI 相关岗位
```

本阶段不计算企业—年度 AI 招聘指标，也不进行企业层面的聚合。

本项目已确定：

1. 招聘数据覆盖2014—2025年。
2. 计算前需要对招聘广告去重。
3. 技能识别和 AI 锚点识别均基于岗位描述。
4. 主锚点包含 AI、ML、NLP、Computer Vision、LLM、Transformer 的中英文表达。
5. 技能 AI 共现率分别按年度、2014—2025年全期和以当年为中心的3年滚动窗口计算；样本首尾年份采用相邻两年窗口。
6. 岗位 AI 相关度采用岗位全部已识别标准技能的 AI 共现率简单平均值。
7. 在连续岗位 AI 相关度基础上识别 AI 相关岗位。
8. A、B、C级技能全部进入正式词典；D级技能保留在候选库。
9. 不强制建设人工标注金标准。
10. 不使用Qwen对歧义词逐条进行语境判定，允许可记录、可检验的测量误差存在。
11. 本地大模型的主要测试和正式推理设备调整为 RTX 4090 24GB。
12. 企业留一法仅作为稳健性检验，不作为主指标的必需步骤。

---

## 2. 核心概念与计算公式

### 2.1 基本分析单位

基本分析单位是一条去重后的招聘广告。每条招聘广告由唯一的 `job_id` 标识。

同一项技能在同一条招聘广告中无论出现多少次，均只计为出现一次。因此，后续统计基于岗位—技能二元关联，而不是词频。

### 2.2 技能 AI 共现率

令：

- `j` 表示招聘岗位；
- `s` 表示标准化技能概念；
- `t` 表示年份；
- `S_j` 表示岗位 `j` 中识别出的标准技能集合；
- `A_j^(k)` 表示在锚点口径 `k` 下，岗位描述是否出现至少一个 AI 锚点。

年度技能 AI 共现率定义为：

$$
w_{s,t}^{AI,k}
=
\frac{\sum_{j\in t} I(s\in S_j)A_j^{(k)}}
{\sum_{j\in t} I(s\in S_j)}
$$

其中，分母是年份 `t` 中包含技能 `s` 的唯一岗位数，分子是其中同时包含至少一个 AI 锚点的唯一岗位数。

### 2.3 岗位 AI 相关度

岗位 `j` 在年份 `t` 和锚点口径 `k` 下的 AI 相关度定义为：

$$
AIscore_{j,t}^{k}
=
\frac{1}{|S_j|}
\sum_{s\in S_j} w_{s,t}^{AI,k}
$$

主指标对岗位全部已识别的标准技能取简单平均。每个 `skill_id` 在同一岗位中只能进入平均值一次。

### 2.4 AI 相关岗位

主口径定义为：

$$
AIjob_{j,t}^{k,0.05}
=
I(AIscore_{j,t}^{k}>0.05)
$$

注意使用严格大于号。同步保留阈值0.10、0.15和不设阈值的连续指标。

### 2.5 主指标与比较指标

为避免九组共现率和两类权重混用，预先固定以下层级：

1. 主技能权重：`anchor_main + annual + raw`。
2. 主岗位得分：`anchor_main + annual + raw + mean_all_skills`。
3. 主AI岗位标识：主岗位得分严格大于0.05。
4. 时间口径比较：`pooled`和`roll3_centered`。
5. 锚点口径比较：`cn_paper`和`babina`。
6. 低频稳健性：`smoothed`。
7. 阈值稳健性：0.10和0.15。
8. 机械相关稳健性：企业留一权重。

所有比较指标均应保留，但不得与主指标使用相同字段名称。

---

## 3. 预期输入数据

原始招聘数据至少应包含：

| 字段 | 必需性 | 说明 |
|---|---:|---|
| `job_id_raw` | 必需 | 原始岗位编号；如果没有，需要重新生成 |
| `posting_date` | 必需 | 招聘发布日期 |
| `year` | 必需 | 从发布日期提取，限定2014—2025年 |
| `job_description` | 必需 | 本项目技能和锚点识别使用的岗位描述 |
| `job_title` | 建议 | 仅用于去重、审计和结果描述，不进入主锚点识别 |
| `company_id` | 建议 | 用于去重和企业留一稳健性检验 |
| `company_name` | 建议 | 企业识别和数据审计 |
| `city` | 建议 | 辅助去重和异质性审计 |
| `industry` | 建议 | 分层抽样和自动质量检验 |
| `source_platform` | 建议 | 识别跨平台重复岗位 |

如果缺少企业、岗位名称或城市字段，需要在去重阶段记录去重能力下降的限制。

### 3.1 字段类型与缺失状态

1. 日期统一转换为ISO格式 `YYYY-MM-DD`；无法解析的日期进入隔离表，不参与按年测算。
2. 年份必须从清洗后的日期重新生成，不直接信任原始年份字段。
3. 所有文本使用UTF-8保存；原始编码及转换失败记录单独保留。
4. 原始空值、空字符串、只含HTML或空白的文本统一标记为 `description_missing=1`。
5. 清洗后少于20个可见字符的文本标记为 `description_too_short=1`，默认保留但不直接删除。
6. 超长岗位描述不在确定性词典匹配阶段截断；只在Qwen候选发现阶段分块。
7. 原始主键永不覆盖；内部 `job_id`单独生成并在全流程保持不变。

### 3.2 内部岗位编号

优先使用以下方式生成稳定 `job_id`：

```text
job_id = SHA256(source_platform | job_id_raw)
```

如果没有可靠的原始岗位编号，则使用：

```text
job_id = SHA256(company_key | job_title_normalized | city_normalized |
                posting_date | text_hash)
```

用于哈希的字段和连接顺序必须写入配置文件。哈希前使用固定分隔符，缺失值使用显式占位符，避免不同字段组合产生相同字符串。

---

## 4. 项目文件与版本结构

最终结果文件和本指南保存在原始文献所在的 `D:\ai渗透率` 项目目录。建议在项目目录内设置以下工作子目录：

```text
D:\ai渗透率\
├─ raw_data\                 原始招聘数据，只读保存
├─ external_dictionary\     ESCO、O*NET、Lightcast原始快照
├─ config\                  指标口径、锚点、阈值和模型配置
├─ prompts\                 Qwen提示词及JSON Schema
├─ intermediate\            清洗、候选词和岗位技能中间表
├─ outputs\                 最终技能词典及岗位层结果
├─ logs\                    数据处理、模型推理和异常日志
└─ 岗位AI技能共现率执行指南.md
```

工作过程中使用子目录管理中间文件。形成正式交付版本时，将最终词典、技能AI共现率、岗位AI得分、岗位AI分类和质量报告同步复制到 `D:\ai渗透率` 根目录，并在文件名中包含版本号；根目录中的原始文献不移动、不覆盖。

任何原始词典、提示词、模型或匹配规则发生变化，都必须生成新版本，不允许覆盖旧版本。

### 4.1 文件格式

1. 百万行以上中间表和结果表使用Parquet，压缩方式默认Zstandard。
2. 需要人工查看或与其他软件交换的小表使用UTF-8 CSV。
3. 不用Excel保存岗位—技能长表或模型逐条输出。
4. 所有Parquet文件必须保存字段类型，禁止依赖自动类型推断重新读取。
5. 文件名不得使用 `final`、`new`、`latest` 等不稳定后缀，统一采用语义版本或运行编号。

### 4.2 运行编号与清单

每次正式运行生成唯一 `run_id`，格式建议为：

```text
YYYYMMDD_HHMM_<dictionary_version>_<config_version>
```

每次运行生成 `run_manifest.json`，至少记录：

```text
run_id
started_at
finished_at
git_commit_or_code_hash
input_file_paths
input_file_sha256
input_row_counts
dictionary_version
anchor_version
prompt_version
model_revision
config_file_sha256
output_file_paths
output_file_sha256
output_row_counts
status
```

### 4.3 推荐技术栈

- 数据清洗和分组统计：Python、Polars或DuckDB；
- 列式存储：PyArrow和Parquet；
- 确定性多模式匹配：Aho-Corasick或等价Trie实现；
- 近似去重：字符n-gram、MinHash或SimHash；
- 本地模型推理：能够固定模型revision并支持JSON Schema约束的推理框架；
- 配置文件：YAML；
- 运行入口：PowerShell 7脚本或Python命令行入口。

---

## 5. 阶段一：招聘数据审计

### 5.1 目标

确认2014—2025年招聘数据是否满足文本识别和岗位共现计算需要。

### 5.2 执行内容

1. 统计每年岗位数量。
2. 统计 `job_description` 的缺失率、空文本率和长度分布。
3. 检查年份是否超出2014—2025年。
4. 检查原始岗位编号是否唯一。
5. 统计企业名称、岗位名称、城市和行业字段的缺失率。
6. 识别HTML标签、乱码、异常编码和截断文本。
7. 统计完全相同岗位描述的数量和比例。
8. 检查同一企业、岗位名称和城市下的重复发布情况。
9. 统计不同招聘平台之间的重复文本。
10. 记录各年份的文本结构是否发生明显变化。

### 5.3 输出

- `data_audit_report.html`或`data_audit_report.md`；
- `data_field_dictionary.xlsx`；
- `year_job_counts.csv`；
- `duplicate_diagnostics.csv`。

### 5.4 阶段检查点

只有在年份、岗位描述和去重字段达到可用状态后，才能进入文本清洗阶段。

以下项目属于阻断性错误：

- 无法确定岗位年份；
- 岗位描述字段整体缺失或读取失败；
- 无法生成稳定岗位编号；
- 同一 `job_id`对应多条内容不同且无法解释的记录；
- 2014—2025年各年度文件口径明显不同但没有来源说明。

以下项目属于警告而非阻断：

- 少量岗位描述过短；
- 部分企业、城市或行业字段缺失；
- 某些年份重复率显著较高；
- 岗位描述长度随年份变化。

审计报告必须明确区分阻断性错误、警告和普通描述统计。

---

## 6. 阶段二：岗位描述清洗与招聘广告去重

### 6.1 文本清洗原则

清洗目标是删除展示性噪声，同时保留技术技能所需的全部字符信息。

必须保留：

- 英文字母和大小写原始信息；
- 数字和版本号；
- `C++`、`C#`、`.NET`、`Node.js`等特殊符号；
- 中英文括号、连字符、斜线和小数点；
- AI、ML、NLP、LLM和Transformer等关键词。

主要处理：

1. HTML实体和标签还原；
2. Unicode规范化；
3. 全角英数字转半角；
4. 连续空白合并；
5. 重复标点压缩；
6. 平台统一模板和无关页脚删除；
7. 保留清洗前文本、清洗后文本和文本哈希。

### 6.1.1 保留三种文本

不得只保留一个覆盖式清洗字段。至少保存：

```text
job_description_raw       原始文本
job_description_clean     去HTML、控制字符和模板后的可读文本
job_description_match     用于词典与锚点匹配的规范化文本
```

`job_description_match`在 `job_description_clean`基础上进行Unicode NFKC规范化、全角英数字转半角和空白统一。英文大小写在匹配器中处理，不直接覆盖可读文本。

### 6.1.2 禁止的清洗操作

不得执行会破坏技能名称的处理，包括：

- 删除全部标点；
- 删除加号、井号和小数点；
- 将所有数字删除；
- 对英文技能做无记录的词干化；
- 使用中文分词结果替代原文；
- 将 `C++`、`C#`、`.NET`、`R`、`Go`等短技能直接过滤；
- 删除括号中的英文全称或缩写。

### 6.2 去重层次

按照从严格到宽松的顺序进行：

1. 原始岗位编号完全重复；
2. 同一企业、岗位名称、城市和岗位描述完全相同；
3. 同一企业、岗位名称、城市下短期重复发布；
4. 跨平台岗位描述完全相同。

同一企业在相隔较长时间后重新发布相同岗位，可能代表新的招聘需求，不应默认跨年度删除。因此，近似重复去重应设置时间窗口，并保留原始记录数和最终保留规则。

### 6.2.0 词典发现文本唯一化

词典候选发现需要尽量覆盖不同的岗位描述表达，因此单独建立文本唯一化语料：

```text
dictionary_discovery_corpus = DISTINCT(source_platform, text_hash)
```

在同一招聘平台内，只要规范化岗位描述 `text_hash`完全相同，就合并为一条词典发现文本，不要求同一企业、岗位名称、城市或发布日期。每组保留一个代表文本，并保存原始出现次数、企业数、年份范围和岗位编号列表。

该规则确定用于：

- 每年按行业抽取1万条基准招聘文本；
- 后续每轮1万条候选发现文本；
- 计算B/C分级使用的不同描述频数 `df_unique_description`；
- 避免Qwen重复处理完全相同的岗位描述。

词典发现文本唯一化与技能AI共现率的招聘广告测量样本是两个不同对象。已经确认：全局相同描述合并只用于词典发现语料；技能AI共现率和岗位AI得分继续使用按下节主规则得到的真实去重招聘广告。因此，共现率表示去重招聘广告中的AI共现比例，而不是不同文本模板中的共现比例。

### 6.2.1 推荐默认去重规则

正式去重分为主规则和近似去重稳健性规则。这里的主规则作用于计算技能AI共现率和岗位AI得分的招聘广告分析主样本，不是Qwen候选发现样本或技能词典主样本。

主规则：

1. 相同平台和相同原始岗位编号只保留一个规范记录。
2. 同一企业、同一规范化岗位名称、同一城市、岗位描述哈希完全相同，且发布日期相差不超过30天，归为同一重复组。
3. 跨平台记录只有在企业、岗位名称、城市和岗位描述哈希均一致，且发布日期相差不超过30天时才合并。
4. 不在不同企业之间仅凭相同岗位描述去重，避免删除模板化但真实存在的招聘需求。
5. 默认不跨自然年度合并重复组；跨年重复另行标记。

近似去重稳健性规则：

1. 仅在同一企业、岗位名称和城市内部比较。
2. 使用清洗后文本的字符5-gram集合。
3. Jaccard相似度不低于0.95且发布日期相差不超过30天时视为近似重复。
4. 近似去重结果不覆盖主结果，应形成独立去重版本用于敏感性比较。

### 6.2.2 重复组保留记录

重复组的规范记录按照以下顺序选择：

1. 岗位描述非缺失；
2. 字段完整度最高；
3. 岗位描述有效字符数最长；
4. 发布日期最早；
5. 原始岗位编号字典序最小。

同时保留重复组的最早日期、最晚日期、原始记录数、平台数和全部原始岗位编号映射。不得物理删除原始记录。

### 6.3 去重结果字段

```text
job_id
job_id_raw
year
job_description_raw
job_description_clean
job_description_match
text_hash
duplicate_group_id
duplicate_reason
records_collapsed
duplicate_version
```

### 6.4 输出

`job_text_clean.parquet`

---

## 7. 阶段三：构建基础技能概念词典

### 7.1 数据源

基础词典依次整合：

1. ESCO技能概念、别名、层级和职业—技能关系；
2. O*NET Software Skills及相关技能字段；
3. Lightcast Skills Taxonomy，如果API申请获批；
4. 已有中英文AI技术、软件、工具和方法清单；
5. 后续由Qwen从中国招聘语料中发现的本地化技能。

Lightcast不是项目实施的必要前提。如果API未获批，先使用ESCO、O*NET和招聘语料构建词典；如果后续获批，将Lightcast作为新版本增量来源。

项目中已有 `D:\ai渗透率\onet Skills.xlsx`。只读审计显示，该文件含1个工作表、62,580条数据记录、894个O*NET职业、35个唯一技能要素，并分别记录重要性（`IM`）和水平（`LV`）量表。它是职业—通用技能评分表，不是包含数万项软件名称的技术技能词典。因此：

1. 仅将35个唯一 `(Element ID, Element Name)`导入为A级通用技能概念；
2. 将其余行保留为O*NET职业—技能—量表关系表，不重复生成技能概念；
3. `Data Value`、样本量、标准误和置信区间不作为招聘文本匹配别名；
4. `Recommend Suppress=Y`的评分关系默认不用于外部合理性检验，但不删除原始行；
5. 为扩充软件、平台和具体技术名称，仍需另行取得O*NET当前发布的 `Software Skills`文件。旧版资料中的 `Technology Skills`在新版本中已改名，实施时按下载版本的实际文件名登记；
6. 不把该工作簿62,580条关系记录误报为62,580项技能；
7. 当前文件大小为3,404,940字节，SHA-256为 `C7AD8C4FD9D541FC40A80627C93155C9419C7BE1BB3721626DE8C2E0BE8D05F3`；将其写入来源日志；
8. 工作表中的 `Date`是职业评分数据日期，不能替代O*NET数据库发布版本。若文件名或下载包未提供release编号，`source_version`暂记为 `UNKNOWN_LOCAL_SNAPSHOT`，待从原下载页面或压缩包元数据核定，不根据文件修改时间猜测版本。

### 7.2 概念表

```text
skill_id
canonical_zh
canonical_en
skill_type
skill_category
definition
source
source_version
source_id
valid_from
valid_to
confidence_tier
dictionary_version
```

### 7.3 别名表

```text
alias_id
skill_id
alias
alias_normalized
language
source
matching_rule
ambiguity_flag
confidence_tier
dictionary_version
```

### 7.4 基本原则

1. 标准技能概念和表面别名必须分表保存。
2. 一个新字符串不等于一个新技能概念。
3. 同一技能的中英文、简称、大小写和拼写变体共享同一个 `skill_id`。
4. 不以最终词典词条数量作为质量目标。
5. 外部词典原始文件、版本、下载日期和校验值必须保存。

### 7.4.1 来源映射与概念合并

外部来源中的每个记录先进入 `source_skill_record`，再映射到内部概念，禁止在导入时直接删除重复项。

```text
source_name
source_version
source_skill_id
source_label
source_description
source_category
internal_skill_id
mapping_type
mapping_evidence
```

概念合并遵循保守原则：只有在技能含义基本相同、可互为别名时才合并。上下位概念、软件与其使用能力、方法与应用场景不应因名称相近而自动合并。

### 7.4.2 稳定技能编号

A级来源技能建议使用UUIDv5生成稳定内部编号：

```text
UUIDv5(project_namespace, source_name | source_version | source_skill_id)
```

多个来源合并为一个概念后保留首个内部编号，其余来源通过映射表关联。B/C级新概念使用固定规范名称、首次发现年份和内部命名空间生成UUIDv5。标准名称修改不得改变既有 `skill_id`。

### 7.4.3 技能类型枚举

`skill_type`使用受控枚举，至少包括：

```text
programming_language
method_algorithm
software_tool
platform_framework_library
data_database
hardware_equipment
domain_knowledge
business_management
general_work_skill
soft_skill
other_skill
```

学历、专业、工作年限、岗位名称、证书名称、福利、工作任务和企业名称不作为技能概念；如研究需要可单独建表，但不进入本项目岗位技能集合。

### 7.4.4 别名激活状态

正式概念进入词典不等于其所有别名都自动参与匹配。别名表增加：

```text
is_active
primary_skill_id
boundary_rule
case_sensitive
```

只有 `is_active=1` 的别名进入正式匹配器。无法建立固定主含义的高歧义别名可以保留在正式词典中，但设置 `is_active=0`，从而避免一条别名同时映射到多个技能概念。

### 7.5 输出

- `skill_concept_base_v1.parquet`；
- `skill_alias_base_v1.parquet`；
- `dictionary_source_log_v1.csv`。

### 7.6 外部英文技能库的Codex中文化流程

ESCO、O*NET及可选Lightcast的英文记录不能直接生成中文匹配词典。本项目由Codex作为一次性词典建设的执行主体，负责读取本地已下载文件或从官方来源下载并固化缺失文件、编写解析脚本、生成中英文标准技能概念与激活别名、完成跨来源合并及自动审计。Qwen不承担外部词典翻译；Qwen仅用于后续中国招聘语料中的候选技能发现和候选到现有概念的辅助映射。

这里的Codex中文化是可审计的文件生产流程，不是让模型直接覆盖正式词典。所有批次必须先写入候选结果，经确定性检查后再生成新词典版本。

#### 7.6.0 本地文件优先与官方版本固化

Codex执行以下工作：

1. 优先检查项目目录已有文件；当前O*NET通用技能输入固定为 `D:\ai渗透率\onet Skills.xlsx`，不得重复下载或覆盖；
2. 对缺失的ESCO和O*NET Software Skills，从官方发布渠道下载可机器读取的原始文件；Lightcast仅在获得合法API权限后接入；
3. 保存原始文件、许可或使用条款快照、下载URL、下载日期、来源版本和SHA-256校验值；本地已有文件同时记录文件大小和最后修改时间；
4. 只解析来源中实际存在的技能、知识、工具或软件字段，不把职业名称、任务描述和工作活动直接当作技能；
5. 保留每个来源的原始ID、层级、定义、别名和关联职业，不在原始层删除重复项；
6. 下载或解析失败时停止该来源的导入，不以搜索结果摘要或第三方转载文件替代正式版本。

#### 7.6.1 建立翻译输入记录

每项外部技能必须提供：

```text
source_name
source_version
source_skill_id
preferred_label_en
alternative_labels_en
description_en
source_category
parent_concepts
related_occupations
```

翻译不能只输入英文技能名称。优先同时提供定义、类别、上位概念和职业使用场景，以降低一词多义造成的错误。

#### 7.6.2 确定性预处理

1. Unicode和空白规范化；
2. 保留原始大小写、标点和产品名称；
3. 完全相同的来源记录只翻译一次；
4. 软件、框架、产品、编程语言和标准缩写默认保留英文原名；
5. 英文原名始终作为正式激活别名保存，不因生成中文译名而删除。

#### 7.6.3 Codex语境化中文化

Codex按固定批次读取7.6.1的完整输入记录，并按照固定JSON Schema输出：

```text
source_skill_id
canonical_zh
alternative_labels_zh
keep_english_label
translation_type
translation_evidence
ambiguity_flag
```

`translation_type`使用：

```text
semantic_translation
proper_name_keep_en
proper_name_with_zh_alias
abbreviation_keep
ambiguous_translation
```

中文化要求：

1. 结合技能定义、类别和使用场景确定中文含义；
2. 不把上下位技能互相替代；
3. 不将工作任务扩写为技能；
4. 不翻译业内普遍直接使用的产品名、框架名和缩写；
5. 中文译名应适合在招聘文本中直接匹配，而不是解释性长句；
6. 如果存在两个不能合并的含义，设置 `ambiguity_flag=1`并分别建立候选映射，不强制合并；
7. 每批输入记录按 `source_name + source_skill_id`稳定排序，批次编号、输入行数和输出行数固定；
8. 每批完成后立即保存JSONL原始输出和解析后的Parquet，不依赖单次对话上下文作为唯一记录；
9. Codex生成的是中文化候选，必须通过7.6.6的确定性检查后才能进入正式词典。

#### 7.6.4 中文别名生成

中文别名包括：

- 规范译名；
- 常见简称；
- 中英文混合表达；
- 招聘市场中稳定使用的同义表达；
- 简繁体及全半角规范化变体。

Codex生成的别名必须经过全体招聘语料文档频数检查。完全没有命中的生成别名保留但设置 `is_active=0`；能够命中的别名才进入后续概念映射和歧义检查。外部来源原始英文标签不受中文命中频数限制，仍作为A级来源别名保留，但是否激活须服从歧义规则。

#### 7.6.5 跨来源概念合并

先进行规范化精确匹配，再进行固定Top-10候选检索和Codex语义判断。Codex只能返回一个现有 `skill_id`、`NEW_CONCEPT`或`AMBIGUOUS`。只有判定为同一技能概念时才合并 `skill_id`；主题相关、上下位关系和共同使用关系不得合并。

来源优先级不用于判断技能真伪，只用于发生字段冲突时选择主标签和定义。所有来源标签和ID均保留在 `source_skill_record`映射表中。

#### 7.6.6 中文化自动检查

1. `canonical_zh`不能为空；
2. 中文规范名不能是完整解释句；
3. 产品和框架的英文原名必须保留；
4. 同一中文别名映射多个概念时必须进入歧义别名流程；
5. 翻译前后来源ID和记录数必须一致；
6. 所有中文化结果保存Codex任务或运行标识、可获得的模型名称、提示词版本、批次编号、输入输出哈希和原始输出；如果应用未暴露精确模型revision，不得虚构该字段；
7. 中文化不会改变外部来源技能的A级身份，但只有激活别名参与岗位匹配。

此外必须检查：每个输入来源ID恰好对应一条中文化状态记录；批次之间无重复或遗漏；同一输入和固定提示词的重跑差异必须写入差异表，不静默覆盖。

#### 7.6.7 中文化输出

- `external_skill_translation_v1.parquet`；
- `external_skill_alias_zh_v1.parquet`；
- `external_skill_ambiguous_v1.parquet`；
- `external_translation_log_v1.jsonl`。

中文化完成后冻结上述文件。后续指标计算只读取已冻结的中英文概念表和激活别名表，不在全量招聘匹配阶段再次调用Codex或Qwen翻译外部词典。

---

## 8. 阶段四：RTX 4090上的Qwen模型测试

### 8.1 模型用途

Qwen用于：

- 从分层招聘样本中发现技能表面形式；
- 判断候选表述是否属于职业技能；
- 将候选表述映射到现有标准技能；
- 为真正的新技能建议标准名称和类型；
- 辅助合并中英文别名和缩写。

Qwen不用于：

- 逐条判断正式词典中歧义词在全量岗位中的语境；
- 直接生成技能AI共现率；
- 直接决定岗位是否属于AI岗位；
- 替代确定性全量词典匹配。

### 8.2 测试设备

正式候选模型在RTX 4090 24GB电脑上测试。测试应覆盖至少两个模型规模或量化版本，但正式生产只能固定一个模型版本。

### 8.3 自动测试指标

在不建设人工金标准的情况下，至少比较：

1. JSON解析成功率；
2. 原文证据片段可回填率；
3. 字符起止位置有效率；
4. 同一文本重复运行的一致性；
5. 与基础词典匹配结果的重合率；
6. 每千条岗位产生的候选技能数量；
7. 新候选技能的重复率；
8. 单位岗位推理时间；
9. 显存峰值；
10. 失败、超时和重试比例。

### 8.4 固定生产环境

确定正式模型后，必须固定：

- 模型仓库和revision/commit；
- 量化版本；
- tokenizer版本；
- 推理框架及版本；
- CUDA和PyTorch版本；
- 提示词版本；
- JSON Schema版本；
- 解码参数；
- 最大输入和输出长度；
- 随机种子；
- 文本截断规则。

### 8.5 输出

- `model_benchmark_report.md`；
- `model_config_v1.yaml`；
- `skill_extraction_prompt_v1.md`；
- `skill_extraction_schema_v1.json`。

### 8.6 推荐推理默认值

模型最终选择前，基准测试统一采用：

```yaml
do_sample: false
temperature: 0
top_p: 1.0
seed: 20260822
max_model_len: 8192
max_input_tokens: 6000
max_output_tokens: 2000
guided_json: true
retry_invalid_json: 1
```

`temperature=0`仍不能替代保存模型版本、推理框架和原始输出。正式运行前应先进行1000条技术基准，再进行1万条预运行；两阶段均通过后才启动候选发现。

### 8.7 长文本分块

岗位描述超过输入上限时按句子或项目符号边界分块：

1. 每块目标不超过5000输入tokens；
2. 相邻块保留约200个中文字符或等价token重叠；
3. 保存 `chunk_id`、原文起止位置和基准偏移；
4. 模型返回的局部字符位置必须换算为岗位全文字符位置；
5. 合并分块结果时按全文跨度和规范技能去重；
6. 不允许仅保留岗位描述前部而静默截断后部。

### 8.8 缓存与断点恢复

缓存键定义为：

```text
SHA256(text_hash | model_revision | prompt_version |
       schema_version | inference_config_hash)
```

每条输入记录保存 `pending`、`success`、`empty`、`schema_failed`、`timeout`或`runtime_failed`状态。失败样本不得从分母中静默消失。正式批处理应支持按缓存键断点续跑。

### 8.9 技能抽取提示词最低要求

正式提示词至少包含以下约束：

```text
任务：仅抽取岗位描述中明确出现的职业技能。

技能包括：编程语言、算法与方法、软件工具、平台框架、数据库、
硬件设备、专业知识、管理技能、通用工作技能和软技能。

不得抽取：岗位名称、学历、专业名称、工作年限、公司名称、福利、
证书名称、纯工作任务，以及原文没有出现但可由常识推断的技能。

每个surface必须是原文中的连续子串；start为包含式起点，end为
不包含式终点，并满足 text[start:end] == surface。

同一技能在同一岗位重复出现时只输出一次；无法确认时不要补写。
只输出符合给定JSON Schema的结果。
```

岗位原文中的任务短语如果同时是业内稳定技能名称，例如数据建模、需求分析或项目管理，可以作为技能；模型必须输出原文短语，后续由概念归一化阶段决定是否保留。

### 8.10 技术基准通过条件

以下指标作为默认最低要求：

- JSON Schema有效率不低于99.5%；
- `text[start:end] == surface`的跨度有效率不低于99%；
- 不在原文中的生成候选比例不高于1%；
- 相同配置重复运行1000条文本时，岗位级技能集合完全一致率不低于95%；
- 失败与超时合计比例不高于1%；
- 所有失败类型均有可重跑日志。

这些是技术一致性门槛，不代表真实语义Precision或Recall。

### 8.11 字符位置运行时回退方案

正式测试阶段先保留Qwen输出 `start/end`的方案。如果1000条技术基准无法达到99%的跨度有效率，不改变技能抽取的研究口径，改用以下工程回退：

1. Qwen只负责输出 `surface`和`evidence`；
2. 程序在模型实际接收的规范化文本块中精确搜索 `surface`；
3. 只有一个命中位置时直接生成 `start/end`；
4. 多个命中位置时优先选择落在 `evidence`句中的位置；
5. 仍无法唯一定位时保留所有候选位置并选择首次位置作为主位置，同时设置 `multiple_span_flag=1`；
6. 使用分块基准偏移换算为岗位全文字符位置；
7. `surface`无法在模型输入文本中找到时才判为跨度失败。

模型直接返回字符位置和程序确定性定位属于同一技能抽取口径，只是实现方式不同，因此可在实际模型测试后选择。

---

## 9. 阶段五：分层抽样和Qwen候选技能发现

### 9.1 抽样原则

不能只对全部年份进行简单随机抽样。候选发现样本应按以下维度分层：

- 年份；
- 行业；
- 岗位名称或职业类别；
- 企业规模；
- 文本长度；
- 技术岗位和非技术岗位；
- 招聘平台。

行业分层直接使用招聘数据自带的行业分类，不另行训练或推断行业。行业字段只做空白、大小写、全半角和已知编码映射等确定性清洗。缺失行业统一进入 `INDUSTRY_MISSING`层，不因行业缺失删除岗位。

### 9.2 迭代抽样

建议采用分批迭代方式：

1. 基准样本按年份和行业分层，每年抽取1万条岗位；2014—2025年最多形成12万条基准招聘文本。
2. 某年份有效岗位不足1万条时取该年份全部有效岗位，不进行有放回抽样。
3. 年内首先向每个非空行业分配 `min(50, 该行业岗位数)` 条，剩余名额按该行业在本年度剩余岗位中的比例分配。
4. 舍入造成的名额差额按照行业剩余岗位数由大到小补齐，确保每年抽样数不超过1万条。
5. 基准样本使用固定随机种子且岗位不重复。
6. 完成基准样本候选发现后，每轮继续抽取1万条尚未进入候选发现样本的岗位。
7. 每轮计算新增表面形式、新增标准技能和新增别名。
8. 连续两轮满足正式停止标准后停止更新，不预先强制复制中文文献的60万条规模。

### 9.2.1 每轮样本构成

在完成按年份、行业构建的12万条以内基准样本后，后续每轮1万条增量样本建议由三部分组成：

1. 40%来自无技能匹配或技能覆盖较低的岗位；
2. 30%来自主AI锚点命中的岗位；
3. 30%来自按年份、行业、岗位类别和平台分层的普通岗位。

同一岗位只在一个候选发现批次中出现。抽样随机种子、抽样概率和分层单元必须保存，以便重新生成相同样本。

### 9.2.2 默认饱和标准

连续两轮同时满足以下条件时，候选发现停止更新：

1. 每1万条新增的有效标准技能概念少于5项；
2. 新增别名带来的全体去重招聘岗位技能覆盖率提升低于0.1个百分点；覆盖率定义为至少匹配到一项正式技能的岗位占全部有效去重岗位的比例；
3. 各年份和主要行业均已至少进入一个批次；
4. 主AI锚点命中岗位中的未覆盖比例不再明显下降。

每轮新增别名合并到临时词典后，对全体去重招聘岗位执行一次确定性匹配，以计算覆盖率提升。停止更新后仍保留未来增量更新接口，不意味着词典永久封闭。

### 9.3 Qwen输出要求

每项技能必须绑定岗位原文证据：

```json
{
  "job_id": "...",
  "skills": [
    {
      "surface": "原文中的连续技能片段",
      "canonical_suggestion": "建议标准名称",
      "skill_type": "method_algorithm",
      "evidence": "包含技能的原句",
      "start": 0,
      "end": 0,
      "existing_skill_id": null
    }
  ]
}
```

模型不得根据常识补写岗位描述中未出现的技能。无法在原文中定位的候选自动进入失败记录，不进入候选词典。

`skill_type`必须使用正式JSON Schema定义的完整枚举，不得使用 `method`、`tool`、`language`等简写。示例中的 `method_algorithm`仅表示其中一个合法取值。

### 9.4 输出

`skill_candidate_surface.parquet`

---

## 10. 阶段六：技能归一化、概念映射与分级

### 10.1 归一化顺序

1. Unicode、大小写和全半角归一化；
2. 中英文符号和空白归一化；
3. 单复数和常见拼写变体处理；
4. 软件版本号与核心产品名称分离保存；
5. 中英文名称、缩写和全称关联；
6. 与基础词典进行精确和近似映射；
7. 使用Qwen判断是现有技能别名还是新技能概念；
8. 对新概念生成稳定 `skill_id`。

### 10.2 技能等级

#### A级

能够与ESCO、O*NET、Lightcast或其他可追溯技能来源对应的技能。

#### B级

外部词典未覆盖，但在中国招聘语料中稳定出现，具有明确原文证据和相对稳定概念边界的技能。

#### C级

低频、新兴或集中于少数年份和行业，但具有明确技能含义的技能，包括新兴AI模型、框架、软件和工程方法。

#### D级

原文证据无效、概念边界不清、不能稳定归一化或明显不是技能的候选。

### 10.3 纳入规则

1. A、B、C级全部进入正式技能词典。
2. D级保留在候选库，不参与正式匹配。
3. `confidence_tier`只表示来源和稳定性，不表示A/B/C技能在岗位匹配时具有不同权重。
4. A/B/C均可参与技能AI共现率和岗位AI相关度计算。
5. 稳健性检验可以暂时排除C级，以识别新兴低频技能对结果的影响，但不改变C级的正式词典身份。

### 10.3.1 自动分级默认规则

分级顺序为A、B、C、D，满足较高等级后不再继续判断。

A级：

- 能够映射至至少一个固定版本外部技能来源；
- 映射关系为同义或同一概念，不只是上下位或主题相关；
- 原始来源ID和映射证据完整。

B级：

- 不能映射到A级来源；
- 原文跨度有效；
- 在至少100个不同的规范化岗位描述文本中出现，即 `df_unique_description>=100`；
- Qwen能够生成唯一的聚合标准名称和技能类型；
- 不在非技能停用表中。

C级满足以下任一规则：

- `10<=df_unique_description<100`且原文跨度有效；
- `df_unique_description>=5`且主AI锚点候选共现率不低于0.50；
- `df_unique_description>=5`，属于新出现的软件、模型、框架或技术方法，能够形成明确标准概念，但尚未达到B级稳定性。

D级：

- 不满足A/B/C；或
- 无法通过原文跨度校验；或
- 主要是任务片段、学历、专业、岗位、公司、福利或经验要求；或
- 无法建立稳定标准名称；或
- 只能通过大模型常识推断，原文没有明确技能表达。

### 10.3.2 频数口径

分级中的 `df_unique_description`一律指包含该表面形式或规范概念的不同规范化岗位描述文本数量，以 `COUNT(DISTINCT text_hash)`计算，不使用模型输出次数，也不受相同文本重复发布次数影响。多个别名映射到同一 `skill_id`后，概念频数按 `text_hash`去重重新计算。

这一定义只用于判断B/C技能是否稳定进入正式词典。正式技能AI共现率仍以最终确定的招聘广告测量样本和唯一 `job_id`计数，不能用 `df_unique_description`替代共现率分母。

候选主AI共现率仅用于C级识别，不直接作为最终技能AI共现率；正式共现率必须在正式词典全量匹配后重新计算。

### 10.3.3 聚合候选的模型复核

Qwen只对聚合后的候选概念进行映射和归一化，不对全量岗位中的歧义实例逐条判定。映射时先从基础词典检索至多10个候选 `skill_id`，模型只能从候选ID中选择或返回 `NEW_CONCEPT`，不得生成不存在的ID。

### 10.3.4 Top-10候选检索机制说明

正式词典可能包含数万至十万项技能，无法把全部技能名称、定义和ID同时放入Qwen提示词。Top-10检索的作用是先用确定性检索或向量检索缩小范围，再让Qwen只在少量可审计候选中判断。

例如，Qwen从岗位中抽取表述`数据建模`。现有词典可能同时包含数据建模、数据库设计、统计建模、业务流程建模等相近概念。检索器先返回最相近的10个候选及其ID、名称和定义，Qwen再判断：

```text
MATCH_EXISTING   与某个候选是同一技能，返回唯一skill_id
NEW_CONCEPT      与10个候选都不是同一技能，需要建立新skill_id
AMBIGUOUS        现有证据不足以在多个候选之间唯一判断
```

推荐执行顺序：

1. 规范化别名精确匹配；唯一命中时直接映射，不调用向量检索。
2. 对无精确命中的候选，将 `surface + canonical_suggestion + evidence + skill_type`编码为检索向量。
3. 将正式词典中的 `canonical_zh + canonical_en + definition + skill_category`编码为词典向量。
4. 使用固定版本的本地中英文或多语言文本嵌入模型，按余弦相似度返回前10项。
5. 把10个候选的 `skill_id`、中英文名称、定义、类别和相似度提供给Qwen。
6. Qwen只能返回一个现有ID、`NEW_CONCEPT`或`AMBIGUOUS`。
7. 相似度并列时全部按确定性次序排序：相似度降序、标准名称字典序、`skill_id`字典序。
8. 保存完整Top-10列表、相似度、Qwen选择和原始输出，保证映射可追溯。

检索模型会决定哪些技能能够进入Qwen的候选列表，因此必须固定：

```text
embedding_model
embedding_model_revision
embedding_dimension
text_template
normalization
vector_index_type
similarity_metric
top_k
tie_break_rule
```

当前不预设相似度硬阈值：即使最高相似度很低，也可把Top-10交给Qwen，由Qwen返回 `NEW_CONCEPT`。这样可以避免不同嵌入模型分数不可直接比较的问题。具体嵌入模型与Qwen模型一样，在RTX 4090测试阶段确定并固定。

### 10.4 输出

- `skill_concept_v1.parquet`；
- `skill_alias_v1.parquet`；
- `skill_candidate_d_v1.parquet`；
- `skill_dictionary_changelog_v1.csv`。

---

## 11. 阶段七：正式词典全量岗位匹配

### 11.1 匹配原则

1. 使用固定版本的正式技能词典和别名表。
2. 使用确定性字符串匹配、Trie或Aho-Corasick等算法。
3. 优先采用最长匹配，避免长技能被拆成多个短技能。
4. 同一 `skill_id` 在同一岗位中只保留一次。
5. 保存实际命中的原始表面形式和字符位置。
6. 对英文简称设置字母数字边界，避免在其他英文单词内部误匹配。
7. 对多义词保留 `ambiguity_flag`，但不使用Qwen逐条进行语境判断。
8. 匹配误差作为测量误差处理，通过后续稳健性和分布检验评估。

### 11.1.1 匹配顺序

1. 加载 `is_active=1` 的别名；
2. 对岗位描述执行一次多模式匹配；
3. 先处理字符跨度更长的命中；
4. 相同跨度、相同 `skill_id`只保留一条；
5. 短别名完全包含于同一概念长别名时，仅保留长别名命中；
6. 不同概念发生嵌套时，默认保留较长跨度，并记录被覆盖候选；
7. 同一技能在岗位多次出现时，保留首次位置并记录 `mention_count`；
8. 生成岗位—技能唯一表后再进行共现统计。

### 11.1.2 英文边界

英文单词和缩写采用ASCII字母数字边界：

```regex
(?<![A-Za-z0-9])TERM(?![A-Za-z0-9])
```

这样允许 `AI算法`、`熟悉NLP`等中英文相邻表达，同时避免在其他英文字母或数字内部命中。对 `C++`、`C#`、`.NET`、`R`、`Go`等特殊技能使用单独规则，不直接套用普通单词边界。

### 11.1.3 歧义别名

1. 能够在词典构建阶段确定固定主含义的别名，只映射到 `primary_skill_id`并设置 `ambiguity_flag=1`。
2. 无法确定固定主含义的别名设置 `is_active=0`，概念仍保留在正式词典，但该别名不参与主匹配。
3. 不允许一个表面命中在主结果中同时扩展为多个 `skill_id`。
4. 可另行生成包含全部激活歧义别名的宽松匹配版本，用于敏感性检验。

### 11.1.4 匹配完整性检查

- 输入岗位数必须与输出岗位摘要表的岗位数一致；
- `job_skill_long`中 `(job_id, skill_id)`必须唯一；
- `surface_form`必须能回填到清洗后文本；
- `mention_count>=1`；
- `confidence_tier`必须与词典当前版本一致。

### 11.2 无技能岗位

如果岗位描述没有匹配到任何正式技能：

- `matched_skill_count=0`；
- `score_eligible=0`；
- 岗位AI相关度记为缺失，而不是0；
- 在时间窗口本身可计算时，三个AI岗位标识均设为0，并设置 `zero_skill_override=1`；
- 不得静默删除该岗位；
- 另行报告各年份无技能岗位占比。

### 11.3 输出

`job_skill_long.parquet`

字段至少包括：

```text
job_id
year
skill_id
surface_form
start
end
match_method
ambiguity_flag
confidence_tier
dictionary_version
```

---

## 12. 阶段八：构建三套AI锚点口径

所有锚点均在清洗后的岗位描述中进行确定性匹配。英文字母匹配不区分大小写，但需要设置字母数字边界。

### 12.1 本项目主锚点 `anchor_main`

包含以下六类概念的中英文表达：

| 锚点组 | 中文关键词 | 英文关键词及缩写 |
|---|---|---|
| AI | 人工智能 | artificial intelligence、AI |
| ML | 机器学习 | machine learning、ML |
| NLP | 自然语言处理 | natural language processing、NLP |
| Computer Vision | 计算机视觉、图像识别 | computer vision、image recognition |
| LLM | 大语言模型、大型语言模型 | large language model、large language models、LLM、LLMs |
| Transformer | Transformer模型、Transformer架构 | Transformer model、Transformer models、Transformer architecture、Transformer architectures |

主锚点集合满足：岗位描述出现任意一个关键词时，`anchor_main=1`。

主口径不使用裸词`大模型`、裸词`Transformer`、裸词`Transformers`或`变换器模型`。Transformer必须与模型或架构限定词共同出现。本项目不使用Qwen逐条消歧，但仍保存锚点命中明细，用于检查行业和年份分布。

### 12.2 中文文献口径 `anchor_cn_paper`

包含：

- 人工智能、artificial intelligence、AI；
- 机器学习、machine learning、ML；
- 自然语言处理、natural language processing、NLP；
- 图像识别、image recognition。

不将独立缩写 `IR`作为锚点，避免与信息检索、投资者关系等含义混淆。

### 12.3 Babina口径 `anchor_babina`

包含：

- 人工智能、artificial intelligence、AI；
- 机器学习、machine learning、ML；
- 自然语言处理、natural language processing、NLP；
- 计算机视觉、computer vision。

这是Babina核心概念在中文招聘语料中的双语适配版本。

### 12.4 锚点词典输出

`ai_anchor_dictionary_v1.csv`

字段：

```text
anchor_version
anchor_group
keyword
keyword_normalized
language
matching_rule
ambiguity_flag
```

### 12.5 岗位锚点标记输出

`job_anchor_flag.parquet`

至少包含：

```text
job_id
year
anchor_main
anchor_cn_paper
anchor_babina
matched_anchor_groups_main
matched_anchor_terms_main
```

### 12.6 锚点规范化与匹配规则

1. 中文关键词在NFKC规范化文本中按连续子串匹配。
2. 英文关键词不区分大小写。
3. 英文短语允许一个或多个空格或连字符，例如 `machine-learning`与`machine learning`。
4. AI、ML、NLP、LLM、LLMs使用ASCII字母数字边界，不在其他英文单词内部命中。
5. 不使用独立 `CV`和`IR`作为正式锚点。
6. `Transformer`只有在与 `model/models/architecture/architectures`或中文`模型/架构`组合时才命中；裸词不命中。
7. 每个岗位保存实际命中的关键词和锚点组，不能只保存0—1结果。
8. 同一锚点重复出现只影响 `anchor_*=1`，不按出现次数增加共现计数。

推荐正则示意：

```regex
artificial[\s-]+intelligence
machine[\s-]+learning
natural[\s-]+language[\s-]+processing
computer[\s-]+vision
image[\s-]+recognition
large[\s-]+language[\s-]+models?
(?<![A-Za-z0-9])AI(?![A-Za-z0-9])
(?<![A-Za-z0-9])ML(?![A-Za-z0-9])
(?<![A-Za-z0-9])NLP(?![A-Za-z0-9])
(?<![A-Za-z0-9])LLMs?(?![A-Za-z0-9])
(?<![A-Za-z0-9])Transformer[\s-]+models?(?![A-Za-z0-9])
(?<![A-Za-z0-9])Transformer[\s-]+architectures?(?![A-Za-z0-9])
```

锚点正则必须建立单元测试，至少覆盖：正确英文短语、大小写变化、中英文相邻、单词内部误命中、复数形式、标点相邻，以及裸词`Transformer`和`大模型`不命中的反例。

### 12.6.1 岗位名称排除规则

岗位名称不进入 `anchor_main`、`anchor_cn_paper`或`anchor_babina`中的任何一个版本。三套锚点全部只在 `job_description_match`中识别。岗位名称仅用于去重、分层抽样、结果描述和质量检查。

### 12.7 锚点自身作为技能

AI核心概念必须同时存在于正式技能词典中。由于岗位出现锚点技能时必然满足 `A_j=1`，该锚点技能的原始共现率通常为1，这是公式的直接结果，不应误判为计算错误。

岗位得分中保留锚点技能，与中文文献及Babina聚合逻辑保持一致。可在稳健性检验中计算删除锚点技能后的岗位得分，以判断结果是否完全由直接关键词驱动。

---

## 13. 阶段九：构造岗位—技能共现计数

### 13.1 年度计数

对每个年份、技能和锚点口径，计算：

```text
n_skill_year       当年包含技能s的唯一岗位数
n_ai_cooccur_year  其中同时包含至少一个AI锚点的唯一岗位数
```

原始年度共现率：

$$
w_{s,t,annual}^{AI,k}
=
\frac{n\_ai\_cooccur\_year}
{n\_skill\_year}
$$

### 13.2 全期计数

使用2014—2025年全部去重岗位：

$$
w_{s,pooled}^{AI,k}
=
\frac{\sum_{t=2014}^{2025}n_{s,t}^{AI,k}}
{\sum_{t=2014}^{2025}n_{s,t}}
$$

全期共现率对所有年份使用同一技能权重，用于获得稳定指标并与Babina的pooled思想比较。

### 13.3 中心三年滚动计数 `roll3_centered`

目标年份`t`的常规窗口定义为：

$$
W_t=\{t-1,t,t+1\}
$$

在2014—2025年样本边界处采用两年窗口：

$$
W_{2014}=\{2014,2015\},\qquad
W_{2025}=\{2024,2025\}
$$

例如：

- 2014年使用2014—2015年；
- 2015年使用2014—2016年；
- 2024年使用2023—2025年；
- 2025年使用2024—2025年。

中心滚动共现率为：

$$
w_{s,t,roll3\_centered}^{AI,k}
=
\frac{\sum_{\tau\in W_t}n_{s,\tau}^{AI,k}}
{\sum_{\tau\in W_t}n_{s,\tau}}
$$

该口径可以减少单年样本稀疏和技能权重缺失，并使2014—2025年每个目标年份均有可用窗口。但它在2014—2024年的多数年份使用`t+1`信息，因此属于双侧中心移动窗口，不是历史权重、滞后指标或实时预测指标。它适用于描述性岗位AI识别和稳健性比较；若后续研究要求严格的信息时序或因果解释，应另行增加只使用`t`及以前信息的口径。

### 13.4 频数与可靠性字段

所有技能均计算共现率，不因A/B/C等级或低频而从正式词典中删除。同时保留：

```text
n_skill
n_ai_cooccur
rare_lt10
rare_lt20
rare_lt50
confidence_tier
```

低频标记用于稳健性检验和结果解释，不改变技能的正式词典身份。

### 13.5 输出

`skill_ai_counts.parquet`

### 13.6 计数实现顺序

计数时使用以下固定顺序：

1. 从 `job_skill_long`选择唯一 `(job_id, skill_id)`；
2. 与 `job_anchor_flag`按 `job_id`一对一连接；
3. 校验连接前后岗位—技能行数不变；
4. 按 `skill_id × year × anchor_version`统计年度分母和分子；
5. 由年度计数汇总全期和`roll3_centered`计数；
6. 不从已经四舍五入的年度比率计算全期或滚动比率；
7. 全部比率都由整数分子和整数分母重新相除；
8. 分子、分母使用64位整数；比率使用64位浮点数；
9. 保存计算前岗位数、岗位—技能行数和分组数日志。

### 13.7 计数不变量

正式结果必须满足：

```text
0 <= n_ai_cooccur <= n_skill
0 <= ai_rate_raw <= 1
年度n_skill之和 = pooled n_skill
年度n_ai_cooccur之和 = pooled n_ai_cooccur
同一skill_id、窗口和锚点版本只有一行
```

中心滚动窗口的计数应等于对应年度整数计数之和；2014年和2025年分别校验两个年度，其他年份校验三个年度。任何不变量失败均属于阻断性错误。

---

## 14. 阶段十：计算技能AI共现率

### 14.1 必须输出的权重

对三套锚点和三种时间窗口，分别输出原始共现率：

| 锚点版本 | 年度 | 全期 | 中心3年窗口 |
|---|---:|---:|---:|
| `anchor_main` | 是 | 是 | 是 |
| `anchor_cn_paper` | 是 | 是 | 是 |
| `anchor_babina` | 是 | 是 | 是 |

合计形成9组基础技能AI共现率。

### 14.2 平滑共现率

为防止低频技能因一次共现得到1的极端值，建议在保留原始共现率的同时计算经验贝叶斯平滑值：

$$
\widetilde w_s
=
\frac{c_s+\alpha}
{n_s+\alpha+\beta}
$$

其中：

- `c_s`为AI共现岗位数；
- `n_s`为技能出现岗位数；
- `α`和`β`为相应锚点、窗口与时期下的Beta先验参数。

原始共现率用于文献可比性，平滑共现率作为低频技能稳健性指标。

默认平滑方法：

1. 对每个 `anchor_version × window_type × period`分别拟合Beta-Binomial先验；
2. 使用该单元内 `n_skill>=5`的技能，通过边际似然最大化估计 `α`和`β`；
3. 约束 `α>0`、`β>0`；
4. 如果优化不收敛或参数异常，回退到Jeffreys先验 `Beta(0.5,0.5)`；
5. 保存 `alpha`、`beta`、拟合状态和回退原因；
6. 不用平滑权重覆盖原始权重。

平滑只改变技能权重，不改变技能是否进入正式词典。

### 14.3 企业留一稳健性

如企业标识可用，仅对主锚点计算企业留一稳健性：

$$
w_{s,t}^{(-f)}
=
\frac{c_{s,t}-c_{s,f,t}}
{n_{s,t}-n_{s,f,t}}
$$

企业留一结果覆盖主锚点下的年度、全期和`roll3_centered`三种窗口，使用原始共现率。中文文献锚点和Babina锚点不计算企业留一版本。企业留一结果只用于检验企业自身岗位是否机械抬高技能权重，不替代主指标。

### 14.4 输出

`skill_ai_relevance.parquet`

建议字段：

```text
skill_id
year
window_type
window_start
window_end
anchor_version
n_skill
n_ai_cooccur
ai_rate_raw
ai_rate_smoothed
rare_lt10
rare_lt20
rare_lt50
confidence_tier
dictionary_version
alpha
beta
smoothing_status
```

---

## 15. 阶段十一：计算岗位AI相关度

### 15.1 基准聚合方法

将 `job_skill_long` 与 `skill_ai_relevance` 按 `skill_id`、年份、时间窗口和锚点版本连接。

岗位AI相关度为：

$$
AIscore_{j,t}^{k,h}
=
\frac{1}{|S_j|}
\sum_{s\in S_j} w_{s,t}^{AI,k,h}
$$

其中：

- `k`表示主锚点、中文文献锚点或Babina锚点；
- `h`表示年度、全期或中心3年窗口；
- `S_j`为岗位中去重后的全部正式技能集合。

### 15.2 年份连接规则

- 年度权重：岗位年份 `t` 连接年份 `t` 的技能权重；
- 全期权重：所有岗位连接2014—2025全期技能权重；
- `roll3_centered`：2015—2024年的岗位年份`t`连接窗口`[t-1,t+1]`；2014年连接`[2014,2015]`；2025年连接`[2024,2025]`。

### 15.3 无权重技能

年度、全期和中心滚动权重均包含岗位所属年份。因此，只要岗位技能已经进入 `job_skill_long`，该技能在对应窗口内的 `n_skill`原则上至少为1。

处理规则：

1. 年度、全期和`roll3_centered`结果中，`weighted_skill_count`应等于`matched_skill_count`；否则视为连接、窗口或词典版本错误。
2. 不将因连接错误产生的缺失权重设为0，也不静默删除该技能后重新取平均。
3. 上述三类结果的 `score_skill_coverage=weighted_skill_count/matched_skill_count`在正式结果中应为1；小于1属于阻断性检查失败。
4. 不设置最低覆盖率门槛，因为对非空岗位该覆盖率按设计必须为1；覆盖率字段保留为连接完整性诊断，而不是样本筛选标准。
5. 企业留一稳健性中，如果某技能仅在被留出的企业出现，则留一分母为0，该技能没有留一权重。留一岗位得分可以对其余有权重技能取平均，但企业留一覆盖率只作描述，不改变主样本。

### 15.4 原始与平滑得分

分别使用 `ai_rate_raw` 和 `ai_rate_smoothed` 计算岗位得分，形成：

- 原始岗位AI相关度；
- 平滑岗位AI相关度。

主文献可比结果使用原始得分；平滑得分用于低频技能稳健性检验。

### 15.5 输出

`job_ai_score.parquet`

建议采用长表结构：

```text
job_id
year
anchor_version
window_type
score_type
ai_score
matched_skill_count
weighted_skill_count
score_skill_coverage
score_eligible
dictionary_version
```

其中：

- `anchor_version`取`main`、`cn_paper`或`babina`；
- `window_type`取`annual`、`pooled`或`roll3_centered`；
- `score_type`取`raw`或`smoothed`。

### 15.6 字段命名规则

长表是规范保存形式。若为分析方便生成宽表，字段名按以下顺序组合：

```text
aiscore_<anchor>_<window>_<scoretype>
aijob_<anchor>_<window>_<scoretype>_<threshold>
```

示例：

```text
aiscore_main_annual_raw
aiscore_main_pooled_raw
aiscore_babina_roll3_centered_smoothed
aijob_main_annual_raw_005
```

不得使用无法识别锚点、窗口和权重类型的通用字段名 `AIscore`或`AIjob`作为最终输出列。

---

## 16. 阶段十二：识别AI相关岗位

### 16.1 阈值

每一组岗位连续得分同步生成：

```text
ai_job_005 = 1(ai_score > 0.05)
ai_job_010 = 1(ai_score > 0.10)
ai_job_015 = 1(ai_score > 0.15)
```

主阈值为0.05；0.10和0.15用于与Babina方法及阈值敏感性比较。

### 16.2 判定条件

通常只有满足以下条件的岗位才能由连续得分生成AI岗位标识：

1. `matched_skill_count>0`；
2. `weighted_skill_count>0`；
3. `ai_score`非缺失。

无技能岗位采用已确认的特殊处理：

- 保留岗位记录；
- `matched_skill_count=0`；
- `score_eligible=0`；
- `ai_score`保持缺失；
- 0.05、0.10和0.15三个AI岗位标识均设为0；
- 设置 `zero_skill_override=1`，明确该0值来自无技能覆盖规则，而不是来自可计算得分。

中心滚动窗口在2014年和2025年使用已确认的两年边界窗口，因而不存在仅因样本边界年份不足而整体缺失的情况。

### 16.3 输出

`job_ai_classification.parquet`

包含连续得分和三个阈值标识，不进行企业年度聚合。

至少保留 `zero_skill_override`和 `window_period_available`，以区分无技能覆盖与窗口数据异常。正常的2014—2025年年度、全期和中心滚动窗口均应有 `window_period_available=1`。

### 16.4 主岗位标识

正式报告中的主AI岗位变量为：

```text
aijob_main_annual_raw_005
```

其他阈值、时间窗口、锚点和平滑版本均明确标记为替代或稳健性指标。岗位得分等于阈值时取0，因为判定条件是严格大于。

同一岗位可能在不同口径下得到不同分类，这不是数据冲突。结果表应保留全部版本，并另外输出不同版本之间的转换矩阵。

---

## 17. 阶段十三：自动化质量检查

本项目不强制建设人工金标准。质量检查以自动约束、稳定性和外部合理性为主。

### 17.1 技能词典检查

1. A/B/C/D数量和占比；
2. 不同来源的技能数量；
3. 同一标准技能的别名数量；
4. 无法归一化候选数量；
5. 新增技能随抽样批次的饱和曲线；
6. 技能名称长度和频数分布；
7. A/B/C对岗位技能覆盖率的贡献。

### 17.2 模型抽取检查

1. JSON解析成功率；
2. 原文证据可回填率；
3. 重复运行一致性；
4. 模型输出中不存在于原文的候选比例；
5. 不同年份和行业的候选发现率；
6. 模型版本和提示词版本的差异。

### 17.3 全量匹配检查

1. 各年份有技能岗位比例；
2. 每条岗位匹配技能数分布；
3. 高频歧义词命中数量；
4. 各等级技能的命中岗位数；
5. 岗位技能覆盖率；
6. 无技能岗位的年份和行业分布。

### 17.4 AI共现率检查

1. 技能出现数和共现数逻辑校验；
2. 共现率必须位于0和1之间；
3. `n_ai_cooccur`不得大于`n_skill`；
4. 低频高共现技能数量；
5. 年度、全期和三年滚动权重相关性；
6. 主锚点、中文文献锚点和Babina锚点的相关性；
7. 2014—2025年高AI相关技能排名变化；
8. LLM和Transformer锚点对2020年后结果的增量影响。

### 17.5 岗位结果检查

1. 岗位AI得分分布；
2. 0.05、0.10、0.15阈值下AI岗位比例；
3. 不同年份AI岗位比例趋势；
4. 不同行业和岗位名称的AI岗位比例；
5. 原始得分和平滑得分相关性；
6. 包含直接AI锚点岗位与不包含锚点岗位的得分差异；
7. A/B/C技能对岗位得分的贡献；
8. 排除C级技能后的岗位分类变化；
9. 主锚点企业留一权重与全样本权重的结果差异。

### 17.6 自动质量门槛

以下检查必须100%通过，否则停止发布结果：

- `job_id`唯一性和引用完整性；
- `(job_id, skill_id)`唯一性；
- 原文跨度回填；
- 分子不大于分母；
- 权重和岗位得分位于0到1；
- 年度计数能够精确加总为全期计数；
- 中心滚动窗口整数计数与对应年度计数之和一致，且首尾年份使用两年、其余年份使用三年；
- 主岗位得分的技能连接覆盖率为1；
- 0.05、0.10、0.15阈值变量具有单调关系，即 `AIjob_015<=AIjob_010<=AIjob_005`；
- 同一配置重新运行得到相同文件行数和关键统计量。

以下检查产生警告但不自动阻断：

- 某年份无技能岗位比例明显升高；
- 某个歧义锚点集中于非技术行业；
- 低频技能大量取得0或1的原始权重；
- A/B/C中某一等级对岗位得分贡献异常集中；
- 不同锚点口径下AI岗位比例差异过大；
- 年度岗位AI比例发生无法由数据覆盖变化解释的断点。

所有警告必须进入质量报告，并给出对应的年份、行业、岗位和技能明细表。

---

## 18. 必须生成的最终结果

本阶段最终至少生成以下文件：

| 文件 | 内容 |
|---|---|
| `skill_concept_v1.parquet` | A/B/C正式技能概念词典 |
| `skill_alias_v1.parquet` | 正式技能别名表 |
| `skill_candidate_d_v1.parquet` | D级候选技能 |
| `ai_anchor_dictionary_v1.csv` | 三套AI锚点词典 |
| `job_skill_long.parquet` | 岗位—技能长表 |
| `job_anchor_flag.parquet` | 岗位的三套锚点标识 |
| `skill_ai_counts.parquet` | 技能出现数和AI共现数 |
| `skill_ai_relevance.parquet` | 九组技能AI共现率及平滑结果 |
| `job_ai_score.parquet` | 岗位连续AI相关度 |
| `job_ai_classification.parquet` | 0.05、0.10、0.15阈值下的AI岗位标识 |
| `quality_control_report.md` | 自动化质量检查结果 |

暂不生成企业—年度AI招聘指标。

### 18.1 结果附带元数据

每个最终结果文件同时生成同名 `.metadata.json`，至少包含：

```text
file_name
run_id
created_at
row_count
column_schema
primary_key
source_files
dictionary_version
anchor_version
config_hash
sha256
```

### 18.2 结果保留精度

1. 整数计数使用64位整数。
2. 权重和岗位得分使用64位浮点数保存，不在数据文件中提前四舍五入。
3. 表格展示时可以保留4—6位小数，但不得用展示值继续计算。
4. 缺失值保持真正的空值，不用-1、999或字符串NA替代。

---

## 19. 推荐执行顺序与阶段确认

### 第一轮：口径与数据准备

1. 冻结指标口径和锚点词典；
2. 审计招聘数据字段；
3. 确定去重规则；
4. 形成清洗后的唯一岗位表。

### 第二轮：技能词典建设

5. 读取并固化本地O*NET Skills文件，由Codex从官方渠道下载缺失的ESCO、O*NET Software Skills及可选Lightcast版本；
6. 由Codex分批形成中英文基础技能概念、激活别名和审计日志；
7. 在RTX 4090上测试并固定Qwen模型；
8. 分层抽样发现候选技能；
9. 完成技能归一化和A/B/C/D分级；
10. 发布正式词典v1。

### 第三轮：全量岗位匹配

11. 使用正式词典匹配全部岗位描述；
12. 生成岗位—技能长表；
13. 生成三套岗位锚点标识；
14. 完成匹配覆盖率和歧义词诊断。

### 第四轮：AI指标计算

15. 计算年度、全期和`roll3_centered`技能共现计数；
16. 计算三套锚点下的技能AI共现率；
17. 计算岗位AI相关度；
18. 使用0.05、0.10和0.15识别AI岗位；
19. 完成全部自动化质量检查。

每一轮完成后应保存阶段结果、配置文件和日志，在确认无误后再进入下一轮。

### 19.1 推荐脚本顺序

```text
00_validate_environment.py
01_audit_raw_jobs.py
02_clean_job_text.py
03_deduplicate_jobs.py
04_import_external_skills.py
05_build_base_dictionary.py
06_benchmark_qwen.py
07_sample_skill_discovery.py
08_extract_skill_candidates.py
09_normalize_and_grade_skills.py
10_build_dictionary_matcher.py
11_match_all_job_skills.py
12_build_anchor_flags.py
13_count_skill_ai_cooccurrence.py
14_estimate_skill_ai_rates.py
15_score_jobs.py
16_classify_ai_jobs.py
17_run_quality_checks.py
18_export_release.py
```

每个脚本应满足：

- 读取配置文件而不是在代码中写死路径和参数；
- 开始时验证输入文件和版本；
- 结束时写出行数、耗时、哈希和状态；
- 对既有输出默认拒绝覆盖；
- 支持 `--dry-run`、`--resume`和 `--run-id`；
- 出错时返回非零退出码。

### 19.2 主配置文件示例

```yaml
project:
  start_year: 2014
  end_year: 2025
  timezone: Asia/Shanghai

dedup:
  exact_window_days: 30
  near_duplicate_enabled: false
  near_duplicate_jaccard: 0.95
  shingle_size: 5
  cross_year_merge: false

dictionary:
  include_tiers: [A, B, C]
  active_alias_only: true
  longest_match: true
  ambiguous_alias_policy: fixed_primary_or_inactive

qwen:
  model_revision: TO_BE_CONFIRMED
  prompt_version: v1
  schema_version: v1
  do_sample: false
  temperature: 0
  max_model_len: 8192
  max_input_tokens: 6000
  max_output_tokens: 2000
  seed: 20260822

cooccurrence:
  windows: [annual, pooled, roll3_centered]
  rolling_years: 3
  roll3_centered_year_offsets: [-1, 0, 1]
  boundary_policy: adjacent_two_years
  anchor_versions: [main, cn_paper, babina]
  compute_raw: true
  compute_smoothed: true

job_score:
  aggregation: mean_all_unique_skills
  thresholds: [0.05, 0.10, 0.15]
  primary_anchor: main
  primary_window: annual
  primary_score_type: raw
  zero_skill_policy: missing

robustness:
  leave_one_firm_out: true
  leave_one_firm_out_anchor_versions: [main]
  leave_one_firm_out_windows: [annual, pooled, roll3_centered]
  leave_one_firm_out_score_types: [raw]
  exclude_tier_c: true
  remove_anchor_skills_from_job_mean: true
```

### 19.3 Qwen候选抽取JSON Schema示例

正式Schema可以在此基础上扩展，但不得删除原文证据和字符位置约束：

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": ["job_id", "skills"],
  "properties": {
    "job_id": {"type": "string", "minLength": 1},
    "skills": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "surface", "canonical_suggestion", "skill_type",
          "evidence", "start", "end", "existing_skill_id"
        ],
        "properties": {
          "surface": {"type": "string", "minLength": 1},
          "canonical_suggestion": {"type": "string", "minLength": 1},
          "skill_type": {
            "type": "string",
            "enum": [
              "programming_language", "method_algorithm", "software_tool",
              "platform_framework_library", "data_database",
              "hardware_equipment", "domain_knowledge",
              "business_management", "general_work_skill",
              "soft_skill", "other_skill"
            ]
          },
          "evidence": {"type": "string", "minLength": 1},
          "start": {"type": "integer", "minimum": 0},
          "end": {"type": "integer", "minimum": 1},
          "existing_skill_id": {
            "oneOf": [
              {"type": "string", "minLength": 1},
              {"type": "null"}
            ]
          }
        }
      }
    }
  }
}
```

Schema校验后必须另做语义无关的程序校验：`end>start`、`text[start:end]==surface`、`job_id`与输入一致、岗位内相同跨度不重复。

### 19.4 共现率计算伪代码

```text
# 一、岗位—技能唯一化
job_skill_unique = DISTINCT(job_id, year, skill_id)

# 二、连接岗位锚点
job_skill_anchor = LEFT JOIN job_skill_unique
                   WITH job_anchor_flag USING(job_id, year)
assert row_count_before == row_count_after

# 三、年度整数计数
FOR anchor_version IN [main, cn_paper, babina]:
    GROUP BY skill_id, year:
        n_skill = COUNT(DISTINCT job_id)
        n_ai_cooccur = COUNT(DISTINCT job_id WHERE anchor_version == 1)
        ai_rate_raw = n_ai_cooccur / n_skill

# 四、全期整数计数
pooled = SUM annual counts BY skill_id, anchor_version
pooled_rate = pooled_n_ai_cooccur / pooled_n_skill

# 五、中心三年整数计数；样本边界使用相邻两年
FOR t IN 2014..2025:
    IF t == 2014:
        window_years = [2014, 2015]
    ELSE IF t == 2025:
        window_years = [2024, 2025]
    ELSE:
        window_years = [t-1, t, t+1]
    roll3_centered_counts(t) = SUM annual counts WHERE year IN window_years
    roll3_centered_rate(t) = n_ai_cooccur / n_skill

# 七、岗位评分
JOIN job_skill_unique WITH matching skill weights
GROUP BY job_id, anchor_version, window_type, score_type:
    ai_score = MEAN(skill_ai_rate)
    matched_skill_count = COUNT(DISTINCT skill_id)

# 八、岗位分类
ai_job_005 = ai_score > 0.05
ai_job_010 = ai_score > 0.10
ai_job_015 = ai_score > 0.15

# 九、无技能岗位覆盖规则
IF matched_skill_count == 0 AND window_period_is_available:
    ai_score = NULL
    ai_job_005 = 0
    ai_job_010 = 0
    ai_job_015 = 0
    zero_skill_override = 1
```

### 19.5 发布前检查清单

正式发布前逐项确认：

```text
[ ] 原始文件哈希和行数已经登记
[ ] 2014—2025年年份范围已校验
[ ] 清洗和去重版本已冻结
[ ] 外部技能词典版本和许可已登记
[ ] Qwen模型revision、提示词和Schema已冻结
[ ] A/B/C/D分级规则已写入配置
[ ] 正式词典和别名表主键唯一
[ ] 歧义别名策略已经固定
[ ] 三套锚点正则单元测试全部通过
[ ] 岗位—技能表(job_id, skill_id)唯一
[ ] 年度、全期和中心滚动计数不变量全部通过
[ ] 主岗位得分连接覆盖率等于1
[ ] 三个AI岗位阈值满足单调关系
[ ] 自动质量报告已生成
[ ] 所有最终文件及元数据哈希已生成
[ ] 正式交付文件已同步到项目根目录
```

---

## 20. 研究口径决策状态

### 20.1 已确认事项

1. 计算中心滚动窗口 `roll3_centered=[t-1,t+1]`；2014年使用2014—2015年，2025年使用2024—2025年。该口径包含未来一年信息，不称为历史权重或滞后指标。
2. 主锚点采用严格LLM与Transformer表达：保留大语言模型、大型语言模型、LLM、LLMs、Transformer模型、Transformer架构及其英文限定短语；删除裸词大模型、裸词Transformer、裸词Transformers和变换器模型。
3. 正式Qwen模型不预先指定，通过RTX 4090 24GB上的技术基准测试确定，确定后固定revision并用于全部正式候选发现。
4. 接受本指南的B/C级自动分级阈值；B级稳定出现定义为同一规范技能在至少100个不同规范化岗位描述文本中重复出现。
5. 基准候选发现样本按年份和行业分层，每年抽取1万条；随后每轮增加1万条，连续两轮新增有效概念少于5项且覆盖率提升低于0.1个百分点后停止。
6. 原始共现率构成主结果，Beta-Binomial平滑作为稳健性。
7. 企业留一稳健性只计算主锚点，覆盖年度、全期和中心三年窗口，使用原始权重。
8. 岗位名称不进入任何锚点版本，只使用岗位描述。
9. 无技能岗位保留，连续得分设为缺失，AI岗位标识设为0，并使用 `zero_skill_override=1`标记。
10. 外部英文技能库由Codex读取本地文件或从官方渠道下载缺失文件、解析并分批中文化，生成中英文标准概念及激活别名；Qwen不用于该环节。
11. 原始行业分类直接使用招聘数据自带行业字段；缺失行业进入统一缺失层。
12. Qwen输出的 `skill_type`必须使用正式JSON Schema枚举；字符位置方案可在实际模型测试时切换为程序确定性定位。
13. 只有词典发现语料按同一平台相同描述全局合并；技能共现率和岗位得分使用真实去重招聘广告主样本。
14. 接受规范化精确匹配优先、固定多语言嵌入模型检索Top-10、再由Qwen返回现有ID、`NEW_CONCEPT`或`AMBIGUOUS`的映射机制。
15. 岗位得分不设置最低覆盖率筛选；年度、全期和中心滚动主计算的连接覆盖率按设计必须为1，小于1视为实现错误。
16. 本地 `onet Skills.xlsx`只贡献35个唯一通用技能概念及其职业评分关系；不得把62,580条职业—技能—量表记录解释为62,580项技能。另行获取O*NET Software Skills以补充具体软件与技术名称。

### 20.2 尚需确认事项

本轮方法口径已全部确认。实际执行阶段仍需通过RTX 4090技术基准确定Qwen与多语言嵌入模型的具体版本；这属于实施参数选择，不改变本指南的统计口径。

---

## 21. 方法依据

本指南综合借鉴：

1. 方颖、王翔宇、叶梦芊、赵西亮《中国企业的人工智能投入：来自招聘大数据的发现》正文及附录；
2. Babina、Fedyk、He和Hodson关于技能AI共现率和岗位AI相关度的方法；
3. 本项目关于中文技能本地化、Qwen候选发现、词典确定性匹配和多时间窗口测算的既定讨论。

本指南的主要改进包括：

- 不将BERT训练设为必要步骤；
- 使用Codex读取本地O*NET文件，并下载、解析、中文化和审计缺失的ESCO、O*NET Software Skills及可选Lightcast，形成可冻结的中英文基础词典；
- 使用RTX 4090上的本地Qwen发现和归一化中国招聘语料中的本土技能；
- 区分新表面形式和新技能概念；
- A/B/C全部进入正式词典；
- 不使用Qwen逐条消歧；
- 同时保留项目主口径、中文文献口径和Babina口径；
- 同时计算年度、全期和中心三年窗口共现率，并对样本首尾年份采用相邻两年边界规则；
- 保留原始与平滑权重、连续岗位得分和多阈值AI岗位标识；
- 暂不进行企业—年度聚合。
