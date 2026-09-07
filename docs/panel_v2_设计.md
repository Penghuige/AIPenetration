# 面板 v2 设计（交接指南 §12–§18 落地）

> 目标：按《岗位AI技能共现率执行指南》口径重建 AI 岗位识别管线，替代 v1 自建管线
> （fused_cities/national）。本文是条款→模块的唯一映射，改口径必须先改本文。

## 1. 指南条款 → 模块映射

| 指南 | 内容 | 模块 |
|---|---|---|
| §6.2 | 岗位去重主规则（platform+job_id_raw 唯一；同企业+岗位名+城市+desc哈希 ≤30天归组） | `panel_v2/dedup.py` |
| §7/词典 | 技能词典 = A 级冻结概念 ∪ 自建技术词（union） | `panel_v2/lexicon.py` |
| §8–§10 | B/C/D 级技能发现（LLM 合规挖掘，周期性） | `gold_standard_eval.py` 缺口模块复用扩展，后期 |
| §12 | 三套锚点（main 六组 / cn_paper 四组 / babina 四组）+ §12.6 匹配 + §12.6.1 岗位名不进锚点 | `panel_v2/anchors.py` |
| §13 | annual/pooled/roll3_centered 计数 + §13.6 顺序 + §13.7 不变量 | `panel_v2/counts.py` |
| §14 | 9 组权重 raw + smoothed（逐 anchor×window×period 单元拟合 Beta-BB，n≥5，回退 Jeffreys，不覆盖 raw）+ §14.3 企业留一（主锚点） | `panel_v2/relevance.py` |
| §15 | 岗位得分 = 去重技能权重简单平均；coverage=1 硬约束；raw+smoothed 双套 | `panel_v2/scoring.py` |
| §16 | 三阈值严格大于；零技能岗位（缺失≠0、zero_skill_override）；主标识 `aijob_main_annual_raw_005` | `panel_v2/scoring.py` |
| §17 | 自动化质量门（不强制金标准；§17.3–17.6 检查 + 转换矩阵） | `panel_v2/quality.py` |
| §18 | 11 件 parquet + 同名 metadata.json + QC 报告 | `panel_v2/export_release.py` |
| §19.1 | 脚本纪律：config 驱动、--dry-run/--resume/--run-id、拒绝覆盖、非零退出 | `panel_v2/run_gz.py` 编排 |

## 2. 关键决策（已定）

1. **主口径 = raw 得分**（§15.4 文献可比），平滑得分与三阈值、三锚点、三口径全部作为
   替代/稳健性列保留；主 AI 岗位变量 = `aijob_main_annual_raw_005`（§16.4）。
   注意：v1 的 is_ai_job（A 加权）与 B(0.15&max0.5) 均**不是**指南口径，
   v2 数字预期与 v1 面板有系统性差异（阈值 0.05 vs 0.15、锚点集合不同）。
2. **先广深后全国**：v2a 范围=广深 2014–2024（eps 无 2025，已在对照清单记录）；
   端到端+守恒核验通过后，扩全国由用户排窗。
3. **union 技能空间**：A 级 22,683 概念为主键空间（skill_id uuid）；自建表
   （6,872 词中未被 A 级别名覆盖者，pytorch/tensorflow/halcon 等）以
   `legacy:<term>` 合成 id 并入，全部技能进词典（§13.4 不因低频删除）。
4. **锚点自身技能**：保留在岗位得分内（§12.7），其 raw 共现率≈1 是公式推论非 bug。
5. **数据规模预算**：广深原始 ~1.01 亿行 → 去重后岗位主表预估 3–7 千万；
   job_skill_long 预估 2–5 亿行（int 编码 + parquet，不物化字符串）。
6. **v1 面板不删除**：v2 验收前现行 CSV 面板与结果库表继续作为发表口径。

## 3. 里程碑

- M1（本轮）：anchors.py + §12.6 含反例单测 + config/panel_v2.yaml + 本设计
- M2：lexicon.py（union 词典 + ai_dict/legacy 加载）、dedup.py（job 主表构建，读侧大扫描）
- M3：scan.py（锚点 flag + 长表，ctid 并行）、counts/relevance/scoring（纯计算层，numpy/pandas）
- M4：quality.py + export_release.py + 广深端到端 + 与 v1/面板锚点交叉核对（守恒后才出数）

## 4. 风险与纪律

- 去重规则依赖 company 字段（eps 有 company_name/company_name2，无 company_id——
  M2 需先探查字段质量，必要时用 ent 表 recruit_id→company_id 映射，注意 ent 重复）
- publish_time 脏数据：年份解析用 `^\d{4}` 白名单（CLAUDE.md §9）
- 30 天归组需要组内排序——用「desc哈希+城市+企业 分组内按时间差压缩」的 SQL 实现，
  避免全表 DISTINCT ON（CLAUDE.md 流式规范）
- 长任务断点=最小工作单元；单实例保护；跑完先守恒核验再交付数字
