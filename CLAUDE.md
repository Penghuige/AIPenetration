# CLAUDE.md

> 创建: 2026-08-27
> 适用对象: Codex、Claude Code、Cursor Agent 等代码代理

本文件是本仓库的代理协作规范。它不负责介绍项目背景；项目目标与运行方式见 `README.md`。
代理在开始实现、重构、修复、评审前，应优先遵守本文件中的硬约束。

## 1. 文档分工

- `README.md` 面向人类读者，说明项目目标、目录结构、快速开始
- `CLAUDE.md` 面向代码代理，说明修改边界、架构约束、验证要求
- 两者冲突时，以用户当次指令为最高优先级；未特别说明时优先遵守本文件

## 2. 项目快照

- 项目主题: AI 岗位探索 / 公司级 AI 投入强度（AIRatio）分析
- 数据源: 独立的 PostgreSQL `eps` 岗位数据库（广东省 21 地级市 job_* 分片表 + ai_dict schema）
- Python 版本: 3.10+
- 由 `Employ26` 仓库的 `src/ai_penetration` 分离而来（2026-08-27），已完全独立，不依赖原仓库

## 3. 必须遵守的架构约束

### 数据存储

- PostgreSQL 是唯一正式数据库，本项目只连接 `eps` 库（默认 `localhost:5432`）
- 连接参数统一来自 `config/database.yaml`（环境变量 `AIPEN_PG_*` 可覆盖）
- 代理**不得**执行 `pg_ctl start/stop/restart`，不得启动/停止/杀掉 PostgreSQL 进程；
  连接失败时报告状态，由用户自行处理
- SQLAlchemy URL 一律通过 `paths.pg_sqlalchemy_url()` 获取，禁止手拼含密码的字符串
- 词典导入、频数计算等写入操作的目标 schema 是 `ai_dict`；
  临时验证数据如需写库，放入独立测试库或先征询用户，禁止污染正式表

### 路径与配置

- 禁止硬编码绝对路径和连接字符串
- 所有路径从 `config.paths.get_project_paths()` 获取：
  - `project_root` / `output_dir` / `report_dir` / `log_dir` / `dict_dir`
  - `pg_connection_params` / `pg_sqlalchemy_url()`
- LLM 配置在 `config/model_runtime.yaml`（环境变量 `AIPEN_LLM_*` 可覆盖），
  client 统一用 `src.model_platform.llm.create_llm_client()`，不要在业务脚本里直连 API
- 运行日志统一写入 `logs/`；中间产物与结构化产出写入 `output/reports/`
- 共享工具（`setup_logging`、`eps_conn_params`、`eps_connect`、
  `DEFAULT_OMEGA_SNAPSHOT`）在 `src/ai_penetration/common.py`，不要在各模块重复实现

### 判定口径一致性（重要）

AI 岗位判定有多条方法链，修改任何一条必须检查其余是否需要同步：

- 方法 A（加权评分）: `ai_scoring.is_ai_job` / `match_skills_scored`
- 方法 B/fused（锚点共现）: `skill_ai_anchor.extract_skills_fast` / `is_ai_fused`
- 口语同形词排除规则（如"深度学习能力"≠深度学习）必须同时体现在
  `ai_scoring._AI_CONTEXT_REQUIRED` 与 `skill_ai_anchor._AMBIGUOUS_AI_TERMS`
- checkpoint（断点文件）按方法区分命名；改动统计口径后必须废弃旧断点

### 流式与大表扫描（2026-09 两轮全国跑修订）

- 扫描策略按表大小分级（HDD + PG 并行的实测最优，勿一刀切）：
  - **≥8GB**：命名游标 + ctid 物理块切片 8 路并行
  - **2-8GB**：命名游标 + 4 路切片
  - **<2GB**：普通 SELECT 客户端直读 + `SET LOCAL max_parallel_workers_per_gather=4`
    （命名游标 DECLARE 会**禁用 PG 并行扫描**，小表下这是 2-3 倍的速度差来源）
- ctid 切片总块数必须用 `pg_relation_size / current_setting('block_size')` 计算，
  **不可用 `pg_class.relpages`**（库未 ANALYZE，实测为 0）
- `SET LOCAL` 必须走匿名 cursor：命名游标会把首条语句包进
  `DECLARE ... FOR <stmt>`，SET 语法直接报错
- 普通客户端游标仍禁止整表 fetchall 超大结果集（内存 OOM 风险按表大小评估）
- 大 join/排序前 `SET LOCAL work_mem = '256MB'~'1GB'`；
  **避免 47M 行级 DISTINCT ON**（排序溢写 tuplestore 会拖爆 HDD，实测 BuffileRead 卡死）
- ent 表约 1% `recruit_id` 重复：任何行业/公司映射必须去重取单条
  （LATERAL LIMIT 1 或"稳定排序+searchsorted 首条"，两者语义已验证等价，重复行行业码 100% 一致）
- 每行 Python 循环里**禁止逐行做字符串键构造 + 多桶 dict 更新**（行业版首跑因此慢 2.2 倍）：
  正确姿势是按 (年,月,行业) 分组计数、批内对数百唯一键统一展开聚合桶（见 `fused_industry_national._process_batch`）；
  行级计数器必须在 **fetch 后立即累加**，不能只在弹出时累加（单批小表会被误判 0 行）

### LLM 批量调用

- 批量 prompt 的返回结果必须校验条数与批次一致，缺失条目显式补空并告警，
  禁止依赖 `zip` 静默截断（参考 `standardize_jobs.standardize_positions`）

## 4. 修改边界

### 优先修改的位置

- 当前任务直接涉及的模块
- `src/tests/` 下相关测试
- `config/` 下相关配置

### 默认不要动

- `GD_SHARDS` 分片映射（探索自 eps 库，见 `load_guangdong.py` 注释）
- 词典文件的已有词条与权重分层（`dicts/*.txt`）；新增词条走单独 commit 并说明依据
- ωsAI 快照默认值 `common.DEFAULT_OMEGA_SNAPSHOT`；换版本需所有入口同步且说明来源

## 5. 导入与运行方式

```bash
# 从项目根目录以 -m 方式运行，禁止 sys.path 补丁、禁止 from module import *
python -m src.ai_penetration --help
python -m src.ai_penetration.company_airatio --method fused --min-jobs 10
python -m src.ai_penetration.per_city_keyword --min-freq 10
```

包内使用相对导入；跨层复用走 `from .common import ...`。

## 6. 代码风格规范

- 变量、函数、模块名英文；注释与 docstring 中文
- 公开类/函数使用 Google 风格 docstring（功能/参数/返回值/异常）
- 新增公开接口带类型提示；新增 `.py` 含模块级 docstring
- 工具: `black`、`isort`、`flake8`

## 7. 验证命令

```bash
python -m compileall -q src config
python -m flake8 src config --select=F,E9 --max-line-length=110
pytest src/tests/ -q
```

涉及公共逻辑（`common.py`、`ai_scoring.py`、`skill_ai_anchor.py`）的改动跑全量 pytest；
局部脚本改动至少完成 compileall + flake8。

## 8. 提交约定

- Conventional Commits（feat/fix/refactor/docs/test/chore）
- 大改前确认工作区干净；checkpoint JSON、日志、CSV 报告不入库（见 `.gitignore`）

## 9. 机器特性与长跑实验规范（2026-09 全国面板两轮的实测教训）

### 硬件画像（规划任何全量跑批前先对照）

- CPU 128 逻辑核 / 内存 137GB，但**数据库盘是机械硬盘（HDD）**：
  聚合读吞吐上限 ~100-150MB/s，多流并发是"分磁头"不是"加带宽"（8 流是上限，小表用 1-2 流）
- CPU 与磁盘谁饱和看证据再调参：`psutil` 双采样磁盘 MB/s + 逐进程 `cpu_percent`；
  双城并发时评分 worker 满载（24×101%）= CPU 端瓶颈，磁盘空闲 = 结构等待，加并发无用
- 本机还承载 WSL vLLM（GPU）：**LLM 挖掘/批量调用先确认 vLLM 空闲**，勿与重型提取任务抢窗口

### 长任务编排（>1 小时必守）

- **断点粒度=最小工作单元**（城市/年），单元完成即落盘；重试前**清空该单元中间态**
- 失败重试必须有上限（实测教训：无上限的死循环白烧 5 小时）；重试间隔内用
  `pg_stat_activity` / CPU 增量判断是"在跑"还是"卡死"
- **"日志静默 ≠ 卡死"**：ctid 切片/并行直读的首批要等物化，大城市 10-40 分钟无日志属正常；
  误判杀进程 = 重跑一整城
- 单实例保护写进生产脚本（扫 `psutil` 同命令行进程，晚启动者退出）；
  勿与 harness 的后台任务重生循环对抗——锁逻辑收敛，后启动者自动让位
- 长跑基准（bench）窗口要扣除物化暖机期，否则测出的是暖机稀释吞吐（会误导调参）
- 架构改动后必须做**新旧实现逐值回归**（小城 1 分钟级即可做），
  并用已验收锚点（如 fused_industry 的 89/98 组）交叉核对后才允许全量点火

### Windows 平台陷阱

- 子进程环境**必须继承完整 `os.environ`** 再追加变量：只传最小 env 会导致
  Windows 子进程 Winsock 解析 `localhost` 失败（报"Non-recoverable failure in name resolution"，
  极具迷惑性，会被误判为 PG 故障）
- 控制台/管道默认 GBK：长任务脚本统一 `python -X utf8` 启动，中文日志输出走文件而非 print
- bash 命令行里传中文参数给 `python -c` 会 mojibake（argv 按 GBK 解码）——
  匹配中文文件名改用内容特征（如 cells 数量）而非字符串匹配
- worker `mmap_mode='r'` 持有的 npy 文件在 Windows 上无法即时 unlink：
  删除操作容错（try/except OSError）+ 进程退出后统一清理

### eps 数据特性（分析口径层面）

- **2024 年数据约止于 10 月初**：Q4 桶普遍 <5000 行，季度/半年度解读 2024 需标注截断
- `publish_time` 有脏数据（"代…"等）：年份解析必须 `^\d{4}` 白名单式匹配，不可 cast
- `city` 字段命名不规范（"天水"无市后缀、区县合并片多县连写名），全国 392 片
  以 `fused_national.enumerate_national_tables` 动态枚举为准，`GD_SHARDS` 仅是广东子集
- 大表默认未 ANALYZE：`reltuples=-1` 不可用于估行数，采样用 `TABLESAMPLE SYSTEM`

## 10. 给代理的简短提醒

- 本项目与 Employ26 已分离：不要再 import Employ26 的任何模块，也不要把新代码加回那边
  （历史代码快照在 Employ26 `archive/ai_penetration/`，通用技能词典已静态化为 `dicts/skill_names_general.txt`）
- 不确定 SQL 影响范围时先用 `EXPLAIN` / 只读查询验证
- 涉及判定口径的结论性数字，写报告前先注明所用 `--method` 与 ω 快照时间戳
- 全量跑批前：读 §3 流式规范 + §9 机器画像，跑完先做守恒核验（三粒度总数相等、
  组内还原、跨面板锚点一致）再交付数字
