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

### 流式与大表扫描

- 全量扫描一律使用 psycopg2 **命名游标**（server-side cursor）+ `fetchmany` 分批，
  禁止普通客户端游标直接迭代大结果集（会整体缓冲进内存导致 OOM）
- 参考 `stream_penetration.py` / `penetration_detail.py` 的既有模式
- 大 join 前 `SET LOCAL work_mem = '1GB'`（见 `company_airatio` / `penetration_detail`）

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

## 9. 给代理的简短提醒

- 本项目与 Employ26 已分离：不要再 import ` Employ26 的任何模块，也不要把新代码加回那边
- 不确定 SQL 影响范围时先用 `EXPLAIN` / 只读查询验证
- 涉及判定口径的结论性数字，写报告前先注明所用 `--method` 与 ω 快照时间戳
