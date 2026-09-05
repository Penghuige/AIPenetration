"""集中化的项目路径与 PostgreSQL 配置管理。

本模块提供 `ProjectPaths` dataclass，将 PostgreSQL 连接参数和输出路径
集中到单一配置对象中。本项目（AIPenetration）的数据源是独立的 eps 数据库，
不再复用 Employ26 主库的连接配置。

配置优先级:
    显式参数 > 环境变量 > config/database.yaml > 模块内默认值

环境变量覆盖:
    AIPEN_PG_HOST     — PostgreSQL 主机地址
    AIPEN_PG_PORT     — PostgreSQL 端口
    AIPEN_PG_DBNAME   — 岗位数据库名（默认 eps）
    AIPEN_PG_USER     — 用户名
    AIPEN_PG_PASSWORD — 密码
    AIPEN_RESULTS_DBNAME — 结果库名（默认 ai_pen_results，与 eps 源库隔离）
    AIPEN_RESULTS_HOST/PORT/USER/PASSWORD — 结果库整体改向（默认沿用源库参数）
    AIPEN_LLM_BASE_URL — LLM OpenAI-compatible API 地址
    AIPEN_LLM_MODEL    — LLM 服务模型名
    AIPEN_LLM_API_KEY  — LLM API key
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus


def _project_root() -> Path:
    """返回项目根目录（config/ 的父目录）。"""
    return Path(__file__).resolve().parent.parent


def _parse_scalar(value: str) -> Any:
    """解析本项目配置文件中使用的简单 YAML 标量。"""
    text = value.strip()
    if text in {"", "null", "None"}:
        return ""
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text


def _simple_yaml_load(path: Path) -> dict[str, Any]:
    """读取不依赖第三方库的简单 YAML 配置。

    只支持映射、列表和标量，覆盖 `config/database.yaml` 的使用范围。
    """
    parsed_lines: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        parsed_lines.append((indent, raw.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for index, (indent, line) in enumerate(parsed_lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"YAML 列表缩进不合法: {line}")
            parent.append(_parse_scalar(line[2:]))
            continue

        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_is_list = False
        if index + 1 < len(parsed_lines):
            next_indent, next_line = parsed_lines[index + 1]
            next_is_list = next_indent > indent and next_line.startswith("- ")

        container: Any = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))

    return root


def load_config_yaml(name: str) -> dict[str, Any]:
    """加载项目根目录 `config/` 下的 YAML 配置文件。

    Args:
        name: 配置文件名，例如 ``database.yaml``。

    Returns:
        dict[str, Any]: 解析结果。文件不存在时返回空字典。
    """
    target = _project_root() / "config" / name
    if not target.exists() or not target.read_text(encoding="utf-8").strip():
        return {}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return _simple_yaml_load(target)


@dataclass(frozen=True)
class ProjectPaths:
    """项目集中配置对象。

    PostgreSQL 连接参数（源库默认即 eps 岗位库）、独立结果库连接参数
    与输出路径的统一来源。每个字段都可通过环境变量覆盖，
    默认值基于项目根目录的相对路径。
    """

    project_root: Path
    pg_connection_params: dict = field(default_factory=dict)
    results_connection_params: dict = field(default_factory=dict)
    output_dir: Path = field(default_factory=Path)
    dict_dir: Path = field(default_factory=Path)
    config_dir: Path = field(default_factory=Path)

    @property
    def pg_host(self) -> str:
        """PostgreSQL 主机地址。"""
        return self.pg_connection_params.get("host", "localhost")

    @property
    def pg_port(self) -> int:
        """PostgreSQL 端口。"""
        return int(self.pg_connection_params.get("port", 5432))

    @property
    def pg_dbname(self) -> str:
        """PostgreSQL 数据库名。"""
        return self.pg_connection_params.get("dbname", "eps")

    @property
    def pg_user(self) -> str:
        """PostgreSQL 用户名。"""
        return self.pg_connection_params.get("user", "postgres")

    @property
    def pg_password(self) -> str:
        """PostgreSQL 密码。"""
        return self.pg_connection_params.get("password", "")

    @property
    def log_dir(self) -> Path:
        """运行日志目录。"""
        return self.project_root / "logs"

    @property
    def report_dir(self) -> Path:
        """报告输出目录。"""
        return self.output_dir / "reports"

    def pg_sqlalchemy_url(self, dbname: str | None = None) -> str:
        """返回 SQLAlchemy 使用的 PostgreSQL 连接 URL。

        Args:
            dbname: 可选目标数据库；为空时使用默认连接参数中的库名。
                凭据会做 URL 编码，避免特殊字符破坏 URL 结构。

        Returns:
            str: SQLAlchemy 连接 URL。
        """
        return self._url_for(self.pg_connection_params, dbname)

    @property
    def results_dbname(self) -> str:
        """结果数据库名（与 eps 源库隔离，存放分析产出表）。"""
        return self.results_connection_params.get("dbname", "ai_pen_results")

    def results_pg_sqlalchemy_url(self) -> str:
        """返回结果库的 SQLAlchemy 连接 URL。

        结果库默认与源库同实例（host/port/user/password 沿用），
        仅 dbname 独立；可通过 AIPEN_RESULTS_* 环境变量整体改向。

        Returns:
            str: SQLAlchemy 连接 URL。
        """
        return self._url_for(self.results_connection_params, None)

    @staticmethod
    def _url_for(params: dict, dbname_override: str | None) -> str:
        """按连接参数 dict 生成 URL 编码后的 SQLAlchemy 连接串。"""
        target_db = dbname_override or params.get("dbname", "eps")
        user = quote_plus(str(params.get("user", "postgres")))
        password = quote_plus(str(params.get("password", "")))
        auth = f"{user}:{password}" if password else user
        return (
            f"postgresql+psycopg2://{auth}@{params.get('host', 'localhost')}"
            f":{params.get('port', 5432)}/{target_db}"
        )


def get_project_paths(
    *,
    pg_params: dict | None = None,
) -> ProjectPaths:
    """构建 `ProjectPaths` 实例，优先使用显式参数，其次环境变量，最后配置文件。

    Args:
        pg_params: 覆盖 PostgreSQL 连接参数（dict，键: host/port/dbname/user/password）。

    Returns:
        ProjectPaths: 冻结的路径与连接配置对象。
    """
    root = _project_root()
    yaml_config = load_config_yaml("database.yaml")
    yaml_database = yaml_config.get("database", {}) if isinstance(yaml_config, dict) else {}
    if pg_params is None:
        pg_params = {}

    def _resolve_setting(key: str, env_key: str, default: Any) -> Any:
        """按 显式参数 > 环境变量 > YAML > 默认值 解析配置值。"""
        if key in pg_params and pg_params[key] is not None:
            return pg_params[key]
        env_val = os.getenv(env_key)
        if env_val:
            return env_val
        if key in yaml_database and yaml_database[key] is not None:
            return yaml_database[key]
        return default

    pg_connection_params = {
        "host": _resolve_setting("host", "AIPEN_PG_HOST", "localhost"),
        "port": int(_resolve_setting("port", "AIPEN_PG_PORT", 5432)),
        "dbname": _resolve_setting("dbname", "AIPEN_PG_DBNAME", "eps"),
        "user": _resolve_setting("user", "AIPEN_PG_USER", "postgres"),
        "password": str(_resolve_setting("password", "AIPEN_PG_PASSWORD", "")),
    }

    # 结果库：默认与源库同实例，仅库名独立（yaml database.results_db）。
    # 各键均可用 AIPEN_RESULTS_* 覆盖，未覆盖时沿用源库参数。
    yaml_results_db = (
        yaml_database.get("results_db") if isinstance(yaml_database, dict) else None
    )
    results_connection_params = {
        "host": os.getenv("AIPEN_RESULTS_HOST") or pg_connection_params["host"],
        "port": int(
            os.getenv("AIPEN_RESULTS_PORT") or pg_connection_params["port"]
        ),
        "dbname": (
            os.getenv("AIPEN_RESULTS_DBNAME")
            or str(yaml_results_db or "ai_pen_results")
        ),
        "user": os.getenv("AIPEN_RESULTS_USER") or pg_connection_params["user"],
        "password": str(
            os.getenv("AIPEN_RESULTS_PASSWORD", pg_connection_params["password"])
        ),
    }

    return ProjectPaths(
        project_root=root,
        pg_connection_params=pg_connection_params,
        results_connection_params=results_connection_params,
        output_dir=root / "output",
        dict_dir=root / "dicts",
        config_dir=root / "config",
    )
