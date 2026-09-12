"""共享工具：日志配置、eps 数据库连接参数与默认产物路径。

本模块收拢原先散落在各入口脚本中重复的 setup_logging 与
`params["dbname"] = "eps"` 连接覆盖。本项目唯一数据源是 eps 数据库，
连接参数统一来自 config/paths.get_project_paths()。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import psycopg2

from config.paths import get_project_paths

# ωsAI 分数快照的统一默认值（各入口一致；如需用其他快照请显式传 --omega-file）
DEFAULT_OMEGA_SNAPSHOT = "output/reports/omega_ai_scores_20260819_105630.json"


def setup_logging(log_path: Path | None = None) -> None:
    """幂等配置根 logger 的终端与可选文件输出。

    可在同一进程内多次调用：不会重复添加 stdout handler；后续首次传入
    ``log_path`` 时仍会补装文件 handler。根 logger 设为 DEBUG，使文件 handler
    能记录 DEBUG；终端 handler 自身限制为 INFO。

    Args:
        log_path: 日志文件路径；为空时仅确保终端输出。
            文件 handler 记录 DEBUG 级，终端记录 INFO 级。
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    has_stdout = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, "stream", None) is sys.stdout
        for h in root.handlers
    )
    if not has_stdout:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)
        root.addHandler(console)

    if log_path is None:
        return

    log_path = Path(log_path).resolve()
    has_file = any(
        isinstance(h, logging.FileHandler)
        and Path(getattr(h, "baseFilename", "")).resolve() == log_path
        for h in root.handlers
    )
    if has_file:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def eps_conn_params() -> dict:
    """返回 eps 岗位数据库的 psycopg2 连接参数（dict 拷贝）。"""
    return dict(get_project_paths().pg_connection_params)


def eps_connect() -> psycopg2.extensions.connection:
    """建立 eps 岗位数据库连接。"""
    return psycopg2.connect(**eps_conn_params())


def resolve_artifact_path(value: str | Path, *, artifact: str) -> Path:
    """解析产物文件路径：绝对/相对当前目录优先，其次回退到项目根相对路径。

    Args:
        value: 用户传入的路径（CLI 参数）。
        artifact: 产物名（用于日志）。

    Returns:
        存在的产物文件路径。

    Raises:
        FileNotFoundError: 两处都找不到该文件时抛出。
    """
    path = Path(value)
    paths = get_project_paths()
    if not path.exists():
        fallback = paths.project_root / value
        if fallback.exists():
            return fallback
    if not path.exists():
        raise FileNotFoundError(
            f"找不到{artifact}文件: {value}（也尝试了项目根相对路径 {paths.project_root / value}）"
        )
    return path
