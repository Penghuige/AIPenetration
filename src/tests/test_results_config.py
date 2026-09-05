"""结果库独立配置与 eps 源库保护（离线测试，不连数据库）。"""
from __future__ import annotations

import pytest

from config.paths import get_project_paths
from src.ai_penetration.import_penetration_results import check_target_db


def _clear_env(monkeypatch):
    """清空结果库相关环境变量，保证解析走配置文件/默认值。"""
    for key in (
        "AIPEN_RESULTS_DBNAME", "AIPEN_RESULTS_HOST", "AIPEN_RESULTS_PORT",
        "AIPEN_RESULTS_USER", "AIPEN_RESULTS_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)


def test_results_dbname_defaults_isolated_from_source(monkeypatch):
    """默认配置下结果库与 eps 源库不同名。"""
    _clear_env(monkeypatch)
    paths = get_project_paths()
    assert paths.pg_dbname == "eps"
    assert paths.results_dbname != paths.pg_dbname
    assert paths.results_dbname == "ai_pen_results"


def test_results_params_follow_source_instance(monkeypatch):
    """结果库默认沿用源库 host/port/user/password，仅库名独立。"""
    _clear_env(monkeypatch)
    paths = get_project_paths()
    for key in ("host", "port", "user"):
        assert paths.results_connection_params[key] == paths.pg_connection_params[key]


def test_results_dbname_env_override(monkeypatch):
    """AIPEN_RESULTS_DBNAME 环境变量优先于配置文件。"""
    monkeypatch.setenv("AIPEN_RESULTS_DBNAME", "my_results")
    paths = get_project_paths()
    assert paths.results_dbname == "my_results"
    assert "my_results" in paths.results_pg_sqlalchemy_url()


def test_check_target_db_rejects_source_collision(monkeypatch):
    """结果库解析回源库时拒绝写入。"""
    _clear_env(monkeypatch)
    paths = get_project_paths()
    with pytest.raises(RuntimeError, match="拒绝"):
        check_target_db(
            dict(paths.results_connection_params, dbname="eps"),
            paths.pg_connection_params,
            allow_source_db=False,
        )


def test_check_target_db_allows_explicit_override():
    """--allow-source-db 显式豁免时放行。"""
    params = {"host": "localhost", "port": 5432, "dbname": "eps"}
    check_target_db(dict(params), dict(params), allow_source_db=True)


def test_check_target_db_passes_when_distinct():
    """结果库与源库不同名时正常放行。"""
    source = {"host": "localhost", "port": 5432, "dbname": "eps"}
    results = {"host": "localhost", "port": 5432, "dbname": "ai_pen_results"}
    check_target_db(results, source, allow_source_db=False)
