"""panel_v2 正式运行的可复现性清单工具（指南 §4.2）。

本模块只记录 provenance，不参与任何指标计算。正式发布可用它把一次运行的
代码/配置、输入文件与输出文件统一绑定到 ``run_manifest.json``。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

from config.paths import get_project_paths, load_config_yaml


def sha256_file(path: Path) -> str:
    """返回文件 SHA-256。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_row_count(path: Path) -> int | None:
    """尽量读取产物逻辑行数；未知格式返回 None。"""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    if path.suffix == ".csv":
        # 发布 CSV 均含一行表头。
        with path.open("r", encoding="utf-8-sig") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    if path.suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if isinstance(obj, (list, dict)):
            return len(obj)
    return None


def _code_fingerprint(root: Path) -> str:
    """无 Git 元数据时，对运行代码与口径配置生成稳定指纹。"""
    h = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "config").glob("*.yaml"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def code_revision(root: Path | None = None) -> str:
    """优先返回 Git HEAD；不可用时返回内容指纹。"""
    root = root or get_project_paths().project_root
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = proc.stdout.strip()
        if sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "sha256:" + _code_fingerprint(root)


def _records(paths: Iterable[Path], root: Path) -> tuple[list[str], dict[str, str], dict[str, int | None]]:
    """把一组文件转换成 manifest 的路径/hash/行数三张表。"""
    names: list[str] = []
    hashes: dict[str, str] = {}
    rows: dict[str, int | None] = {}
    for path in paths:
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"run manifest 输入/输出文件不存在: {p}")
        try:
            name = p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            name = str(p.resolve())
        names.append(name)
        hashes[name] = sha256_file(p)
        rows[name] = artifact_row_count(p)
    return names, hashes, rows


def write_run_manifest(
    release_dir: Path,
    *,
    run_id: str,
    started_at: datetime,
    input_paths: Iterable[Path],
    output_paths: Iterable[Path],
    dictionary_version: str,
    anchor_version: str,
    status: str = "complete",
) -> Path:
    """生成指南 §4.2 的正式 ``run_manifest.json``。

    本函数不修改任何数据产物，只在所有输入/输出均存在时记录其 SHA-256 与
    行数，因此缺件会 fail-closed。
    """
    paths = get_project_paths()
    root = paths.project_root
    inp, inp_sha, inp_rows = _records(input_paths, root)
    out, out_sha, out_rows = _records(output_paths, root)

    config_files = sorted(paths.config_dir.glob("*.yaml"))
    config_sha = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in config_files
    }
    model_cfg = load_config_yaml("model_runtime.yaml")
    llm_cfg = model_cfg.get("llm", {}) if isinstance(model_cfg, dict) else {}

    manifest = {
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit_or_code_hash": code_revision(root),
        "input_file_paths": inp,
        "input_file_sha256": inp_sha,
        "input_row_counts": inp_rows,
        "dictionary_version": dictionary_version,
        "anchor_version": anchor_version,
        "prompt_version": "not_used_in_v2h_release_rebuild",
        "model_revision": str(llm_cfg.get("model", "not_used")),
        "config_file_sha256": config_sha,
        "output_file_paths": out,
        "output_file_sha256": out_sha,
        "output_row_counts": out_rows,
        "status": status,
    }
    release_dir.mkdir(parents=True, exist_ok=True)
    target = release_dir / "run_manifest.json"
    tmp = release_dir / ".run_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target
