"""Helpers for loading frontend assets from the local project and workspace roots."""

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def dist_asset_path(filename: str) -> Path:
    return project_root() / "dist" / filename


def shared_css_text() -> str:
    local_css = project_root() / "ts" / "reinforcement-learning.css"
    if local_css.exists():
        return local_css.read_text()
    return (workspace_root() / "ts" / "reinforcement-learning.css").read_text()
