"""
Agent Runtime - 结构化偏好
==========================

memory/preferences.json 保存"机器读"偏好，独立于 USER.md 自由文本：
  {
    "preferred_search_source": "arxiv",       # arxiv / semantic_scholar / openreview
    "figure_analysis_default": "on_request",  # on_request / always / never
    "default_domain": "llm_agent",            # llm_agent / comms / shared
    "language": "zh",                          # zh / en
    "reflector_enabled": true
  }

读：每次 workflow 开跑前 load_preferences()，用偏好补 args 默认值。
写：update_preference(key, value)，原子写。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("ScholarMind.Runtime.Preferences")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PREF_PATH = _PROJECT_ROOT / "memory" / "preferences.json"

DEFAULT_PREFERENCES: dict[str, Any] = {
    "preferred_search_source": "arxiv",
    "figure_analysis_default": "on_request",
    "default_domain": "llm_agent",
    "language": "zh",
    "reflector_enabled": True,
}

ALLOWED_VALUES: dict[str, set] = {
    "preferred_search_source": {"arxiv", "semantic_scholar", "openreview", "crossref", "dblp"},
    "figure_analysis_default": {"on_request", "always", "never"},
    "default_domain": {"llm_agent", "comms", "shared"},
    "language": {"zh", "en"},
}


def load_preferences(path: Path | None = None) -> dict[str, Any]:
    """读取偏好，缺失字段用默认值补齐（不写盘）。"""
    p = path or _PREF_PATH
    data: dict[str, Any] = dict(DEFAULT_PREFERENCES)
    if p.exists():
        try:
            data.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            logger.warning("preferences read failed (%s), using defaults", e)
    return data


def save_preferences(prefs: dict[str, Any], path: Path | None = None) -> None:
    """原子写。"""
    p = path or _PREF_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    logger.info("preferences saved to %s", p)


def update_preference(key: str, value: Any, path: Path | None = None) -> dict[str, Any]:
    """更新一个偏好键。若值不在 ALLOWED_VALUES 限定集内会抛 ValueError。"""
    if key not in DEFAULT_PREFERENCES:
        raise ValueError(
            f"unknown preference key: {key!r}. "
            f"Allowed: {sorted(DEFAULT_PREFERENCES.keys())}"
        )
    if key in ALLOWED_VALUES and value not in ALLOWED_VALUES[key]:
        raise ValueError(
            f"value {value!r} not allowed for {key}. "
            f"Allowed: {sorted(ALLOWED_VALUES[key])}"
        )
    prefs = load_preferences(path)
    prefs[key] = value
    save_preferences(prefs, path)
    return prefs
