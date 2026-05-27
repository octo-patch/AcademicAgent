"""LLM 模型确认日志：每个 (caller, requested, returned) 三元组只打印一次，避免批量任务刷屏。"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_seen: set[tuple[str, str, str]] = set()


def log_model_once(caller: str, requested: str, response) -> None:
    returned = getattr(response, "model", "<unknown>") or "<unknown>"
    key = (caller, requested, returned)
    if key in _seen:
        return
    _seen.add(key)
    if requested == returned:
        logger.info("[LLM] %s -> %s (server confirmed)", caller, returned)
    else:
        logger.warning(
            "[LLM] %s requested=%s but server returned=%s (silent remap)",
            caller, requested, returned,
        )
