"""
通用 Notion REST 客户端
======================

从 src/evaluation/notion_sync.py:124-136 + scripts/notion_sync_engineering_recap.py:98-136
抽取整合，作为后续所有 Notion 同步的统一入口。

特性：
  - Bearer token + Notion-Version 头
  - 自动批量 append（≤100 blocks/批 + 0.4s 间隔避速率限制）
  - 失败优雅降级：所有外部调用经 _safe()，永不抛错
  - 父页面按名称查找（Search API），结果缓存到 data/recorder_state/_parent.json
  - 配置加载优先级：环境变量 > data/evaluation/notion_config.json（向后兼容）

设计准则：
  调用方拿到 None 表示同步失败，本地处理仍继续。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

logger = logging.getLogger("AcademicAgent.Recorder.Notion")

_API = "https://api.notion.com/v1"
_DEFAULT_VERSION = "2022-06-28"
_DEFAULT_TIMEOUT = 30.0
_BATCH_SIZE = 100
_BATCH_DELAY = 0.4

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PARENT_CACHE = _PROJECT_ROOT / "data" / "recorder_state" / "_parent.json"


# ============================================================
# 配置
# ============================================================

@dataclass
class NotionConfig:
    """Notion 同步配置。"""
    api_key: str = ""
    parent_page_id: str = ""
    parent_page_name: str = ""
    notion_version: str = _DEFAULT_VERSION
    enabled: bool = False

    @property
    def is_ready(self) -> bool:
        return self.enabled and bool(self.api_key) and bool(self.parent_page_id or self.parent_page_name)


def load_notion_config() -> NotionConfig:
    """加载 Notion 配置。

    优先级：
      1. 环境变量 NOTION_API_KEY / NOTION_PARENT_PAGE_ID / NOTION_PARENT_PAGE_NAME
      2. data/evaluation/notion_config.json（向后兼容现有 notion_sync.py）

    若调用方未先 load_dotenv()，这里兜底加载一次项目根 .env —— 避免脚本从 Python
    直接 import RefactorRecorder 时静默走本地 md 路径（Phase 5-7 踩过这个坑）。
    """
    if not os.getenv("NOTION_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ROOT / ".env", override=False)
        except Exception:  # noqa: BLE001
            pass

    cfg = NotionConfig()

    # 1. 环境变量（首选）
    cfg.api_key = os.getenv("NOTION_API_KEY", "")
    cfg.parent_page_id = os.getenv("NOTION_PARENT_PAGE_ID", "")
    cfg.parent_page_name = os.getenv("NOTION_PARENT_PAGE_NAME", "")
    cfg.notion_version = os.getenv("NOTION_VERSION", _DEFAULT_VERSION)

    # 2. 向后兼容：data/evaluation/notion_config.json
    legacy_cfg = _PROJECT_ROOT / "data" / "evaluation" / "notion_config.json"
    if legacy_cfg.exists():
        try:
            data = json.loads(legacy_cfg.read_text(encoding="utf-8"))
            if not cfg.api_key:
                cfg.api_key = data.get("api_key", "")
            if not cfg.parent_page_id:
                cfg.parent_page_id = data.get("parent_page_id", "")
            if not cfg.parent_page_name:
                cfg.parent_page_name = data.get("parent_page_name", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("legacy notion_config.json 解析失败: %s", exc)

    cfg.enabled = bool(cfg.api_key) and bool(cfg.parent_page_id or cfg.parent_page_name)
    return cfg


# ============================================================
# 客户端
# ============================================================

class NotionClient:
    """通用 Notion REST 客户端。

    使用示例：
        client = NotionClient(load_notion_config())
        if client.ready:
            parent_id = client.resolve_parent()
            page = client.create_subpage(parent_id, "🔧 KG 治理", first_blocks=[...])
            client.append_blocks(page["id"], more_blocks)
    """

    def __init__(self, config: NotionConfig | None = None, log_path: Path | None = None):
        self.config = config or load_notion_config()
        self._log_path = log_path
        self._parent_id_cache: str | None = None

    # ---------- 基础设施 ----------

    @property
    def ready(self) -> bool:
        return self.config.is_ready

    def _log(self, msg: str) -> None:
        logger.info("[notion] %s", msg)
        if self._log_path:
            try:
                from datetime import datetime, timezone
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(f"{ts} {msg}\n")
            except Exception:  # noqa: BLE001
                pass

    def _safe(self, label: str, fn: Callable, *args, **kwargs) -> Any:
        """包裹一切 Notion 调用：异常 -> 记日志 -> 返回 None。"""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._log(f"FAIL {label}: {exc}")
            return None

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Notion-Version": self.config.notion_version,
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict) -> dict:
        resp = httpx.post(
            f"{_API}{path}", headers=self._headers(),
            json=payload, timeout=_DEFAULT_TIMEOUT,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _patch(self, path: str, payload: dict) -> dict:
        resp = httpx.patch(
            f"{_API}{path}", headers=self._headers(),
            json=payload, timeout=_DEFAULT_TIMEOUT,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _get(self, path: str) -> dict:
        resp = httpx.get(
            f"{_API}{path}", headers=self._headers(), timeout=_DEFAULT_TIMEOUT,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    # ---------- 父页面解析 ----------

    def resolve_parent(self) -> str | None:
        """返回父页面的 page_id。

        优先级：
          1. config.parent_page_id（直接给）
          2. data/recorder_state/_parent.json 缓存
          3. config.parent_page_name → Notion Search API
        """
        if self.config.parent_page_id:
            return self.config.parent_page_id

        # 缓存命中
        if _PARENT_CACHE.exists():
            try:
                data = json.loads(_PARENT_CACHE.read_text(encoding="utf-8"))
                if data.get("name") == self.config.parent_page_name and data.get("page_id"):
                    self._parent_id_cache = data["page_id"]
                    return self._parent_id_cache
            except Exception:  # noqa: BLE001
                pass

        # Search API
        if not self.config.parent_page_name:
            self._log("FAIL resolve_parent: 既无 page_id 也无 page_name")
            return None

        result = self._safe(
            "search parent",
            self._post,
            "/search",
            {
                "query": self.config.parent_page_name,
                "filter": {"value": "page", "property": "object"},
            },
        )
        if not result:
            return None

        for item in result.get("results", []):
            title_blocks = (
                item.get("properties", {}).get("title", {}).get("title")
                or item.get("properties", {}).get("Name", {}).get("title")
                or []
            )
            title = "".join(b.get("plain_text", "") for b in title_blocks)
            if title.strip() == self.config.parent_page_name.strip():
                pid = item["id"]
                self._parent_id_cache = pid
                _PARENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
                _PARENT_CACHE.write_text(json.dumps(
                    {"name": self.config.parent_page_name, "page_id": pid,
                     "url": item.get("url", "")},
                    ensure_ascii=False, indent=2,
                ), encoding="utf-8")
                self._log(f"OK resolved parent {self.config.parent_page_name!r} -> {pid}")
                return pid

        self._log(f"FAIL parent page {self.config.parent_page_name!r} not found in Notion search results")
        return None

    # ---------- 页面操作 ----------

    def create_subpage(
        self,
        parent_id: str,
        title: str,
        first_blocks: list[dict],
        icon_emoji: str | None = None,
    ) -> dict | None:
        """在 parent_id 下创建一个子页面。返回 Notion 响应或 None。"""
        # 延迟导入避免循环
        from .block_builders import rich_text

        payload: dict[str, Any] = {
            "parent": {"type": "page_id", "page_id": parent_id},
            "properties": {"title": {"title": rich_text(title)}},
            "children": first_blocks[:_BATCH_SIZE],
        }
        if icon_emoji:
            payload["icon"] = {"type": "emoji", "emoji": icon_emoji}
        return self._safe(f"create_subpage {title!r}", self._post, "/pages", payload)

    def append_blocks(self, page_id: str, blocks: list[dict]) -> bool:
        """分批追加 blocks（每批 ≤100）。返回是否全部成功。"""
        if not blocks:
            return True
        ok = True
        for i in range(0, len(blocks), _BATCH_SIZE):
            batch = blocks[i: i + _BATCH_SIZE]
            res = self._safe(
                f"append_blocks batch {i // _BATCH_SIZE}",
                self._patch,
                f"/blocks/{page_id}/children",
                {"children": batch},
            )
            if res is None:
                ok = False
                break
            time.sleep(_BATCH_DELAY)
        return ok

    def update_page_icon_title(
        self,
        page_id: str,
        *,
        icon_emoji: str | None = None,
        title: str | None = None,
    ) -> dict | None:
        from .block_builders import rich_text
        payload: dict[str, Any] = {}
        if icon_emoji:
            payload["icon"] = {"type": "emoji", "emoji": icon_emoji}
        if title is not None:
            payload["properties"] = {"title": {"title": rich_text(title)}}
        if not payload:
            return None
        return self._safe(
            f"update_page {page_id}",
            self._patch,
            f"/pages/{page_id}",
            payload,
        )
