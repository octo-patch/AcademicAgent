"""
Notion block 构造器（统一收口）
==============================

从两处合并：
  - src/evaluation/notion_sync.py:54-81 （_rich/_heading/_paragraph/_bullet/_callout/_code）
  - scripts/notion_sync_engineering_recap.py:27-91 （加上 numbered/quote/divider/code_block/table）

所有调用方（RefactorRecorder、eval notion 同步、一次性脚本）都走这里，避免重复维护。

Notion API 约束：
  - 单个 rich_text content ≤ 2000 字符（实际我们留 1900 安全裕度）
  - 单次 children append ≤ 100 blocks
  - emoji 必须是单字符 unicode
"""

from __future__ import annotations

from typing import Iterable

_MAX_CHUNK = 1900


# ============================================================
# rich_text — 自动分段
# ============================================================

def rich_text(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    code: bool = False,
    color: str | None = None,
) -> list[dict]:
    """构造 Notion rich_text 数组。超长自动分段（≤1900 char/段）。

    color: Notion 支持 "default" / "red" / "blue" / ... / "red_background" / ...
    """
    if not text:
        return [{"type": "text", "text": {"content": ""}}]
    text = str(text)
    parts = []
    while text:
        chunk = text[:_MAX_CHUNK]
        text = text[_MAX_CHUNK:]
        ann: dict = {"bold": bold, "italic": italic, "code": code}
        if color:
            ann["color"] = color
        parts.append({"type": "text", "text": {"content": chunk}, "annotations": ann})
    return parts


# ============================================================
# 单 block 构造
# ============================================================

def heading(text: str, level: int = 2) -> dict:
    """Notion 只支持 heading_1 / heading_2 / heading_3"""
    lv = max(1, min(level, 3))
    key = f"heading_{lv}"
    return {"object": "block", "type": key, key: {"rich_text": rich_text(text)}}


def paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": rich_text(text)}}


def bullet(text: str) -> dict:
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": rich_text(text)}}


def numbered(text: str) -> dict:
    return {"object": "block", "type": "numbered_list_item",
            "numbered_list_item": {"rich_text": rich_text(text)}}


def callout(text: str, emoji: str = "💡", color: str = "blue_background") -> dict:
    return {"object": "block", "type": "callout",
            "callout": {"rich_text": rich_text(text),
                        "icon": {"emoji": emoji}, "color": color}}


def quote(text: str) -> dict:
    return {"object": "block", "type": "quote",
            "quote": {"rich_text": rich_text(text)}}


def divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def code_block(text: str, language: str = "plain text") -> dict:
    """语言名见 Notion API 文档（plain text / python / json / bash / markdown / yaml ...）"""
    return {"object": "block", "type": "code",
            "code": {"rich_text": rich_text(text), "language": language}}


def table(headers: list[str], rows: list[list[str]]) -> dict:
    """完整 table block（含 header 行 + data 行）。"""
    width = len(headers)
    children: list[dict] = [{
        "object": "block", "type": "table_row",
        "table_row": {"cells": [rich_text(h, bold=True) for h in headers]},
    }]
    for row in rows:
        cells = [rich_text(str(c)) for c in row[:width]]
        while len(cells) < width:
            cells.append(rich_text(""))
        children.append({
            "object": "block", "type": "table_row",
            "table_row": {"cells": cells},
        })
    return {
        "object": "block", "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": True,
            "has_row_header": False,
            "children": children,
        },
    }


def toggle(summary: str, children: list[dict]) -> dict:
    """toggle 折叠块（可包含子 blocks）。"""
    return {
        "object": "block", "type": "toggle",
        "toggle": {
            "rich_text": rich_text(summary),
            "children": children,
        },
    }


# ============================================================
# 组合工具
# ============================================================

def kv_bullets(pairs: Iterable[tuple[str, object]]) -> list[dict]:
    """把 (key, value) 序列变成一组 bullet block。"""
    return [bullet(f"{k}: {v}") for k, v in pairs]


def status_callout(status: str) -> dict:
    """根据状态字符串选合适的 callout。"""
    s = (status or "").lower()
    if s in ("ok", "pass", "success", "completed"):
        return callout("✓ 通过", emoji="🟢", color="green_background")
    if s in ("warn", "warning"):
        return callout("⚠ 警告", emoji="🟡", color="yellow_background")
    if s in ("fail", "failed", "error"):
        return callout("✗ 失败", emoji="🔴", color="red_background")
    return callout(f"状态: {status}", emoji="ℹ️", color="gray_background")
