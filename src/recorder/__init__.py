"""
src.recorder — 改进步骤 + 验证结果的双写记录器
==================================================

每个 area（如 "kg-quality"、"pdf-multimodal"）对应：
  - 一个 Notion 子页（在 NOTION_PARENT_PAGE_NAME 下）
  - 一份本地 md (docs/refactor/<area>.md)
  - 一份状态缓存 (data/recorder_state/<area>.json)

主类：
  RefactorRecorder(area).log_step(...)         记录一次代码改动
  RefactorRecorder(area).log_validation(...)   记录一次验证 run
  RefactorRecorder(area).close(...)            结束阶段，标记完成

双写策略：
  本地 md 是 source of truth，Notion 失败不阻塞本地写入。
"""

from .refactor_recorder import RefactorRecorder
from .notion_client import NotionClient, NotionConfig, load_notion_config

__all__ = [
    "RefactorRecorder",
    "NotionClient",
    "NotionConfig",
    "load_notion_config",
]
