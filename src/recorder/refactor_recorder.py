"""
RefactorRecorder —— 重构 area 的双写记录器
==========================================

每个 area（如 "infra-recorder"、"kg-quality"、"pdf-multimodal"）对应：
  - Notion 子页（在 NOTION_PARENT_PAGE_NAME 下；首次调用时创建）
  - 本地 md：docs/refactor/<area>.md
  - 状态缓存：data/recorder_state/<area>.json （存 page_id + step 计数）

主要 API：
  recorder = RefactorRecorder(area="kg-quality", title="🧠 知识图谱治理升级")
  recorder.log_step(
      title="Atomic save 改造",
      problem="graph_store.save() 写崩了会丢整图",
      approach="改 tempfile + os.replace 原子写",
      files_changed=["src/knowledge/graph_store.py:316-317"],
      effect="崩溃后旧 JSON 仍然完整",
      risks_resolved=["#9 工程隐患"],
  )
  recorder.log_validation(
      step="atomic-save-smoke",
      command="pytest tests/test_knowledge_graph.py::test_atomic_save",
      output="...",
      status="ok",  # ok | warn | fail
  )
  recorder.close(status="completed", summary="3 个改动 + 5 个验证全通过")

设计准则：
  - 本地 md 是 source of truth；Notion 失败不阻塞 md 写入
  - md 用 YYYY-MM-DD HH:MM 时间戳的 `##` 小节，与 docs/experiments/ 风格一致
  - 状态文件记录 step 计数 + last_synced_at，方便后续 idempotence 与查询
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from . import block_builders as bb
from .notion_client import NotionClient, NotionConfig, load_notion_config

logger = logging.getLogger("AcademicAgent.Recorder")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_STATE_DIR = _PROJECT_ROOT / "data" / "recorder_state"
_DOCS_DIR = _PROJECT_ROOT / "docs" / "refactor"


def _now_local_str() -> str:
    """YYYY-MM-DD HH:MM 本地时间字符串（md 标题用）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _now_utc_iso() -> str:
    """UTC ISO（state 文件用）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class _AreaState:
    """每个 area 的状态文件结构。"""
    area: str
    title: str
    icon_emoji: str = "🔧"
    notion_page_id: str = ""
    notion_url: str = ""
    created_at: str = ""
    last_synced_at: str = ""
    step_count: int = 0
    validation_count: int = 0
    closed: bool = False

    def to_dict(self) -> dict:
        return {
            "area": self.area,
            "title": self.title,
            "icon_emoji": self.icon_emoji,
            "notion_page_id": self.notion_page_id,
            "notion_url": self.notion_url,
            "created_at": self.created_at,
            "last_synced_at": self.last_synced_at,
            "step_count": self.step_count,
            "validation_count": self.validation_count,
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_AreaState":
        return cls(
            area=d.get("area", ""),
            title=d.get("title", ""),
            icon_emoji=d.get("icon_emoji", "🔧"),
            notion_page_id=d.get("notion_page_id", ""),
            notion_url=d.get("notion_url", ""),
            created_at=d.get("created_at", ""),
            last_synced_at=d.get("last_synced_at", ""),
            step_count=d.get("step_count", 0),
            validation_count=d.get("validation_count", 0),
            closed=d.get("closed", False),
        )


class RefactorRecorder:
    """改进步骤 + 验证结果的双写记录器。

    一个进程内可以持有多个 RefactorRecorder（每个 area 一个）。
    每次 log_* 调用都同步写本地 md + 追加到 Notion 子页。
    """

    def __init__(
        self,
        area: str,
        title: str | None = None,
        icon_emoji: str | None = None,
        *,
        config: NotionConfig | None = None,
        notion: NotionClient | None = None,
    ):
        if not area or any(c in area for c in r'\/:*?"<>|'):
            raise ValueError(f"invalid area name: {area!r}")
        self.area = area
        # title / icon_emoji 为 None 时不覆盖 state；首次创建 area 时用默认值
        self._explicit_title = title
        self._explicit_icon = icon_emoji
        self.title = title or area.replace("-", " ").title()
        self.icon_emoji = icon_emoji or "🔧"

        # 路径
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self._state_path = _STATE_DIR / f"{area}.json"
        self._md_path = _DOCS_DIR / f"{area}.md"
        self._notion_log_path = _STATE_DIR / f"{area}.notion.log"

        # 状态
        self._state = self._load_state()
        # 把已恢复的 state 反映到实例（避免后续 _write_md_header 用错值）
        self.title = self._state.title or self.title
        self.icon_emoji = self._state.icon_emoji or self.icon_emoji

        # Notion 客户端
        self._notion = notion or NotionClient(
            config or load_notion_config(),
            log_path=self._notion_log_path,
        )

        # 首次：初始化 md 头 + Notion 子页
        if not self._md_path.exists():
            self._write_md_header()
        if self._notion.ready and not self._state.notion_page_id:
            self._create_notion_subpage()

    # ============================================================
    # 公开 API
    # ============================================================

    def log_step(
        self,
        title: str,
        problem: str = "",
        approach: str = "",
        files_changed: Iterable[str] = (),
        effect: str = "",
        risks_resolved: Iterable[str] = (),
        extra_notes: str = "",
    ) -> None:
        """记录一次代码改动。"""
        self._state.step_count += 1
        ts = _now_local_str()
        step_id = f"step-{self._state.step_count:03d}"

        # 1. 写本地 md
        lines: list[str] = [
            "",
            f"## [{ts}] step {self._state.step_count}: {title}",
            "",
        ]
        if problem:
            lines += [f"**问题**：{problem}", ""]
        if approach:
            lines += [f"**方案**：{approach}", ""]
        files_list = list(files_changed)
        if files_list:
            lines += ["**改动文件**："]
            lines += [f"- `{f}`" for f in files_list]
            lines += [""]
        if effect:
            lines += [f"**效果**：{effect}", ""]
        risks_list = list(risks_resolved)
        if risks_list:
            lines += ["**解决的隐患**：" + ", ".join(risks_list), ""]
        if extra_notes:
            lines += [extra_notes, ""]
        self._append_md("\n".join(lines))

        # 2. 写 Notion
        blocks = self._build_step_blocks(
            ts, step_id, title, problem, approach, files_list, effect, risks_list, extra_notes,
        )
        self._push_blocks(blocks)
        self._save_state()
        logger.info("[recorder/%s] logged step %d: %s", self.area, self._state.step_count, title)

    def log_validation(
        self,
        step: str,
        command: str = "",
        output: str = "",
        status: str = "ok",
        notes: str = "",
        artifacts: Iterable[str] = (),
    ) -> None:
        """记录一次验证 run。status: ok | warn | fail"""
        self._state.validation_count += 1
        ts = _now_local_str()
        val_id = f"val-{self._state.validation_count:03d}"

        # 1. md
        emoji = {"ok": "✓", "pass": "✓", "warn": "⚠", "fail": "✗"}.get(status.lower(), "•")
        lines: list[str] = [
            "",
            f"### [{ts}] {emoji} validation {self._state.validation_count}: {step}",
            "",
        ]
        if command:
            lines += ["**命令**：", "```bash", command, "```", ""]
        if output:
            short = output if len(output) <= 2000 else output[:2000] + f"\n... (truncated, total {len(output)} chars)"
            lines += ["**输出**：", "```", short, "```", ""]
        lines += [f"**状态**：`{status}`", ""]
        if notes:
            lines += [f"**备注**：{notes}", ""]
        artifacts_list = list(artifacts)
        if artifacts_list:
            lines += ["**产物**："]
            lines += [f"- `{a}`" for a in artifacts_list]
            lines += [""]
        self._append_md("\n".join(lines))

        # 2. Notion
        blocks = self._build_validation_blocks(
            ts, val_id, step, command, output, status, notes, artifacts_list,
        )
        self._push_blocks(blocks)
        self._save_state()
        logger.info("[recorder/%s] logged validation %d: %s [%s]",
                    self.area, self._state.validation_count, step, status)

    def close(self, status: str = "completed", summary: str = "") -> None:
        """标记 area 完成。"""
        self._state.closed = True
        ts = _now_local_str()

        lines = [
            "",
            "---",
            "",
            f"## [{ts}] 阶段关闭 · {status}",
            "",
        ]
        if summary:
            lines += [summary, ""]
        lines += [
            f"- 总步骤数: **{self._state.step_count}**",
            f"- 总验证数: **{self._state.validation_count}**",
            "",
        ]
        self._append_md("\n".join(lines))

        # Notion: callout + 总结
        emoji = "✅" if status == "completed" else "⏹️"
        color = "green_background" if status == "completed" else "gray_background"
        blocks = [
            bb.divider(),
            bb.callout(
                f"[{ts}] 阶段关闭 · {status}\n"
                f"步骤数: {self._state.step_count}  ·  验证数: {self._state.validation_count}",
                emoji=emoji, color=color,
            ),
        ]
        if summary:
            blocks.append(bb.paragraph(summary))
        self._push_blocks(blocks)
        self._save_state()
        logger.info("[recorder/%s] closed (%s)", self.area, status)

    # ============================================================
    # 内部：本地 md
    # ============================================================

    def _write_md_header(self) -> None:
        # 避免 icon + title 已含 emoji 时双重前缀
        title_line = self.title
        if self.icon_emoji and not self.title.lstrip().startswith(self.icon_emoji):
            title_line = f"{self.icon_emoji} {self.title}"
        header = (
            f"# {title_line}\n"
            f"\n"
            f"**area**: `{self.area}`  \n"
            f"**created**: {_now_utc_iso()}  \n"
            f"**parent (Notion)**: `{self._state.notion_page_id or '(pending)'}`\n"
            f"\n"
            f"> 本文件由 `src.recorder.RefactorRecorder` 自动维护。所有 step / validation 双写到 Notion + 本文件。\n"
        )
        self._md_path.write_text(header, encoding="utf-8")

    def _append_md(self, text: str) -> None:
        with open(self._md_path, "a", encoding="utf-8") as f:
            f.write(text + ("\n" if not text.endswith("\n") else ""))

    # ============================================================
    # 内部：Notion blocks
    # ============================================================

    def _create_notion_subpage(self) -> None:
        parent_id = self._notion.resolve_parent()
        if not parent_id:
            logger.warning(
                "[recorder/%s] cannot resolve Notion parent; will continue local-only",
                self.area,
            )
            return

        first_blocks = [
            bb.callout(
                f"area = {self.area} · 由 src.recorder.RefactorRecorder 自动维护。"
                f"本页镜像 docs/refactor/{self.area}.md，本地是 source of truth。",
                emoji=self.icon_emoji, color="blue_background",
            ),
            bb.paragraph(f"创建时间（UTC）: {_now_utc_iso()}"),
            bb.divider(),
        ]
        page = self._notion.create_subpage(
            parent_id, self.title,
            first_blocks=first_blocks,
            icon_emoji=self.icon_emoji,
        )
        if page:
            self._state.notion_page_id = page.get("id", "")
            self._state.notion_url = page.get("url", "")
            self._state.created_at = _now_utc_iso()
            self._save_state()
            # 重新写 md header（带上 page_id）
            self._rewrite_md_header()
            logger.info("[recorder/%s] notion subpage created: %s",
                        self.area, self._state.notion_url)

    def _rewrite_md_header(self) -> None:
        """更新 md 头部的 (pending) 占位为真实 page_id。"""
        if not self._md_path.exists():
            return
        text = self._md_path.read_text(encoding="utf-8")
        text = text.replace(
            "**parent (Notion)**: `(pending)`",
            f"**parent (Notion)**: [{self._state.notion_page_id}]({self._state.notion_url})",
        )
        self._md_path.write_text(text, encoding="utf-8")

    def _push_blocks(self, blocks: list[dict]) -> None:
        if not self._notion.ready or not self._state.notion_page_id:
            return
        ok = self._notion.append_blocks(self._state.notion_page_id, blocks)
        if ok:
            self._state.last_synced_at = _now_utc_iso()

    def _build_step_blocks(
        self,
        ts: str,
        step_id: str,
        title: str,
        problem: str,
        approach: str,
        files_changed: list[str],
        effect: str,
        risks_resolved: list[str],
        extra_notes: str,
    ) -> list[dict]:
        blocks: list[dict] = [
            bb.heading(f"[{ts}] {step_id}: {title}", level=2),
        ]
        if problem:
            blocks += [bb.paragraph("问题"), bb.callout(problem, emoji="❓", color="orange_background")]
        if approach:
            blocks += [bb.paragraph("方案"), bb.paragraph(approach)]
        if files_changed:
            blocks.append(bb.paragraph("改动文件"))
            blocks += [bb.bullet(f) for f in files_changed]
        if effect:
            blocks += [bb.paragraph("效果"), bb.callout(effect, emoji="✨", color="green_background")]
        if risks_resolved:
            blocks.append(bb.paragraph("解决的隐患"))
            blocks += [bb.bullet(r) for r in risks_resolved]
        if extra_notes:
            blocks.append(bb.paragraph(extra_notes))
        blocks.append(bb.divider())
        return blocks

    def _build_validation_blocks(
        self,
        ts: str,
        val_id: str,
        step: str,
        command: str,
        output: str,
        status: str,
        notes: str,
        artifacts: list[str],
    ) -> list[dict]:
        s = (status or "").lower()
        if s in ("ok", "pass", "success"):
            emoji, color = "🟢", "green_background"
        elif s in ("warn", "warning"):
            emoji, color = "🟡", "yellow_background"
        elif s in ("fail", "failed", "error"):
            emoji, color = "🔴", "red_background"
        else:
            emoji, color = "ℹ️", "gray_background"

        blocks: list[dict] = [
            bb.heading(f"[{ts}] {val_id}: {step}", level=3),
            bb.callout(f"状态: {status}", emoji=emoji, color=color),
        ]
        if command:
            blocks += [bb.paragraph("命令"), bb.code_block(command, language="bash")]
        if output:
            short = output if len(output) <= 1800 else output[:1800] + f"\n... (truncated, total {len(output)} chars)"
            blocks += [bb.paragraph("输出"), bb.code_block(short, language="plain text")]
        if notes:
            blocks += [bb.paragraph(f"备注: {notes}")]
        if artifacts:
            blocks.append(bb.paragraph("产物"))
            blocks += [bb.bullet(a) for a in artifacts]
        return blocks

    # ============================================================
    # 内部：state
    # ============================================================

    def _load_state(self) -> _AreaState:
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                st = _AreaState.from_dict(data)
                # 仅当调用方显式提供新值时覆盖（允许升级 title/icon）
                if self._explicit_title and st.title != self._explicit_title:
                    st.title = self._explicit_title
                if self._explicit_icon and st.icon_emoji != self._explicit_icon:
                    st.icon_emoji = self._explicit_icon
                return st
            except Exception as exc:  # noqa: BLE001
                logger.warning("[recorder/%s] state load failed, starting fresh: %s",
                               self.area, exc)
        return _AreaState(area=self.area, title=self.title, icon_emoji=self.icon_emoji)

    def _save_state(self) -> None:
        # 原子写
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self._state_path)
