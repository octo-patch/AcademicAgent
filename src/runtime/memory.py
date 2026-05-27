"""
Agent Runtime - 长期记忆读写 API
=================================

不重写记忆系统，只把对 memory/ 目录的读写聚到一处，让 workflow / reflector
都走同一份代码，避免乱写、避免相对路径漂移。

memory 目录结构（约定）：
  memory/
    MEMORY.md             经验索引（带预算约束）
    USER.md               用户画像
    preferences.json      结构化偏好（独立于 USER.md 自由文本）
    experiences/
      analysis_patterns.md
      search_strategies.md
      success_patterns.md   新（runtime 自动 append）
      failure_patterns.md   新
      api_quirks.md         新
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("ScholarMind.Runtime.Memory")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MEMORY_DIR = _PROJECT_ROOT / "memory"
_EXP_DIR = _MEMORY_DIR / "experiences"


class Memory:
    """对 memory/ 目录的薄封装。"""

    def __init__(self, root: Path | None = None):
        self.root = root or _MEMORY_DIR
        self.exp_dir = self.root / "experiences"
        self.root.mkdir(parents=True, exist_ok=True)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 用户画像 / 索引 ----------

    def read_user(self) -> str:
        p = self.root / "USER.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def read_index(self) -> str:
        p = self.root / "MEMORY.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # ---------- 经验文件 ----------

    def list_experiences(self) -> list[str]:
        return sorted(p.name for p in self.exp_dir.glob("*.md"))

    def read_experience(self, category: str) -> str:
        p = self.exp_dir / f"{category}.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def search_experiences(self, keyword: str, limit: int = 5) -> list[tuple[str, str]]:
        """跨经验文件按关键词找匹配行；返回 [(filename, line), ...]。"""
        kw = keyword.lower().strip()
        if not kw:
            return []
        hits: list[tuple[str, str]] = []
        for p in self.exp_dir.glob("*.md"):
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if kw in line.lower():
                        hits.append((p.name, line.strip()))
                        if len(hits) >= limit:
                            return hits
            except Exception as e:  # noqa: BLE001
                logger.warning("read %s failed: %s", p, e)
        return hits

    def append_experience(self, category: str, content: str) -> Path:
        """追加一条带时间戳的经验记录到 experiences/<category>.md。

        每条独占一个二级小节 `## [YYYY-MM-DD HH:MM]`。
        """
        if not category or any(c in category for c in r'\/:*?"<>|'):
            raise ValueError(f"invalid category: {category!r}")
        p = self.exp_dir / f"{category}.md"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = f"\n## [{ts}]\n\n{content.strip()}\n"
        if not p.exists():
            p.write_text(
                f"# {category.replace('_', ' ').title()}\n\n"
                f"<!-- runtime 自动追加；超过 2000 字时头部老条目会被压缩 -->\n"
                + block,
                encoding="utf-8",
            )
        else:
            with open(p, "a", encoding="utf-8") as f:
                f.write(block)
        logger.info("append_experience %s/%s", category, ts)
        self._maybe_trim(p)
        return p

    def _maybe_trim(self, path: Path, max_chars: int = 2000) -> None:
        """文件超长时把最老的若干小节合并成"压缩摘要"占位。

        策略简单：超 max_chars 时保留最后 N 个 `## [...]` 小节 + 在头部
        写一行 "[truncated N entries before {date}]" 占位。
        不调 LLM，纯文本处理，幂等。
        """
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return
        if len(text) <= max_chars:
            return

        # split 出所有 `## [YYYY-...]` 小节
        import re
        sections = re.split(r"(?=^## \[)", text, flags=re.MULTILINE)
        header, sections = sections[0], sections[1:]
        if len(sections) < 6:
            return

        keep = sections[-5:]
        dropped = len(sections) - len(keep)
        first_kept_match = re.search(r"\[([\d\-: ]+)\]", keep[0]) if keep else None
        cut_at = first_kept_match.group(1) if first_kept_match else "earlier"
        compacted = (
            f"\n## [compaction]\n\n_truncated {dropped} earlier entries before "
            f"{cut_at} to keep file < {max_chars} chars_\n"
        )
        new_text = header.rstrip() + "\n" + compacted + "".join(keep)
        path.write_text(new_text, encoding="utf-8")
        logger.info("trimmed %s: dropped %d old sections", path.name, dropped)


_DEFAULT: Memory | None = None


def get_memory() -> Memory:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Memory()
    return _DEFAULT
