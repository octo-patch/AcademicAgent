"""
Agent Runtime - 经验沉淀
=========================

跑完 workflow 后把 reflector 的总结按状态归类到 memory/experiences/：
  - status=ok   → success_patterns.md
  - status=failed → failure_patterns.md
  - 任何带 retry 的步骤 → 同时进 api_quirks.md（标记"重试后成功"的 API 行为）

调用：Phase 5 的 workflow runner / cli 跑完后调 record_run_log(run_log)。
"""

from __future__ import annotations

import logging
from typing import Any

from .memory import get_memory

logger = logging.getLogger("ScholarMind.Runtime.Experience")


def record_run_log(run_log: dict[str, Any]) -> list[str]:
    """根据 run_log 自动 append 到对应经验文件。

    返回被写入的文件名列表（方便上层 RefactorRecorder 知道产物在哪）。
    """
    mem = get_memory()
    written: list[str] = []

    wf = run_log.get("workflow", "?")
    status = run_log.get("status", "?")
    summary = run_log.get("summary", "") or "(no summary)"
    steps = run_log.get("steps", [])

    # 1) 成败分桶
    bucket = "success_patterns" if status == "ok" else "failure_patterns"
    content_parts = [
        f"- workflow: `{wf}` run={run_log.get('run_id','')} status={status}",
        f"- 总结: {summary}",
    ]
    failures = [s for s in steps if s.get("status") == "failed"]
    if failures:
        content_parts.append("- 失败步骤:")
        for s in failures[:3]:
            content_parts.append(
                f"  - `{s.get('name')}` ({s.get('tool')}): {s.get('error','')[:160]}"
            )

    final_state = run_log.get("final_state") or {}
    if final_state:
        sample = ", ".join(
            f"{k}={_short(v)}" for k, v in list(final_state.items())[:6]
        )
        content_parts.append(f"- 末态: {sample}")

    p = mem.append_experience(bucket, "\n".join(content_parts))
    written.append(p.name)

    # 2) API quirks：含重试的步骤
    retried = [s for s in steps if s.get("attempts", 1) > 1]
    if retried:
        quirks = ["- workflow: `%s`" % wf]
        for s in retried:
            quirks.append(
                f"- `{s.get('tool')}` 重试 {s.get('attempts')} 次后 status={s.get('status')}"
                + (f"; 末错 {s.get('error','')[:120]}" if s.get("error") else "")
            )
        p2 = mem.append_experience("api_quirks", "\n".join(quirks))
        written.append(p2.name)

    return written


def _short(v: Any, max_len: int = 80) -> str:
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= max_len else s[: max_len - 3] + "..."
