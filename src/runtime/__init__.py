"""
Agent Runtime —— Phase 5 收口包

把"现在散给宿主的责任"收回到 Python：
  - WorkflowRunner (workflow.py): 按 spec 顺序跑工具，带 retry/skip/abort
  - Reflector (reflector.py): 规则化重试判断 + 跑完一句话总结
  - Experience (experience.py): 把 run_log 归档到 memory/experiences/
  - Memory (memory.py): 对 memory/ 的统一读写 API
  - Preferences (preferences.py): 结构化偏好 memory/preferences.json
  - Workflows (workflows_registry.py): paper_analysis_flow / knowledge_build_flow
  - Tools (tools.py): 把 MCP 工具 wrap 成内部 async callable

宿主侧调用：from src.runtime import run_workflow
CLI 侧调用：python -m src.runtime run paper_analysis_flow --pdf path/to.pdf
"""

from __future__ import annotations

from .experience import record_run_log
from .memory import Memory, get_memory
from .preferences import (
    DEFAULT_PREFERENCES,
    load_preferences,
    save_preferences,
    update_preference,
)
from .reflector import Reflector
from .workflow import (
    Workflow,
    WorkflowContext,
    WorkflowStep,
    get_tool,
    known_tools,
    register_tool,
)
from .workflows_registry import all_workflows, get_workflow


async def run_workflow(
    name: str,
    initial_state: dict | None = None,
    *,
    use_reflector: bool | None = None,
    record_experience: bool = True,
) -> dict:
    """对外便捷入口：跑一个注册过的 workflow。"""
    wf = get_workflow(name)
    if wf is None:
        raise ValueError(
            f"unknown workflow: {name!r}; known={list(all_workflows())}"
        )

    prefs = load_preferences()
    enabled = (
        prefs.get("reflector_enabled", True)
        if use_reflector is None
        else bool(use_reflector)
    )
    reflector = Reflector() if enabled else None

    run_log = await wf.run(initial_state or {}, reflector=reflector)

    if record_experience:
        try:
            run_log["experience_files"] = record_run_log(run_log)
        except Exception as e:  # noqa: BLE001
            run_log["experience_files"] = []
            run_log["experience_error"] = str(e)

    return run_log


__all__ = [
    "Workflow", "WorkflowContext", "WorkflowStep",
    "Reflector",
    "Memory", "get_memory",
    "load_preferences", "save_preferences", "update_preference",
    "DEFAULT_PREFERENCES",
    "record_run_log",
    "get_workflow", "all_workflows",
    "register_tool", "get_tool", "known_tools",
    "run_workflow",
]
