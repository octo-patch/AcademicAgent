"""一次性脚本：标记 agent-runtime area 已完成。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.recorder import RefactorRecorder

recorder = RefactorRecorder(area="agent-runtime")
recorder.log_step(
    title="CLAUDE.md / paper-analysis workflow 文档同步",
    problem="Phase 5 落地后，宿主侧的 CLAUDE.md / .agents/workflows/paper-analysis.md 仍只指向 MCP 散调用，没有暴露 runtime 一键入口。",
    approach=(
        "CLAUDE.md 记忆系统章节加结构化偏好 + 经验文件指引 + 'Agent Runtime' 子章节；"
        ".agents/workflows/paper-analysis.md 在前置准备后插入'一键路径'指引。"
    ),
    files_changed=[
        "CLAUDE.md",
        ".agents/workflows/paper-analysis.md",
    ],
    effect="宿主下次进入会话即可看到 runtime CLI 入口和经验文件位置，不再绕路调单工具。",
    risks_resolved=["#6 Agent 不是 Agent"],
)
recorder.close(
    status="completed",
    summary=(
        "Phase 5 完成：Workflow Runner（retry/skip/abort）+ Lite Reflector + 经验沉淀 + "
        "结构化偏好 + Memory API + 内置 paper_analysis_flow / knowledge_build_flow + CLI。"
        "13 个单元测试全过；smoke 验证（retry 成功 / 401 abort / prefs roundtrip / pytest）全过。"
    ),
)
