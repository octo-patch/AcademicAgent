"""
Phase 5 smoke 验证脚本
========================

不依赖 LLM/MCP，跑两个用例：
  1. happy path: 注册三个假工具，retry_once 中第一次失败、第二次成功，
     最终 status=ok，run_log 入 memory/experiences/。
  2. abort path: 故意触发认证类错误（401），reflector 否决重试，run_log
     入 failure_patterns.md。

同时每步走 RefactorRecorder("agent-runtime") 同步 Notion + 本地 md。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.recorder import RefactorRecorder
from src.runtime import (
    Reflector,
    Workflow,
    WorkflowStep,
    record_run_log,
    register_tool,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------- 假工具 ----------

_RETRY_CALLS = {"n": 0}


async def fake_ingest(pdf_path: str):
    return {"status": "ok", "report": f"ingested {Path(pdf_path).name}"}


async def fake_flaky_extract(pdf_path: str, domain: str = ""):
    _RETRY_CALLS["n"] += 1
    if _RETRY_CALLS["n"] == 1:
        raise RuntimeError("temporary timeout from LLM")
    return {"status": "ok", "node_count": 12, "edge_count": 18, "domain": domain or "llm_agent"}


async def fake_query(query: str, domain: str = ""):
    return {"status": "ok", "report": f"matched 3 nodes for '{query}'"}


async def fake_auth_failure(pdf_path: str, **kwargs):
    raise RuntimeError("401 Unauthorized: invalid api key")


register_tool("__smoke_ingest", fake_ingest)
register_tool("__smoke_extract_flaky", fake_flaky_extract)
register_tool("__smoke_query", fake_query)
register_tool("__smoke_auth_fail", fake_auth_failure)


# ---------- 两条 smoke workflow ----------


def happy_workflow() -> Workflow:
    return (
        Workflow(name="smoke_happy", description="retry-once 触发后第二次成功")
        .add(WorkflowStep(
            name="ingest", tool="__smoke_ingest",
            args_fn=lambda s: {"pdf_path": s["pdf_path"]},
        ))
        .add(WorkflowStep(
            name="extract", tool="__smoke_extract_flaky",
            args_fn=lambda s: {"pdf_path": s["pdf_path"], "domain": s.get("domain", "")},
            capture=lambda raw: {"node_count": raw["node_count"]},
            on_failure="retry_once",
        ))
        .add(WorkflowStep(
            name="query_back", tool="__smoke_query",
            args_fn=lambda s: {"query": s.get("paper_title", "smoke")},
            on_failure="skip",
        ))
    )


def abort_workflow() -> Workflow:
    return (
        Workflow(name="smoke_abort", description="401 错误：反思器否决重试")
        .add(WorkflowStep(
            name="extract", tool="__smoke_auth_fail",
            args_fn=lambda s: {"pdf_path": s["pdf_path"]},
            on_failure="retry_once",
        ))
        .add(WorkflowStep(
            name="query_back", tool="__smoke_query",
            args_fn=lambda s: {"query": "after-abort"},
        ))
    )


# ---------- 主流程 ----------


async def main():
    recorder = RefactorRecorder(
        area="agent-runtime",
        title="🤖 Agent Runtime + 记忆闭环",
        icon_emoji="🤖",
    )

    # 步骤记录
    recorder.log_step(
        title="src/runtime/ 包落地",
        problem=(
            "之前散落给宿主的责任（步骤编排、失败重试、经验沉淀、偏好读取）"
            "在 Python 端完全没有承载；evaluation 的 WorkflowSpec 又是评估专用，"
            "Agent 跑不起来。"
        ),
        approach=(
            "新建 src/runtime/ 包：workflow.WorkflowRunner（含 retry_once/skip/abort）、"
            "reflector.Reflector（规则化 should_retry + summarize_run）、"
            "experience.record_run_log（按状态归档到 memory/experiences/）、"
            "memory.Memory（统一读写 + 自动 trim）、preferences.json + update_preference、"
            "workflows_registry 内置 paper_analysis_flow / knowledge_build_flow、"
            "tools.py 把 MCP @tool 函数 wrap 为内部 async callable。"
        ),
        files_changed=[
            "src/runtime/__init__.py",
            "src/runtime/workflow.py",
            "src/runtime/reflector.py",
            "src/runtime/experience.py",
            "src/runtime/memory.py",
            "src/runtime/preferences.py",
            "src/runtime/tools.py",
            "src/runtime/workflows_registry.py",
            "src/runtime/cli.py",
            "src/runtime/__main__.py",
            "memory/preferences.json",
            "tests/test_runtime.py",
            "scripts/smoke_runtime_phase5.py",
        ],
        effect=(
            "宿主可以一句话调 run_workflow('paper_analysis_flow', state)，"
            "失败按策略重试/跳过/中止，成功/失败自动写到 memory/experiences/。"
        ),
        risks_resolved=["#6 Agent 不是 Agent"],
    )

    # ---- happy path ----
    _RETRY_CALLS["n"] = 0
    wf1 = happy_workflow()
    reflector = Reflector()
    rl1 = await wf1.run(
        {"pdf_path": "data/papers/MemGPT.pdf", "paper_title": "MemGPT", "domain": "llm_agent"},
        reflector=reflector,
    )
    files1 = record_run_log(rl1)
    recorder.log_validation(
        step="smoke-happy-retry-success",
        command="python scripts/smoke_runtime_phase5.py  # subcase: happy",
        output=(
            f"status={rl1['status']}\n"
            f"steps={[ (s['name'], s['status'], s['attempts']) for s in rl1['steps'] ]}\n"
            f"final_state={rl1['final_state']}\n"
            f"summary={rl1['summary']}\n"
            f"experience_files={files1}"
        ),
        status="ok" if rl1["status"] == "ok" and any("ok" == s["status"] and s["attempts"] == 2 for s in rl1["steps"]) else "warn",
        notes="第二次重试后成功；reflector 因 'temporary timeout' 匹配 OK 模式而放行。",
        artifacts=[
            *(f"memory/experiences/{f}" for f in files1),
        ],
    )

    # ---- abort path ----
    wf2 = abort_workflow()
    rl2 = await wf2.run({"pdf_path": "data/papers/MemGPT.pdf"}, reflector=reflector)
    files2 = record_run_log(rl2)
    recorder.log_validation(
        step="smoke-abort-on-auth-error",
        command="python scripts/smoke_runtime_phase5.py  # subcase: abort",
        output=(
            f"status={rl2['status']}\n"
            f"steps={[ (s['name'], s['status'], s['attempts']) for s in rl2['steps'] ]}\n"
            f"summary={rl2['summary']}\n"
            f"experience_files={files2}"
        ),
        status="ok" if rl2["status"] == "failed" and len(rl2["steps"]) == 1 else "fail",
        notes="reflector 在 401 错误下否决重试，后续步骤不执行（验证非死循环 + abort 正确）。",
        artifacts=[
            *(f"memory/experiences/{f}" for f in files2),
        ],
    )

    # ---- prefs roundtrip ----
    from src.runtime.preferences import load_preferences, update_preference
    before = load_preferences()
    update_preference("default_domain", "comms")
    after = load_preferences()
    update_preference("default_domain", before["default_domain"])  # 还原

    recorder.log_validation(
        step="preferences-roundtrip",
        command="update_preference('default_domain', 'comms') ; restore",
        output=f"before={before}\nafter={after}\nrestored={load_preferences()}",
        status="ok" if after["default_domain"] == "comms" else "fail",
        notes="可结构化读 / 写偏好；ALLOWED_VALUES 守门已在 unit test 覆盖。",
    )

    # ---- pytest summary ----
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_runtime.py", "-q"],
        cwd=str(_ROOT), capture_output=True, text=True,
    )
    recorder.log_validation(
        step="pytest-runtime",
        command="python -m pytest tests/test_runtime.py -q",
        output=(proc.stdout + "\n" + proc.stderr)[-1600:],
        status="ok" if proc.returncode == 0 else "fail",
        notes="13 个单元测试覆盖 workflow / reflector / memory / experience / preferences。",
    )

    print("\n[smoke] happy run_log:")
    print(json.dumps(rl1, ensure_ascii=False, indent=2)[:1200])
    print("\n[smoke] abort run_log:")
    print(json.dumps(rl2, ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
