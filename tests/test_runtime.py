"""Phase 5 验证：Agent Runtime（workflow / reflector / experience / preferences）。

测试边界：
  - 不依赖真实 LLM / 真实 MCP；注册假工具走通流程
  - Memory 用 tmp_path，不污染项目 memory/
  - Preferences 用 tmp_path
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from src.runtime.memory import Memory
from src.runtime.preferences import (
    DEFAULT_PREFERENCES,
    load_preferences,
    save_preferences,
    update_preference,
)
from src.runtime.reflector import Reflector
from src.runtime.workflow import (
    Workflow,
    WorkflowStep,
    register_tool,
    known_tools,
)


# ============================================================
# Workflow runner
# ============================================================


def test_workflow_runs_steps_in_order_and_threads_state():
    async def tool_a(x: int):
        return {"y": x * 2}

    async def tool_b(y: int):
        return {"z": y + 1}

    register_tool("__t_a", tool_a)
    register_tool("__t_b", tool_b)

    wf = Workflow(name="t_basic")
    wf.add(WorkflowStep(
        name="a", tool="__t_a",
        args_fn=lambda s: {"x": s.get("x", 0)},
        capture=lambda raw: {"y": raw["y"]},
    ))
    wf.add(WorkflowStep(
        name="b", tool="__t_b",
        args_fn=lambda s: {"y": s["y"]},
        capture=lambda raw: {"z": raw["z"]},
    ))

    run_log = asyncio.run(wf.run({"x": 3}))
    assert run_log["status"] == "ok"
    assert run_log["final_state"]["z"] == 7
    assert [s["status"] for s in run_log["steps"]] == ["ok", "ok"]
    assert run_log["steps"][0]["captured"] == {"y": 6}


def test_workflow_retries_once_then_succeeds():
    """retry_once 策略：第一次抛错、第二次成功，最终 status=ok 且 attempts=2。"""
    state = {"calls": 0}

    async def flaky():
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("temporary timeout while connecting")
        return {"ok": True}

    register_tool("__t_flaky", flaky)
    wf = Workflow(name="t_flaky").add(WorkflowStep(
        name="step", tool="__t_flaky", on_failure="retry_once",
    ))
    # 反射器不否决重试（错误信息匹配 retry_ok）
    rl = asyncio.run(wf.run({}, reflector=Reflector()))
    assert rl["status"] == "ok"
    assert rl["steps"][0]["attempts"] == 2


def test_workflow_skip_keeps_running():
    async def boom():
        raise RuntimeError("intentional")

    async def good():
        return {"ok": True}

    register_tool("__t_boom", boom)
    register_tool("__t_good", good)
    wf = (Workflow(name="t_skip")
          .add(WorkflowStep(name="bad", tool="__t_boom", on_failure="skip"))
          .add(WorkflowStep(name="good", tool="__t_good")))
    rl = asyncio.run(wf.run({}))
    # 整体应是 ok（skip 不算 failed），但 step1 标记 skipped
    assert rl["steps"][0]["status"] == "skipped"
    assert rl["steps"][1]["status"] == "ok"
    assert rl["status"] == "ok"


def test_workflow_abort_stops_chain():
    """on_failure=abort：失败后链路中止。"""
    async def boom():
        raise RuntimeError("invalid api key 401")  # reflector 应否决重试

    async def good():
        return {"ok": True}

    register_tool("__t_boom2", boom)
    register_tool("__t_good2", good)
    wf = (Workflow(name="t_abort")
          .add(WorkflowStep(name="bad", tool="__t_boom2",
                            on_failure="abort"))
          .add(WorkflowStep(name="after", tool="__t_good2")))
    rl = asyncio.run(wf.run({}, reflector=Reflector()))
    assert rl["status"] == "failed"
    # 第二步不应执行
    assert len(rl["steps"]) == 1


def test_unknown_tool_fails_gracefully():
    wf = Workflow(name="t_unk").add(
        WorkflowStep(name="x", tool="__nope_does_not_exist__")
    )
    rl = asyncio.run(wf.run({}))
    assert rl["status"] == "failed"
    assert "not registered" in rl["steps"][0]["error"]


# ============================================================
# Reflector
# ============================================================


def test_reflector_should_retry_rules():
    r = Reflector()
    # 鼓励类
    assert asyncio.run(r.should_retry(
        {"tool": "add_paper_to_graph", "error": "ExtractionError: JSON decode failed"},
        attempt_count=1,
    ))
    # 否决类
    assert not asyncio.run(r.should_retry(
        {"tool": "ingest_local_pdf", "error": "401 Unauthorized: invalid api key"},
        attempt_count=1,
    ))
    # attempts 已用满
    assert not asyncio.run(r.should_retry(
        {"tool": "x", "error": "timeout"},
        attempt_count=2,
    ))


def test_reflector_deterministic_summary_contains_status_and_steps():
    r = Reflector()
    rl = {
        "workflow": "demo",
        "status": "ok",
        "steps": [
            {"name": "a", "status": "ok", "attempts": 2, "elapsed_ms": 100},
            {"name": "b", "status": "ok", "attempts": 1, "elapsed_ms": 50},
        ],
    }
    s = asyncio.run(r.summarize_run(rl))
    assert "demo" in s and "2/2" in s and "重试" in s


# ============================================================
# Memory & Experience
# ============================================================


def test_memory_append_and_search(tmp_path: Path):
    mem = Memory(root=tmp_path)
    p = mem.append_experience("success_patterns", "first entry")
    assert p.exists()
    mem.append_experience("success_patterns", "second entry with 'ReAct'")

    hits = mem.search_experiences("React", limit=5)
    names = [h[0] for h in hits]
    assert "success_patterns.md" in names


def test_memory_trim_compacts_old_entries(tmp_path: Path):
    mem = Memory(root=tmp_path)
    # 写一堆短 entry 直到超过 2000 字符
    long_filler = "x" * 300
    for i in range(12):
        mem.append_experience("noise", f"entry {i} body: {long_filler}")
    text = (tmp_path / "experiences" / "noise.md").read_text(encoding="utf-8")
    assert "[compaction]" in text or len(text) <= 2200
    # 至少保留 5 条最近的
    assert text.count("## [") >= 5


def test_record_run_log_writes_buckets(tmp_path: Path, monkeypatch):
    """run_log 入 experiences/success_patterns.md 或 failure_patterns.md。"""
    # patch Memory 默认目录
    import src.runtime.memory as m
    fake = Memory(root=tmp_path)
    monkeypatch.setattr(m, "_DEFAULT", fake)
    from src.runtime.experience import record_run_log

    run_log_ok = {
        "run_id": "r1", "workflow": "wf_ok", "status": "ok",
        "summary": "all green",
        "steps": [{"name": "a", "status": "ok", "attempts": 1, "elapsed_ms": 10}],
        "final_state": {"k": 1},
    }
    files = record_run_log(run_log_ok)
    assert "success_patterns.md" in files

    run_log_fail = {
        "run_id": "r2", "workflow": "wf_fail", "status": "failed",
        "summary": "step bad failed",
        "steps": [{"name": "bad", "status": "failed",
                   "error": "boom", "tool": "x",
                   "attempts": 2, "elapsed_ms": 20}],
        "final_state": {},
    }
    files = record_run_log(run_log_fail)
    assert "failure_patterns.md" in files
    assert "api_quirks.md" in files  # attempts>1 触发


# ============================================================
# Preferences
# ============================================================


def test_preferences_load_default_and_update(tmp_path: Path):
    p = tmp_path / "preferences.json"
    prefs = load_preferences(p)
    assert prefs == DEFAULT_PREFERENCES

    save_preferences({**prefs, "default_domain": "comms"}, p)
    loaded = load_preferences(p)
    assert loaded["default_domain"] == "comms"


def test_preferences_update_validates_enum(tmp_path: Path):
    p = tmp_path / "preferences.json"
    save_preferences(DEFAULT_PREFERENCES, p)
    with pytest.raises(ValueError):
        update_preference("default_domain", "not_a_domain", p)
    with pytest.raises(ValueError):
        update_preference("__no_such_key__", "x", p)


def test_preferences_update_accepts_allowed(tmp_path: Path):
    p = tmp_path / "preferences.json"
    save_preferences(DEFAULT_PREFERENCES, p)
    new = update_preference("default_domain", "comms", p)
    assert new["default_domain"] == "comms"
