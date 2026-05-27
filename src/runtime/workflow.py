"""
Agent Runtime - Workflow 引擎
=============================

把 .agents/workflows/*.md 的"剧本"落到 Python 端，Workflow runner 顺序执行
注册过的 WorkflowStep；每步带失败策略（skip / retry_once / abort）；
跑完产出可序列化的 run_log，供 reflector / RefactorRecorder 消费。

设计来源（与 src/evaluation/workflows/registry.py 的对照）：
  evaluation 端那套是为可重放的 L2 评估写的（fix 序列、capture 函数纯净），
  本模块把它升级为"通用运行时"：
    - WorkflowContext 是跨步骤共享的可变状态（args + 累积产物）
    - WorkflowStep.on_failure 给每步声明失败策略
    - Workflow.run() 是 async（实际工具都是 async），失败时按策略走
    - run_log: dict[]，便于反思器读 + recorder 写
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

logger = logging.getLogger("ScholarMind.Runtime.Workflow")

OnFailure = str  # Literal: 'skip' | 'retry_once' | 'abort'

# ============================================================
# 工具注册：内部 Python 函数（不是 MCP 协议层）
# ============================================================

ToolFn = Callable[..., Awaitable[Any]]
_TOOLS: dict[str, ToolFn] = {}


def register_tool(name: str, fn: ToolFn) -> None:
    """注册一个可被 WorkflowStep.tool 引用的 async callable。"""
    _TOOLS[name] = fn


def get_tool(name: str) -> ToolFn | None:
    return _TOOLS.get(name)


def known_tools() -> list[str]:
    return sorted(_TOOLS.keys())


# ============================================================
# Context & Step
# ============================================================


@dataclass
class WorkflowContext:
    """跨步骤的可变状态：初始 args + 每步 capture 产物。

    用 dict 接口而不是 __getattr__ 是因为 workflow 的产物字段不固定，
    把所有读写都聚到 `state` dict 上避免 dataclass 字段爆炸。
    """
    state: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def merge(self, updates: dict[str, Any]) -> None:
        self.state.update(updates)


@dataclass
class WorkflowStep:
    """工作流中的一步。

    name        : 步骤标识（log / capture 用）
    tool        : 工具名，需在 register_tool 注册
    args_fn     : (ctx_state) -> dict，本步工具调用参数
    capture     : (raw_result) -> dict，把要传给后续步骤的键值挑出来
    on_failure  : 失败处理策略：skip / retry_once / abort（默认 retry_once）
    description : 文档字符串（写入 run_log，可读）
    """
    name: str
    tool: str
    args_fn: Callable[[dict[str, Any]], dict[str, Any]] = lambda _ctx: {}
    capture: Callable[[Any], dict[str, Any]] = lambda _raw: {}
    on_failure: OnFailure = "retry_once"
    description: str = ""

    def __post_init__(self) -> None:
        if self.on_failure not in {"skip", "retry_once", "abort"}:
            raise ValueError(
                f"WorkflowStep[{self.name}].on_failure={self.on_failure!r}; "
                "must be one of skip/retry_once/abort"
            )


@dataclass
class StepResult:
    name: str
    tool: str
    status: str  # ok / failed / skipped
    attempts: int
    elapsed_ms: int
    args: dict[str, Any] = field(default_factory=dict)
    captured: dict[str, Any] = field(default_factory=dict)
    raw_summary: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tool": self.tool,
            "status": self.status,
            "attempts": self.attempts,
            "elapsed_ms": self.elapsed_ms,
            "args": self.args,
            "captured": self.captured,
            "raw_summary": self.raw_summary[:600],
            "error": self.error[:600],
        }


# ============================================================
# Workflow
# ============================================================


@dataclass
class Workflow:
    name: str
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)

    def add(self, step: WorkflowStep) -> "Workflow":
        self.steps.append(step)
        return self

    async def run(
        self,
        initial_state: dict[str, Any] | None = None,
        *,
        reflector=None,
    ) -> dict:
        """执行整条 workflow。

        reflector: 可选的 Reflector 实例；
          - reflector.should_retry(step_result_dict, attempt_count) -> bool
          - reflector.summarize_run(run_log) -> str

        返回 run_log = {
            "run_id", "workflow", "started_at", "ended_at", "status",
            "steps": [StepResult.to_dict, ...],
            "final_state", "summary"
        }
        """
        ctx = WorkflowContext(state=dict(initial_state or {}))
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        step_results: list[StepResult] = []
        run_status = "ok"

        logger.info("[workflow/%s/%s] start", self.name, run_id)

        for step in self.steps:
            sr = await self._run_step(step, ctx, reflector)
            step_results.append(sr)
            if sr.status == "failed":
                # on_failure = abort 才中断；skip 已在 _run_step 内变成 skipped 状态
                run_status = "failed"
                logger.warning(
                    "[workflow/%s/%s] abort after step '%s'", self.name, run_id, step.name
                )
                break

        ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_log = {
            "run_id": run_id,
            "workflow": self.name,
            "description": self.description,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": run_status,
            "steps": [sr.to_dict() for sr in step_results],
            "final_state": dict(ctx.state),
            "summary": "",
        }

        # 反思器写总结（失败也写）
        if reflector is not None:
            try:
                run_log["summary"] = await _maybe_await(
                    reflector.summarize_run(run_log)
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("[workflow/%s/%s] reflector.summarize_run failed: %s",
                               self.name, run_id, e)
                run_log["summary"] = f"(summary failed: {e})"

        logger.info("[workflow/%s/%s] end status=%s", self.name, run_id, run_status)
        return run_log

    async def _run_step(
        self,
        step: WorkflowStep,
        ctx: WorkflowContext,
        reflector,
    ) -> StepResult:
        fn = get_tool(step.tool)
        if fn is None:
            return StepResult(
                name=step.name, tool=step.tool, status="failed",
                attempts=0, elapsed_ms=0,
                error=f"tool not registered: {step.tool}",
            )

        try:
            args = step.args_fn(ctx.state) or {}
        except Exception as e:  # noqa: BLE001
            return StepResult(
                name=step.name, tool=step.tool, status="failed",
                attempts=0, elapsed_ms=0,
                error=f"args_fn raised: {type(e).__name__}: {e}",
            )

        max_attempts = 2 if step.on_failure == "retry_once" else 1
        attempts = 0
        last_err = ""
        t0 = time.time()

        while attempts < max_attempts:
            attempts += 1
            try:
                logger.info("[step/%s] attempt %d tool=%s args=%s",
                            step.name, attempts, step.tool,
                            {k: _short(v) for k, v in args.items()})
                raw = await fn(**args)
                elapsed_ms = int((time.time() - t0) * 1000)
                captured: dict[str, Any] = {}
                try:
                    captured = step.capture(raw) or {}
                except Exception as cap_e:  # noqa: BLE001
                    logger.warning("[step/%s] capture failed: %s", step.name, cap_e)
                ctx.merge(captured)
                return StepResult(
                    name=step.name, tool=step.tool, status="ok",
                    attempts=attempts, elapsed_ms=elapsed_ms,
                    args=args, captured=captured,
                    raw_summary=_short(raw, max_len=1200),
                )
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                logger.warning("[step/%s] attempt %d failed: %s\n%s",
                               step.name, attempts, last_err,
                               traceback.format_exc(limit=2))
                # reflector 可以否决重试（如非瞬时错误）
                if attempts < max_attempts and reflector is not None:
                    try:
                        partial = {
                            "name": step.name, "tool": step.tool,
                            "status": "failed", "error": last_err,
                            "args": args, "attempts": attempts,
                        }
                        retry = bool(await _maybe_await(
                            reflector.should_retry(partial, attempts)
                        ))
                        if not retry:
                            logger.info("[step/%s] reflector vetoed retry", step.name)
                            break
                    except Exception as ref_e:  # noqa: BLE001
                        logger.warning("[step/%s] reflector.should_retry crashed: %s",
                                       step.name, ref_e)

        elapsed_ms = int((time.time() - t0) * 1000)
        if step.on_failure == "skip":
            logger.info("[step/%s] failed but on_failure=skip — continue", step.name)
            return StepResult(
                name=step.name, tool=step.tool, status="skipped",
                attempts=attempts, elapsed_ms=elapsed_ms,
                args=args, error=last_err,
            )
        return StepResult(
            name=step.name, tool=step.tool, status="failed",
            attempts=attempts, elapsed_ms=elapsed_ms,
            args=args, error=last_err,
        )


# ============================================================
# 辅助
# ============================================================


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


def _short(v: Any, max_len: int = 80) -> str:
    s = repr(v) if not isinstance(v, str) else v
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s
