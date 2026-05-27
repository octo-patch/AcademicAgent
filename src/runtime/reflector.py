"""
Agent Runtime - Lite 反思器
==========================

不是用 LLM 决定整个分支（成本贵 + 不可控），而是基于规则 + 错误类别 + 步骤类型
来做"是否重试"与"跑完一句话总结"两件事。

设计目标：
  - should_retry: 在 reflector 视角下决定第 N 次是否重试（runner 内部还有一个 max_attempts
    上限，two-layer 保险防死循环）。
  - summarize_run: 跑完后给一句 100 字内的中文总结，可写进 memory/experiences/。

关键规则（实测痛点驱动）：
  - LLM JSON parse 失败：重试 1 次（temperature 默认 0，提高随机性可能解套）
  - extractor 返回 nodes=0：重试 1 次（在 prompt 加"上次抽取为空"提示——
    本模块只产生"建议"，不直接改 prompt；下游 worker 可以读 retry_hint 字段）
  - 工具 API 限流：runner 已有 backoff，本模块不再重试（避免叠加）
  - 认证 / 配置错误（API key 缺失等）：不重试，立即 abort
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("ScholarMind.Runtime.Reflector")


# 错误关键词分桶
_RETRY_OK_PATTERNS = [
    r"json.*decode",
    r"parse.*fail",
    r"extractionerror",
    r"timeout",
    r"connection",
    r"temporar",
    r"empty.*nodes",
    r"nodes=0",
]
_RETRY_VETO_PATTERNS = [
    r"401",
    r"403",
    r"unauthorized",
    r"forbidden",
    r"invalid api key",
    r"missing.*api.*key",
    r"no such file",
    r"file not found",
    r"tool not registered",
    r"keyerror",
    r"typeerror.*missing",
]


class Reflector:
    """Lite 反思器。可被 Workflow.run 引用。"""

    def __init__(self, *, llm_summarizer=None):
        """llm_summarizer: 可选 async (run_log: dict) -> str；不传则用 deterministic 总结。"""
        self._llm_summarizer = llm_summarizer

    # --------- 重试判定 ---------

    async def should_retry(self, step_result: dict, attempt_count: int) -> bool:
        err = (step_result.get("error") or "").lower()
        tool = (step_result.get("tool") or "").lower()

        if attempt_count >= 2:
            return False

        # 否决类：明显非瞬时错误
        for pat in _RETRY_VETO_PATTERNS:
            if re.search(pat, err):
                logger.info("reflector: veto retry by pattern %r on tool=%s",
                            pat, tool)
                return False

        # 鼓励类
        for pat in _RETRY_OK_PATTERNS:
            if re.search(pat, err):
                return True

        # extractor 类工具默认重试一次（empty/parse 问题常见）
        if "add_paper_to_graph" in tool or "extract" in tool:
            return True

        return False

    # --------- 跑后总结 ---------

    async def summarize_run(self, run_log: dict) -> str:
        if self._llm_summarizer is not None:
            try:
                return await self._llm_summarizer(run_log)
            except Exception as e:  # noqa: BLE001
                logger.warning("llm_summarizer failed, fallback: %s", e)

        return self._deterministic_summary(run_log)

    @staticmethod
    def _deterministic_summary(run_log: dict) -> str:
        wf = run_log.get("workflow", "")
        status = run_log.get("status", "")
        steps = run_log.get("steps", [])
        ok = sum(1 for s in steps if s.get("status") == "ok")
        failed = [s for s in steps if s.get("status") == "failed"]
        skipped = [s for s in steps if s.get("status") == "skipped"]
        retried = sum(1 for s in steps if s.get("attempts", 1) > 1)
        ms = sum(int(s.get("elapsed_ms", 0)) for s in steps)

        parts = [f"[{wf}] 状态={status}; 步骤 {ok}/{len(steps)} 成功"]
        if retried:
            parts.append(f"重试 {retried} 次")
        if failed:
            parts.append(
                "失败: "
                + ", ".join(f"{s['name']}({s.get('error','')[:80]})" for s in failed[:2])
            )
        if skipped:
            parts.append("跳过: " + ", ".join(s["name"] for s in skipped[:3]))
        parts.append(f"总耗时 {ms} ms")
        return "; ".join(parts)
