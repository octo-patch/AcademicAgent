"""
评测监控 / 记录管道（Phase 6.3）
================================

把评测 run 的"进度 + 摘要 + 失败"通过 `RefactorRecorder("eval-runs")` 统一同步到
Notion + 本地 `docs/refactor/eval-runs.md`，与重构步骤记录走同一管道。

替代旧的 `notion_sync.py`：不再维护独立 Notion 镜像逻辑，所有写入都走 recorder。

提供两个入口：
  - ProgressMonitor(total_tasks): 实时打印进度条 + 收集 timing
  - record_run_to_refactor(run_dir, run_summary, contamination): run 结束后
    把摘要 + 失败 + 指标分位写到 recorder area="eval-runs"
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("AcademicAgent.Eval.Monitor")


# ============================================================
# 进度条（无依赖 minimal）
# ============================================================


class ProgressMonitor:
    """轻量进度跟踪。

    每 tick(name, status) 一次更新内部计数，打印形如：
      [eval] 5/12 ok=4 failed=1 skipped=0  | last: smoke_l1_extract (ok, 832ms)
    """

    def __init__(self, total: int, *, stream=None, prefix: str = "[eval]"):
        self.total = max(0, int(total))
        self.done = 0
        self.ok = 0
        self.failed = 0
        self.skipped = 0
        self._started_at = time.time()
        self._stream = stream if stream is not None else sys.stderr
        self._prefix = prefix

    def tick(self, name: str, status: str, elapsed_ms: int = 0) -> None:
        s = (status or "").lower()
        if s in ("ok", "pass", "success"):
            self.ok += 1
        elif s in ("skip", "skipped"):
            self.skipped += 1
        else:
            self.failed += 1
        self.done += 1
        self._render(last=f"{name} ({status}, {elapsed_ms}ms)")

    def _render(self, last: str = "") -> None:
        line = (
            f"{self._prefix} {self.done}/{self.total} "
            f"ok={self.ok} failed={self.failed} skipped={self.skipped}"
        )
        if last:
            line += f"  | last: {last}"
        try:
            print(line, file=self._stream, flush=True)
        except Exception:  # noqa: BLE001
            pass

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "done": self.done,
            "ok": self.ok,
            "failed": self.failed,
            "skipped": self.skipped,
            "elapsed_s": round(time.time() - self._started_at, 2),
        }


# ============================================================
# RefactorRecorder 写入
# ============================================================


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def _format_metrics_inline(by_metric: dict) -> str:
    """从 run_summary["by_metric"] 取每个指标的 mean，拼成单行文本。"""
    if not by_metric:
        return "(no metrics)"
    parts = []
    for name, agg in sorted(by_metric.items()):
        if isinstance(agg, dict) and "mean" in agg:
            parts.append(f"{name}={agg['mean']:.3f}(n={agg.get('n', '?')})")
    return ", ".join(parts) or "(empty)"


def record_run_to_refactor(
    run_dir: str | Path,
    run_summary: dict[str, Any],
    contamination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把 run 摘要写到 RefactorRecorder(area="eval-runs")。

    与"重构步骤"共享同一记录管道：
      - log_step：记录这次 run 是"哪个数据集/模型/git hash + 总数 + 关键指标"
      - log_validation：每个 failure_card 转一条 validation 记录（status=fail/warn）

    返回 {area, notion_url?, written_step, written_validations}。
    """
    from src.recorder import RefactorRecorder

    run_dir = Path(run_dir)
    contam = contamination or {}
    rid = run_summary.get("run_id", run_dir.name)

    recorder = RefactorRecorder(
        area="eval-runs",
        title="📊 评测 run 记录",
        icon_emoji="📊",
    )

    # ----- log_step: run 总览 -----
    totals = run_summary.get("totals", {})
    config = run_summary.get("config") or {}
    by_metric = run_summary.get("by_metric", {})
    latency = run_summary.get("latency", {}).get("__all__", {})
    cost = run_summary.get("cost", {})

    problem_lines = []
    if contam.get("contaminated"):
        problem_lines.append(
            "⚠️ CONTAMINATED：受保护文件被改动 "
            + ", ".join(contam.get("changed_paths", []))
        )
    if totals.get("failed", 0):
        problem_lines.append(f"失败任务 {totals['failed']} 个")
    problem = "\n".join(problem_lines) or "无致命问题"

    effect_lines = [
        f"任务总数: {totals.get('total', '?')}  "
        f"ok={totals.get('ok', 0)}  "
        f"failed={totals.get('failed', 0)}  "
        f"skipped={totals.get('skipped', 0)}",
        f"指标: {_format_metrics_inline(by_metric)}",
    ]
    if latency:
        effect_lines.append(
            f"延迟: p50={latency.get('p50', 0):.0f}ms  "
            f"p90={latency.get('p90', 0):.0f}ms  "
            f"max={latency.get('max', 0):.0f}ms"
        )
    if cost:
        effect_lines.append(
            f"成本: total_usd={cost.get('total_usd', 0):.4f}  "
            f"avg_per_task={cost.get('avg_cost_usd_per_task', 0):.4f}"
        )

    recorder.log_step(
        title=f"eval run {rid}",
        problem=problem,
        approach=(
            f"dataset={config.get('dataset_path', '-')} "
            f"version={config.get('dataset_version', '-')} "
            f"model={config.get('model', '-')} "
            f"code={config.get('code_hash', '-')} "
            f"offline={config.get('offline', False)}"
        ),
        files_changed=[
            f"data/evaluation/runs/{rid}/run_summary.json",
            f"data/evaluation/runs/{rid}/metrics.json",
            f"data/evaluation/runs/{rid}/failure_cards.jsonl",
        ],
        effect="\n".join(effect_lines),
        risks_resolved=([] if not contam.get("contaminated") else ["污染检测告警"]),
    )

    # ----- log_validation：失败卡片每条一条 -----
    cards = _read_jsonl(run_dir / "failure_cards.jsonl")
    written_validations = 0
    for c in cards[:20]:  # 限制 20 条避免 Notion 块爆
        sev = (c.get("severity") or "warn").lower()
        status = "fail" if sev in ("critical", "high", "fail") else "warn"
        recorder.log_validation(
            step=f"{c.get('task_id','?')} · {c.get('category','?')}",
            command=c.get("repro_command", ""),
            output=(c.get("hypothesis", "") + "\n\n" + c.get("regression_test_suggestion", "")).strip(),
            status=status,
            notes=f"card_id={c.get('card_id','?')}",
        )
        written_validations += 1

    state_path = Path("data/recorder_state/eval-runs.json")
    notion_url = ""
    if state_path.exists():
        try:
            notion_url = json.loads(state_path.read_text(encoding="utf-8")).get("notion_url", "")
        except Exception:  # noqa: BLE001
            pass

    return {
        "area": "eval-runs",
        "notion_url": notion_url,
        "written_step": True,
        "written_validations": written_validations,
        "local_md": "docs/refactor/eval-runs.md",
    }


# ============================================================
# 帮助：从 task_results 派生进度 ticks（runner 内部用）
# ============================================================


def derive_progress(task_results: Iterable[dict]) -> ProgressMonitor:
    """给定一组已完成的 task_result dict，重建进度统计（用于 sync 后补写）。"""
    results = list(task_results)
    mon = ProgressMonitor(total=len(results))
    for r in results:
        mon.tick(
            name=r.get("task_id", "?"),
            status=r.get("status", "?"),
            elapsed_ms=int(r.get("latency_ms", 0)),
        )
    return mon
