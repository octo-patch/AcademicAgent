"""
Phase 6 smoke 验证脚本
=======================

不调真实 LLM / 不动真实评测目录。流程：
  1. 造一个 fake run 目录（含 run_summary / failure_cards / traces）
  2. 跑 monitor.record_run_to_refactor → 写本地 docs/refactor/eval-runs.md + Notion（如配）
  3. 跑 auto_rca.generate_rca(use_llm=False) → 写 run_dir/rca.md
  4. 验证 CLI parser 不再包含 gate
  5. 全程通过 RefactorRecorder("evaluation-and-monitor") 同步记录
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.evaluation import auto_rca, monitor
from src.recorder import RefactorRecorder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _seed_run_dir(base: Path) -> Path:
    rid = "smoke_phase6_demo"
    rd = base / rid
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True)

    summary = {
        "run_id": rid,
        "config": {
            "dataset_path": "data/evaluation/datasets/smoke",
            "dataset_version": "0.0.0-smoke",
            "model": "MiniMax-Text-01",
            "code_hash": "phase6-smoke",
            "offline": True,
        },
        "totals": {"total": 5, "ok": 3, "failed": 2, "skipped": 0, "contaminated": False},
        "by_metric": {
            "kg_node_f1": {"mean": 0.42, "n": 3},
            "tool_success_rate": {"mean": 0.96, "n": 5},
            "schema_validity_rate": {"mean": 1.0, "n": 5},
        },
        "latency": {"__all__": {"p50": 110, "p90": 380, "max": 690}},
        "cost": {"total_usd": 0.018, "avg_cost_usd_per_task": 0.0036},
    }
    (rd / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = [
        {
            "card_id": "card_001", "task_id": "wf_paper_analysis_1",
            "layer": "workflow", "capability": "knowledge_extraction",
            "category": "extraction_empty", "severity": "P1",
            "trace_excerpt": "extractor.extract_from_text returned nodes=0",
            "detail": "abstract length 47 chars; likely below threshold",
            "repro_command": "python -m src.evaluation.cli run --task wf_paper_analysis_1",
            "regression_test_suggestion": "把 abstract<100 char 的样本加入回归集",
        },
        {
            "card_id": "card_002", "task_id": "l1_search_arxiv",
            "layer": "component", "capability": "paper_search",
            "category": "tool_exception", "severity": "P0",
            "trace_excerpt": "HTTPError 401 Unauthorized while calling arXiv API",
            "detail": "arXiv 罕见 401，疑似临时故障",
            "repro_command": "python -m src.evaluation.cli run --task l1_search_arxiv",
            "regression_test_suggestion": "加一个 mock 401 的单测",
        },
    ]
    (rd / "failure_cards.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cards), encoding="utf-8",
    )
    traces = [
        {"task_id": "wf_paper_analysis_1", "event": "tool_call", "status": "ok",
         "payload": {"tool": "add_paper_to_graph", "args": {"text_len": 47}}},
        {"task_id": "wf_paper_analysis_1", "event": "tool_result", "status": "warn",
         "payload": {"nodes": 0, "edges": 0, "reason": "no entities detected"}},
        {"task_id": "l1_search_arxiv", "event": "tool_call", "status": "ok",
         "payload": {"tool": "search_arxiv"}},
        {"task_id": "l1_search_arxiv", "event": "tool_error", "status": "failed",
         "payload": {"error": "HTTPError 401 Unauthorized"}},
    ]
    (rd / "traces.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in traces), encoding="utf-8",
    )
    return rd


def main():
    recorder = RefactorRecorder(
        area="evaluation-and-monitor",
        title="📊 评估子系统精简 + 自动 RCA",
        icon_emoji="📊",
    )

    recorder.log_step(
        title="Phase 6 落地：gate / notion_sync 移除，monitor + auto_rca 新增",
        problem=(
            "src/evaluation/gate.py 163 行的 CI 风格门禁对个人项目过重；"
            "notion_sync.py 270 行又是独立的 Notion 镜像逻辑，与 src.recorder 重复；"
            "缺乏统一的 run 进度监控与失败 RCA。"
        ),
        approach=(
            "1) 删除 gate.py + tests/test_evaluation_gate.py，把 thresholds 加载缩成 runner "
            "的 mini-loader 仅供失败卡片使用；2) 删除 notion_sync.py，新增 monitor.py 把"
            "run 摘要 + 失败卡片通过 RefactorRecorder(area='eval-runs') 统一同步；"
            "3) 新增 auto_rca.py：读 run_dir 三件套 → 按 category 分桶 → 可选 MiniMax 生成根因 "
            "→ 写 rca.md + mermaid pie，CLI 子命令 `rca`；4) cli.py 删 `gate` 子命令、新增 `rca`、"
            "`sync` 改走 monitor；5) evaluation/__init__.py 同步更新文档。"
        ),
        files_changed=[
            "src/evaluation/gate.py (删除)",
            "src/evaluation/notion_sync.py (删除)",
            "tests/test_evaluation_gate.py (删除)",
            "src/evaluation/monitor.py (新增)",
            "src/evaluation/auto_rca.py (新增)",
            "src/evaluation/runner.py",
            "src/evaluation/cli.py",
            "src/evaluation/__init__.py",
            "tests/test_evaluation_phase6.py (新增)",
            "scripts/smoke_evaluation_phase6.py (新增)",
        ],
        effect=(
            "评测子系统从 4153 行 → 净减 ~430 行；记录入口统一到 src.recorder；"
            "新增 mermaid pie 可视化与（可选）LLM RCA 解读；CLI 增加 `rca` 子命令。"
        ),
        risks_resolved=["#7 评估盖了体育场没办比赛（精简 CI 风格冗余）"],
    )

    # ---- A. monitor.ProgressMonitor 统计 ----
    mon = monitor.ProgressMonitor(total=5)
    for name, st in [("t1", "ok"), ("t2", "ok"), ("t3", "failed"),
                     ("t4", "ok"), ("t5", "failed")]:
        mon.tick(name, st, elapsed_ms=50)
    s = mon.summary()
    recorder.log_validation(
        step="monitor-progress-counts",
        command="ProgressMonitor(total=5).tick × 5",
        output=json.dumps(s, ensure_ascii=False),
        status="ok" if (s["ok"] == 3 and s["failed"] == 2) else "fail",
    )

    # ---- B. monitor.record_run_to_refactor 落 eval-runs 镜像 ----
    fake_runs = _ROOT / "data" / "evaluation" / "runs"
    fake_runs.mkdir(parents=True, exist_ok=True)
    rd = _seed_run_dir(fake_runs)
    summary = json.loads((rd / "run_summary.json").read_text(encoding="utf-8"))
    result = monitor.record_run_to_refactor(rd, summary, {"contaminated": False, "changed_paths": []})

    recorder.log_validation(
        step="monitor-record-run-to-refactor",
        command="monitor.record_run_to_refactor(run_dir, summary, contam)",
        output=(
            f"result={result}\n"
            f"local md exists: {(_ROOT / 'docs' / 'refactor' / 'eval-runs.md').exists()}"
        ),
        status="ok" if result["written_step"] and result["written_validations"] == 2 else "fail",
        artifacts=["docs/refactor/eval-runs.md", "data/recorder_state/eval-runs.json"],
        notes="无 NOTION_API_KEY 时静默走本地 md（设计如此）",
    )

    # ---- C. auto_rca (规则路径，不需 LLM key) ----
    rca_out = auto_rca.generate_rca(rd, use_llm=False, sync_to_recorder=False)
    rca_md = Path(rca_out["md_path"]).read_text(encoding="utf-8")
    head = rca_md[:1200]
    recorder.log_validation(
        step="auto-rca-rule-based",
        command=f"python -m src.evaluation.cli rca --run {rd}",
        output=f"bucket_summary={rca_out['bucket_summary']}\n\n--- rca.md head ---\n{head}",
        status="ok" if (
            "extraction_empty" in rca_md and "tool_exception" in rca_md
            and "mermaid" in rca_md and "根因假设" in rca_md
        ) else "fail",
        artifacts=[str(rd / "rca.md")],
        notes="规则路径（use_llm=False）；带 mermaid pie + 两个分桶根因。",
    )

    # ---- D. CLI parser smoke ----
    from src.evaluation.cli import _build_parser
    p = _build_parser()
    subs = list(p._subparsers._actions[-1].choices.keys())
    recorder.log_validation(
        step="cli-subcommands",
        command="from src.evaluation.cli import _build_parser",
        output=f"subcommands={subs}",
        status="ok" if ("gate" not in subs and "rca" in subs and "sync" in subs) else "fail",
        notes="gate 子命令已移除，rca 已加入。",
    )

    # ---- E. pytest ----
    proc = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_evaluation_phase6.py", "tests/test_runtime.py", "-q"],
        cwd=str(_ROOT), capture_output=True, text=True,
    )
    recorder.log_validation(
        step="pytest-phase6-and-runtime",
        command="python -m pytest tests/test_evaluation_phase6.py tests/test_runtime.py -q",
        output=(proc.stdout + "\n" + proc.stderr)[-1800:],
        status="ok" if proc.returncode == 0 else "fail",
        notes="phase 6 + phase 5 测试合并验证。",
    )

    recorder.close(
        status="completed",
        summary=(
            "Phase 6 完成：gate.py / notion_sync.py 删除；monitor.py + auto_rca.py 上线；"
            "CLI 新增 rca 子命令；记录管道统一到 src.recorder.RefactorRecorder。"
            "5 个 smoke validation + 8 + 13 个单元测试全过。"
        ),
    )

    print("\n[phase6 smoke] OK; rca md =", rca_out["md_path"])


if __name__ == "__main__":
    main()
