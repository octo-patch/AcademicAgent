"""Phase 6 验证：评估子系统精简 + monitor + auto_rca。

测试边界：
  - 不调真实 LLM；auto_rca 走 use_llm=False 规则路径
  - 不动真实 Notion；recorder 没有 NOTION_API_KEY 时只写本地 md
  - 不动真实 eval runs；构造临时 run_dir
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation import auto_rca, monitor


def _seed_run_dir(tmp_path: Path) -> Path:
    """造一个 fake run 目录，包含 run_summary / failure_cards / traces。"""
    run_dir = tmp_path / "2026-05-21_0830_smoke"
    run_dir.mkdir()

    summary = {
        "run_id": "2026-05-21_0830_smoke",
        "config": {
            "dataset_path": "data/evaluation/datasets/smoke",
            "dataset_version": "1.0.0",
            "model": "MiniMax-Text-01",
            "code_hash": "abc1234",
            "offline": True,
        },
        "totals": {"total": 4, "ok": 2, "failed": 2, "skipped": 0, "contaminated": False},
        "by_metric": {
            "kg_node_f1": {"mean": 0.42, "n": 3},
            "tool_success_rate": {"mean": 1.0, "n": 4},
        },
        "latency": {"__all__": {"p50": 100, "p90": 220, "max": 480}},
        "cost": {"total_usd": 0.01, "avg_cost_usd_per_task": 0.0025},
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cards = [
        {
            "card_id": "c1", "task_id": "t_extract_1", "category": "extraction_empty",
            "severity": "P1", "trace_excerpt": "extractor returned nodes=0",
            "detail": "abstract too short",
            "repro_command": "python -m src.evaluation.cli run --task t_extract_1",
            "regression_test_suggestion": "加入这条 PDF 到 smoke 数据集",
        },
        {
            "card_id": "c2", "task_id": "t_tool_x", "category": "tool_exception",
            "severity": "P0", "trace_excerpt": "401 Unauthorized",
            "detail": "API key missing", "repro_command": "...",
            "regression_test_suggestion": "在 setup 加 mock key",
        },
    ]
    (run_dir / "failure_cards.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cards),
        encoding="utf-8",
    )

    traces = [
        {"task_id": "t_extract_1", "event": "tool_call", "status": "ok",
         "payload": {"tool": "add_paper_to_graph"}},
        {"task_id": "t_extract_1", "event": "tool_result", "status": "warn",
         "payload": {"nodes": 0, "edges": 0}},
        {"task_id": "t_tool_x", "event": "tool_error", "status": "failed",
         "payload": {"error": "401 Unauthorized"}},
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in traces),
        encoding="utf-8",
    )

    return run_dir


# ============================================================
# monitor
# ============================================================


def test_progress_monitor_counts_correctly(capsys):
    mon = monitor.ProgressMonitor(total=3)
    mon.tick("t1", "ok", elapsed_ms=10)
    mon.tick("t2", "failed", elapsed_ms=20)
    mon.tick("t3", "skipped", elapsed_ms=5)
    s = mon.summary()
    assert s == {"total": 3, "done": 3, "ok": 1, "failed": 1, "skipped": 1,
                 "elapsed_s": s["elapsed_s"]}


def test_record_run_to_refactor_writes_local_md(tmp_path: Path, monkeypatch):
    """没有 NOTION_API_KEY 时，recorder 应只写本地 md，不抛错。"""
    monkeypatch.delenv("NOTION_API_KEY", raising=False)

    # 让 RefactorRecorder 的全局 _STATE_DIR / _DOCS_DIR 指到 tmp_path
    import src.recorder.refactor_recorder as rr
    monkeypatch.setattr(rr, "_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(rr, "_DOCS_DIR", tmp_path / "docs_refactor")

    run_dir = _seed_run_dir(tmp_path)
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    contam = {"contaminated": False, "changed_paths": []}

    result = monitor.record_run_to_refactor(run_dir, summary, contam)
    assert result["area"] == "eval-runs"
    assert result["written_step"] is True
    assert result["written_validations"] == 2  # 两张卡

    md = tmp_path / "docs_refactor" / "eval-runs.md"
    assert md.exists()
    text = md.read_text(encoding="utf-8")
    assert "eval run" in text
    assert "t_extract_1" in text or "extraction_empty" in text


# ============================================================
# auto_rca
# ============================================================


def test_auto_rca_rule_based_generates_md(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)  # 强制走规则路径
    run_dir = _seed_run_dir(tmp_path)

    out = auto_rca.generate_rca(run_dir, use_llm=False, sync_to_recorder=False)
    assert out["bucket_summary"] == {"extraction_empty": 1, "tool_exception": 1}
    assert out["total_failures"] == 2
    md = Path(out["md_path"])
    assert md.exists()
    text = md.read_text(encoding="utf-8")
    assert "mermaid" in text
    assert "extraction_empty" in text
    assert "tool_exception" in text
    assert "根因假设" in text
    assert "规则表" in text  # use_llm=False 时应说明来源


def test_auto_rca_handles_no_failures(tmp_path: Path):
    """无失败卡片时不应崩；md 显示"无失败任务"。"""
    run_dir = tmp_path / "2026-05-21_clean"
    run_dir.mkdir()
    (run_dir / "run_summary.json").write_text(
        json.dumps({"run_id": "2026-05-21_clean",
                    "totals": {"total": 1, "ok": 1, "failed": 0, "skipped": 0}}),
        encoding="utf-8",
    )
    (run_dir / "failure_cards.jsonl").write_text("", encoding="utf-8")
    (run_dir / "traces.jsonl").write_text("", encoding="utf-8")

    out = auto_rca.generate_rca(run_dir, use_llm=False, sync_to_recorder=False)
    assert out["total_failures"] == 0
    text = Path(out["md_path"]).read_text(encoding="utf-8")
    assert "无失败任务" in text


def test_auto_rca_marks_contamination(tmp_path: Path):
    run_dir = _seed_run_dir(tmp_path)
    # 改 summary 标 contaminated
    s = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    s["totals"]["contaminated"] = True
    (run_dir / "run_summary.json").write_text(
        json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out = auto_rca.generate_rca(run_dir, use_llm=False, sync_to_recorder=False)
    text = Path(out["md_path"]).read_text(encoding="utf-8")
    assert "CONTAMINATED" in text


# ============================================================
# CLI parser
# ============================================================


def test_cli_no_longer_has_gate_subcommand():
    """Phase 6.1：gate 子命令已移除，rca 已加入。"""
    from src.evaluation.cli import _build_parser
    p = _build_parser()
    subs_action = p._subparsers._actions[-1]
    names = set(subs_action.choices.keys())
    assert "gate" not in names
    assert "rca" in names
    assert "sync" in names
    assert "run" in names


def test_gate_module_is_gone():
    """gate.py 已删除。"""
    with pytest.raises(ModuleNotFoundError):
        __import__("src.evaluation.gate", fromlist=[""])


def test_notion_sync_module_is_gone():
    """notion_sync.py 已删除（功能并入 monitor.record_run_to_refactor）。"""
    with pytest.raises(ModuleNotFoundError):
        __import__("src.evaluation.notion_sync", fromlist=[""])
