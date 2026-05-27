from __future__ import annotations

"""
AcademicAgent 评测统一 CLI。

子命令：
  run              运行评测（按档位/层/任务过滤），落盘 artifacts + 失败卡片 + 报告
  report           对一个 run 目录重新生成 HTML/MD 报告
  sync             把一个 run 目录通过 RefactorRecorder 同步到 Notion（本地优先，失败不阻断）
  rca              对一个 run 目录跑 LLM 自动 RCA，输出 rca.md
  list             列出数据集中的任务
  validate         校验数据集，返回非零退出码表示有问题
  promote-failures 把失败卡片转成新 gold 任务的候选（数据飞轮）

【Phase 6.1 变更】`gate` 子命令已移除（个人项目不再做 CI/CD 风格门禁）。
thresholds.yaml 保留为失败卡片"低于基线"判定的输入；不再决定退出码。

示例：
  python -m src.evaluation.cli run --dataset data/evaluation/datasets/smoke \\
      --out data/evaluation/runs --tier smoke --offline
  python -m src.evaluation.cli rca --run data/evaluation/runs/<run_id>
  python -m src.evaluation.cli validate --dataset data/evaluation/datasets/smoke
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .dataset import load_dataset
from .schema import EvalLayer, Tier

logger = logging.getLogger("AcademicAgent.Eval.CLI")

_LAYER_MAP = {
    "component": EvalLayer.COMPONENT, "l1": EvalLayer.COMPONENT,
    "workflow": EvalLayer.WORKFLOW, "l2": EvalLayer.WORKFLOW,
    "e2e": EvalLayer.E2E, "l3": EvalLayer.E2E,
}


# ------------------------------------------------------------------ #
# run
# ------------------------------------------------------------------ #

def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_evaluation

    tier = Tier(args.tier) if args.tier else None
    layers = None
    if args.layers:
        layers = [_LAYER_MAP[x.strip().lower()] for x in args.layers.split(",")]
    task_ids = [t.strip() for t in args.task.split(",")] if args.task else None

    summary = asyncio.run(run_evaluation(
        dataset_path=args.dataset, out_dir=args.out, tier=tier, layers=layers,
        offline=args.offline, run_id=args.run_id, model=args.model,
        enable_e2e=args.enable_e2e, notion=args.notion, task_ids=task_ids,
    ))
    totals = summary["totals"]
    print(f"\nrun 完成: {summary['run_id']}")
    print(f"  目录: data/evaluation/runs/{summary['run_id']}")
    print(f"  任务: {totals['total']} 总 / {totals['ok']} ok / "
          f"{totals['failed']} failed / {totals['skipped']} skipped")
    if totals.get("contaminated"):
        print("  [!] CONTAMINATED — 受保护文件被改动")
        return 2
    # 失败卡片数量提示（不再有 gate exit code，但保留失败信号）
    cards_file = Path(args.out) / summary["run_id"] / "failure_cards.jsonl"
    if cards_file.exists():
        try:
            n = sum(1 for line in cards_file.read_text(encoding="utf-8").splitlines() if line.strip())
            if n:
                print(f"  失败卡片: {n} 张（详见 failure_cards.jsonl / failures.md）")
        except Exception:  # noqa: BLE001
            pass
    return 0 if totals["failed"] == 0 else 1


# ------------------------------------------------------------------ #
# report
# ------------------------------------------------------------------ #

def _cmd_report(args: argparse.Namespace) -> int:
    from .reporting.html import generate_html_report
    from .reporting.markdown import generate_markdown_report

    run_dir = Path(args.run)
    html = generate_html_report(run_dir)
    md = generate_markdown_report(run_dir)
    print(f"报告已生成: {html}  {md}")
    return 0


# ------------------------------------------------------------------ #
# sync
# ------------------------------------------------------------------ #

def _cmd_sync(args: argparse.Namespace) -> int:
    """通过 RefactorRecorder("eval-runs") 同步到 Notion + 本地 md 镜像。"""
    from .monitor import record_run_to_refactor

    run_dir = Path(args.run)
    summary_file = run_dir / "run_summary.json"
    if not summary_file.exists():
        print(f"run_summary.json 不存在: {summary_file}", file=sys.stderr)
        return 1
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    contam = {"contaminated": bool(summary.get("totals", {}).get("contaminated"))}
    result = record_run_to_refactor(run_dir, summary, contam)
    print(f"已记录到 RefactorRecorder area=eval-runs (本地: docs/refactor/eval-runs.md)")
    if result.get("notion_url"):
        print(f"Notion: {result['notion_url']}")
    return 0


# ------------------------------------------------------------------ #
# rca
# ------------------------------------------------------------------ #

def _cmd_rca(args: argparse.Namespace) -> int:
    """对 run 目录跑 LLM 自动 RCA，输出 rca.md。"""
    from .auto_rca import generate_rca

    run_dir = Path(args.run)
    out = generate_rca(run_dir, use_llm=not args.no_llm)
    print(f"RCA 报告: {out['md_path']}")
    print(f"失败分桶: {out['bucket_summary']}")
    if out.get("notion_url"):
        print(f"Notion: {out['notion_url']}")
    return 0


# ------------------------------------------------------------------ #
# list
# ------------------------------------------------------------------ #

def _cmd_list(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    tier = Tier(args.tier) if args.tier else None
    tasks = dataset.filter(tier=tier)
    print(f"数据集 {args.dataset}  版本 {dataset.version.version}  共 {len(tasks)} 任务"
          + (f"（档位 {tier.value}）" if tier else ""))
    for t in tasks:
        target = (t.target.tool or t.target.workflow or "e2e")
        flags = []
        if t.requires_api:
            flags.append("api")
        if t.requires_llm:
            flags.append("llm")
        flag_str = f" [{'/'.join(flags)}]" if flags else ""
        print(f"  {t.task_id:22s} {t.layer.value:18s} {t.capability.value:16s} "
              f"{target:22s}{flag_str}")
    return 0


# ------------------------------------------------------------------ #
# validate
# ------------------------------------------------------------------ #

def _cmd_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    problems = dataset.validate()
    if not problems:
        print(f"[OK] 数据集校验通过: {args.dataset} "
              f"({len(dataset.tasks)} 任务, 版本 {dataset.version.version})")
        return 0
    print(f"[FAIL] 数据集校验失败: {args.dataset}")
    for p in problems:
        print(f"  - {p}")
    return 1


# ------------------------------------------------------------------ #
# promote-failures
# ------------------------------------------------------------------ #

def _cmd_promote_failures(args: argparse.Namespace) -> int:
    """把失败卡片转成新 gold 任务候选 —— 数据飞轮。

    本期实现：打印每张卡片的摘要 + 建议的回归测试，输出一个候选清单。
    人工挑选后再编辑数据集（保留对 gold label 的最终判断权）。
    """
    cards_path = Path(args.cards)
    if not cards_path.exists():
        print(f"失败卡片文件不存在: {cards_path}")
        return 1
    cards = [json.loads(line) for line in cards_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cards:
        print("没有失败卡片。")
        return 0
    print(f"# 失败卡片 -> 回归任务候选（共 {len(cards)} 张）\n")
    for c in cards:
        print(f"## [{c['severity']}] {c['card_id']} — {c['task_id']} ({c['category']})")
        print(f"   回归测试建议: {c['regression_test_suggestion']}")
        print(f"   复现: {c['repro_command']}")
        print()
    print("下一步：人工挑选要纳入回归集的卡片，编辑 datasets/<tier>/tasks.jsonl 与对应 gold 文件。")
    return 0


# ------------------------------------------------------------------ #
# parser
# ------------------------------------------------------------------ #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluation.cli",
        description="AcademicAgent 评测统一 CLI",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)

    # run
    pr = sub.add_parser("run", help="运行评测")
    pr.add_argument("--dataset", required=True)
    pr.add_argument("--out", default="data/evaluation/runs")
    pr.add_argument("--tier", choices=["smoke", "full"], default=None)
    pr.add_argument("--layers", default=None,
                    help="逗号分隔: component,workflow,e2e（默认 component,workflow）")
    pr.add_argument("--task", default=None, help="只跑指定 task_id（逗号分隔）")
    pr.add_argument("--offline", action="store_true", help="跳过所有需要 API/LLM 的任务")
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--model", default=None)
    pr.add_argument("--notion", action="store_true", help="run 后自动通过 RefactorRecorder 同步")
    pr.add_argument("--enable-e2e", action="store_true", help="启用 L3 端到端层")
    pr.set_defaults(func=_cmd_run)

    # report
    prp = sub.add_parser("report", help="重新生成报告")
    prp.add_argument("--run", required=True)
    prp.set_defaults(func=_cmd_report)

    # sync
    ps = sub.add_parser("sync", help="通过 RefactorRecorder 同步 run 到 Notion + 本地 md")
    ps.add_argument("--run", required=True)
    ps.set_defaults(func=_cmd_sync)

    # rca
    pra = sub.add_parser("rca", help="对 run 目录跑 LLM 自动 RCA")
    pra.add_argument("--run", required=True)
    pra.add_argument("--no-llm", action="store_true",
                     help="跳过 LLM 调用，只做规则化分桶")
    pra.set_defaults(func=_cmd_rca)

    # list
    pl = sub.add_parser("list", help="列出数据集任务")
    pl.add_argument("--dataset", required=True)
    pl.add_argument("--tier", choices=["smoke", "full"], default=None)
    pl.set_defaults(func=_cmd_list)

    # validate
    pv = sub.add_parser("validate", help="校验数据集")
    pv.add_argument("--dataset", required=True)
    pv.set_defaults(func=_cmd_validate)

    # promote-failures
    pf = sub.add_parser("promote-failures", help="失败卡片 -> 回归任务候选")
    pf.add_argument("--cards", required=True, help="failure_cards/<run_id>.jsonl 路径")
    pf.set_defaults(func=_cmd_promote_failures)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
