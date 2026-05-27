"""
Agent Runtime - CLI
====================

用法：
  python -m src.runtime run <workflow> [--key value ...]
  python -m src.runtime list
  python -m src.runtime prefs                            # 列当前偏好
  python -m src.runtime prefs set <key> <value>          # 更新偏好

示例：
  python -m src.runtime run paper_analysis_flow --pdf_path data/papers/MemGPT.pdf
  python -m src.runtime run knowledge_build_flow --focus_area "agent memory"
  python -m src.runtime prefs set default_domain llm_agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from . import run_workflow
from .preferences import load_preferences, update_preference
from .workflows_registry import all_workflows


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RUN_DIR = _PROJECT_ROOT / "data" / "runtime_runs"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _parse_kvs(rest: list[str]) -> dict[str, str]:
    """把 --key value --key2 value2 转成 {key: value}。"""
    state: dict[str, str] = {}
    i = 0
    while i < len(rest):
        token = rest[i]
        if token.startswith("--"):
            key = token[2:]
            if i + 1 >= len(rest) or rest[i + 1].startswith("--"):
                state[key] = "true"
                i += 1
            else:
                state[key] = rest[i + 1]
                i += 2
        else:
            i += 1
    return state


def _save_run_log(run_log: dict) -> Path:
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    rid = run_log.get("run_id", "unknown")
    wf = run_log.get("workflow", "wf")
    p = _RUN_DIR / f"{wf}_{rid}.json"
    p.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def cmd_list(_args: argparse.Namespace) -> int:
    print("已注册 workflow：")
    for name, wf in all_workflows().items():
        print(f"  - {name}: {wf.description}")
        for s in wf.steps:
            print(f"      · {s.name} (tool={s.tool}, on_failure={s.on_failure})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    name = args.workflow
    initial_state = _parse_kvs(args.rest)
    print(f"[runtime] run workflow={name} state={initial_state}")
    run_log = asyncio.run(
        run_workflow(
            name,
            initial_state,
            use_reflector=None if args.reflector is None else args.reflector == "on",
            record_experience=not args.no_experience,
        )
    )
    saved = _save_run_log(run_log)
    print(f"\n[runtime] status={run_log['status']}  run_id={run_log['run_id']}")
    print(f"[runtime] summary: {run_log.get('summary','')}")
    print(f"[runtime] log saved: {saved}")
    if run_log.get("experience_files"):
        print(f"[runtime] experience updated: {run_log['experience_files']}")
    return 0 if run_log["status"] == "ok" else 1


def cmd_prefs(args: argparse.Namespace) -> int:
    if args.action == "show":
        print(json.dumps(load_preferences(), ensure_ascii=False, indent=2))
        return 0
    if args.action == "set":
        if not args.key or args.value is None:
            print("usage: prefs set <key> <value>", file=sys.stderr)
            return 2
        # 类型尽力：true/false 转 bool，纯数字转 int
        v: object = args.value
        if v in ("true", "false"):
            v = v == "true"
        elif isinstance(v, str) and v.isdigit():
            v = int(v)
        new_prefs = update_preference(args.key, v)
        print(json.dumps(new_prefs, ensure_ascii=False, indent=2))
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.runtime")
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出已注册 workflow")

    run = sub.add_parser("run", help="跑一个 workflow")
    run.add_argument("workflow", help="workflow 名（见 list）")
    run.add_argument("--reflector", choices=["on", "off"], default=None,
                     help="是否启用反思器；默认按 preferences.json")
    run.add_argument("--no-experience", action="store_true",
                     help="不把 run_log 归档到 memory/experiences/")
    run.add_argument("rest", nargs=argparse.REMAINDER,
                     help="initial_state，--key value 形式")

    prefs = sub.add_parser("prefs", help="管理偏好")
    prefs.add_argument("action", choices=["show", "set"], nargs="?", default="show")
    prefs.add_argument("key", nargs="?", default="")
    prefs.add_argument("value", nargs="?")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "prefs":
        return cmd_prefs(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
