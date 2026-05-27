"""
src.recorder CLI —— 不进 Claude Code 也能记录的命令行入口

用法：
  # 记录一次代码改动
  python -m src.recorder log \
      --area kg-quality \
      --title "atomic save 改造" \
      --problem "graph_store.save() 非原子" \
      --approach "改 tempfile + os.replace" \
      --file src/knowledge/graph_store.py:316 \
      --effect "崩溃后 JSON 不丢" \
      --risk "#9 工程隐患"

  # 记录一次验证
  python -m src.recorder validate \
      --area kg-quality \
      --step atomic-save-smoke \
      --command "pytest tests/test_knowledge_graph.py" \
      --output-file /tmp/pytest.log \
      --status ok

  # 关闭 area
  python -m src.recorder close --area kg-quality --summary "全部通过"

  # 检查 Notion 是否就绪、父页面是否能解析
  python -m src.recorder status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .notion_client import load_notion_config, NotionClient
from .refactor_recorder import RefactorRecorder


def _read_output(args: argparse.Namespace) -> str:
    if args.output_file:
        return Path(args.output_file).read_text(encoding="utf-8")
    return args.output or ""


# log：记录一次代码改动
def cmd_log(args: argparse.Namespace) -> int:
    recorder = RefactorRecorder(
        area=args.area,
        title=args.title_area,
        icon_emoji=args.icon,
    )
    recorder.log_step(
        title=args.title,
        problem=args.problem,
        approach=args.approach,
        files_changed=args.file or [],
        effect=args.effect,
        risks_resolved=args.risk or [],
        extra_notes=args.notes or "",
    )
    print(f"OK logged step in area={args.area}")
    print(f"  md     : {recorder._md_path}")
    if recorder._state.notion_url:
        print(f"  notion : {recorder._state.notion_url}")
    return 0

# validate：记录一次验证
def cmd_validate(args: argparse.Namespace) -> int:
    recorder = RefactorRecorder(
        area=args.area,
        title=args.title_area,
        icon_emoji=args.icon,
    )
    output = _read_output(args)
    recorder.log_validation(
        step=args.step,
        command=args.command or "",
        output=output,
        status=args.status,
        notes=args.notes or "",
        artifacts=args.artifact or [],
    )
    print(f"OK logged validation step={args.step!r} status={args.status} in area={args.area}")
    if recorder._state.notion_url:
        print(f"  notion : {recorder._state.notion_url}")
    return 0


# close：关闭一个 area
def cmd_close(args: argparse.Namespace) -> int:
    recorder = RefactorRecorder(
        area=args.area,
        title=args.title_area,
        icon_emoji=args.icon,
    )
    recorder.close(status=args.status, summary=args.summary or "")
    print(f"OK closed area={args.area} status={args.status}")
    return 0


# status：检查 Notion 配置
def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_notion_config()
    print("Notion config:")
    print(f"  api_key set      : {'yes' if cfg.api_key else 'NO'}")
    print(f"  parent_page_id   : {cfg.parent_page_id or '(none)'}")
    print(f"  parent_page_name : {cfg.parent_page_name or '(none)'}")
    print(f"  notion_version   : {cfg.notion_version}")
    print(f"  enabled          : {cfg.enabled}")

    if not cfg.is_ready:
        print("\n→ Notion 未就绪：本地 md 仍会写入，Notion 同步跳过。")
        return 0

    client = NotionClient(cfg)
    parent = client.resolve_parent()
    if parent:
        print(f"\n→ parent resolved: {parent}")
        return 0
    print("\n→ FAIL parent 未解析；请检查 token 权限或 page name 是否正确。")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.recorder",
        description="ScholarMind RefactorRecorder CLI",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--area", required=True, help="area slug, e.g. kg-quality")
        sp.add_argument("--title-area", dest="title_area",
                        help="area 页标题（首次创建生效），默认用 area")
        sp.add_argument("--icon", help="area 图标 emoji（首次创建生效），默认 🔧")

    # log
    sp = sub.add_parser("log", help="记录一次代码改动")
    add_common(sp)
    sp.add_argument("--title", required=True)
    sp.add_argument("--problem", default="")
    sp.add_argument("--approach", default="")
    sp.add_argument("--file", action="append", help="改动文件（可重复）")
    sp.add_argument("--effect", default="")
    sp.add_argument("--risk", action="append", help="解决的隐患（可重复）")
    sp.add_argument("--notes", default="")
    sp.set_defaults(func=cmd_log)

    # validate
    sp = sub.add_parser("validate", help="记录一次验证 run")
    add_common(sp)
    sp.add_argument("--step", required=True)
    sp.add_argument("--command", default="")
    sp.add_argument("--output", default="", help="直接给字符串输出")
    sp.add_argument("--output-file", default="", help="从文件读输出")
    sp.add_argument("--status", default="ok", choices=["ok", "warn", "fail"])
    sp.add_argument("--notes", default="")
    sp.add_argument("--artifact", action="append", help="产物路径（可重复）")
    sp.set_defaults(func=cmd_validate)

    # close
    sp = sub.add_parser("close", help="关闭 area")
    add_common(sp)
    sp.add_argument("--status", default="completed",
                    choices=["completed", "abandoned", "paused"])
    sp.add_argument("--summary", default="")
    sp.set_defaults(func=cmd_close)

    # status
    sp = sub.add_parser("status", help="检查 Notion 配置 + 父页面解析")
    sp.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
