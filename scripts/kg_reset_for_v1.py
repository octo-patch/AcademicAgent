"""
KG 重置（gold dataset v1.0 起点）
==================================

一次性脚本：
  1. 备份 data/kg_{llm_agent,comms}.json 到 data/_archive/ + 日期戳
  2. 写空 KG（保留 metadata.domain，nodes/edges 清零）
  3. 删除 data/semantic_index_{llm_agent,comms}.npz（避免与空 KG 不一致）

幂等：备份文件已存在则跳过，不会覆盖更早的备份。

运行：
    python scripts/kg_reset_for_v1.py [--dry-run] [--date YYYY-MM-DD]

执行后所有 v1.0 入库走 paper_analysis_flow，30 篇论文重抽得到唯一存在的双 KG。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_DIR = DATA_DIR / "_archive"

DOMAINS = ["llm_agent", "comms"]


def _backup_one(src: Path, archive_path: Path, dry_run: bool) -> str:
    if not src.exists():
        return f"  - {src.name}: 源文件不存在，跳过"
    if archive_path.exists():
        return f"  - {src.name}: 备份已存在 ({archive_path.name})，跳过覆盖"
    if dry_run:
        return f"  - {src.name} -> {archive_path.relative_to(PROJECT_ROOT)} (dry-run)"
    shutil.copy2(src, archive_path)
    return f"  - {src.name} -> {archive_path.relative_to(PROJECT_ROOT)}"


def _write_empty_kg(path: Path, domain: str, dry_run: bool) -> str:
    payload = {
        "metadata": {
            "node_count": 0,
            "edge_count": 0,
            "domain": domain,
            "version": "1.2",
            "source": "scripts/kg_reset_for_v1.py",
        },
        "nodes": [],
        "edges": [],
    }
    if dry_run:
        return f"  - {path.name}: 将写空 KG (domain={domain}) (dry-run)"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"  - {path.name}: 已写空 KG (domain={domain})"


def _clear_semantic_cache(domain: str, dry_run: bool) -> str:
    cache = DATA_DIR / f"semantic_index_{domain}.npz"
    if not cache.exists():
        return f"  - {cache.name}: 不存在，跳过"
    if dry_run:
        return f"  - {cache.name}: 将删除 (dry-run)"
    cache.unlink()
    return f"  - {cache.name}: 已删除"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只打印将做的事，不写盘")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="备份文件名的日期戳，默认今日 (e.g. 2026-05-21)",
    )
    args = parser.parse_args()

    print(f"KG reset for v1.0 ({'DRY RUN' if args.dry_run else 'EXECUTE'})")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"date stamp   = {args.date}")
    print()

    if not args.dry_run:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] 备份双 KG")
    archive_map = {
        "llm_agent": ARCHIVE_DIR / f"kg_llm_agent_polluted_{args.date}.json",
        "comms": ARCHIVE_DIR / f"kg_comms_old_{args.date}.json",
    }
    for d in DOMAINS:
        src = DATA_DIR / f"kg_{d}.json"
        print(_backup_one(src, archive_map[d], args.dry_run))

    print()
    print("[2/3] 写空 KG")
    for d in DOMAINS:
        print(_write_empty_kg(DATA_DIR / f"kg_{d}.json", domain=d, dry_run=args.dry_run))

    print()
    print("[3/3] 清空 semantic_index 缓存")
    for d in DOMAINS:
        print(_clear_semantic_cache(d, args.dry_run))

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
