"""Render a markdown pivot table from chunk_model_sweep results.jsonl.

Usage:
  python scripts/experiments/summarize_chunk_model_sweep.py --tag <run_tag>
  python scripts/experiments/summarize_chunk_model_sweep.py --tag latest  # 最新 tag
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

BASE = REPO_ROOT / "data" / "experiments" / "chunk_model_sweep"


def _resolve_tag(tag: str) -> Path:
    if tag == "latest":
        candidates = sorted(BASE.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = [c for c in candidates if c.is_dir() and (c / "results.jsonl").exists()]
        if not candidates:
            raise FileNotFoundError(f"no runs found under {BASE}")
        return candidates[0]
    return BASE / tag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="run tag, or 'latest'")
    args = parser.parse_args(argv)

    run_dir = _resolve_tag(args.tag)
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        print(f"[error] no results.jsonl at {results_path}")
        return 1

    rows = [json.loads(l) for l in open(results_path, encoding="utf-8") if l.strip()]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append(f"# Chunk × Model Sweep — {run_dir.name}")
    lines.append("")
    lines.append(f"- started: {config.get('started_at')}")
    lines.append(f"- total cells: {len(rows)}")
    lines.append(f"- ok: {sum(1 for r in rows if r['ok'])}")
    lines.append(f"- failed: {sum(1 for r in rows if not r['ok'])}")
    lines.append("")

    # ----- Phase 1: chunk sweep -----
    p1 = [r for r in rows if r.get("phase") == "1_chunk_sweep"]
    if p1:
        lines.append("## Phase 1 — chunk size sensitivity (model fixed = MiniMax-M2)")
        lines.append("")
        slugs = sorted({r["slug"] for r in p1})
        for slug in slugs:
            sub = sorted([r for r in p1 if r["slug"] == slug], key=lambda r: r["chunk_size"])
            text_chars = sub[0]["text_chars"] if sub else 0
            lines.append(f"### slug = `{slug}` (text_chars={text_chars})")
            lines.append("")
            lines.append("| chunk_size | n_chunks | ok | nodes | edges | confidence | warnings | latency_s | lat/chunk | tokens_out |")
            lines.append("|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in sub:
                lines.append(
                    f"| {r['chunk_size']} | {r['n_chunks']} | "
                    f"{'✓' if r['ok'] else '✗'} | "
                    f"{r['nodes']} | {r['edges']} | "
                    f"{r['confidence']:.2f} | {r['warnings']} | "
                    f"{r['latency_s']:.1f} | {r['latency_per_chunk_s']:.1f} | "
                    f"{r['tokens_out_est']} |"
                )
                if r.get("error"):
                    lines.append(f"|  | | | | | | | | | _err: {r['error'][:80]}_ |")
            lines.append("")

    # ----- Phase 2: model compare -----
    p2 = [r for r in rows if r.get("phase") == "2_model_compare"]
    if p2:
        lines.append("## Phase 2 — model comparison (best chunk size from phase 1)")
        lines.append("")
        chunk_size = p2[0]["chunk_size"] if p2 else "?"
        lines.append(f"chunk_size = **{chunk_size}**")
        lines.append("")
        slugs = sorted({r["slug"] for r in p2})
        for slug in slugs:
            sub = [r for r in p2 if r["slug"] == slug]
            sub.sort(key=lambda r: r["model"])
            lines.append(f"### slug = `{slug}`")
            lines.append("")
            lines.append("| model | n_chunks | ok | nodes | edges | confidence | warnings | latency_s | lat/chunk | tokens_out |")
            lines.append("|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in sub:
                lines.append(
                    f"| {r['model']} | {r['n_chunks']} | "
                    f"{'✓' if r['ok'] else '✗'} | "
                    f"{r['nodes']} | {r['edges']} | "
                    f"{r['confidence']:.2f} | {r['warnings']} | "
                    f"{r['latency_s']:.1f} | {r['latency_per_chunk_s']:.1f} | "
                    f"{r['tokens_out_est']} |"
                )
                if r.get("error"):
                    lines.append(f"|  | | | | | | | | | _err: {r['error'][:80]}_ |")
            lines.append("")

    out_path = run_dir / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {out_path.relative_to(REPO_ROOT)}")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
