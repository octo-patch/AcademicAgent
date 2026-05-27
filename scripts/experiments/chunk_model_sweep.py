"""Chunk-size × base-model sweep on a few representative papers.

直接调用 KnowledgeExtractor.extract_from_text，绕过 evaluation runner，
专心比对：分块大小 + 基模 对抽取量 / 延迟 / 成本 / parse 成功率的影响。

输出：
  data/experiments/chunk_model_sweep/<run_tag>/
    config.json
    results.jsonl    （每 cell 一条）
    summary.md       （透视表）

Usage:
  python scripts/experiments/chunk_model_sweep.py --phase 1   # chunk sweep
  python scripts/experiments/chunk_model_sweep.py --phase 2 --chunk-size 8000  # model sweep
  python scripts/experiments/chunk_model_sweep.py --phase all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 强制 UTF-8 输出
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

PAPERS = {
    "rhs_slam": ("EE_papers/Reconfigurable Holographic Surface Aided Wireless Simultaneous Localization and Mapping.pdf",
                  "RHS-aided Wireless SLAM"),
    "memgpt": ("data/papers/MemGPT.pdf", "MemGPT: Towards LLMs as Operating Systems"),
    "react": ("data/papers/arxiv_2210_03629.pdf", "ReAct: Synergizing Reasoning and Acting"),
}

PHASE1_CHUNK_SIZES = [2000, 4000, 8000, 16000, 30000]
PHASE2_MODELS = ["MiniMax-M2", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"]

OUT_BASE = REPO_ROOT / "data" / "experiments" / "chunk_model_sweep"


def _pdf_text(pdf_path: str) -> str:
    """完整 PDF text。复用 src.core.pdf_parser。"""
    from src.core.pdf_parser import PDFParser
    abs_path = REPO_ROOT / pdf_path
    if not abs_path.exists():
        abs_path = Path(pdf_path)
    with PDFParser(str(abs_path)) as parser:
        return parser.extract_full_text() or ""


async def _run_one_cell(
    slug: str,
    pdf_path: str,
    title: str,
    chunk_size: int,
    model: str,
) -> dict:
    """单 cell：(slug, chunk_size, model) -> result dict."""
    from src.knowledge.extractor import KnowledgeExtractor, ExtractionError

    text = _pdf_text(pdf_path)
    text_len = len(text)

    extractor = KnowledgeExtractor(
        model=model,
        max_chunk_chars=chunk_size,
    )

    n_chunks = len(extractor._split_text(text))

    t0 = time.time()
    err = None
    err_cat = None
    nodes_n = edges_n = 0
    tokens_in_est = tokens_out_est = 0
    confidence = 0.0
    fallback_used = False
    n_warnings = 0

    try:
        result = await extractor.extract_from_text(
            text=text,
            paper_title=title,
            paper_year="2024",
        )
        nodes_n = result.node_count
        edges_n = result.edge_count
        confidence = result.extraction_confidence
        fallback_used = bool(getattr(result, "fallback_used", False))
        n_warnings = len(getattr(result, "parse_warnings", []) or [])
        # 粗估 tokens
        tokens_in_est = text_len // 4
        tokens_out_est = len(result.raw_llm_output or "") // 4
    except ExtractionError as e:
        err = str(e)[:300]
        err_cat = "extraction_error"
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:300]
        err_cat = type(e).__name__

    latency_s = time.time() - t0

    return {
        "slug": slug,
        "title": title,
        "text_chars": text_len,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "model": model,
        "ok": err is None and (nodes_n > 0 or edges_n > 0),
        "nodes": nodes_n,
        "edges": edges_n,
        "confidence": confidence,
        "fallback_used": fallback_used,
        "warnings": n_warnings,
        "tokens_in_est": tokens_in_est,
        "tokens_out_est": tokens_out_est,
        "latency_s": round(latency_s, 2),
        "latency_per_chunk_s": round(latency_s / max(n_chunks, 1), 2),
        "error": err,
        "error_category": err_cat,
    }


def _write_jsonl_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_phase1(out_dir: Path, slugs: list[str], chunk_sizes: list[int], model: str) -> None:
    results_file = out_dir / "results.jsonl"
    print(f"[phase1] chunk sensitivity sweep on {slugs} × {chunk_sizes} × {model}")
    print(f"         -> {results_file.relative_to(REPO_ROOT)}")
    for slug in slugs:
        pdf, title = PAPERS[slug]
        for cs in chunk_sizes:
            print(f"  [run]  slug={slug:18s} chunk={cs:5d} model={model}", flush=True)
            rec = await _run_one_cell(slug, pdf, title, cs, model)
            rec["phase"] = "1_chunk_sweep"
            _write_jsonl_append(results_file, rec)
            print(f"         -> nodes={rec['nodes']:3d} edges={rec['edges']:3d} "
                  f"chunks={rec['n_chunks']:2d} lat={rec['latency_s']:5.1f}s "
                  f"ok={rec['ok']} {rec.get('error_category') or ''}", flush=True)


async def run_phase2(out_dir: Path, slugs: list[str], chunk_size: int, models: list[str]) -> None:
    results_file = out_dir / "results.jsonl"
    print(f"[phase2] model comparison on {slugs} × chunk={chunk_size} × {models}")
    print(f"         -> {results_file.relative_to(REPO_ROOT)}")
    for slug in slugs:
        pdf, title = PAPERS[slug]
        for model in models:
            print(f"  [run]  slug={slug:18s} chunk={chunk_size:5d} model={model}", flush=True)
            rec = await _run_one_cell(slug, pdf, title, chunk_size, model)
            rec["phase"] = "2_model_compare"
            _write_jsonl_append(results_file, rec)
            print(f"         -> nodes={rec['nodes']:3d} edges={rec['edges']:3d} "
                  f"chunks={rec['n_chunks']:2d} lat={rec['latency_s']:6.1f}s "
                  f"ok={rec['ok']} {rec.get('error_category') or ''}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["1", "2", "all"])
    parser.add_argument("--chunk-size", type=int, default=8000,
                        help="phase 2 用的 chunk size（默认 8000）")
    parser.add_argument("--phase1-slugs", default="rhs_slam",
                        help="phase 1 用哪些论文（逗号分隔，默认 rhs_slam）")
    parser.add_argument("--phase2-slugs", default="rhs_slam,memgpt",
                        help="phase 2 用哪些论文（默认 rhs_slam,memgpt）")
    parser.add_argument("--tag", default=None, help="run tag, 默认 timestamp")
    args = parser.parse_args(argv)

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "tag": tag,
        "phase": args.phase,
        "chunk_size_phase2": args.chunk_size,
        "phase1_slugs": args.phase1_slugs.split(","),
        "phase1_chunk_sizes": PHASE1_CHUNK_SIZES,
        "phase1_model": "MiniMax-M2",
        "phase2_slugs": args.phase2_slugs.split(","),
        "phase2_models": PHASE2_MODELS,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[init] run tag={tag} -> {out_dir.relative_to(REPO_ROOT)}\n")

    async def _go():
        if args.phase in ("1", "all"):
            await run_phase1(
                out_dir,
                slugs=config["phase1_slugs"],
                chunk_sizes=PHASE1_CHUNK_SIZES,
                model="MiniMax-M2",
            )
        if args.phase in ("2", "all"):
            await run_phase2(
                out_dir,
                slugs=config["phase2_slugs"],
                chunk_size=args.chunk_size,
                models=PHASE2_MODELS,
            )

    asyncio.run(_go())

    print(f"\n[done] results: {(out_dir / 'results.jsonl').relative_to(REPO_ROOT)}")
    print(f"[tip]  跑 python scripts/experiments/summarize_chunk_model_sweep.py --tag {tag} 看透视表")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
