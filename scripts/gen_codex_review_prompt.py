"""Generate a self-contained gold-review prompt that can be pasted into Codex / Claude / GPT.

Reads:
  - docs/gold_review_prompt.md (template)
  - datasets/personal_v1/annotation_guide.md
  - datasets/personal_v1/gold/<slug>.json
  - datasets/personal_v1/fixtures/papers_text/<slug>.txt (PDF text used by extractor)
  - (optional) data/evaluation/runs/<run_id>/sidecar/kg_extract_<slug>/data/knowledge_graph.json

Writes:
  - datasets/personal_v1/gold_reviews/_pending/<slug>.prompt.md (ready to paste)

Usage:
  python scripts/gen_codex_review_prompt.py --slug react
  python scripts/gen_codex_review_prompt.py --slug react --kg-from-run personal_v1_1_full_2026_05_22
  python scripts/gen_codex_review_prompt.py --all          # 一次性生成 30 个 prompt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DS = REPO_ROOT / "datasets" / "personal_v1"
PROMPT_TPL = REPO_ROOT / "docs" / "gold_review_prompt.md"
ANNOT_GUIDE = DS / "annotation_guide.md"
GOLD_DIR = DS / "gold"
TXT_DIR = DS / "fixtures" / "papers_text"
OUT_DIR = DS / "gold_reviews" / "_pending"


def _extract_template_block(text: str) -> str:
    """从 docs/gold_review_prompt.md 抽出 ```...``` 围栏内的完整 prompt 模板。"""
    start = text.find("```\n你是 ScholarMind")
    if start < 0:
        raise RuntimeError("template block not found in gold_review_prompt.md")
    start += 4  # 跳过 ```\n
    end = text.find("\n```", start)
    if end < 0:
        raise RuntimeError("template close fence not found")
    return text[start:end]


def _kg_extraction_for(slug: str, run_id: str | None) -> str:
    if not run_id:
        return "(未提供：跳过资料 D)"
    kg_path = REPO_ROOT / "data" / "evaluation" / "runs" / run_id / "sidecar" / f"kg_extract_{slug}" / "data" / "knowledge_graph.json"
    if not kg_path.exists():
        return f"(KG not found at {kg_path.relative_to(REPO_ROOT)})"
    kg = json.loads(kg_path.read_text(encoding="utf-8"))
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    lines = [f"该 run 抽出 {len(nodes)} 节点 / {len(edges)} 边", "", "Nodes:"]
    for n in nodes[:40]:
        lines.append(f"  - {n.get('label', '?')} ({n.get('node_type', '?')})")
    if len(nodes) > 40:
        lines.append(f"  ... +{len(nodes) - 40} more")
    lines.append("")
    lines.append("Edges:")
    for e in edges[:30]:
        s = e.get("source_label") or e.get("source_id", "?")
        d = e.get("target_label") or e.get("target_id", "?")
        r = e.get("relation_type", "?")
        lines.append(f"  - {s} --[{r}]--> {d}")
    if len(edges) > 30:
        lines.append(f"  ... +{len(edges) - 30} more")
    return "\n".join(lines)


def build_prompt(slug: str, run_id: str | None = None) -> str:
    tpl_full = PROMPT_TPL.read_text(encoding="utf-8")
    body = _extract_template_block(tpl_full)

    gold_path = GOLD_DIR / f"{slug}.json"
    if not gold_path.exists():
        raise FileNotFoundError(f"gold not found: {gold_path}")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    txt_path = TXT_DIR / f"{slug}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"paper text fixture not found: {txt_path}")
    paper_text = txt_path.read_text(encoding="utf-8")

    annot = ANNOT_GUIDE.read_text(encoding="utf-8")

    filled = (
        body
        .replace("<<<INSERT_ANNOTATION_GUIDE_HERE>>>", annot)
        .replace("<<<SLUG>>>", slug)
        .replace("<<<TITLE>>>", gold.get("paper_title", ""))
        .replace("<<<INSERT_GOLD_JSON_HERE>>>", json.dumps(gold, ensure_ascii=False, indent=2))
        .replace("<<<INSERT_PAPER_TEXT_HERE>>>", paper_text)
        .replace("<<<INSERT_CURRENT_KG_EXTRACTION_HERE>>>", _kg_extraction_for(slug, run_id))
    )
    return filled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", help="单篇 slug (gold/<slug>.json)")
    parser.add_argument("--all", action="store_true", help="对 30 篇全集生成")
    parser.add_argument("--kg-from-run", default=None,
                        help="可选：从指定 evaluation run 的 sidecar 里读 KG 抽取结果作为资料 D")
    args = parser.parse_args(argv)

    if not (args.slug or args.all):
        parser.error("--slug 或 --all 至少给一个")

    slugs: list[str]
    if args.all:
        slugs = sorted(p.stem for p in GOLD_DIR.glob("*.json") if p.name != "retrieval.jsonl")
    else:
        slugs = [args.slug]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for slug in slugs:
        try:
            prompt = build_prompt(slug, run_id=args.kg_from_run)
        except FileNotFoundError as exc:
            print(f"[skip] {slug}: {exc}")
            continue
        out_path = OUT_DIR / f"{slug}.prompt.md"
        out_path.write_text(prompt, encoding="utf-8")
        print(f"[ok] {slug:35s} {len(prompt):6d} chars -> {out_path.relative_to(REPO_ROOT)}")

    print(f"\n=== {len(slugs)} prompt(s) generated in {OUT_DIR.relative_to(REPO_ROOT)} ===")
    print("把单个 .prompt.md 整文件复制到 codex/claude/gpt 输入框，把返回的 JSON 存到")
    print("  datasets/personal_v1/gold_reviews/<slug>.<reviewer>.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
