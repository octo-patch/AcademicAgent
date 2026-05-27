"""Build a self-contained side-by-side HTML viewer from spot_check_results.json."""
from __future__ import annotations

import html
import json
from pathlib import Path

IN_PATH = Path("data/spot_check/spot_check_results.json")
OUT_PATH = Path("data/spot_check/spot_check.html")

NODE_TYPE_COLORS = {
    "paper": "#fde68a", "author": "#fbcfe8", "concept": "#bae6fd",
    "method": "#bbf7d0", "dataset": "#fed7aa", "metric": "#ddd6fe",
    "tool": "#fecaca",
}


def _esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def render_nodes(nodes: list) -> str:
    if not nodes:
        return "<div class='empty'>（无节点）</div>"
    rows = []
    by_type: dict[str, list] = {}
    for n in nodes:
        by_type.setdefault(n.get("node_type", "?"), []).append(n)
    for t, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        color = NODE_TYPE_COLORS.get(t, "#e5e7eb")
        chips = "".join(
            f"<span class='chip' style='background:{color}'>{_esc(n.get('label','?'))}</span>"
            for n in items
        )
        rows.append(
            f"<div class='type-row'>"
            f"<span class='type-name'>{_esc(t)} <small>({len(items)})</small></span>"
            f"<span class='chips'>{chips}</span></div>"
        )
    return "<div class='nodes'>" + "".join(rows) + "</div>"


def render_edges(edges: list) -> str:
    if not edges:
        return "<div class='empty'>（无边）</div>"
    items = "".join(
        f"<li><b>{_esc(e.get('source_label','?'))}</b> "
        f"<span class='rel'>―{_esc(e.get('relation_type','?'))}→</span> "
        f"<b>{_esc(e.get('target_label','?'))}</b></li>"
        for e in edges
    )
    return f"<ul class='edges'>{items}</ul>"


def render_model_cell(r: dict) -> str:
    if not r.get("ok"):
        return (
            f"<div class='cell fail'>"
            f"<div class='status'>❌ FAIL</div>"
            f"<div class='err'>{_esc(r.get('error',''))}</div></div>"
        )
    parse_warn = ""
    if r.get("parse_error"):
        parse_warn = f"<div class='parse-warn'>⚠ parse: {_esc(r['parse_error'][:200])}</div>"
    nodes_html = render_nodes(r.get("nodes", []))
    edges_html = render_edges(r.get("edges", []))
    return (
        f"<div class='cell'>"
        f"<div class='meta'>nodes={len(r.get('nodes',[]))} · edges={len(r.get('edges',[]))} · "
        f"in={r.get('in_tokens',0)}/out={r.get('out_tokens',0)}tok · {r.get('latency_s','?')}s · "
        f"server={_esc(r.get('server_model',''))}</div>"
        f"{parse_warn}"
        f"<details open><summary>Nodes</summary>{nodes_html}</details>"
        f"<details><summary>Edges</summary>{edges_html}</details>"
        f"</div>"
    )


def main() -> int:
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    models = data["models"]
    papers = data["papers"]
    by_key: dict[tuple[str, str], dict] = {(r["paper"], r["model"]): r for r in data["results"]}

    # summary table
    summary_rows = []
    for m in models:
        rs = [r for r in data["results"] if r["model"] == m]
        ok = [r for r in rs if r["ok"]]
        total_nodes = sum(len(r.get("nodes", [])) for r in ok)
        total_edges = sum(len(r.get("edges", [])) for r in ok)
        parse_err = sum(1 for r in ok if r.get("parse_error"))
        in_t = sum(r["in_tokens"] for r in rs)
        out_t = sum(r["out_tokens"] for r in rs)
        avg_lat = sum(r["latency_s"] for r in rs) / max(1, len(rs))
        summary_rows.append(
            f"<tr><td><code>{_esc(m)}</code></td><td>{len(ok)}/{len(papers)}</td>"
            f"<td>{total_nodes}</td><td>{total_edges}</td><td>{parse_err}</td>"
            f"<td>{in_t:,}</td><td>{out_t:,}</td><td>{avg_lat:.1f}s</td></tr>"
        )

    # main grid: one row per paper, columns = models
    paper_blocks = []
    for p in papers:
        cells_html = "".join(
            render_model_cell(by_key.get((p["title"], m), {"ok": False, "error": "no result"}))
            for m in models
        )
        col_headers = "".join(f"<th>{_esc(m)}</th>" for m in models)
        paper_blocks.append(
            f"<section class='paper'>"
            f"<h2>{_esc(p['title'])}</h2>"
            f"<div class='subtitle'><code>{_esc(p['pdf'])}</code></div>"
            f"<table class='grid'><thead><tr>{col_headers}</tr></thead>"
            f"<tbody><tr>{''.join(f'<td>{c}</td>' for c in [render_model_cell(by_key.get((p['title'], m), {'ok': False, 'error': 'no result'})) for m in models])}</tr></tbody>"
            f"</table>"
            f"</section>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>KG Extraction Spot Check</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1800px; margin: 20px auto; padding: 0 20px; color: #1f2937; }}
  h1 {{ color: #111827; }}
  h2 {{ margin-top: 36px; font-size: 18px; border-bottom: 2px solid #6366f1; padding-bottom: 4px; }}
  .subtitle {{ color: #6b7280; font-size: 12px; margin-bottom: 8px; }}
  table.summary {{ border-collapse: collapse; margin: 16px 0 24px; font-size: 13px; }}
  table.summary th, table.summary td {{ border: 1px solid #d1d5db; padding: 6px 10px; text-align: left; }}
  table.summary th {{ background: #f3f4f6; }}
  table.grid {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  table.grid th {{ background: #1f2937; color: #fff; padding: 8px; font-size: 13px; border: 1px solid #4b5563; }}
  table.grid td {{ border: 1px solid #d1d5db; vertical-align: top; padding: 0; width: 25%; }}
  .cell {{ padding: 10px; min-height: 200px; font-size: 12px; }}
  .cell.fail {{ background: #fee2e2; }}
  .meta {{ color: #6b7280; font-size: 11px; font-family: monospace; margin-bottom: 6px; }}
  .parse-warn {{ background: #fef3c7; color: #92400e; padding: 4px 6px; border-radius: 3px; font-size: 11px; margin: 4px 0; }}
  .err {{ color: #991b1b; font-family: monospace; font-size: 11px; word-break: break-all; }}
  .status {{ font-weight: bold; margin-bottom: 6px; }}
  details {{ margin-top: 8px; }}
  summary {{ cursor: pointer; font-weight: bold; color: #374151; padding: 2px 0; }}
  .nodes .type-row {{ margin: 4px 0; }}
  .type-name {{ display: inline-block; font-size: 11px; color: #4b5563; font-family: monospace; margin-right: 6px; }}
  .chips {{}}
  .chip {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin: 1px 2px; }}
  .edges {{ padding-left: 18px; font-size: 11px; line-height: 1.5; margin: 4px 0; }}
  .edges .rel {{ color: #6366f1; font-family: monospace; font-size: 10px; }}
  .empty {{ color: #9ca3af; font-style: italic; font-size: 11px; }}
</style>
</head><body>
<h1>📊 MiniMax 模型 KG 抽取对比 (5 篇通信论文 × 4 模型)</h1>
<p>用相同 prompt + 相同 schema (PAPER/AUTHOR/CONCEPT/METHOD/DATASET/METRIC/TOOL) 抽取，对比四个 MiniMax 模型的输出。Reasoning 模型的 <code>&lt;think&gt;</code> 块已被剥离。</p>
<table class="summary">
<thead><tr><th>模型</th><th>成功率</th><th>总节点</th><th>总边</th><th>解析错误</th><th>in tokens</th><th>out tokens</th><th>avg 延时</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody>
</table>
{''.join(paper_blocks)}
<p style="color:#9ca3af; font-size:11px; margin-top:40px;">Generated by scripts/build_spot_check_html.py · raw data: data/spot_check/spot_check_results.json</p>
</body></html>"""

    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"HTML saved to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
