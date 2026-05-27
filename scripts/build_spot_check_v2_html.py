"""Build v2 spot-check HTML: 3 M2 models head-to-head + Text-01 dropout evidence section."""
from __future__ import annotations

import html
import json
from pathlib import Path

V2_PATH = Path("data/spot_check_v2/spot_check_v2_results.json")
V1_PATH = Path("data/spot_check/spot_check_results.json")
OUT_PATH = Path("data/spot_check_v2/spot_check_v2.html")

ACTIVE_MODELS = ["MiniMax-M2", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"]

NODE_COLORS = {
    "paper": "#fde68a", "author": "#fbcfe8", "concept": "#bae6fd",
    "method": "#bbf7d0", "dataset": "#fed7aa", "metric": "#ddd6fe",
    "tool": "#fecaca",
}


def _esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def render_nodes(nodes: list) -> str:
    if not nodes:
        return "<div class='empty'>（无节点）</div>"
    by_type: dict[str, list] = {}
    for n in nodes:
        by_type.setdefault(n.get("node_type", "?"), []).append(n)
    rows = []
    for t, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        color = NODE_COLORS.get(t, "#e5e7eb")
        chips = "".join(
            f"<span class='chip' style='background:{color}' title='{_esc(n.get('node_type','?'))}'>{_esc(n.get('label','?'))}</span>"
            for n in items
        )
        rows.append(
            f"<div class='type-row'><span class='type-name'>{_esc(t)} ({len(items)})</span>"
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


def render_badges(d: dict) -> str:
    badges = []
    badges.append("<span class='badge ok'>HTTP 200</span>" if d.get("http_ok") else "<span class='badge fail'>HTTP FAIL</span>")
    if d.get("has_open_think"):
        if d.get("truncated_thinking"):
            badges.append("<span class='badge fail'>think truncated</span>")
        elif d.get("has_close_think"):
            badges.append("<span class='badge warn'>think stripped</span>")
    else:
        badges.append("<span class='badge neutral'>no think</span>")
    if d.get("has_markdown_fence"):
        badges.append("<span class='badge warn'>fence stripped</span>")
    rt = d.get("root_type") or "—"
    rt_class = {"object": "ok", "array": "fail", "other": "fail"}.get(rt, "neutral")
    badges.append(f"<span class='badge {rt_class}'>root={rt}</span>")
    pp = d.get("parse_path", "failed")
    pp_class = {"pydantic_strict": "ok", "json_loads_fallback": "warn", "failed": "fail"}.get(pp, "fail")
    badges.append(f"<span class='badge {pp_class}'>{pp}</span>")
    out_t = d.get("out_tokens", 0)
    if out_t >= 16000:
        badges.append("<span class='badge fail'>HIT max_tokens</span>")
    return "<div class='badges'>" + "".join(badges) + "</div>"


def render_cell(d: dict) -> str:
    if not d.get("http_ok"):
        return (
            f"<div class='cell fail-cell'>"
            f"{render_badges(d)}"
            f"<div class='err'>{_esc(d.get('error',''))[:500]}</div>"
            f"</div>"
        )
    parse_warn = ""
    if d.get("parse_error"):
        parse_warn = f"<div class='parse-warn'>⚠ {_esc(d['parse_error'][:300])}</div>"
    return (
        f"<div class='cell'>"
        f"{render_badges(d)}"
        f"<div class='meta'>n={len(d.get('nodes',[]))} · e={len(d.get('edges',[]))} · "
        f"in={d.get('in_tokens',0)}/out={d.get('out_tokens',0)}tok · {d.get('latency_s','?')}s</div>"
        f"{parse_warn}"
        f"<details open><summary>Nodes ({len(d.get('nodes',[]))})</summary>{render_nodes(d.get('nodes',[]))}</details>"
        f"<details><summary>Edges ({len(d.get('edges',[]))})</summary>{render_edges(d.get('edges',[]))}</details>"
        f"<details><summary>Raw (first 600 chars)</summary><pre>{_esc(d.get('raw_first_600',''))}</pre></details>"
        f"<details><summary>Raw (last 300 chars)</summary><pre>{_esc(d.get('raw_last_300',''))}</pre></details>"
        f"</div>"
    )


def main() -> int:
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8")) if V1_PATH.exists() else None

    papers = v2["papers"]
    diags = v2["diags"]
    key = {(d["paper"], d["model"]): d for d in diags}

    # ===== v1 → v2 改善对照表（仅 3 个活跃模型）=====
    cmp_rows = []
    for m in ACTIVE_MODELS:
        v2_rs = [d for d in diags if d["model"] == m]
        v2_strict = sum(1 for r in v2_rs if r["parse_path"] == "pydantic_strict")
        v2_nodes = sum(len(r.get("nodes", [])) for r in v2_rs)
        v2_edges = sum(len(r.get("edges", [])) for r in v2_rs)
        v2_lat = sum(r["latency_s"] for r in v2_rs) / max(1, len(v2_rs))
        v1_strict = 0
        v1_nodes = v1_edges = 0
        if v1:
            v1_rs = [r for r in v1["results"] if r["model"] == m]
            v1_strict = sum(1 for r in v1_rs if r["ok"] and not r.get("parse_error") and len(r.get("nodes", [])) > 0)
            v1_nodes = sum(len(r.get("nodes", [])) for r in v1_rs if r["ok"])
            v1_edges = sum(len(r.get("edges", [])) for r in v1_rs if r["ok"])
        cmp_rows.append(
            f"<tr><td><code>{_esc(m)}</code></td>"
            f"<td>{v1_strict}/5 → <b>{v2_strict}/5</b></td>"
            f"<td>{v1_nodes} → <b>{v2_nodes}</b></td>"
            f"<td>{v1_edges} → <b>{v2_edges}</b></td>"
            f"<td>{v2_lat:.1f}s</td></tr>"
        )

    fixes_doc = v2.get("fixes", {})
    fixes_html = "".join(f"<li><code>{_esc(k)}</code>: {_esc(v)}</li>" for k, v in fixes_doc.items())

    # ===== 主对比网格 =====
    paper_blocks = []
    for p in papers:
        header = "".join(f"<th>{_esc(m)}</th>" for m in ACTIVE_MODELS)
        cells = "".join(
            f"<td>{render_cell(key.get((p['title'], m), {'http_ok': False, 'error': 'no result'}))}</td>"
            for m in ACTIVE_MODELS
        )
        paper_blocks.append(
            f"<section class='paper'>"
            f"<h2>{_esc(p['title'])}</h2>"
            f"<div class='subtitle'><code>{_esc(p['pdf'])}</code></div>"
            f"<table class='grid'><thead><tr>{header}</tr></thead><tbody><tr>{cells}</tr></tbody></table>"
            f"</section>"
        )

    # ===== Text-01 淘汰证据 =====
    text01_v2 = [d for d in diags if d["model"] == "MiniMax-Text-01"]
    text01_first_err = next((d["error"] for d in text01_v2 if d.get("error")), "")
    # Partial Text-01 rerun data (4 papers, hardcoded from text01_rerun.log)
    text01_rerun_rows = [
        ("A_Cluster-Based_Statistical_Channel_Model_for_Integrated_Sensing", "HTTP 200, parsed object, 0 nodes / 0 edges", 134, 6.4),
        ("Angle-Based_SLAM_on_5G_mmWave_Systems", "HTTP 200, parsed object, 0 nodes / 0 edges", 4070, 104.4),
        ("A_Multipath-Aided_Localization_Method_for_MIMO-OFDM", "HTTP 200, parsed object, 0 nodes / 0 edges", 9705, 270.4),
        ("5G_mm_Wave_Downlink_Vehicular_Positioning", "❌ Hit max_tokens=16000, parse failed", 16000, 462.4),
        ("An_Asymptotically_Efficient_Estimator_AOA", "Task killed before completion", "—", "—"),
    ]
    text01_rerun_html = "<table class='summary'><thead><tr><th>论文</th><th>结果</th><th>out tokens</th><th>延时 (s)</th></tr></thead><tbody>"
    for paper, result, tok, lat in text01_rerun_rows:
        text01_rerun_html += f"<tr><td>{_esc(paper)}</td><td>{_esc(result)}</td><td>{_esc(tok)}</td><td>{_esc(lat)}</td></tr>"
    text01_rerun_html += "</tbody></table>"

    css = """
    body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1900px; margin: 20px auto; padding: 0 20px; color: #1f2937; }
    h1 { color: #111827; }
    h2 { margin-top: 36px; font-size: 18px; border-bottom: 2px solid #6366f1; padding-bottom: 4px; }
    h3 { margin-top: 24px; color: #374151; }
    .subtitle { color: #6b7280; font-size: 12px; margin-bottom: 8px; }
    table.summary, table.cmp { border-collapse: collapse; margin: 12px 0 24px; font-size: 13px; }
    table.summary th, table.summary td, table.cmp th, table.cmp td { border: 1px solid #d1d5db; padding: 6px 12px; text-align: left; }
    table.summary th, table.cmp th { background: #f3f4f6; }
    table.cmp td b { color: #059669; font-weight: 700; }
    table.grid { border-collapse: collapse; width: 100%; table-layout: fixed; }
    table.grid th { background: #1f2937; color: #fff; padding: 8px; font-size: 12px; border: 1px solid #4b5563; }
    table.grid td { border: 1px solid #d1d5db; vertical-align: top; padding: 0; width: 33.3%; }
    .cell { padding: 10px; min-height: 200px; font-size: 12px; }
    .cell.fail-cell { background: #fee2e2; }
    .badges { margin-bottom: 8px; }
    .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-right: 3px; font-family: monospace; }
    .badge.ok { background: #d1fae5; color: #065f46; }
    .badge.warn { background: #fef3c7; color: #92400e; }
    .badge.fail { background: #fee2e2; color: #991b1b; }
    .badge.neutral { background: #e5e7eb; color: #374151; }
    .meta { color: #6b7280; font-size: 11px; font-family: monospace; margin-bottom: 6px; }
    .parse-warn { background: #fef3c7; color: #92400e; padding: 4px 6px; border-radius: 3px; font-size: 11px; margin: 4px 0; }
    .err { color: #991b1b; font-family: monospace; font-size: 11px; word-break: break-word; }
    details { margin-top: 6px; }
    summary { cursor: pointer; font-weight: bold; color: #374151; padding: 2px 0; font-size: 12px; }
    pre { background: #f9fafb; padding: 6px; border-radius: 3px; font-size: 10px; max-height: 220px; overflow: auto; white-space: pre-wrap; word-break: break-word; }
    .nodes .type-row { margin: 4px 0; }
    .type-name { display: inline-block; font-size: 10px; color: #4b5563; font-family: monospace; margin-right: 6px; }
    .chip { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin: 1px 2px; }
    .edges { padding-left: 18px; font-size: 11px; line-height: 1.5; margin: 4px 0; }
    .edges .rel { color: #6366f1; font-family: monospace; font-size: 10px; }
    .empty { color: #9ca3af; font-style: italic; font-size: 11px; }
    .fixes-doc ul { padding-left: 18px; }
    .fixes-doc li { margin: 4px 0; font-size: 13px; }
    .dropout { background: #fee2e2; padding: 16px; border-radius: 6px; border-left: 4px solid #dc2626; margin: 24px 0; }
    .dropout h3 { margin-top: 0; color: #991b1b; }
    .dropout pre { background: #fef2f2; }
    """

    html_doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Spot-check v2: 3 模型工程修复后对比</title>
<style>{css}</style></head><body>
<h1>📊 MiniMax KG 抽取 spot-check v2（工程修复后，3 模型对比）</h1>
<p>v1 的失败 80%+ 是工程问题（schema 描述超限、schema 类型限制、prompt 不够明确、token 截断）。v2 应用了工程修复后，<b>MiniMax-M2、M2.7、M2.7-highspeed 三个模型都能稳定产出 KG</b>。<b>MiniMax-Text-01 在排除工程问题后仍然无法产出有效 KG，已淘汰</b>（见底部证据章节）。</p>

<h3>🔧 工程修复清单</h3>
<div class='fixes-doc'><ul>{fixes_html}</ul></div>

<h3>📈 v1 → v2 改善对照（3 个活跃模型）</h3>
<table class='cmp'>
<thead><tr><th>模型</th><th>有效输出 (v1 → v2)</th><th>总节点 (v1 → v2)</th><th>总边 (v1 → v2)</th><th>v2 平均延时</th></tr></thead>
<tbody>{''.join(cmp_rows)}</tbody>
</table>
<p style="font-size:12px; color:#6b7280;">「有效输出」= 模型给出了非空、严格符合 schema 的 KG。</p>

<h3>📋 5 篇论文 × 3 模型 详细对比</h3>
{''.join(paper_blocks)}

<div class='dropout'>
<h3>❌ MiniMax-Text-01 淘汰证据</h3>
<p><b>第一轮 v2（应用 F1a + F1b）的 400 错误</b>：</p>
<pre>{_esc(text01_first_err[:600])}</pre>
<p>修复 F1c（<code>type:integer</code> → <code>type:number</code>）后单独重跑 Text-01：</p>
{text01_rerun_html}
<p><b>结论</b>：</p>
<ol>
<li>4 个独立的 MiniMax-Text-01 schema 限制（描述长度 / 可空类型 / integer 类型 / 推测 schema-name 包裹）依次踩坑</li>
<li>即使修了前 3 个，仍出现 0 节点 / 0 边的可疑结果（Paper 3 烧了 9705 个 output token 仍然空手）— 怀疑 schema name 包裹（<code>{{"kg_extraction": {{"nodes": [...], "edges": [...]}}}}</code>），但已无修复价值</li>
<li>速度灾难：单篇 6 – 462 秒，105 篇预计 6 – 13 小时</li>
<li>对比同样 prompt + 同样 schema，<code>MiniMax-M2</code> 单篇 10-40 秒、产出 21-28 节点，<b>差距巨大</b></li>
</ol>
</div>

<p style="color:#9ca3af; font-size:11px; margin-top:40px;">
Generated by scripts/build_spot_check_v2_html.py · raw: data/spot_check_v2/spot_check_v2_results.json
</p></body></html>"""

    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"HTML saved to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
