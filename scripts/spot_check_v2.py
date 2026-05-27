"""Spot-check v2: 4 engineering fixes applied, full diagnostics per call.

Fixes vs v1 (scripts/spot_check_kg.py):
  F1. SHORT_SCHEMA — inline JSON Schema with all descriptions <30 chars (Text-01 200-char limit)
  F2. STRUCTURED_PROMPT — explicit "MUST be {nodes:[], edges:[]} object, NOT flat array" + example
  F3. max_tokens=16000 — give reasoning models room for <think> + complete JSON
  F4. Truncated <think> detection — if <think> present but no </think>, mark as TRUNCATED, not parse error

Diagnostics recorded per call:
  - http_ok, error
  - has_open_think, has_close_think, truncated_thinking (F4 trigger)
  - has_markdown_fence
  - parse_path: pydantic_strict | json_loads_fallback | failed
  - root_type: object | array | other  (M2 array-bug detector)
  - fixes_applied: [list of fix labels that contributed]
  - in_tokens, out_tokens, latency_s, server_model
  - nodes_count, edges_count
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from src.core.pdf_parser import PDFParser

load_dotenv()
logging.basicConfig(level=logging.WARNING)

MODELS = [
    "MiniMax-Text-01",
    "MiniMax-M2",
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
]

PAPERS = [
    "A_Cluster-Based_Statistical_Channel_Model_for_Integrated_Sensing_and_Communication_Channels.pdf",
    "Angle-Based_SLAM_on_5G_mmWave_Systems_Design_Implementation_and_Measurement (1).pdf",
    "A_Multipath-Aided_Localization_Method_for_MIMO-OFDM_Systems_via_Tensor_Decomposition.pdf",
    "5G_mm_Wave_Downlink_Vehicular_Positioning.pdf",
    "An_Asymptotically_Efficient_Estimator_in_Closed-Form_for_3-D_AOA_Localization_Using_a_Sensor_Network.pdf",
]

EE_DIR = Path("EE_papers")
OUT_DIR = Path("data/spot_check_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHARS = 12000
MAX_TOKENS = 16000  # F3

NODE_ENUM = ["paper", "author", "concept", "method", "dataset", "metric", "tool"]
REL_ENUM = ["proposes", "uses", "improves", "extends", "cites"]

# F1: SHORT_SCHEMA — bypasses TWO MiniMax schema bugs:
#   (a) description fields >200 chars → kept all descriptions <30 chars
#   (b) nullable via `type: [X, "null"]` array syntax → use plain types, optional via NOT in `required`
SHORT_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "entity name (English)"},
                    "node_type": {"type": "string", "enum": NODE_ENUM, "description": "entity type"},
                    "first_seen_year": {"type": "number", "description": "year first appeared"},
                    "superseded_by": {"type": "string", "description": "newer replacement"},
                },
                "required": ["label", "node_type"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_label": {"type": "string", "description": "source name"},
                    "source_type": {"type": "string", "enum": NODE_ENUM, "description": "source type"},
                    "target_label": {"type": "string", "description": "target name"},
                    "target_type": {"type": "string", "enum": NODE_ENUM, "description": "target type"},
                    "relation_type": {"type": "string", "enum": REL_ENUM, "description": "relation"},
                    "established_year": {"type": "number", "description": "year established"},
                },
                "required": ["source_label", "source_type", "target_label", "target_type", "relation_type"],
            },
        },
    },
    "required": ["nodes", "edges"],
}

# F2: STRUCTURED_PROMPT — explicit structure example, anti-array directive
STRUCTURED_PROMPT = """你是通信/感知/定位领域的论文知识图谱抽取专家。

从下方论文文本中抽取实体（节点）和关系（边），按以下规范输出。

## 允许的节点类型 (node_type)
paper / author / concept / method / dataset / metric / tool

## 允许的关系类型 (relation_type)
proposes / uses / improves / extends / cites

## 规则
1. 所有实体标签使用标准英文术语（ISAC、SLAM、MUSIC、PMBM、OFDM、CRLB 等缩写优先）。
2. 只抽取上述 7 种 node_type 和 5 种 relation_type；其他类型一律不要。
3. 不要伪造文本未明确支持的实体或关系。

## 论文标题
{paper_title}

## 论文文本
{text}

## 输出结构（⚠️ 必须严格遵守）
返回**一个 JSON 对象**，恰好包含两个顶层字段：`nodes`（数组）+ `edges`（数组）。

正确格式范例：
```json
{{
  "nodes": [
    {{"label": "OFDM", "node_type": "method"}},
    {{"label": "ISAC", "node_type": "concept"}}
  ],
  "edges": [
    {{"source_label": "OFDM", "source_type": "method",
      "target_label": "ISAC", "target_type": "concept",
      "relation_type": "uses"}}
  ]
}}
```

❌ 错误：直接返回数组 `[{{...}}, {{...}}]`（必须是带 nodes/edges 字段的对象）
❌ 错误：把 nodes 和 edges 混在同一个数组里
❌ 错误：在 JSON 外加任何解释文字

只输出 JSON，不要任何其他内容。"""


@dataclass
class Diag:
    model: str
    paper: str
    http_ok: bool = False
    error: str = ""
    in_tokens: int = 0
    out_tokens: int = 0
    latency_s: float = 0.0
    server_model: str = ""
    raw: str = ""
    has_open_think: bool = False
    has_close_think: bool = False
    truncated_thinking: bool = False  # F4
    has_markdown_fence: bool = False
    root_type: str = ""  # object | array | other
    parse_path: str = "failed"  # pydantic_strict | json_loads_fallback | failed
    parse_error: str = ""
    fixes_applied: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)


def strip_think_and_fence(raw: str, diag: Diag) -> str:
    """Apply F4 + fence stripping. Returns cleaned content. Updates diag."""
    s = raw
    diag.has_open_think = "<think>" in s
    diag.has_close_think = "</think>" in s
    if diag.has_open_think and not diag.has_close_think:
        diag.truncated_thinking = True
        diag.fixes_applied.append("F4:truncated_think_detected")
        return ""  # don't try to parse truncated thinking
    if diag.has_close_think:
        s = re.split(r"</think>", s, maxsplit=1)[1]
        diag.fixes_applied.append("F-strip:think_block")
    s = s.strip()
    if s.startswith("```"):
        diag.has_markdown_fence = True
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        diag.fixes_applied.append("F-strip:markdown_fence")
    return s.strip()


def parse_and_classify(cleaned: str, diag: Diag) -> None:
    """Try strict object parse; fall back to soft JSON; record root_type."""
    if not cleaned:
        diag.parse_path = "failed"
        diag.parse_error = "empty after cleaning"
        return
    # Soft JSON first to detect root type
    try:
        obj = json.loads(cleaned)
    except Exception as e:
        diag.parse_path = "failed"
        diag.parse_error = f"json.loads: {str(e)[:200]}"
        diag.root_type = "other"
        return
    # Root type classification
    if isinstance(obj, dict):
        diag.root_type = "object"
        nodes = obj.get("nodes") or []
        edges = obj.get("edges") or []
        if isinstance(nodes, list) and isinstance(edges, list):
            diag.parse_path = "pydantic_strict"
            diag.nodes = nodes
            diag.edges = edges
        else:
            diag.parse_path = "failed"
            diag.parse_error = "object missing nodes/edges arrays"
    elif isinstance(obj, list):
        diag.root_type = "array"
        # F2 fix didn't take — model still returned flat array. Try to salvage by classifying items.
        diag.parse_path = "json_loads_fallback"
        salvaged_nodes, salvaged_edges = [], []
        for item in obj:
            if not isinstance(item, dict):
                continue
            if "relation_type" in item or "source_label" in item:
                salvaged_edges.append(item)
            elif "node_type" in item or "label" in item:
                salvaged_nodes.append(item)
        diag.nodes = salvaged_nodes
        diag.edges = salvaged_edges
        diag.parse_error = f"root was array (M2 structure bug), salvaged {len(salvaged_nodes)}+{len(salvaged_edges)}"
        diag.fixes_applied.append("F2:array_salvage_attempted")
    else:
        diag.root_type = "other"
        diag.parse_path = "failed"
        diag.parse_error = f"unexpected root type: {type(obj).__name__}"


def call_one(client: OpenAI, model: str, paper_title: str, text: str) -> Diag:
    diag = Diag(model=model, paper=paper_title)
    prompt = STRUCTURED_PROMPT.format(paper_title=paper_title, text=text)
    diag.fixes_applied.append("F1a:short_descriptions")  # always applied
    diag.fixes_applied.append("F1b:no_nullable_arrays")
    diag.fixes_applied.append("F1c:number_not_integer")
    diag.fixes_applied.append("F2:structured_prompt")
    diag.fixes_applied.append("F3:max_tokens_16k")

    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "kg_extraction", "schema": SHORT_SCHEMA},
            },
        )
    except Exception as e:
        diag.error = f"{type(e).__name__}: {str(e)[:400]}"
        diag.latency_s = round(time.time() - t0, 2)
        return diag

    diag.latency_s = round(time.time() - t0, 2)
    diag.http_ok = True
    diag.server_model = getattr(r, "model", "") or ""
    if r.usage:
        diag.in_tokens = r.usage.prompt_tokens
        diag.out_tokens = r.usage.completion_tokens
    diag.raw = r.choices[0].message.content or ""
    cleaned = strip_think_and_fence(diag.raw, diag)
    parse_and_classify(cleaned, diag)
    return diag


def diag_to_dict(d: Diag) -> dict:
    return {
        "model": d.model, "paper": d.paper,
        "http_ok": d.http_ok, "error": d.error,
        "in_tokens": d.in_tokens, "out_tokens": d.out_tokens,
        "latency_s": d.latency_s, "server_model": d.server_model,
        "has_open_think": d.has_open_think, "has_close_think": d.has_close_think,
        "truncated_thinking": d.truncated_thinking,
        "has_markdown_fence": d.has_markdown_fence,
        "root_type": d.root_type, "parse_path": d.parse_path,
        "parse_error": d.parse_error,
        "fixes_applied": d.fixes_applied,
        "nodes": d.nodes, "edges": d.edges,
        "raw_first_600": d.raw[:600],
        "raw_last_300": d.raw[-300:] if len(d.raw) > 300 else "",
    }


def main() -> int:
    api_key = os.getenv("MINIMAX_API_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    print("Extracting paper texts...")
    paper_texts: dict[str, tuple[str, str]] = {}
    for pdf in PAPERS:
        text = PDFParser(str(EE_DIR / pdf)).extract_full_text()
        title = pdf.replace(".pdf", "").replace("_", " ").replace(" (1)", "")
        paper_texts[pdf] = (title, text[:MAX_CHARS])
        print(f"  {pdf[:55]:<55} {len(text)} -> {min(len(text), MAX_CHARS)}")

    diags: list[Diag] = []
    total = len(PAPERS) * len(MODELS)
    n = 0
    print(f"\nRunning {total} extractions (v2 with 4 fixes)...")
    for pdf in PAPERS:
        title, text = paper_texts[pdf]
        for model in MODELS:
            n += 1
            print(f"  [{n:2d}/{total}] {model:<26} {pdf[:48]}...", end=" ", flush=True)
            d = call_one(client, model, title, text)
            diags.append(d)
            if d.http_ok:
                tag = "[OK]" if d.parse_path == "pydantic_strict" else ("[FB]" if d.parse_path == "json_loads_fallback" else "[XX]")
                print(f"{tag} n={len(d.nodes)} e={len(d.edges)} root={d.root_type} in/out={d.in_tokens}/{d.out_tokens} {d.latency_s}s parse={d.parse_path}")
            else:
                print(f"[XX] HTTP_FAIL: {d.error[:100]}")

    payload = {
        "version": "v2",
        "fixes": {
            "F1a": "Short JSON schema descriptions (<30 chars) — bypasses MiniMax 200-char description limit",
            "F1b": "Plain types instead of nullable type-arrays — bypasses MiniMax not supporting `type: [X, \"null\"]`",
            "F1c": "`type: number` instead of `type: integer` — Text-01 only accepts {string,number,boolean,object,array}, rejects `integer`",
            "F2": "Explicit 'must be {nodes,edges} object, NOT array' in prompt with example",
            "F3": f"max_tokens={MAX_TOKENS} for reasoning models",
            "F4": "Detect truncated <think> (open without close) and mark as failure",
            "F-strip:think_block": "Remove <think>...</think> from output",
            "F-strip:markdown_fence": "Remove ```json...``` fences",
            "F2:array_salvage_attempted": "When M2 still returns array, classify items by field shape into nodes/edges",
        },
        "models": MODELS,
        "papers": [{"pdf": pdf, "title": paper_texts[pdf][0]} for pdf in PAPERS],
        "diags": [diag_to_dict(d) for d in diags],
    }
    out_json = OUT_DIR / "spot_check_v2_results.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults: {out_json}")

    print("\n=== Per-model summary ===")
    for m in MODELS:
        ds = [d for d in diags if d.model == m]
        http_ok = sum(1 for d in ds if d.http_ok)
        strict = sum(1 for d in ds if d.parse_path == "pydantic_strict")
        fallback = sum(1 for d in ds if d.parse_path == "json_loads_fallback")
        failed = sum(1 for d in ds if d.parse_path == "failed")
        trunc = sum(1 for d in ds if d.truncated_thinking)
        arr = sum(1 for d in ds if d.root_type == "array")
        nodes = sum(len(d.nodes) for d in ds)
        edges = sum(len(d.edges) for d in ds)
        in_t = sum(d.in_tokens for d in ds)
        out_t = sum(d.out_tokens for d in ds)
        lat = sum(d.latency_s for d in ds) / max(1, len(ds))
        print(f"  {m:<26} http={http_ok}/5 strict={strict} fb={fallback} fail={failed} trunc={trunc} root_arr={arr} | nodes={nodes} edges={edges} tok_in={in_t} tok_out={out_t} avg_lat={lat:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
