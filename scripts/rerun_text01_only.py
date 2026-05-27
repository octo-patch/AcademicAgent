"""Re-run only MiniMax-Text-01 after F1c fix; merge into v2 results.
Standalone — no cross-script imports.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from src.core.pdf_parser import PDFParser

load_dotenv()

PAPERS = [
    "A_Cluster-Based_Statistical_Channel_Model_for_Integrated_Sensing_and_Communication_Channels.pdf",
    "Angle-Based_SLAM_on_5G_mmWave_Systems_Design_Implementation_and_Measurement (1).pdf",
    "A_Multipath-Aided_Localization_Method_for_MIMO-OFDM_Systems_via_Tensor_Decomposition.pdf",
    "5G_mm_Wave_Downlink_Vehicular_Positioning.pdf",
    "An_Asymptotically_Efficient_Estimator_in_Closed-Form_for_3-D_AOA_Localization_Using_a_Sensor_Network.pdf",
]
EE_DIR = Path("EE_papers")
MAX_CHARS = 12000

NODE_ENUM = ["paper", "author", "concept", "method", "dataset", "metric", "tool"]
REL_ENUM = ["proposes", "uses", "improves", "extends", "cites"]

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

## 输出结构（必须严格遵守）
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

只输出 JSON，不要任何其他内容。"""


def strip_and_parse(raw: str) -> dict:
    diag = {"has_open_think": "<think>" in raw, "has_close_think": "</think>" in raw,
            "truncated_thinking": False, "has_markdown_fence": False,
            "root_type": "", "parse_path": "failed", "parse_error": "", "nodes": [], "edges": []}
    s = raw
    if diag["has_open_think"] and not diag["has_close_think"]:
        diag["truncated_thinking"] = True
        return diag
    if diag["has_close_think"]:
        s = re.split(r"</think>", s, maxsplit=1)[1]
    s = s.strip()
    if s.startswith("```"):
        diag["has_markdown_fence"] = True
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    if not s:
        diag["parse_error"] = "empty after cleaning"
        return diag
    try:
        obj = json.loads(s)
    except Exception as e:
        diag["parse_error"] = f"json.loads: {str(e)[:200]}"
        return diag
    if isinstance(obj, dict):
        diag["root_type"] = "object"
        nodes = obj.get("nodes") or []
        edges = obj.get("edges") or []
        if isinstance(nodes, list) and isinstance(edges, list):
            diag["parse_path"] = "pydantic_strict"
            diag["nodes"] = nodes
            diag["edges"] = edges
    elif isinstance(obj, list):
        diag["root_type"] = "array"
        diag["parse_path"] = "json_loads_fallback"
        for item in obj:
            if isinstance(item, dict):
                if "relation_type" in item or "source_label" in item:
                    diag["edges"].append(item)
                elif "node_type" in item or "label" in item:
                    diag["nodes"].append(item)
    return diag


def main() -> int:
    client = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"),
                    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"))

    V2_PATH = Path("data/spot_check_v2/spot_check_v2_results.json")
    payload = json.loads(V2_PATH.read_text(encoding="utf-8"))
    new_diags = [d for d in payload["diags"] if d["model"] != "MiniMax-Text-01"]

    print(f"Rerunning MiniMax-Text-01 on {len(PAPERS)} papers (after F1c fix)...")
    for i, pdf in enumerate(PAPERS, 1):
        text = PDFParser(str(EE_DIR / pdf)).extract_full_text()[:MAX_CHARS]
        title = pdf.replace(".pdf", "").replace("_", " ").replace(" (1)", "")
        print(f"  [{i}/{len(PAPERS)}] {pdf[:55]}...", end=" ", flush=True)
        prompt = STRUCTURED_PROMPT.format(paper_title=title, text=text)
        rec = {"model": "MiniMax-Text-01", "paper": title, "http_ok": False, "error": "",
               "in_tokens": 0, "out_tokens": 0, "latency_s": 0.0, "server_model": "",
               "has_open_think": False, "has_close_think": False, "truncated_thinking": False,
               "has_markdown_fence": False, "root_type": "", "parse_path": "failed",
               "parse_error": "", "fixes_applied": ["F1a:short_descriptions", "F1b:no_nullable_arrays",
                                                     "F1c:number_not_integer", "F2:structured_prompt",
                                                     "F3:max_tokens_16k"],
               "nodes": [], "edges": [], "raw_first_600": "", "raw_last_300": ""}
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model="MiniMax-Text-01", max_tokens=16000, temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "kg_extraction", "schema": SHORT_SCHEMA}},
            )
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            rec["latency_s"] = round(time.time() - t0, 2)
            print(f"[XX] HTTP_FAIL: {rec['error'][:80]}")
            new_diags.append(rec)
            continue
        rec["latency_s"] = round(time.time() - t0, 2)
        rec["http_ok"] = True
        rec["server_model"] = getattr(r, "model", "") or ""
        if r.usage:
            rec["in_tokens"] = r.usage.prompt_tokens
            rec["out_tokens"] = r.usage.completion_tokens
        raw = r.choices[0].message.content or ""
        rec["raw_first_600"] = raw[:600]
        rec["raw_last_300"] = raw[-300:] if len(raw) > 300 else ""
        diag = strip_and_parse(raw)
        rec.update({k: v for k, v in diag.items() if k in rec})
        tag = "[OK]" if rec["parse_path"] == "pydantic_strict" else ("[FB]" if rec["parse_path"] == "json_loads_fallback" else "[XX]")
        print(f"{tag} n={len(rec['nodes'])} e={len(rec['edges'])} root={rec['root_type']} in/out={rec['in_tokens']}/{rec['out_tokens']} {rec['latency_s']}s")
        new_diags.append(rec)

    payload["diags"] = new_diags
    payload.setdefault("fixes", {})["F1c"] = "`type: number` instead of `type: integer` — Text-01 only accepts {string,number,boolean,object,array}, rejects `integer`"
    V2_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Updated Text-01 stats ===")
    rs = [d for d in new_diags if d["model"] == "MiniMax-Text-01"]
    http_ok = sum(1 for d in rs if d["http_ok"])
    strict = sum(1 for d in rs if d["parse_path"] == "pydantic_strict")
    nodes = sum(len(d["nodes"]) for d in rs)
    edges = sum(len(d["edges"]) for d in rs)
    print(f"  http_ok={http_ok}/5  strict={strict}  nodes={nodes}  edges={edges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
