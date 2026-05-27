"""Spot-check: run KG extraction on N papers across M MiniMax models, compare side-by-side.

Self-contained — does NOT modify production extractor.
Handles <think>...</think> + ```json``` markdown fence stripping for reasoning models.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from src.core.pdf_parser import PDFParser
from src.knowledge.schema import ExtractionOutput, SCHEMA_DESCRIPTION

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

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
OUT_DIR = Path("data/spot_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHARS = 12000  # matches extractor.py default

EXTRACTION_PROMPT = """你是一名通信/感知/定位领域的学术论文知识图谱抽取专家，熟悉 5G/6G、ISAC、SLAM、多径定位、AOA/TDOA/RSS 估计、MIMO、信道建模、Kalman/PMBM 滤波等主题。

你的任务：根据下方的 Schema（架构），从给定的文本中抽取实体（节点）和关系（边）。

{schema}

## 规则
1. 只能抽取上方 Schema 中定义过的实体和关系类型。
2. 所有的实体标签（Entity labels）强制使用 **英文**（即使原文是中文，也必须使用标准英文专业术语）。
3. 规范化实体名称：使用最常见/标准的英文名称或缩写形式（例如：ISAC、SLAM、MUSIC、PMBM、OFDM、CRLB）。
4. 每个实体必须具备 Schema 中明确定义的 `node_type`。
5. 每条关系必须包含 `source_label`, `source_type`, `target_label`, `target_type`, 以及 `relation_type`。
6. 必须精确：只能抽取文本中明确陈述或强烈暗示的内容。
7. 绝对不要（Do NOT）伪造或幻觉出文本不支持的实体或关系。

## 论文上下文
- **标题**: {paper_title}

## 待抽取的文本
{text}

## 输出格式
严格按照 JSON Schema 约束输出，不要输出任何其他解释性文本。
"""


@dataclass
class CallResult:
    model: str
    paper: str
    ok: bool
    error: str = ""
    raw: str = ""
    parsed_nodes: list = None
    parsed_edges: list = None
    parse_error: str = ""
    in_tokens: int = 0
    out_tokens: int = 0
    latency_s: float = 0.0
    server_model: str = ""


def strip_reasoning(content: str) -> str:
    """Strip <think>...</think> blocks and ```json...``` markdown fences."""
    # remove <think>...</think>
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # strip leading ```json or ``` and trailing ```
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def call_model(client: OpenAI, model: str, paper_title: str, text: str) -> CallResult:
    prompt = EXTRACTION_PROMPT.format(
        schema=SCHEMA_DESCRIPTION,
        paper_title=paper_title,
        text=text,
    )
    res = CallResult(model=model, paper=paper_title, ok=False)
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            max_tokens=8000,  # generous, accommodate reasoning + JSON
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_output",
                    "schema": ExtractionOutput.model_json_schema(),
                },
            },
        )
    except Exception as e:
        res.error = f"{type(e).__name__}: {str(e)[:300]}"
        res.latency_s = round(time.time() - t0, 2)
        return res

    res.latency_s = round(time.time() - t0, 2)
    res.server_model = getattr(r, "model", "") or ""
    res.in_tokens = r.usage.prompt_tokens if r.usage else 0
    res.out_tokens = r.usage.completion_tokens if r.usage else 0
    raw = r.choices[0].message.content or ""
    res.raw = raw
    cleaned = strip_reasoning(raw)
    res.ok = True
    try:
        parsed = ExtractionOutput.model_validate_json(cleaned)
        res.parsed_nodes = [n.model_dump() for n in parsed.nodes]
        res.parsed_edges = [e.model_dump() for e in parsed.edges]
    except Exception as e:
        # try json.loads as a softer fallback
        try:
            obj = json.loads(cleaned)
            res.parsed_nodes = obj.get("nodes", [])
            res.parsed_edges = obj.get("edges", [])
            res.parse_error = f"pydantic failed, json.loads ok: {str(e)[:150]}"
        except Exception as e2:
            res.parse_error = f"pydantic: {str(e)[:150]} | json: {str(e2)[:150]}"
    return res


def main() -> int:
    api_key = os.getenv("MINIMAX_API_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # extract text from each PDF once
    paper_texts: dict[str, tuple[str, str]] = {}
    print("Extracting paper texts...")
    for pdf_name in PAPERS:
        path = EE_DIR / pdf_name
        parser = PDFParser(str(path))
        text = parser.extract_full_text()
        title = pdf_name.replace(".pdf", "").replace("_", " ").replace(" (1)", "")
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
        paper_texts[pdf_name] = (title, text)
        print(f"  {pdf_name[:60]:<60} {len(text)} chars")

    # run all (paper x model) combinations
    results: list[CallResult] = []
    total = len(PAPERS) * len(MODELS)
    n = 0
    print(f"\nRunning {total} extractions...")
    for pdf_name in PAPERS:
        title, text = paper_texts[pdf_name]
        for model in MODELS:
            n += 1
            print(f"  [{n}/{total}] {model:<26} on {pdf_name[:50]}...", end=" ", flush=True)
            r = call_model(client, model, title, text)
            results.append(r)
            if r.ok:
                nc = len(r.parsed_nodes or [])
                ec = len(r.parsed_edges or [])
                pe = " (parse_err)" if r.parse_error else ""
                print(f"OK nodes={nc} edges={ec} in/out={r.in_tokens}/{r.out_tokens} {r.latency_s}s{pe}")
            else:
                print(f"FAIL: {r.error[:100]}")

    # save raw + parsed
    payload = {
        "models": MODELS,
        "papers": [
            {"pdf": pdf, "title": paper_texts[pdf][0]} for pdf in PAPERS
        ],
        "results": [
            {
                "model": r.model, "paper": r.paper, "ok": r.ok, "error": r.error,
                "server_model": r.server_model,
                "in_tokens": r.in_tokens, "out_tokens": r.out_tokens, "latency_s": r.latency_s,
                "parse_error": r.parse_error,
                "nodes": r.parsed_nodes or [], "edges": r.parsed_edges or [],
                "raw_first_500": r.raw[:500],
            } for r in results
        ],
    }
    out_json = OUT_DIR / "spot_check_results.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_json}")

    # quick summary
    print("\n=== Summary ===")
    for model in MODELS:
        ok = [r for r in results if r.model == model and r.ok]
        n_nodes = sum(len(r.parsed_nodes or []) for r in ok)
        n_edges = sum(len(r.parsed_edges or []) for r in ok)
        n_parse_err = sum(1 for r in ok if r.parse_error)
        in_t = sum(r.in_tokens for r in results if r.model == model)
        out_t = sum(r.out_tokens for r in results if r.model == model)
        avg_lat = sum(r.latency_s for r in results if r.model == model) / max(1, len([r for r in results if r.model == model]))
        print(f"  {model:<26} ok={len(ok)}/{len(PAPERS)}  nodes={n_nodes}  edges={n_edges}  parse_err={n_parse_err}  tokens_in={in_t}  tokens_out={out_t}  avg_lat={avg_lat:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
