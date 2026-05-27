"""Production smoke test: run KnowledgeExtractor on 1 paper after F-strip + F3 + F2 fixes."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.pdf_parser import PDFParser
from src.knowledge.extractor import KnowledgeExtractor

PDF = "EE_papers/5G_mm_Wave_Downlink_Vehicular_Positioning.pdf"


async def main() -> int:
    print(f"=== Production smoke test ===")
    print(f"Paper: {PDF}")
    text = PDFParser(PDF).extract_full_text()
    print(f"PDF text: {len(text)} chars (will be auto-chunked)")

    extractor = KnowledgeExtractor()
    print(f"Model  : {extractor.model}")
    print(f"max_tokens: {extractor.max_tokens}")
    print(f"max_chunk_chars: {extractor.max_chunk_chars} (paper will be split)")
    print()

    t0 = time.time()
    result = await extractor.extract_from_text(
        text=text[:12000],  # cap to 1 chunk to avoid multi-chunk for smoke test
        paper_title="5G mmWave Downlink Vehicular Positioning",
        paper_year="2018",
    )
    elapsed = time.time() - t0

    print(f"\n=== Result ===")
    print(f"elapsed         : {elapsed:.1f}s")
    print(f"nodes           : {len(result.nodes)}")
    print(f"edges           : {len(result.edges)}")
    print(f"confidence      : {result.extraction_confidence}")
    if result.nodes:
        print(f"\nFirst 5 nodes:")
        for n in result.nodes[:5]:
            print(f"  - {n.label} ({n.node_type.value})")
    if result.edges:
        print(f"\nFirst 5 edges:")
        for e in result.edges[:5]:
            print(f"  - {e.source_id} -[{e.relation_type.value}]-> {e.target_id}")
    if not result.nodes and not result.edges:
        print(f"\nFAILED: empty result")
        print(f"raw_llm_output (first 500 chars): {result.raw_llm_output[:500]!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
