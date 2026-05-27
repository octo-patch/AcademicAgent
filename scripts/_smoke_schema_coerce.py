"""Phase 1 smoke: load legacy KG with new coerce + domain logic, no data loss."""
from __future__ import annotations

import collections
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from src.knowledge.schema import KGNode, KGEdge, NodeType


def main() -> None:
    src = Path("data/knowledge_graph.json")
    data = json.loads(src.read_text(encoding="utf-8"))

    nodes_loaded: list[KGNode] = []
    type_dist: collections.Counter[str] = collections.Counter()
    for n in data["nodes"]:
        node = KGNode.from_dict(n)
        nodes_loaded.append(node)
        type_dist[node.node_type.value] += 1
    other_count = type_dist.get("other", 0)

    edges_loaded: list[KGEdge] = []
    rel_dist: collections.Counter[str] = collections.Counter()
    for e in data["edges"]:
        edge = KGEdge.from_dict(e)
        edges_loaded.append(edge)
        rel_dist[edge.relation_type.value] += 1

    print("---SUMMARY---")
    print(f"nodes loaded: {len(nodes_loaded)} / {len(data['nodes'])}")
    print(f"type dist (after coerce): {dict(type_dist)}")
    print(f"OTHER fallback count: {other_count}")
    print(f"edges loaded: {len(edges_loaded)} / {len(data['edges'])}")
    print(f"rel dist: {dict(rel_dist)}")

    # Round-trip check: serialize back and confirm
    sample = nodes_loaded[0]
    rt = KGNode.from_dict(sample.to_dict())
    assert rt.node_id == sample.node_id, f"round-trip mismatch: {sample.node_id} vs {rt.node_id}"
    assert rt.domain == sample.domain
    print(f"round-trip OK on sample node {sample.node_id} (domain={sample.domain})")


if __name__ == "__main__":
    main()
