"""Merge data/kg_llm_agent.json + data/kg_comms.json into
datasets/personal_v1/fixtures/seed_kg_combined.json.

每个节点的 properties.domain 标记原始域，supplying retrieved_domains for
cross_domain_leakage metric. 节点 id 冲突时按 'llm_agent_' / 'comms_' 前缀消歧。

Output schema mirrors KnowledgeGraphStore JSON: {metadata, nodes, edges}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "datasets" / "personal_v1" / "fixtures" / "seed_kg_combined.json"

SRC = {
    "llm_agent": REPO_ROOT / "data" / "kg_llm_agent.json",
    "comms": REPO_ROOT / "data" / "kg_comms.json",
}


def main() -> int:
    merged_nodes: list[dict] = []
    merged_edges: list[dict] = []
    seen_ids: set[str] = set()
    id_remap: dict[tuple[str, str], str] = {}  # (domain, orig_id) -> final_id

    for dom, path in SRC.items():
        if not path.exists():
            print(f"[warn] {path} missing, skipping {dom}")
            continue
        kg = json.loads(path.read_text(encoding="utf-8"))
        nodes = kg.get("nodes", []) or []
        edges = kg.get("edges", []) or []
        print(f"[info] {dom}: {len(nodes)} nodes, {len(edges)} edges")

        for n in nodes:
            orig_id = n.get("node_id") or n.get("id") or ""
            if not orig_id:
                continue
            final_id = orig_id if orig_id not in seen_ids else f"{dom}__{orig_id}"
            id_remap[(dom, orig_id)] = final_id
            seen_ids.add(final_id)
            n2 = dict(n)
            n2["node_id"] = final_id
            props = dict(n2.get("properties") or {})
            props["domain"] = dom
            n2["properties"] = props
            merged_nodes.append(n2)

        for e in edges:
            src = e.get("source_id") or e.get("src") or ""
            dst = e.get("target_id") or e.get("dst") or ""
            new_src = id_remap.get((dom, src), src)
            new_dst = id_remap.get((dom, dst), dst)
            e2 = dict(e)
            e2["source_id"] = new_src
            e2["target_id"] = new_dst
            merged_edges.append(e2)

    out = {
        "metadata": {
            "node_count": len(merged_nodes),
            "edge_count": len(merged_edges),
            "domain": "shared",
            "version": "1.0",
            "source": "datasets/personal_v1/fixtures/seed_kg_combined.json (merged from kg_llm_agent + kg_comms)",
        },
        "nodes": merged_nodes,
        "edges": merged_edges,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(REPO_ROOT)}: {len(merged_nodes)} nodes / {len(merged_edges)} edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
