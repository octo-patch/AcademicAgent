"""快速跑 v1.0 两份 gold 样本的 metric，用于审核：
- kg_node_f1_semantic (alias + embedding)
- must_have_recall
- precision_on_known
"""
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.evaluation.metrics.kg_semantic import (
    kg_node_f1_semantic, must_have_recall, precision_on_known,
)
from src.knowledge.graph_store import KnowledgeGraphStore


def load_paper_nodes(domain, paper_title_hints):
    store = KnowledgeGraphStore(graph_path=f"data/kg_{domain}.json", domain=domain)
    paper_nodes = []
    for n in store._nodes.values():
        src = (n.source_paper or "")
        # also check properties.source_papers
        srcs = (n.properties.get("source_papers") or []) + [src]
        for hint in paper_title_hints:
            if any(hint in str(s) for s in srcs):
                paper_nodes.append({"label": n.label, "node_type": n.node_type.value})
                break
    return paper_nodes


def run_one(label, gold_path, domain, paper_title_hints):
    print(f"\n===== {label} =====")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    extracted = load_paper_nodes(domain, paper_title_hints)
    print(f"extracted {len(extracted)} nodes attributed to this paper")

    # Run metrics WITHOUT embedding (faster, alias-only) and WITH (full semantic)
    for use_emb in (False, True):
        tag = "alias+embed" if use_emb else "alias only"
        f1 = kg_node_f1_semantic(extracted, gold, use_embedding=use_emb)
        mh = must_have_recall(extracted, gold, use_embedding=use_emb)
        pk = precision_on_known(extracted, gold, use_embedding=use_emb)
        print(f"  [{tag:12s}]  kg_node_f1_semantic = {f1.value:.3f}  ({f1.notes})")
        print(f"  [{tag:12s}]  must_have_recall    = {mh.value:.3f}  ({mh.notes})")
        print(f"  [{tag:12s}]  precision_on_known  = {pk.value:.3f}  ({pk.notes})")
        print()


GOLD_DIR = PROJECT_ROOT / "datasets" / "personal_v1" / "gold"

run_one("ReAct", GOLD_DIR / "react.json", "llm_agent",
        ["ReAct: Synergizing", "ReAct", "arxiv_2210_03629"])

run_one("Channel-SLAM", GOLD_DIR / "channel_slam.json", "comms",
        ["ChannelSLAM", "Channel-SLAM", "Multipath Assisted Positioning",
         "Multipath_Assisted_Positioning"])
