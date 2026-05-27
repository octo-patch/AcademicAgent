"""跑所有 datasets/personal_v1/gold/*.json 的 metric，输出汇总表。"""
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


# slug -> (domain, paper_title_hints)
PAPER_MAP = {
    "react":              ("llm_agent", ["ReAct: Synergizing", "ReAct"]),
    "toolformer":         ("llm_agent", ["Toolformer"]),
    "reflexion":          ("llm_agent", ["Reflexion"]),
    "self_refine":        ("llm_agent", ["Self-Refine"]),
    "voyager":            ("llm_agent", ["Voyager", "VOYAGER"]),
    "tot":                ("llm_agent", ["Tree of Thoughts"]),
    "generative_agents":  ("llm_agent", ["Generative Agents", "Interactive Simulacra"]),
    "autogen":            ("llm_agent", ["AutoGen"]),
    "agentbench":         ("llm_agent", ["AgentBench"]),
    "coala":              ("llm_agent", ["Cognitive Architectures", "CoALA"]),
    "memgpt":             ("llm_agent", ["MemGPT"]),
    "llm_agent_survey":   ("llm_agent", ["Survey on Large Language Model based Autonomous Agents"]),
    "os_copilot":         ("llm_agent", ["OS-Copilot"]),
    "appagent":           ("llm_agent", ["AppAgent"]),
    "swe_agent":          ("llm_agent", ["SWE-agent", "Agent-Computer Interfaces"]),
    "channel_slam":       ("comms", ["ChannelSLAM", "Channel-SLAM",
                                     "Multipath_Assisted_Positioning",
                                     "Multipath Assisted"]),
}


def load_paper_nodes(domain, hints):
    store = KnowledgeGraphStore(graph_path=f"data/kg_{domain}.json", domain=domain)
    out = []
    for n in store._nodes.values():
        srcs = (n.properties.get("source_papers") or []) + [n.source_paper or ""]
        for h in hints:
            if any(h in str(s) for s in srcs):
                out.append({"label": n.label, "node_type": n.node_type.value})
                break
    return out


GOLD_DIR = PROJECT_ROOT / "datasets" / "personal_v1" / "gold"

print(f"{'slug':22s}{'#ext':>5}{'#gold_n':>9}{'must':>7}{'F1':>7}{'rec':>7}{'prec':>7}{'p_known':>9}{'tiers (exact/alias/embed/miss)':>34}")
print("-" * 100)

for slug, (domain, hints) in PAPER_MAP.items():
    gold_path = GOLD_DIR / f"{slug}.json"
    if not gold_path.exists():
        continue
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    n_gold = (len(gold.get("must_have", []))
              + len(gold.get("should_have", []))
              + len(gold.get("may_have", [])))
    extracted = load_paper_nodes(domain, hints)
    f1 = kg_node_f1_semantic(extracted, gold, use_embedding=True)
    mh = must_have_recall(extracted, gold, use_embedding=True)
    pk = precision_on_known(extracted, gold, use_embedding=True)
    # Parse the notes JSON for precision/recall and tier breakdown
    notes = json.loads(f1.notes) if f1.notes else {}
    tiers = notes.get("match_tiers", {})
    tier_str = (f"{tiers.get('exact',0):>3}/{tiers.get('alias',0):>3}/"
                f"{tiers.get('embed',0):>3}/{tiers.get('miss',0):>3}")
    print(f"{slug:22s}{len(extracted):5d}{n_gold:9d}"
          f"{mh.value:7.3f}{f1.value:7.3f}"
          f"{notes.get('recall',0):7.3f}{notes.get('precision',0):7.3f}"
          f"{pk.value:9.3f}{tier_str:>34}")
