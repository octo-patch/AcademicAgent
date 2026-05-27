"""批量 dump KG 抽到的节点 + abstract 文本，给批量 gold 标注做准备。

输出：每篇论文一段，用 --- 隔开。
"""
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pdf_parser import PDFParser

# (slug, domain, paper_title_hints[], pdf_path)
JOBS = [
    # Wave 3: Comms batch (14 papers; ChannelSLAM 已完成)
    ("graph_slam_los_bias", "comms",
     ["Graph-Based Algorithm for Robust Sequential Localization"],
     "EE_papers/A_Graph-Based_Algorithm_for_Robust_Sequential_Localization_Exploiting_Multipath_for_Obstructed-LOS-Bias_Mitigation.pdf"),
    ("isac_cluster_channel", "comms",
     ["Cluster-Based Statistical Channel Model"],
     "EE_papers/A_Cluster-Based_Statistical_Channel_Model_for_Integrated_Sensing_and_Communication_Channels.pdf"),
    ("isac_massive_mimo_tensor", "comms",
     ["Integrated Sensing and Communication With Massive MIMO"],
     "EE_papers/Integrated_Sensing_and_Communication_With_Massive_MIMO_A_Unified_Tensor_Approach_for_Channel_and_Target_Parameter_Estimation.pdf"),
    ("5g_sync_pos_map", "comms",
     ["5G Synchronization, Positioning and Mapping", "5G Synchronization"],
     "EE_papers/5G_Synchronization_Positioning_and_Mapping_From_Diffuse_Multipath.pdf"),
    ("5g_mmwave_vehicular", "comms",
     ["5G mmWave Downlink Vehicular"],
     "EE_papers/5G_mm_Wave_Downlink_Vehicular_Positioning.pdf"),
    ("doa_range_ula", "comms",
     ["DOA and Range Estimation"],
     "EE_papers/DOA_and_Range_Estimation_Using_a_Uniform_Linear_Antenna_Array_Without_A_Priori_Knowledge_of_the_Source_Number.pdf"),
    ("multipath_mimo_ofdm_tensor", "comms",
     ["Multipath-Aided Localization Method for MIMO-OFDM"],
     "EE_papers/A_Multipath-Aided_Localization_Method_for_MIMO-OFDM_Systems_via_Tensor_Decomposition.pdf"),
    ("ris_channel_estimation", "comms",
     ["Channel Estimation With Reconfigurable Intelligent Surfaces"],
     "EE_papers/Channel_Estimation_With_Reconfigurable_Intelligent_SurfacesA_General_Framework.pdf"),
    ("mds_localization_survey", "comms",
     ["State-of-the-Art Survey on Multidimensional Scaling-Based Localization"],
     "EE_papers/A_State-of-the-Art_Survey_on_Multidimensional_Scaling-Based_Localization_Techniques.pdf"),
    ("music_ml_crb", "comms",
     ["MUSIC, Maximum Likelihood, and Cramer-Rao Bound"],
     "EE_papers/MUSIC_maximum_likelihood_and_Cramer-Rao_bound.pdf"),
    ("slam_part_i", "comms",
     ["Simultaneous localization and mapping-part I", "Part I"],
     "EE_papers/Simultaneous localization and mapping-part I.pdf"),
    ("angle_slam_5g_mmwave", "comms",
     ["Angle-Based SLAM on 5G mmWave"],
     "EE_papers/Angle-Based_SLAM_on_5G_mmWave_Systems_Design_Implementation_and_Measurement (1).pdf"),
    ("ek_pmbm_filter", "comms",
     ["EK-PMBM Filter for Bistatic mmWave Radio SLAM"],
     "EE_papers/A_Computationally_Efficient_EK-PMBM_Filter_for_Bistatic_mmWave_Radio_SLAM.pdf"),
    ("rhs_slam", "comms",
     ["Reconfigurable Holographic Surface Aided Wireless Simultaneous Localization"],
     "EE_papers/Reconfigurable Holographic Surface Aided Wireless Simultaneous Localization and Mapping.pdf"),
]


def get_paper_nodes(domain, hints):
    kg = json.loads((PROJECT_ROOT / f"data/kg_{domain}.json").read_text(encoding="utf-8"))
    nodes = []
    for n in kg["nodes"]:
        srcs = (n.get("properties", {}).get("source_papers") or []) + [n.get("source_paper") or ""]
        for h in hints:
            if any(h in str(s) for s in srcs):
                nodes.append(n)
                break
    return nodes


for slug, domain, hints, pdf in JOBS:
    print("=" * 78)
    print(f"## {slug}  domain={domain}")
    print(f"## pdf={pdf}")
    print()
    # KG nodes
    nodes = get_paper_nodes(domain, hints)
    print(f"### KG (this paper contributed {len(nodes)} nodes):")
    by_type = {}
    for n in nodes:
        by_type.setdefault(n["node_type"], []).append(n["label"])
    for t in sorted(by_type):
        labels = sorted(by_type[t])
        print(f"  [{t}] {labels}")
    print()

    # Abstract
    try:
        with PDFParser(str(PROJECT_ROOT / pdf)) as p:
            text = p.extract_full_text()
        # find abstract: between "Abstract" and "Introduction"
        low = text.lower()
        i = low.find("abstract")
        j = low.find("introduction", i + 8 if i >= 0 else 0)
        if 0 <= i < j:
            abstract = text[i:j][:2500]
        else:
            abstract = text[:2500]
        print(f"### Abstract / early text (first 2500 chars):")
        print(abstract)
    except Exception as e:
        print(f"PDF read failed: {e}")
    print()

