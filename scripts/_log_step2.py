"""一次性脚本：把 Step 2 (Comms 15 篇 ingest) 写进 RefactorRecorder。"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.recorder import RefactorRecorder

rec = RefactorRecorder(area="gold-dataset-v1")
rec.log_validation(
    step="Step 2: 15 Comms papers 批量 ingest 完成",
    command="python scripts/ingest_comms_v1.py (+ --only music_ml_crb --force for retry, then --start 12)",
    output=(
        "15/15 OK. music_ml_crb 首跑因 LLM 输出 JSON 在 column 392 损坏"
        " (math symbols 转义问题) failed, retry 一次后 lenient fallback 救回. "
        "Per-paper delta nodes: channel_slam +49, graph_slam_los_bias +52, "
        "isac_cluster_channel +46, isac_massive_mimo_tensor +54, "
        "5g_sync_pos_map +16, 5g_mmwave_vehicular +23, doa_range_ula +34, "
        "multipath_mimo_ofdm_tensor +26, ris_channel_estimation +71, "
        "mds_localization_survey +133 (survey), music_ml_crb +25 (retry), "
        "slam_part_i +45, angle_slam_5g_mmwave +25, ek_pmbm_filter +54, "
        "rhs_slam +21. Final Comms KG: 674 nodes / 1130 edges."
    ),
    status="ok",
    notes=(
        "Comms 节点密度 45/篇 vs LLM Agent 110/篇 (Comms 论文更聚焦/数学密集). "
        "MUSIC 经典论文需 retry 体现 reasoning 模型在数学密集长文本上偶尔 JSON 损坏的 quirk. "
        "两 domain 总计 30 papers => 2326 nodes / 3098 edges"
    ),
    artifacts=[
        "data/kg_comms.json",
        "data/runtime_runs/ingest_comms_v1.progress.json",
    ],
)
print("Step 2 logged")
