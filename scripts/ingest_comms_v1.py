"""
v1.0 gold dataset - Comms 域 15 篇论文 ingest
=============================================

handoff §3.5 定稿的 15 篇通信论文（来源 EE_papers/，不复制到 data/papers/，
workflow 接受任意 pdf_path 绝对/相对路径）。

要求与 ingest_llm_agent_v1.py 一致：串行、progress.json 持久化、
任一篇失败 abort。domain="comms"。

运行：
    python scripts/ingest_comms_v1.py
    python scripts/ingest_comms_v1.py --dry-run
    python scripts/ingest_comms_v1.py --only channel_slam,slam_part_i

退出码：0 全部成功；1 部分失败。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROGRESS_PATH = PROJECT_ROOT / "data" / "runtime_runs" / "ingest_comms_v1.progress.json"
EE_PAPERS_DIR = PROJECT_ROOT / "EE_papers"


@dataclass(frozen=True)
class PaperJob:
    slug: str
    title: str
    pdf: str         # 相对 EE_papers/
    archetype: str   # SLAM / ISAC / DoA / RIS / Multipath / Survey
    year: str = ""


# handoff §3.5 定稿 15 篇
JOBS: list[PaperJob] = [
    PaperJob("channel_slam",
             "ChannelSLAM: Multipath Assisted Positioning with Simultaneous Localization and Mapping",
             "[ChannelSLAM]Multipath_Assisted_Positioning_with_Simultaneous_Localization_and_Mapping.pdf",
             "SLAM classic", "2016"),
    PaperJob("graph_slam_los_bias",
             "A Graph-Based Algorithm for Robust Sequential Localization Exploiting Multipath for Obstructed-LOS-Bias Mitigation",
             "A_Graph-Based_Algorithm_for_Robust_Sequential_Localization_Exploiting_Multipath_for_Obstructed-LOS-Bias_Mitigation.pdf",
             "SLAM graph", "2018"),
    PaperJob("isac_cluster_channel",
             "A Cluster-Based Statistical Channel Model for Integrated Sensing and Communication Channels",
             "A_Cluster-Based_Statistical_Channel_Model_for_Integrated_Sensing_and_Communication_Channels.pdf",
             "ISAC channel", "2023"),
    PaperJob("isac_massive_mimo_tensor",
             "Integrated Sensing and Communication With Massive MIMO: A Unified Tensor Approach for Channel and Target Parameter Estimation",
             "Integrated_Sensing_and_Communication_With_Massive_MIMO_A_Unified_Tensor_Approach_for_Channel_and_Target_Parameter_Estimation.pdf",
             "ISAC tensor", "2023"),
    PaperJob("5g_sync_pos_map",
             "5G Synchronization, Positioning and Mapping From Diffuse Multipath",
             "5G_Synchronization_Positioning_and_Mapping_From_Diffuse_Multipath.pdf",
             "5G localization + multipath", "2021"),
    PaperJob("5g_mmwave_vehicular",
             "5G mmWave Downlink Vehicular Positioning",
             "5G_mm_Wave_Downlink_Vehicular_Positioning.pdf",
             "mmWave vehicular positioning", "2018"),
    PaperJob("doa_range_ula",
             "DOA and Range Estimation Using a Uniform Linear Antenna Array Without A Priori Knowledge of the Source Number",
             "DOA_and_Range_Estimation_Using_a_Uniform_Linear_Antenna_Array_Without_A_Priori_Knowledge_of_the_Source_Number.pdf",
             "DoA + Range", "2018"),
    PaperJob("multipath_mimo_ofdm_tensor",
             "A Multipath-Aided Localization Method for MIMO-OFDM Systems via Tensor Decomposition",
             "A_Multipath-Aided_Localization_Method_for_MIMO-OFDM_Systems_via_Tensor_Decomposition.pdf",
             "Multipath + tensor", "2020"),
    PaperJob("ris_channel_estimation",
             "Channel Estimation With Reconfigurable Intelligent Surfaces: A General Framework",
             "Channel_Estimation_With_Reconfigurable_Intelligent_SurfacesA_General_Framework.pdf",
             "RIS channel estimation", "2022"),
    PaperJob("mds_localization_survey",
             "A State-of-the-Art Survey on Multidimensional Scaling-Based Localization Techniques",
             "A_State-of-the-Art_Survey_on_Multidimensional_Scaling-Based_Localization_Techniques.pdf",
             "Survey MDS", "2019"),
    PaperJob("music_ml_crb",
             "MUSIC, Maximum Likelihood, and Cramer-Rao Bound",
             "MUSIC_maximum_likelihood_and_Cramer-Rao_bound.pdf",
             "DoA classic MUSIC + CRB", "1989"),
    PaperJob("slam_part_i",
             "Simultaneous Localization and Mapping (Part I)",
             "Simultaneous localization and mapping-part I.pdf",
             "SLAM classic survey Part I", "2006"),
    PaperJob("angle_slam_5g_mmwave",
             "Angle-Based SLAM on 5G mmWave Systems: Design, Implementation and Measurement",
             "Angle-Based_SLAM_on_5G_mmWave_Systems_Design_Implementation_and_Measurement (1).pdf",
             "mmWave SLAM measurement", "2022"),
    PaperJob("ek_pmbm_filter",
             "A Computationally Efficient EK-PMBM Filter for Bistatic mmWave Radio SLAM",
             "A_Computationally_Efficient_EK-PMBM_Filter_for_Bistatic_mmWave_Radio_SLAM.pdf",
             "EK-PMBM filter", "2023"),
    PaperJob("rhs_slam",
             "Reconfigurable Holographic Surface Aided Wireless Simultaneous Localization and Mapping",
             "Reconfigurable Holographic Surface Aided Wireless Simultaneous Localization and Mapping.pdf",
             "RHS + SLAM frontier", "2024"),
]


@dataclass
class JobResult:
    slug: str
    status: str
    started_at: str = ""
    finished_at: str = ""
    kg_node_count: int = 0
    kg_edge_count: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    step_summary: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def _load_progress() -> dict[str, JobResult]:
    if not PROGRESS_PATH.exists():
        return {}
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        return {s: JobResult(**rec) for s, rec in data.items()}
    except Exception:  # noqa: BLE001
        return {}


def _save_progress(state: dict[str, JobResult]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps({s: r.to_dict() for s, r in state.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def _run_one(job: PaperJob) -> JobResult:
    from src.runtime import run_workflow

    pdf_path = EE_PAPERS_DIR / job.pdf
    if not pdf_path.exists():
        return JobResult(slug=job.slug, status="failed", error=f"PDF missing: {pdf_path}")

    started = time.time()
    result = await run_workflow(
        "paper_analysis_flow",
        {
            "pdf_path": str(pdf_path),
            "domain": "comms",
            "hint_title": job.title,
            "paper_year": job.year,
        },
    )
    elapsed = time.time() - started

    fs = result.get("final_state", {}) or {}
    steps_summary = [
        {"name": s.get("name"), "status": s.get("status"), "latency_ms": s.get("latency_ms", 0)}
        for s in (result.get("steps", []) or [])
    ]
    failed = any(s.get("status") in ("failed", "error") for s in (result.get("steps", []) or []))

    status = "failed" if failed else "ok"
    error_msg = ""
    if failed:
        for s in result.get("steps", []) or []:
            if s.get("status") in ("failed", "error") and s.get("error"):
                error_msg = (s["error"] or "")[:400]
                break

    return JobResult(
        slug=job.slug,
        status=status,
        started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        kg_node_count=int(fs.get("kg_node_count") or 0),
        kg_edge_count=int(fs.get("kg_edge_count") or 0),
        elapsed_s=round(elapsed, 2),
        error=error_msg,
        step_summary=steps_summary,
    )


def _filter_jobs(args) -> list[PaperJob]:
    jobs = list(JOBS)
    if args.only:
        wanted = {x.strip().lower() for x in args.only.split(",")}
        jobs = [j for j in jobs if j.slug in wanted]
    if args.start is not None:
        jobs = jobs[args.start - 1:]
    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


async def _amain(args) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    progress = _load_progress()
    jobs = _filter_jobs(args)

    print(f"v1.0 Comms ingest: {len(jobs)} 篇 (full list = {len(JOBS)})")
    print(f"progress file: {PROGRESS_PATH.relative_to(PROJECT_ROOT)}")
    if args.dry_run:
        print("\n[DRY RUN]\n")
        for j in jobs:
            prev = progress.get(j.slug)
            note = f"[{prev.status}]" if prev else "[new]"
            print(f"  {note} {j.slug}: {j.title[:60]}  ({j.pdf[:60]})")
        return 0

    failures: list[str] = []
    print()
    for i, job in enumerate(jobs, 1):
        prev = progress.get(job.slug)
        if prev and prev.status == "ok" and not args.force:
            print(f"  [{i}/{len(jobs)}] {job.slug}: skip (already ok)")
            continue
        print(f"  [{i}/{len(jobs)}] {job.slug}: {job.title[:60]} ...", flush=True)
        try:
            r = await _run_one(job)
        except Exception as exc:  # noqa: BLE001
            r = JobResult(slug=job.slug, status="failed",
                          error=f"{type(exc).__name__}: {exc}"[:400])
        progress[job.slug] = r
        _save_progress(progress)

        if r.status == "ok":
            print(f"      OK  KG nodes/edges = {r.kg_node_count}/{r.kg_edge_count}"
                  f"  elapsed={r.elapsed_s}s")
        else:
            print(f"      FAIL  {r.error}")
            failures.append(job.slug)
            if not args.continue_on_failure:
                print(f"\n中止：等手动处理 {job.slug} 后再续跑（progress 已保存）。")
                return 1

        if i < len(jobs):
            await asyncio.sleep(2.0)

    print()
    if failures:
        print(f"失败 {len(failures)} / {len(jobs)}: {failures}")
        return 1
    print(f"全部 {len(jobs)} 篇 ingest 完成。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="仅跑这些 slug（逗号分隔）")
    parser.add_argument("--start", type=int, default=None, help="从第 N 篇开始（1-indexed）")
    parser.add_argument("--limit", type=int, default=None, help="最多跑 N 篇")
    parser.add_argument("--force", action="store_true", help="忽略 progress.json，重跑")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
