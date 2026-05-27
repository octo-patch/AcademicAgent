"""
v1.0 gold dataset - LLM Agent 域 15 篇论文 ingest
=================================================

按 handoff §3.4 的 15 篇定稿，逐篇跑 ``paper_analysis_flow``：
  precheck → ingest_local_pdf → add_paper_to_graph → query_knowledge

要求：
  - 串行（MiniMax-M2 限速 + 便于看抽取质量），不并发
  - 默认从第 1 篇开始；若 ReAct 已 ingest，可用 ``--skip-existing`` 让 precheck
    决定是否跳过（precheck 命中 ≥0.55 视为已存在）。
  - 每篇跑完打印 KG 累计 node/edge 数 + ingest_report 头部，
    任何一篇失败立即 abort，等手动处理后再续跑。
  - 进度落 ``data/runtime_runs/ingest_llm_agent_v1.progress.json``，
    支持下次接续（已成功的跳过）。

运行：
    python scripts/ingest_llm_agent_v1.py                # 跑全部 15 篇
    python scripts/ingest_llm_agent_v1.py --only ReAct   # 只跑 1 篇（试运行）
    python scripts/ingest_llm_agent_v1.py --start 8      # 从第 8 篇开始
    python scripts/ingest_llm_agent_v1.py --dry-run      # 只打印计划

退出码：0 全部成功；1 部分失败（剩余在 progress.json 中标 pending）。
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
# 确保 "src.runtime" / "src.knowledge" 等能 import（脚本可能从任意 cwd 启动）
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROGRESS_PATH = PROJECT_ROOT / "data" / "runtime_runs" / "ingest_llm_agent_v1.progress.json"


@dataclass(frozen=True)
class PaperJob:
    slug: str        # short id (e.g. "react")
    title: str       # full paper title
    pdf: str         # filename under data/papers/
    arxiv_id: str = ""
    year: str = ""


# handoff §3.4 定稿 15 篇
JOBS: list[PaperJob] = [
    PaperJob("react",         "ReAct: Synergizing Reasoning and Acting in Language Models",
             "arxiv_2210_03629.pdf",                "2210.03629", "2022"),
    PaperJob("toolformer",    "Toolformer: Language Models Can Teach Themselves to Use Tools",
             "Toolformer_arxiv_2302_04761.pdf",     "2302.04761", "2023"),
    PaperJob("reflexion",     "Reflexion: Language Agents with Verbal Reinforcement Learning",
             "Reflexion_arxiv_2303_11366.pdf",      "2303.11366", "2023"),
    PaperJob("self_refine",   "Self-Refine: Iterative Refinement with Self-Feedback",
             "Self_Refine_arxiv_2303_17651.pdf",    "2303.17651", "2023"),
    PaperJob("voyager",       "Voyager: An Open-Ended Embodied Agent with Large Language Models",
             "Voyager_arxiv_2305_16291.pdf",        "2305.16291", "2023"),
    PaperJob("tot",           "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
             "Tree_of_Thoughts_arxiv_2305_10601.pdf", "2305.10601", "2023"),
    PaperJob("generative_agents", "Generative Agents: Interactive Simulacra of Human Behavior",
             "Generative_Agents.pdf",               "2304.03442", "2023"),
    PaperJob("autogen",       "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
             "AutoGen_arxiv_2308_08155.pdf",        "2308.08155", "2023"),
    PaperJob("agentbench",    "AgentBench: Evaluating LLMs as Agents",
             "AgentBench_arxiv_2308_03688.pdf",     "2308.03688", "2023"),
    PaperJob("coala",         "Cognitive Architectures for Language Agents",
             "CoALA_arxiv_2309_02427.pdf",          "2309.02427", "2023"),
    PaperJob("memgpt",        "MemGPT: Towards LLMs as Operating Systems",
             "MemGPT.pdf",                          "2310.08560", "2023"),
    PaperJob("llm_agent_survey", "A Survey on Large Language Model based Autonomous Agents",
             "LLM_Agent_Survey.pdf",                "2308.11432", "2024"),
    PaperJob("os_copilot",    "OS-Copilot: Towards Generalist Computer Agents with Self-Improvement",
             "OS_Copilot_arxiv_2402_07456.pdf",     "2402.07456", "2024"),
    PaperJob("appagent",      "AppAgent: Multimodal Agents as Smartphone Users",
             "AppAgent_arxiv_2312_13771.pdf",       "2312.13771", "2023"),
    PaperJob("swe_agent",     "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering",
             "SWE_Agent_arxiv_2405_15793.pdf",      "2405.15793", "2024"),
]


@dataclass
class JobResult:
    slug: str
    status: str            # "ok" / "failed" / "skipped" / "pending"
    started_at: str = ""
    finished_at: str = ""
    kg_node_count: int = 0
    kg_edge_count: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    step_summary: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


# ----------------------------------------------------------------------
# Progress 持久化
# ----------------------------------------------------------------------


def _load_progress() -> dict[str, JobResult]:
    if not PROGRESS_PATH.exists():
        return {}
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        return {
            slug: JobResult(**rec) for slug, rec in data.items()
        }
    except Exception:  # noqa: BLE001
        return {}


def _save_progress(state: dict[str, JobResult]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps({s: r.to_dict() for s, r in state.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------
# 单篇执行
# ----------------------------------------------------------------------


async def _run_one(job: PaperJob) -> JobResult:
    from src.runtime import run_workflow

    pdf_path = PROJECT_ROOT / "data" / "papers" / job.pdf
    if not pdf_path.exists():
        return JobResult(slug=job.slug, status="failed", error=f"PDF missing: {pdf_path}")

    started = time.time()
    result = await run_workflow(
        "paper_analysis_flow",
        {
            "pdf_path": str(pdf_path),
            "domain": "llm_agent",
            "hint_title": job.title,
            "hint_arxiv_id": job.arxiv_id,
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


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------


def _filter_jobs(args) -> list[PaperJob]:
    jobs = list(JOBS)
    if args.only:
        wanted = {x.strip().lower() for x in args.only.split(",")}
        jobs = [j for j in jobs if j.slug in wanted]
    if args.start is not None:
        jobs = jobs[args.start - 1:]  # 1-indexed
    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


async def _amain(args) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    progress = _load_progress()
    jobs = _filter_jobs(args)

    print(f"v1.0 LLM Agent ingest: {len(jobs)} 篇 (full list = {len(JOBS)})")
    print(f"progress file: {PROGRESS_PATH.relative_to(PROJECT_ROOT)}")
    if args.dry_run:
        print("\n[DRY RUN]\n")
        for j in jobs:
            prev = progress.get(j.slug)
            note = f"[{prev.status}]" if prev else "[new]"
            print(f"  {note} {j.slug}: {j.title}  ({j.pdf})")
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

        # 礼貌停顿，避免连击 MiniMax
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
    parser.add_argument("--only", help="仅跑这些 slug（逗号分隔，如 react,toolformer）")
    parser.add_argument("--start", type=int, default=None, help="从第 N 篇开始（1-indexed）")
    parser.add_argument("--limit", type=int, default=None, help="最多跑 N 篇")
    parser.add_argument("--force", action="store_true", help="忽略 progress.json，已成功的也重跑")
    parser.add_argument("--continue-on-failure", action="store_true", help="某篇失败不中止")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划")
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
