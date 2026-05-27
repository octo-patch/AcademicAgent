"""
Agent Runtime - 内置 Workflow 注册
====================================

把 .agents/workflows/*.md 描述的科研工作流落到 Python：

  paper_analysis_flow:
    输入: pdf_path（或 paper_text + paper_title）
    步骤:
      1) kg_semantic_precheck — 跑前查 KG，避免重复入库（skip on fail）
      2) [if pdf_path] ingest_local_pdf — 注册到 paper_registry（retry once）
      3) add_paper_to_graph    — LLM 抽取 + 入图（retry once）
      4) query_knowledge       — 入图后回查，让 final_state 带 KG 命中
    输出: run_log，含 paper_title / node_count / kg_hits 等

  knowledge_build_flow:
    输入: 无（读 KG 现状）
    步骤:
      1) detect_gaps           — 盲区检测
      2) analyze_knowledge     — 学习路径
    输出: run_log，含 gap_summary / path_summary

这两条流程都不替代宿主 Claude——它们只是把"一键端到端执行"封装好。
"""

from __future__ import annotations

import logging

from .preferences import load_preferences
from .tools import register_default_tools
from .workflow import Workflow, WorkflowStep

logger = logging.getLogger("ScholarMind.Runtime.Workflows")


_REGISTRY: dict[str, Workflow] = {}


def get_workflow(name: str) -> Workflow | None:
    _ensure_registered()
    return _REGISTRY.get(name)


def all_workflows() -> dict[str, Workflow]:
    _ensure_registered()
    return dict(_REGISTRY)


# ------------------------------------------------------------------
# Workflow 定义
# ------------------------------------------------------------------


def _build_paper_analysis_flow() -> Workflow:
    """端到端跑一篇 PDF：precheck → ingest → extract → 回查。"""

    def _precheck_args(state: dict) -> dict:
        prefs = load_preferences()
        domain = state.get("domain") or prefs.get("default_domain", "llm_agent")
        # 给一个最朴素的 query：用 hint_title 或 pdf 文件名
        q = state.get("paper_title") or state.get("hint_title") or ""
        if not q and state.get("pdf_path"):
            from pathlib import Path
            q = Path(state["pdf_path"]).stem
        state["_resolved_domain"] = domain
        return {"query": q or "paper", "domain": domain}

    def _capture_precheck(raw: dict) -> dict:
        return {
            "precheck_hits": raw.get("hits", []),
            "precheck_existing": raw.get("looks_existing", False),
        }

    def _ingest_args(state: dict) -> dict:
        if not state.get("pdf_path"):
            raise ValueError("paper_analysis_flow: pdf_path 必填（无 PDF 走 add_paper_to_graph 直传 text）")
        return {
            "pdf_path": state["pdf_path"],
            "hint_title": state.get("hint_title", ""),
            "hint_arxiv_id": state.get("hint_arxiv_id", ""),
            "domain": state.get("_resolved_domain") or state.get("domain") or "",
            "add_to_kg": False,  # 我们后面手动调 add_paper_to_graph，便于分步看
            "first_seen_date": state.get("first_seen_date", ""),
        }

    def _capture_ingest(raw: dict) -> dict:
        report = raw.get("report", "") or ""
        return {
            "ingest_report_head": report[:400],
            "intake_status": raw.get("status", ""),
        }

    def _extract_args(state: dict) -> dict:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "text": state.get("paper_text", ""),
            "paper_title": state.get("paper_title") or state.get("hint_title") or "",
            "paper_year": state.get("paper_year", ""),
            "domain": state.get("_resolved_domain") or state.get("domain") or "",
            "first_seen_year": state.get("first_seen_date", ""),
        }

    def _capture_extract(raw: dict) -> dict:
        return {
            "kg_node_count": raw.get("node_count", 0),
            "kg_edge_count": raw.get("edge_count", 0),
            "kg_domain": raw.get("domain", ""),
        }

    def _query_args(state: dict) -> dict:
        q = state.get("paper_title") or state.get("hint_title") or "paper"
        return {"query": q, "domain": state.get("_resolved_domain") or ""}

    def _capture_query(raw: dict) -> dict:
        report = raw.get("report", "") or ""
        return {"query_report_head": report[:500]}

    return Workflow(
        name="paper_analysis_flow",
        description="一篇 PDF 端到端入库：precheck → ingest → extract → 回查",
        steps=[
            WorkflowStep(
                name="precheck",
                tool="kg_semantic_precheck",
                args_fn=_precheck_args,
                capture=_capture_precheck,
                on_failure="skip",  # KG 为空时不阻塞
                description="跑前在 KG 里搜一下，避免重复入库",
            ),
            WorkflowStep(
                name="ingest",
                tool="ingest_local_pdf",
                args_fn=_ingest_args,
                capture=_capture_ingest,
                on_failure="retry_once",
                description="注册到 paper_registry，去重 + 元数据抽取",
            ),
            WorkflowStep(
                name="extract",
                tool="add_paper_to_graph",
                args_fn=_extract_args,
                capture=_capture_extract,
                on_failure="retry_once",
                description="LLM 抽取实体/关系并入图",
            ),
            WorkflowStep(
                name="query_back",
                tool="query_knowledge",
                args_fn=_query_args,
                capture=_capture_query,
                on_failure="skip",
                description="入图后用 title 回查，验证可达性",
            ),
        ],
    )


def _build_knowledge_build_flow() -> Workflow:
    def _gaps_args(state: dict) -> dict:
        prefs = load_preferences()
        return {"domain": state.get("domain") or prefs.get("default_domain", "llm_agent")}

    def _capture_gaps(raw: dict) -> dict:
        return {"gap_report_head": (raw.get("report") or "")[:500]}

    def _analyze_args(state: dict) -> dict:
        prefs = load_preferences()
        return {
            "domain": state.get("domain") or prefs.get("default_domain", "llm_agent"),
            "focus_area": state.get("focus_area") or state.get("user_interest", ""),
            "max_items": int(state.get("max_items", 10)),
        }

    def _capture_analyze(raw: dict) -> dict:
        return {"learning_path_head": (raw.get("report") or "")[:600]}

    return Workflow(
        name="knowledge_build_flow",
        description="盲区检测 + 学习路径推荐（基于现有 KG）",
        steps=[
            WorkflowStep(
                name="detect_gaps",
                tool="detect_gaps",
                args_fn=_gaps_args,
                capture=_capture_gaps,
                on_failure="skip",
                description="找出图谱中孤立 / 低连通的概念",
            ),
            WorkflowStep(
                name="learning_path",
                tool="analyze_knowledge",
                args_fn=_analyze_args,
                capture=_capture_analyze,
                on_failure="retry_once",
                description="基于盲区给出建议阅读路径",
            ),
        ],
    )


_REGISTERED = False


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    register_default_tools()
    _REGISTRY["paper_analysis_flow"] = _build_paper_analysis_flow()
    _REGISTRY["knowledge_build_flow"] = _build_knowledge_build_flow()
    _REGISTERED = True
