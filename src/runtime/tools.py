"""
Agent Runtime - 工具适配层
===========================

把 src/mcp_servers/*.py 里的 @mcp.tool() async 函数适配为 Python 内部调用，
注册到 src/runtime/workflow.py 的 _TOOLS 字典，让 WorkflowStep.tool 名字
能找到对应可 await 的函数。

为什么不直接调 MCP 协议：runtime 跑在同一个 Python 进程内，
直接调内部函数省 stdio 一跳；并且 MCP 端口可以独立保留给宿主用。
"""

from __future__ import annotations

import logging
from typing import Any

from .workflow import register_tool

logger = logging.getLogger("ScholarMind.Runtime.Tools")


# ------------------------------------------------------------------
# Lazy import wrappers —— 第一次调用时才导入对应模块，避免加载 MCP fastmcp
# ------------------------------------------------------------------


async def ingest_local_pdf(
    pdf_path: str,
    hint_title: str = "",
    hint_arxiv_id: str = "",
    domain: str = "",
    add_to_kg: bool = False,
    first_seen_date: str = "",
) -> dict[str, Any]:
    from src.mcp_servers.paper_search import ingest_local_pdf as _impl
    text = await _impl(
        pdf_path=pdf_path,
        hint_title=hint_title,
        hint_arxiv_id=hint_arxiv_id,
        domain=domain,
        add_to_kg=add_to_kg,
        first_seen_date=first_seen_date,
    )
    return {
        "status": "ok" if "✅" in text or "intake" in text.lower() else "warn",
        "report": text,
        "added_to_kg": add_to_kg,
    }


async def add_paper_to_graph(
    text: str = "",
    pdf_path: str = "",
    paper_title: str = "",
    paper_year: str = "",
    domain: str = "",
    first_seen_year: str = "",
) -> dict[str, Any]:
    from src.mcp_servers.knowledge_graph import add_paper_to_graph as _impl, get_store
    report = await _impl(
        text=text, pdf_path=pdf_path, paper_title=paper_title,
        paper_year=paper_year, domain=domain, first_seen_year=first_seen_year,
    )
    if report.lstrip().startswith("⚠️"):
        raise RuntimeError(f"add_paper_to_graph reported failure: {report[:300]}")
    store = get_store(domain)
    return {
        "status": "ok",
        "report": report,
        "node_count": store.node_count,
        "edge_count": store.edge_count,
        "domain": store.domain,
    }


async def query_knowledge(query: str, domain: str = "") -> dict[str, Any]:
    from src.mcp_servers.knowledge_graph import query_knowledge as _impl
    report = await _impl(query=query, domain=domain)
    return {"status": "ok", "report": report}


async def kg_semantic_precheck(
    query: str, domain: str = "", top_k: int = 5
) -> dict[str, Any]:
    """workflow 跑前在 KG 里搜一下，避免重复入库。

    返回 top-k 节点 + 一个 bool `looks_existing`（首条相似度 ≥ 0.55 视为已存在）。
    domain 留空走默认。
    """
    from src.mcp_servers.knowledge_graph import get_store
    store = get_store(domain)
    if store.node_count == 0:
        return {"status": "ok", "hits": [], "looks_existing": False, "node_count": 0}
    hits = store.semantic_search(query, top_k=top_k)
    looks_existing = bool(hits and hits[0].get("similarity", 0) >= 0.55)
    return {
        "status": "ok",
        "hits": hits,
        "looks_existing": looks_existing,
        "node_count": store.node_count,
    }


async def detect_gaps(domain: str = "") -> dict[str, Any]:
    # learning_path MCP 当前只用 legacy 单图路径；domain 暂忽略
    from src.mcp_servers.learning_path import detect_gaps as _impl
    report = await _impl()
    return {"status": "ok", "report": report}


async def analyze_knowledge(
    domain: str = "",
    focus_area: str = "",
    max_items: int = 10,
) -> dict[str, Any]:
    from src.mcp_servers.learning_path import analyze_knowledge as _impl
    report = await _impl(focus_area=focus_area, max_items=int(max_items))
    return {"status": "ok", "report": report}


# ------------------------------------------------------------------
# 注册
# ------------------------------------------------------------------


def register_default_tools() -> None:
    register_tool("ingest_local_pdf", ingest_local_pdf)
    register_tool("add_paper_to_graph", add_paper_to_graph)
    register_tool("query_knowledge", query_knowledge)
    register_tool("kg_semantic_precheck", kg_semantic_precheck)
    register_tool("detect_gaps", detect_gaps)
    register_tool("analyze_knowledge", analyze_knowledge)
    logger.info("default runtime tools registered")
