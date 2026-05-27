from __future__ import annotations

"""
knowledge_graph adapter —— 封装知识图谱抽取与查询能力。

关键点（直接修复「静默 0.0」问题）：
  add_paper_to_graph 跑 KnowledgeExtractor，若 extraction_confidence==0.0
  且 raw_llm_output 不是合法 JSON（即它其实是一段异常文本，如 MiniMax 400
  "description too long"），则置 ok=False / error_category="llm_api_error"，
  而不是伪装成「成功但空」的抽取。

所有操作都在 ctx.sandbox.graph_path 上进行 —— 绝不碰真实 data/knowledge_graph.json。
"""

import json

from .base import AdapterContext, ToolCallResult, register_adapter


def _looks_like_api_error(raw_output: str) -> bool:
    """raw_llm_output 在抽取失败时存的是 str(exception)。
    若它不是合法 JSON，就认为这是一段 API 错误文本而非真实的空抽取。"""
    if not raw_output:
        return True
    text = raw_output.strip()
    try:
        json.loads(text)
        return False
    except Exception:  # noqa: BLE001
        # 进一步：尝试宽松提取 { } 块
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                json.loads(text[start:end + 1])
                return False
            except Exception:  # noqa: BLE001
                pass
        return True


@register_adapter("add_paper_to_graph")
async def add_paper_to_graph(args: dict, ctx: AdapterContext) -> ToolCallResult:
    """从文本抽取实体/关系并写入隔离图谱。

    raw: {"nodes": [...], "edges": [...], "confidence": float,
          "node_count": int, "edge_count": int}
    """
    from pathlib import Path

    from src.knowledge.extractor import KnowledgeExtractor
    from src.knowledge.graph_store import KnowledgeGraphStore

    text = args.get("text", "")
    # v1.0 personal gold：tasks.jsonl 用 text_file 引用 fixture（项目根相对路径），
    # 避免把大段 PDF text 内嵌到 tasks.jsonl 里
    if not text:
        text_file = args.get("text_file", "")
        if text_file:
            fp = Path(text_file)
            if fp.exists():
                text = fp.read_text(encoding="utf-8")
    paper_title = args.get("paper_title", "")
    paper_year = args.get("paper_year", "")

    store = KnowledgeGraphStore(graph_path=ctx.sandbox.graph_path)
    extractor = KnowledgeExtractor()

    try:
        result = await extractor.extract_from_text(
            text=text, paper_title=paper_title, paper_year=paper_year,
        )
    except Exception as exc:  # noqa: BLE001 — extractor 理论上自吞异常，这里兜底
        return ToolCallResult.failure(
            "add_paper_to_graph", f"extractor raised: {exc}", "llm_api_error",
        )

    tokens_in = max(0, len(text) // 4)
    tokens_out = max(0, len(result.raw_llm_output or "") // 4)

    # —— 显式暴露 MiniMax API 报错（不伪装成空成功）——
    if result.extraction_confidence == 0.0 and not result.nodes and not result.edges:
        if _looks_like_api_error(result.raw_llm_output):
            return ToolCallResult(
                ok=False, tool="add_paper_to_graph",
                raw={"nodes": [], "edges": [], "confidence": 0.0,
                     "node_count": 0, "edge_count": 0,
                     "raw_llm_output": (result.raw_llm_output or "")[:500]},
                error=f"extraction returned 0 confidence; raw output looks like an "
                      f"API error: {(result.raw_llm_output or '')[:200]}",
                error_category="llm_api_error",
                tokens_in=tokens_in, tokens_out=tokens_out,
            )
        # 合法 JSON 但确实空 —— 归类为 empty_extraction（仍是失败，需要可见）
        return ToolCallResult(
            ok=False, tool="add_paper_to_graph",
            raw={"nodes": [], "edges": [], "confidence": 0.0,
                 "node_count": 0, "edge_count": 0},
            error="extraction produced no nodes or edges",
            error_category="empty_extraction",
            tokens_in=tokens_in, tokens_out=tokens_out,
        )

    # 写入隔离图谱
    for node in result.nodes:
        store.add_node(node)
    for edge in result.edges:
        store.add_edge(edge)
    if result.nodes or result.edges:
        store.save(ctx.sandbox.graph_path)

    return ToolCallResult(
        ok=True, tool="add_paper_to_graph",
        raw={
            "nodes": [n.to_dict() for n in result.nodes],
            "edges": [e.to_dict() for e in result.edges],
            "confidence": result.extraction_confidence,
            "node_count": result.node_count,
            "edge_count": result.edge_count,
        },
        text=f"{paper_title!r}: {result.node_count} nodes, {result.edge_count} edges, "
             f"conf={result.extraction_confidence}",
        tokens_in=tokens_in, tokens_out=tokens_out,
    )


@register_adapter("query_knowledge")
async def query_knowledge(args: dict, ctx: AdapterContext) -> ToolCallResult:
    """在隔离图谱（通常已 seed）上做检索。

    args:
      query             : 查询字符串
      top_k             : 主检索取多少（默认 10）
      mode              : "semantic" (默认, 用 store.semantic_search) | "substring" (旧的 search_nodes)
      expand_neighbors  : bool，是否对每个命中节点做 1-hop 邻居扩展（默认 True）
      neighbor_depth    : int，扩展深度（默认 1）

    raw 输出：
      matched_labels / matched_node_ids   : 直接命中（不含邻居扩展）
      retrieved / retrieved_domains       : 用于评测指标，**含邻居扩展**，按相似度+插入序去重
      direct_hits / expanded_hits         : 调试用计数
    """
    from src.knowledge.graph_store import KnowledgeGraphStore

    query = args.get("query", "")
    top_k = int(args.get("top_k", 10))
    mode = args.get("mode", "semantic")
    expand_neighbors = bool(args.get("expand_neighbors", True))
    neighbor_depth = int(args.get("neighbor_depth", 1))

    store = KnowledgeGraphStore(graph_path=ctx.sandbox.graph_path)

    # 主检索：semantic（默认）或 substring（兼容旧行为）
    if mode == "substring":
        nodes = store.search_nodes(query)
        direct_node_ids = [n.node_id for n in nodes][:top_k]
        direct_labels = [n.label for n in nodes][:top_k]
    else:
        hits = store.semantic_search(query, top_k=top_k)
        direct_node_ids = [h["node_id"] for h in hits]
        direct_labels = [h["label"] for h in hits]

    # 邻居扩展：对每个直接命中节点取 1-hop 邻居
    expanded_ids: list[str] = list(direct_node_ids)
    expanded_labels: list[str] = list(direct_labels)
    seen_ids: set[str] = set(direct_node_ids)

    if expand_neighbors:
        for nid in list(direct_node_ids):  # 拷贝，遍历时不变
            for neighbor_node, _edge in store.query_neighbors(nid, depth=neighbor_depth):
                if neighbor_node.node_id not in seen_ids:
                    seen_ids.add(neighbor_node.node_id)
                    expanded_ids.append(neighbor_node.node_id)
                    expanded_labels.append(neighbor_node.label)

    # 收集 domain（用于 cross_domain_leakage）
    def _domain_of(node_id: str) -> str:
        node = store.get_node(node_id)
        if node is None:
            return ""
        return str((node.properties or {}).get("domain", ""))

    expanded_domains = [_domain_of(nid) for nid in expanded_ids]

    return ToolCallResult(
        ok=True, tool="query_knowledge",
        raw={
            "matched_labels": direct_labels,        # 仅直接命中（旧字段语义保留）
            "matched_node_ids": direct_node_ids,
            "retrieved": expanded_labels,            # 评测用：含邻居
            "retrieved_domains": expanded_domains,
            "direct_hits": len(direct_node_ids),
            "expanded_hits": len(expanded_labels) - len(direct_node_ids),
            "mode": mode,
            "expand_neighbors": expand_neighbors,
        },
        text=f"query={query!r} mode={mode} direct={len(direct_node_ids)} "
             f"+neighbors={len(expanded_labels) - len(direct_node_ids)}",
    )


@register_adapter("get_related_concepts")
async def get_related_concepts(args: dict, ctx: AdapterContext) -> ToolCallResult:
    """取某概念的多跳邻居。

    raw: {"neighbors": [labels]}
    """
    from src.knowledge.graph_store import KnowledgeGraphStore

    concept = args.get("concept_name", "")
    depth = int(args.get("depth", 2))
    store = KnowledgeGraphStore(graph_path=ctx.sandbox.graph_path)
    # 先按关键词定位起点节点
    matches = store.search_nodes(concept)
    if not matches:
        return ToolCallResult(
            ok=True, tool="get_related_concepts",
            raw={"neighbors": []},
            text=f"concept {concept!r} not found in graph",
        )
    start = matches[0]
    neighbors = store.query_neighbors(start.node_id, depth=depth)
    labels = [node.label for node, _edge in neighbors]
    return ToolCallResult(
        ok=True, tool="get_related_concepts",
        raw={"neighbors": labels, "start": start.label},
        text=f"{concept!r} -> {len(labels)} related concepts",
    )


@register_adapter("get_graph_stats")
async def get_graph_stats(args: dict, ctx: AdapterContext) -> ToolCallResult:
    """隔离图谱的统计摘要。"""
    from src.knowledge.graph_store import KnowledgeGraphStore

    store = KnowledgeGraphStore(graph_path=ctx.sandbox.graph_path)
    stats = store.get_stats()
    return ToolCallResult(
        ok=True, tool="get_graph_stats",
        raw={"stats": stats,
             "node_count": stats.get("total_nodes", 0),
             "edge_count": stats.get("total_edges", 0)},
        text=f"{stats.get('total_nodes', 0)} nodes, {stats.get('total_edges', 0)} edges",
    )
