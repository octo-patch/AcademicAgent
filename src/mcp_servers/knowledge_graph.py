from __future__ import annotations

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""
ScholarMind - 知识图谱 MCP Server
====================================

功能：
  1. add_paper_to_graph    — 从论文文本中抽取知识并入图
  2. query_knowledge        — 自然语言查询知识图谱
  3. get_graph_stats         — 获取图谱统计摘要
  4. get_related_concepts    — 查询某概念的相关实体

技术架构：
  用户请求 → Claude Code → MCP Protocol → 本 Server
                                            ├── extractor.py (LLM 抽取)
                                            └── graph_store.py (NetworkX 存储)

工程要点：
  - 图谱持久化到 data/knowledge_graph.json
  - 每次 add_paper 后自动保存
  - 查询结果格式化为 Markdown，方便 Claude 直接展示给用户
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

from ._mcp_compat import make_fastmcp_or_stub
from src.knowledge.schema import NodeType, RelationType, KNOWN_DOMAINS, normalize_domain
from src.knowledge.graph_store import KnowledgeGraphStore
from src.knowledge.extractor import KnowledgeExtractor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ScholarMind.KnowledgeGraph")

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(os.getenv("SCHOLARMIND_DATA_DIR",
                          str(Path(__file__).parent.parent.parent / "data")))

# 默认活动 KG（domain），来自环境变量或 "llm_agent"
DEFAULT_DOMAIN = normalize_domain(os.getenv("SCHOLARMIND_ACTIVE_KG", "llm_agent"))

# 旧单图路径（向后兼容：迁移脚本未跑时，shared store 退化加载此文件）
LEGACY_GRAPH_PATH = DATA_DIR / "knowledge_graph.json"


def _path_for(domain: str) -> Path:
    """每个 domain 一个 JSON 文件。"""
    return DATA_DIR / f"kg_{domain}.json"


# ============================================================
# Store registry（lazy 多图谱）
# ============================================================
_STORES: dict[str, KnowledgeGraphStore] = {}


def get_store(domain: str = "") -> KnowledgeGraphStore:
    """获取（或懒加载）指定 domain 的 store。

    路径策略：
      - 标准路径 data/kg_<domain>.json 存在 → 用它
      - 否则若 domain=shared 且 legacy data/knowledge_graph.json 存在
        → 用 legacy 路径（向后兼容，迁移前过渡）
      - 否则用标准路径（首次写入会创建）
    """
    domain = normalize_domain(domain) if domain else DEFAULT_DOMAIN
    if domain in _STORES:
        return _STORES[domain]

    path = _path_for(domain)
    if not path.exists() and domain == "shared" and LEGACY_GRAPH_PATH.exists():
        logger.warning(
            "kg_shared.json 不存在，回退加载 legacy %s（请尽快跑 migrate_kg_to_dual.py）",
            LEGACY_GRAPH_PATH,
        )
        path = LEGACY_GRAPH_PATH

    store = KnowledgeGraphStore(graph_path=path, domain=domain)
    _STORES[domain] = store
    logger.info("store registered [%s]: %s (%d nodes / %d edges)",
                domain, path, store.node_count, store.edge_count)
    return store


# ============================================================
# 全局实例
# ============================================================
mcp = make_fastmcp_or_stub(
    "ScholarMind-KnowledgeGraph",
    instructions=(
        "学术知识图谱管理服务（多域：llm_agent / comms / shared）。"
        "支持从论文中自动抽取实体和关系、构建知识图谱、"
        "查询概念关联、检测知识盲区。"
        "所有数据基于真实论文内容，不编造信息。"
        f"默认活动 KG: {DEFAULT_DOMAIN}（可通过 domain 参数切换或设置 SCHOLARMIND_ACTIVE_KG 环境变量）。"
    ),
)

extractor = KnowledgeExtractor()

# 按 domain 缓存 EntityLinker。sentence-transformer encoder 与 per-store 的
# embedding cache 在多次 add_paper_to_graph 之间复用，避免每篇都重新加载模型。
from src.knowledge.entity_linker import EntityLinker as _EntityLinker

_LINKER_CACHE: dict[str, _EntityLinker] = {}


def _get_linker_for(graph_store) -> _EntityLinker:
    domain = getattr(graph_store, "domain", "shared")
    linker = _LINKER_CACHE.get(domain)
    if linker is None:
        linker = _EntityLinker(store=graph_store, enable_semantic=True)
        _LINKER_CACHE[domain] = linker
    else:
        # store 实例可能被 reset 重建过，刷新引用
        linker.store = graph_store
    return linker


# ============================================================
# MCP Tools
# ============================================================


@mcp.tool()
async def add_paper_to_graph(
    text: str = "",
    pdf_path: str = "",
    paper_title: str = "",
    paper_year: str = "",
    domain: str = "",
    first_seen_year: str = "",
) -> str:
    """
    从论文文本中抽取知识实体和关系，添加到知识图谱中。

    适合使用的场景：
    - 用户说"把这篇论文加入知识图谱"
    - 需要积累论文中的关键概念、方法、评估指标
    - 构建个人学术知识网络

    两种输入方式（任选其一）：
    1. pdf_path: 提供本地 PDF 文件路径，Server 自动提取文本（推荐，避免传输大量文本）
    2. text: 直接提供论文文本内容（适合只传摘要或特定章节）

    Args:
        text: 论文文本内容（摘要、全文或特定章节均可）
        pdf_path: 本地 PDF 文件路径（推荐方式，Server 内部提取文本）
        paper_title: 论文标题（用于溯源）
        paper_year: 论文年份
        domain: 目标知识图谱域，"llm_agent" / "comms" / "shared"。
                空字符串则用默认（环境变量 SCHOLARMIND_ACTIVE_KG，默认 llm_agent）。
        first_seen_year: 可选，手动指定"你首次看到这篇论文的年份"，覆盖 LLM 自推断。
                格式 "YYYY" 或 "YYYY-MM-DD"。常用于"论文是 2017 发表，但我 2026 才读"的场景。

    Returns:
        抽取结果摘要 + 入图统计 + 解析告警（若有）
    """
    graph_store = get_store(domain)

    # 解析 first_seen_year override
    override_year: int | None = None
    if first_seen_year:
        try:
            override_year = int(first_seen_year.split("-")[0].strip())
        except (ValueError, AttributeError):
            return (
                f"⚠️ **first_seen_year 格式无效**: {first_seen_year!r}\n\n"
                f"请用 YYYY (如 2026) 或 YYYY-MM-DD (如 2026-01-20)。"
            )
    # 如果提供了 pdf_path，从 PDF 中提取文本
    if pdf_path and not text:
        from src.core.pdf_parser import PDFParser
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return f"⚠️ **PDF 文件不存在**: {pdf_path}"
        try:
            with PDFParser(str(pdf_file)) as parser:
                text = parser.extract_full_text()
                if not paper_title:
                    meta = parser.get_metadata()
                    paper_title = meta.title or ""
            logger.info(f"从 PDF 提取文本: {len(text)} 字符, title='{paper_title}'")
        except Exception as e:
            return f"⚠️ **PDF 解析失败**: {type(e).__name__}: {e}"

    logger.info(f"知识抽取请求: title='{paper_title}', text_len={len(text)}")

    if len(text) < 50:
        return (
            "⚠️ **文本过短**（少于 50 字符）。\n\n"
            "请提供 pdf_path 参数指向本地 PDF 文件，或通过 text 参数提供论文摘要/正文。"
        )


    try:
        # Step 1: LLM 抽取（带 entity-linker 语义兜底）
        linker = _get_linker_for(graph_store)
        try:
            result = await extractor.extract_from_text(
                text=text,
                paper_title=paper_title,
                paper_year=paper_year,
                first_seen_year_override=override_year,
                linker=linker,
            )
        except Exception as parse_err:
            # ExtractionError 或其它 LLM 调用异常：明确报告失败，不再静默"零节点"
            from src.knowledge.extractor import ExtractionError
            if isinstance(parse_err, ExtractionError):
                return (
                    f"⚠️ **LLM 输出解析失败**\n\n"
                    f"详情：{parse_err}\n\n"
                    f"建议：1) 检查 MINIMAX_MODEL 是否为 reasoning 模型；2) 重试一次；3) 看 logs/。"
                )
            raise

        if not result.nodes:
            return (
                "⚠️ **未抽取到有效实体**。\n\n"
                "可能原因：\n"
                "- 文本不包含可识别的学术实体\n"
                "- 文本过短或内容不够具体\n"
                "- 建议提供论文的 Abstract + Introduction + Method 章节"
                + (f"\n\n**解析警告** ({len(result.parse_warnings)}):\n"
                   + "\n".join(f"- {w}" for w in result.parse_warnings[:5])
                   if result.parse_warnings else "")
            )

        # Step 2: 写入图谱
        added_nodes = 0
        added_edges = 0

        for node in result.nodes:
            node_id = graph_store.add_node(node)
            if node_id:
                added_nodes += 1

        for edge in result.edges:
            edge_id = graph_store.add_edge(edge)
            if edge_id:
                added_edges += 1

        # Step 3: 持久化
        graph_store.save()

        # Step 4: 格式化输出
        summary = result.to_summary()
        graph_summary = (
            f"\n### 📈 图谱更新 [{graph_store.domain}]\n"
            f"- 新增/合并节点: **{added_nodes}**\n"
            f"- 新增关系: **{added_edges}**\n"
            f"- 图谱总节点: **{graph_store.node_count}**\n"
            f"- 图谱总关系: **{graph_store.edge_count}**\n"
        )
        if override_year:
            graph_summary += f"- first_seen_year 手动覆盖: **{override_year}**\n"
        if result.had_parse_fallback:
            graph_summary += (
                f"- ⚠️ **解析降级**：严格 Pydantic 失败，走宽松 fallback "
                f"(置信度已自动 ×0.8 = {result.extraction_confidence:.2f})\n"
            )
        if result.parse_warnings:
            graph_summary += f"- 解析警告 ({len(result.parse_warnings)})\n"
        graph_summary += "\n### 🔍 抽取的实体\n"

        for node in result.nodes:
            graph_summary += (
                f"- **{node.label}** ({node.node_type.value})"
            )
            if node.properties:
                props = ", ".join(
                    f"{k}={v}"
                    for k, v in node.properties.items()
                    if isinstance(v, (str, int, float)) and v
                )
                if props:
                    graph_summary += f" — {props}"
            graph_summary += "\n"

        graph_summary += "\n### 🔗 抽取的关系\n"
        for edge in result.edges:
            source_node = graph_store.get_node(edge.source_id)
            target_node = graph_store.get_node(edge.target_id)
            source_label = source_node.label if source_node else edge.source_id
            target_label = target_node.label if target_node else edge.target_id
            graph_summary += (
                f"- {source_label} **—{edge.relation_type.value}→** "
                f"{target_label}\n"
            )

        logger.info(
            f"知识入图完成: +{added_nodes} 节点, +{added_edges} 边"
        )
        return summary + graph_summary

    except Exception as e:
        logger.exception(f"知识抽取失败: {e}")
        return f"⚠️ **知识抽取失败**: {type(e).__name__}: {e}"


@mcp.tool()
async def query_knowledge(query: str, domain: str = "") -> str:
    """
    查询知识图谱中的信息。

    适合使用的场景：
    - 用户问"ReAct 和什么方法相关？"
    - 用户问"我读过的论文中有哪些概念？"
    - 需要查找某个方法/概念在知识图谱中的位置

    Args:
        query: 查询关键词或概念名称
        domain: 目标知识图谱域，留空用默认

    Returns:
        查询结果（匹配的节点和关系）
    """
    graph_store = get_store(domain)
    logger.info(f"知识查询 [{graph_store.domain}]: '{query}'")

    if graph_store.node_count == 0:
        return (
            "📭 **知识图谱为空**\n\n"
            "还没有任何论文被加入图谱。\n"
            "使用 `add_paper_to_graph` 工具添加论文内容。"
        )

    # 搜索匹配节点
    matching_nodes = graph_store.search_nodes(query)

    if not matching_nodes:
        return (
            f"🔍 未找到与 \"{query}\" 相关的实体。\n\n"
            f"当前图谱包含 {graph_store.node_count} 个节点。\n"
            f"尝试使用更具体的关键词，或使用 `get_graph_stats` 查看图谱概览。"
        )

    result = f"## 🔍 查询结果: \"{query}\"\n\n"
    result += f"找到 **{len(matching_nodes)}** 个匹配实体：\n\n"

    for node in matching_nodes[:10]:  # 最多展示10个
        result += f"### 📌 {node.label} ({node.node_type.value})\n"

        # 属性
        if node.properties:
            for key, value in node.properties.items():
                if isinstance(value, (str, int, float)) and value:
                    result += f"- **{key}**: {value}\n"

        # 关联关系
        neighbors = graph_store.query_neighbors(node.node_id, depth=1)
        if neighbors:
            result += "\n**关联实体**：\n"
            for n_node, n_edge in neighbors[:8]:
                direction = "→" if n_edge.source_id == node.node_id else "←"
                result += (
                    f"  - {direction} **{n_node.label}** "
                    f"({n_node.node_type.value}) "
                    f"[{n_edge.relation_type.value}]\n"
                )

        result += "\n---\n\n"

    return result


@mcp.tool()
async def get_graph_stats(domain: str = "") -> str:
    """
    获取知识图谱的统计摘要。

    适合使用的场景：
    - 用户问"我的知识图谱现在是什么样的？"
    - 需要了解当前积累了多少知识
    - 查看知识图谱的整体概览

    Args:
        domain: 目标知识图谱域，留空用默认。
                传 "all" 可查看所有 domain 的汇总（暂时返回每个 domain 单独的摘要）。

    Returns:
        图谱统计信息（节点/边数量、类型分布、核心节点等）
    """
    # all: 聚合所有已知 domain 的简要统计
    if domain.strip().lower() == "all":
        parts: list[str] = ["## 📚 全部 KG 域统计\n"]
        for d in KNOWN_DOMAINS:
            s = get_store(d)
            parts.append(
                f"### [{d}] {s.node_count} 节点 / {s.edge_count} 边\n"
            )
        return "\n".join(parts)
    graph_store = get_store(domain)
    logger.info(f"获取图谱统计 [{graph_store.domain}]")

    if graph_store.node_count == 0:
        return (
            "📭 **知识图谱为空**\n\n"
            "还没有任何论文被加入图谱。\n"
            "使用 `add_paper_to_graph` 工具添加论文内容。"
        )

    return graph_store.to_markdown()


@mcp.tool()
async def get_related_concepts(
    concept_name: str,
    depth: int = 2,
    domain: str = "",
) -> str:
    """
    查询某个概念的关联实体网络。

    与 query_knowledge 的区别：
    - query_knowledge: 关键词搜索，返回匹配的节点
    - get_related_concepts: 图遍历，返回多跳关联的完整子图

    适合使用的场景：
    - 用户问"和 Agent Memory 相关的概念有哪些？"
    - 需要探索知识图谱中某个概念的周边网络
    - 了解某个方法的上下游关系

    Args:
        concept_name: 概念名称（如 "ReAct", "Agent Memory"）
        depth: 查询深度（1=直接相关, 2=两跳关联, 默认2）
        domain: 目标知识图谱域，留空用默认

    Returns:
        概念的关联网络描述
    """
    graph_store = get_store(domain)
    logger.info(f"关联概念查询 [{graph_store.domain}]: '{concept_name}', depth={depth}")

    # 先搜索到目标节点
    matching = graph_store.search_nodes(concept_name)

    if not matching:
        return (
            f"🔍 未找到概念 \"{concept_name}\"。\n\n"
            f"尝试使用不同的拼写或缩写。"
        )

    # 取最佳匹配
    target = matching[0]
    result = (
        f"## 🌐 关联概念网络: {target.label}\n\n"
        f"**类型**: {target.node_type.value}\n"
        f"**查询深度**: {depth} 跳\n\n"
    )

    # 多跳查询
    neighbors = graph_store.query_neighbors(target.node_id, depth=depth)

    if not neighbors:
        result += "该概念目前没有关联实体。\n"
        result += "建议添加更多论文来丰富知识图谱。\n"
        return result

    # 按类型分组展示
    by_type: dict[str, list] = {}
    for n_node, n_edge in neighbors:
        type_name = n_node.node_type.value
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append((n_node, n_edge))

    for type_name, items in by_type.items():
        result += f"### {type_name} ({len(items)})\n"
        for n_node, n_edge in items:
            direction = "→" if n_edge.source_id == target.node_id else "←"
            result += (
                f"- {direction} **{n_node.label}** "
                f"[{n_edge.relation_type.value}]"
            )
            if n_edge.confidence < 0.7:
                result += f" ⚠️低置信度({n_edge.confidence:.2f})"
            result += "\n"
        result += "\n"

    result += (
        f"---\n"
        f"📊 共找到 **{len(neighbors)}** 个关联实体 "
        f"(涵盖 {len(by_type)} 种类型)\n"
    )
    return result


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    logger.info("ScholarMind Knowledge Graph MCP Server 启动中...")
    logger.info(f"默认活动 KG: {DEFAULT_DOMAIN}")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    default_store = get_store(DEFAULT_DOMAIN)
    logger.info(
        f"默认 KG [{DEFAULT_DOMAIN}]: {default_store.node_count} 节点, "
        f"{default_store.edge_count} 边"
    )
    mcp.run()
