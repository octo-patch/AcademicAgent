from __future__ import annotations

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""
ScholarMind - 论文搜索 MCP Server
=================================

功能：
  1. search_papers       — 关键词搜索论文（Semantic Scholar API）
  2. get_paper_details    — 获取单篇论文详细信息
  3. get_related_papers   — 获取引用/被引论文
  4. search_arxiv         — 搜索 arXiv 预印本

技术要点（学习笔记）：
  - 这是一个 MCP Server，作为独立进程运行
  - Claude Code 通过 stdio 与它通信
  - @mcp.tool() 装饰器注册工具函数
  - 函数的 docstring 决定了 Claude 何时调用这个工具
  - type hints 决定了参数的 JSON Schema

工程设计原则：
  1. 所有论文数据来自真实 API，绝不由 LLM 生成 → 防止幻觉
  2. 每篇论文包含可验证的唯一标识（Paper ID / DOI / arXiv ID）
  3. 错误信息返回人类可读文本，让 Claude 能据此调整策略
  4. 请求限流 + 超时控制 → 防止天价账单和无限阻塞
"""

import os
import json
import asyncio
import logging
from typing import Optional

import httpx
from dotenv import load_dotenv

from ._mcp_compat import make_fastmcp_or_stub

# ============================================================
# 环境加载与日志配置
# ============================================================
load_dotenv()  # 从 .env 文件加载 API Key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ScholarMind.PaperSearch")

# ============================================================
# MCP Server 初始化
# ============================================================
mcp = make_fastmcp_or_stub(
    "ScholarMind-PaperSearch",
    instructions=(
        "学术论文搜索与检索服务。"
        "通过 Semantic Scholar API 和 arXiv API 搜索真实论文，"
        "确保每篇论文都可以通过 DOI/arXiv ID 验证真实性。"
    ),
)

# ============================================================
# 常量配置
# ============================================================
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
# 【踩坑记录】arXiv API 已从 HTTP 迁移到 HTTPS
ARXIV_API_BASE = "https://export.arxiv.org/api/query"

# 请求配置
# 【说明】Semantic Scholar API Key 申请被拒，项目统一使用无 Key 模式。
# 无 Key 模式限额 1 req/s，通过主动限速 + Exponential Backoff 双重保护。
REQUEST_TIMEOUT = 30.0  # 秒
MAX_RESULTS_PER_SEARCH = 10  # 单次搜索最大返回数

# 重试配置
# 【工程思考】Exponential Backoff 是外部 API 集成的标准模式
# 初始等 1 秒，然后 2 秒、4 秒...指数增长
# 这样既能快速重试偶发错误，又不会在持续限流时频繁骚扰 API
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # 秒

# ============================================================
# 主动限速器 (Proactive Rate Limiter)
#
# 【踩坑记录】Semantic Scholar 无 Key 模式限额仅 1 req/s。
# 如果只靠被动的 Exponential Backoff（429 触发后才等待），
# 用户连续调用两次 MCP 工具就会立即触发限流。
#
# 解决方案：主动限速 — 在每次请求前检查距上次请求的时间间隔，
# 不足则自动 sleep。这比被动 backoff 更友好：
#   - 被动策略：请求 → 429 → 等 1s → 请求 → 429 → 等 2s（用户感知到报错）
#   - 主动策略：请求 → 等 1.5s → 请求 → 成功（用户无感知）
# ============================================================
import time

# 无 Key 模式: 限额 1 req/s → 保守设 1.5s 间隔
MIN_REQUEST_INTERVAL = 1.5
_last_request_time: float = 0.0  # 上次请求的时间戳

# 【修复竞态条件】asyncio.Lock 确保"读时间→判断→等待→写时间"是原子操作
# 没有这个锁，两个并发任务可能同时读到旧的 _last_request_time，
# 都判断"不用等"，然后同时发请求 → 触发 429。
# 这是经典的 TOCTOU (Time-of-Check-Time-of-Use) 竞态。
_rate_limit_lock = asyncio.Lock()


async def _rate_limit_wait():
    """主动限速：确保两次请求间隔不低于 MIN_REQUEST_INTERVAL（并发安全）"""
    global _last_request_time
    async with _rate_limit_lock:  # ← 加锁，同一时刻只有一个任务能进入
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            wait = MIN_REQUEST_INTERVAL - elapsed
            logger.debug(f"主动限速: 等待 {wait:.1f}s")
            await asyncio.sleep(wait)
        _last_request_time = time.monotonic()


# ============================================================
# HTTP 请求工具函数
#
# 【工程思考】为什么封装统一请求函数？
# 1. 集中处理 API Key 注入
# 2. 统一超时和错误处理
# 3. 主动限速 + 被动重试双重保护
# ============================================================
async def _semantic_scholar_request(
    endpoint: str, params: dict | None = None
) -> dict:
    """
    向 Semantic Scholar API 发送请求（带主动限速 + Exponential Backoff 重试）

    双重限流保护：
    1. 主动：请求前自动等待至 MIN_REQUEST_INTERVAL（无感知）
    2. 被动：遇到 429 时 Exponential Backoff 重试（1s → 2s → 4s）

    Raises:
        httpx.HTTPStatusError: 重试耗尽后仍失败
        httpx.TimeoutException: 请求超时
    """
    # 主动限速（在发请求之前等待，从源头避免 429）
    await _rate_limit_wait()

    url = f"{SEMANTIC_SCHOLAR_BASE}/{endpoint}"
    headers = {}

    last_exception = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT, follow_redirects=True
            ) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code == 429 and attempt < MAX_RETRIES:
                wait_time = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"API 限流 (429), 等待 {wait_time}s 后重试 "
                    f"(第 {attempt + 1}/{MAX_RETRIES} 次)"
                )
                await asyncio.sleep(wait_time)
                _last_request_time = time.monotonic()  # 重试后刷新时间戳
                continue
            raise

    raise last_exception  # type: ignore


def _format_authors(authors: list[dict], max_display: int = 3) -> str:
    """格式化作者列表，超过 max_display 人时显示 et al."""
    names = [a.get("name", "Unknown") for a in authors[:max_display]]
    result = ", ".join(names)
    if len(authors) > max_display:
        result += " et al."
    return result


def _truncate(text: str | None, max_len: int = 300) -> str:
    """安全截断文本，处理 None"""
    if not text:
        return "（无）"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ============================================================
# MCP 工具定义
#
# 【工程思考】工具设计三原则：
# 1. docstring 越详细越好 → Claude 靠它决定何时调用
# 2. 返回格式化 markdown → Claude 能直接展示给用户
# 3. 异常返回友好提示 → Claude 能据此调整策略（如换关键词）
# ============================================================


@mcp.tool()
async def search_papers(
    query: str,
    limit: int = 5,
    year_range: str | None = None,
    fields_of_study: str | None = None,
) -> str:
    """
    通过 Semantic Scholar API 搜索学术论文。**备选搜索源**——首选 `search_arxiv`。

    ⚠️ **优先级提示** (Phase 4.1)：本工具无 API Key 时极易触发 429 限流（1 req/s）；
    建议默认走 `search_arxiv`（公开 API 无限速）或 `search_crossref`（DOI 元数据强）。
    仅在以下场景才用本工具：
    - 需要引用数（citationCount）/ TL;DR / 完整 venue 信息
    - 已经有 Paper ID 想拿详情（→ 用 get_paper_details）
    - arXiv + CrossRef 都搜不到的论文

    不适合使用的场景：
    - 用户已经提供了论文 PDF（→ 用 ingest_local_pdf）
    - 用户在讨论已知论文的细节

    Args:
        query: 搜索关键词。使用英文效果最好。
               例如 "LLM Agent memory"、"agent tool use" 或 "multi-agent collaboration"
        limit: 返回论文数量，1-10 之间，默认 5
        year_range: 可选的发表年份过滤，格式 "2020-2025" 或 "2024-"
        fields_of_study: 可选的学科过滤，如 "Computer Science"、"Engineering"

    Returns:
        格式化的论文搜索结果（Markdown），每篇包含标题、作者、年份、引用数、
        Paper ID 和摘要摘录。
    """
    logger.info(f"搜索论文: query='{query}', limit={limit}, year='{year_range}'")

    # 参数校验
    limit = max(1, min(limit, MAX_RESULTS_PER_SEARCH))

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,citationCount,abstract,url,venue,externalIds",
    }
    if year_range:
        params["year"] = year_range
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    try:
        data = await _semantic_scholar_request("paper/search", params)
        papers = data.get("data", [])

        if not papers:
            return (
                f"## 未找到结果\n\n"
                f"关键词 \"{query}\" 未找到相关论文。\n"
                f"**建议**：\n"
                f"- 尝试使用更通用的英文关键词\n"
                f"- 减少过滤条件（如去掉年份限制）\n"
                f"- 尝试同义词（如 'agent memory' → 'long-term memory'）"
            )

        # 格式化结果
        results = []
        for i, paper in enumerate(papers, 1):
            authors = _format_authors(paper.get("authors", []))
            abstract = _truncate(paper.get("abstract"), 250)

            # 提取可验证的外部 ID
            ext_ids = paper.get("externalIds") or {}
            doi = ext_ids.get("DOI", "")
            arxiv_id = ext_ids.get("ArXiv", "")
            paper_id = paper.get("paperId", "N/A")

            # 构建验证链接
            verify_links = []
            if doi:
                verify_links.append(f"[DOI](https://doi.org/{doi})")
            if arxiv_id:
                verify_links.append(f"[arXiv](https://arxiv.org/abs/{arxiv_id})")
            verify_str = " | ".join(verify_links) if verify_links else "无外部链接"

            result = (
                f"### [{i}] {paper.get('title', 'N/A')}\n"
                f"- **作者**: {authors}\n"
                f"- **年份**: {paper.get('year', 'N/A')} "
                f"| **期刊/会议**: {paper.get('venue', 'N/A') or '未知'}\n"
                f"- **引用数**: {paper.get('citationCount', 0)}\n"
                f"- **Paper ID**: `{paper_id}`\n"
                f"- **验证链接**: {verify_str}\n"
                f"- **摘要**: {abstract}"
            )
            results.append(result)

        total = data.get("total", len(papers))
        header = (
            f"## 🔍 搜索结果: \"{query}\"\n\n"
            f"共找到约 {total} 篇相关论文，展示前 {len(papers)} 篇：\n"
        )

        logger.info(f"搜索完成: 返回 {len(papers)} 篇论文 (共约 {total} 篇)")
        return header + "\n\n---\n\n".join(results)

    except httpx.TimeoutException:
        logger.warning(f"搜索超时: query='{query}'")
        # 【异常反馈闭环】错误消息面向 Agent 而非用户，引导 LLM 自动降级
        return (
            "⚠️ **Semantic Scholar 搜索超时**\n\n"
            "Semantic Scholar API 响应超时（30秒），网络不稳定或服务繁忙。\n\n"
            "**请立即调用 `search_arxiv` 工具进行备选搜索**，"
            f"使用相同的查询词: \"{query}\"。\n"
            "arXiv API 通常更稳定，且覆盖了大多数计算机科学与大模型 Agent 相关论文。"
        )
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        logger.error(f"API 错误: status={status}, query='{query}'")
        if status == 429:
            # 【异常反馈闭环】429 限流 → 引导 Agent 无缝切换到 arXiv
            return (
                "⚠️ **Semantic Scholar API 限流 (HTTP 429)**\n\n"
                "当前请求频率超过 Semantic Scholar 的免费限额（1 req/s）。\n\n"
                "**请立即调用 `search_arxiv` 工具作为备选**，"
                f"使用相同的查询词: \"{query}\"。\n"
                "arXiv 搜索无限流限制，可以直接获取结果。"
            )
        return f"⚠️ **API 错误** (HTTP {status})\n\n请稍后重试。"
    except Exception as e:
        logger.exception(f"未知错误: {e}")
        return f"⚠️ **未知错误**: {type(e).__name__}: {e}"


@mcp.tool()
async def get_paper_details(paper_id: str) -> str:
    """
    获取一篇论文的完整详细信息。需要先通过 search_papers 获取 Paper ID。

    适合使用的场景：
    - 用户想深入了解搜索结果中的某篇论文
    - 需要获取论文的完整摘要、PDF 链接或引用信息

    Args:
        paper_id: Semantic Scholar 的论文 ID（从 search_papers 结果中获取的 Paper ID）

    Returns:
        论文的完整详细信息，包括摘要、TL;DR、引用统计、PDF 链接和外部 ID。
    """
    logger.info(f"获取论文详情: paper_id='{paper_id}'")

    fields = (
        "title,authors,year,abstract,citationCount,referenceCount,"
        "url,venue,publicationDate,fieldsOfStudy,tldr,"
        "openAccessPdf,externalIds"
    )

    try:
        paper = await _semantic_scholar_request(
            f"paper/{paper_id}", {"fields": fields}
        )

        authors = _format_authors(paper.get("authors", []), max_display=10)

        # TL;DR (Semantic Scholar 的 AI 生成摘要)
        tldr_obj = paper.get("tldr")
        tldr_text = tldr_obj.get("text", "无自动摘要") if tldr_obj else "无自动摘要"

        # PDF 链接
        pdf_obj = paper.get("openAccessPdf")
        pdf_url = pdf_obj.get("url", "无公开 PDF") if pdf_obj else "无公开 PDF"

        # 外部标识（用于验证论文真实性）
        ext_ids = paper.get("externalIds") or {}
        doi = ext_ids.get("DOI", "N/A")
        arxiv_id = ext_ids.get("ArXiv", "N/A")

        # 研究领域
        fields_list = paper.get("fieldsOfStudy") or []
        fields_str = ", ".join(fields_list) if fields_list else "未分类"

        result = (
            f"## 📄 {paper.get('title', 'N/A')}\n\n"
            f"| 属性 | 值 |\n"
            f"|:---|:---|\n"
            f"| **作者** | {authors} |\n"
            f"| **年份** | {paper.get('year', 'N/A')} |\n"
            f"| **发表于** | {paper.get('venue', 'N/A') or '未知'} |\n"
            f"| **发表日期** | {paper.get('publicationDate', 'N/A')} |\n"
            f"| **引用数** | {paper.get('citationCount', 0)} |\n"
            f"| **参考文献数** | {paper.get('referenceCount', 0)} |\n"
            f"| **研究领域** | {fields_str} |\n"
            f"| **DOI** | {doi} |\n"
            f"| **arXiv ID** | {arxiv_id} |\n"
            f"| **PDF 链接** | {pdf_url} |\n"
            f"| **Semantic Scholar** | {paper.get('url', 'N/A')} |\n\n"
            f"### TL;DR\n{tldr_text}\n\n"
            f"### 完整摘要\n{paper.get('abstract', '无摘要')}"
        )

        logger.info(f"获取详情成功: '{paper.get('title', '')[:50]}'")
        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"⚠️ **论文未找到**\n\nPaper ID `{paper_id}` 不存在。请检查 ID 是否正确。"
        return f"⚠️ **API 错误** (HTTP {e.response.status_code})"
    except Exception as e:
        logger.exception(f"获取详情失败: {e}")
        return f"⚠️ **获取失败**: {type(e).__name__}: {e}"


@mcp.tool()
async def get_related_papers(
    paper_id: str,
    relation: str = "references",
    limit: int = 5,
) -> str:
    """
    获取与指定论文相关的论文列表（引用或被引用关系）。

    适合使用的场景：
    - 用户想扩展阅读某篇论文的相关工作
    - 追溯某篇论文的理论基础（references）
    - 查看某篇论文的影响力和后续发展（citations）

    Args:
        paper_id: 源论文的 Semantic Scholar Paper ID
        relation: 关系类型
                  - "references": 该论文引用了谁（向前追溯，了解理论基础）
                  - "citations": 谁引用了该论文（向后追踪，了解后续发展）
        limit: 返回数量，1-10，默认 5

    Returns:
        相关论文列表，按引用数降序排列。
    """
    logger.info(
        f"获取相关论文: paper_id='{paper_id}', relation='{relation}', limit={limit}"
    )

    if relation not in ("references", "citations"):
        return "⚠️ relation 参数必须是 'references' 或 'citations'"

    limit = max(1, min(limit, MAX_RESULTS_PER_SEARCH))
    endpoint = f"paper/{paper_id}/{relation}"
    params = {
        "fields": "title,authors,year,citationCount,abstract",
        "limit": limit,
    }

    try:
        data = await _semantic_scholar_request(endpoint, params)
        items = data.get("data", [])

        if not items:
            relation_cn = "引用的论文" if relation == "references" else "引用它的论文"
            return f"未找到该论文{relation_cn}。"

        # 提取论文对象（references 和 citations 的数据结构略有不同）
        key = "citedPaper" if relation == "references" else "citingPaper"

        results = []
        for i, item in enumerate(items, 1):
            paper = item.get(key, {})
            if not paper or not paper.get("title"):
                continue

            authors = _format_authors(paper.get("authors", []))
            abstract = _truncate(paper.get("abstract"), 150)

            result = (
                f"**[{i}]** {paper.get('title', 'N/A')}\n"
                f"   - 作者: {authors} | 年份: {paper.get('year', 'N/A')} "
                f"| 引用数: {paper.get('citationCount', 0)}\n"
                f"   - 摘要: {abstract}"
            )
            results.append(result)

        relation_cn = "参考文献（该论文引用的）" if relation == "references" else "后续引用（引用该论文的）"
        header = f"## 📚 {relation_cn}\n\n"

        logger.info(f"获取相关论文完成: 返回 {len(results)} 篇")
        return header + "\n\n".join(results)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"⚠️ Paper ID `{paper_id}` 不存在。"
        return f"⚠️ API 错误 (HTTP {e.response.status_code})"
    except Exception as e:
        logger.exception(f"获取相关论文失败: {e}")
        return f"⚠️ **获取失败**: {type(e).__name__}: {e}"


@mcp.tool()
async def search_arxiv(
    query: str,
    limit: int = 5,
    sort_by: str = "relevance",
    category: str = "",
    author: str = "",
    recent_days: int = 0,
) -> str:
    """
    通过 arXiv API 搜索预印本论文。**推荐作为搜索主路径**（公开 API 无 Key 无限速）。

    Phase 4.1 增强：
    - category 过滤（cs.AI / cs.CL / cs.LG / cs.IR / eess.SP ...）
    - author 过滤（作者名）
    - recent_days 过滤（最近 N 天的提交，配合 sortBy="submittedDate"）

    适合使用的场景：
    - 用户找 LLM Agent / 通信 / SLAM 等领域的论文（默认走这里）
    - 想要最新预印本
    - 按作者列搜某个研究者发表

    与 search_papers 的对比：
    - **search_arxiv（本工具）**：公开无 key，覆盖 CS/物理/数学/工程
    - **search_papers (Semantic Scholar)**：备选，无 API key 易触 429；引用数据完整时调它

    Args:
        query: 关键词（英文最佳）
        limit: 返回数量，1-10
        sort_by: relevance / lastUpdatedDate / submittedDate
        category: arXiv 分类前缀，如 "cs.AI" / "cs.CL" / "eess.SP"。空则不过滤
        author: 作者姓名（部分匹配）
        recent_days: 仅返回最近 N 天提交的论文。>0 时建议 sort_by="submittedDate"

    Returns:
        arXiv 论文列表（标题/作者/arXiv ID/摘要/PDF 链接）
    """
    logger.info(
        f"arXiv 搜索: q='{query}', limit={limit}, cat={category!r}, "
        f"author={author!r}, recent_days={recent_days}"
    )

    limit = max(1, min(limit, MAX_RESULTS_PER_SEARCH))

    # 构造 search_query（arXiv 用 AND 组合多字段）
    parts = []
    if query:
        parts.append(f"all:{query}")
    if category:
        parts.append(f"cat:{category}")
    if author:
        parts.append(f"au:{author}")
    if not parts:
        return "⚠️ 至少提供 query / category / author 中的一个参数"

    search_query = " AND ".join(parts)
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": limit if recent_days <= 0 else min(limit * 3, 30),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT, follow_redirects=True
        ) as client:
            response = await client.get(ARXIV_API_BASE, params=params)
            response.raise_for_status()
            xml_text = response.text

        # 简易 XML 解析（arXiv 返回 Atom XML）
        # 【工程思考】为什么不用 lxml？
        # 减少依赖。arXiv 的 XML 结构简单，用标准库 xml 足够。
        # 如果后续需要复杂解析，再引入 feedparser 或 lxml。
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entries = root.findall("atom:entry", ns)

        # recent_days 后过滤
        if recent_days > 0 and entries:
            from datetime import datetime as _dt, timedelta as _td, timezone as _tz
            cutoff = _dt.now(_tz.utc) - _td(days=recent_days)
            filtered = []
            for e in entries:
                pub = e.findtext("atom:published", "", ns)[:10]
                if pub:
                    try:
                        d = _dt.strptime(pub, "%Y-%m-%d").replace(tzinfo=_tz.utc)
                        if d >= cutoff:
                            filtered.append(e)
                    except ValueError:
                        pass
            entries = filtered[:limit]

        if not entries:
            note = ""
            if recent_days > 0:
                note = f"（最近 {recent_days} 天内无新提交）"
            return f"## arXiv 搜索未找到结果\n\n关键词: \"{query}\" {note}"

        results = []
        for i, entry in enumerate(entries, 1):
            title = entry.findtext("atom:title", "N/A", ns).strip().replace("\n", " ")

            # 提取作者
            author_elems = entry.findall("atom:author/atom:name", ns)
            authors = ", ".join(a.text for a in author_elems[:3])
            if len(author_elems) > 3:
                authors += " et al."

            # 提取 arXiv ID
            entry_id = entry.findtext("atom:id", "", ns)
            arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id

            # 发布日期
            published = entry.findtext("atom:published", "N/A", ns)[:10]

            # 摘要
            summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
            summary = _truncate(summary, 250)

            # PDF 链接
            pdf_link = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_link = link.get("href", "")
                    break

            result = (
                f"### [{i}] {title}\n"
                f"- **作者**: {authors}\n"
                f"- **发布日期**: {published}\n"
                f"- **arXiv ID**: `{arxiv_id}`\n"
                f"- **链接**: [arXiv]({entry_id})"
                f"{f' | [PDF]({pdf_link})' if pdf_link else ''}\n"
                f"- **摘要**: {summary}"
            )
            results.append(result)

        header = f"## 📄 arXiv 搜索结果: \"{query}\"\n\n展示 {len(results)} 篇：\n"

        logger.info(f"arXiv 搜索完成: 返回 {len(results)} 篇")
        return header + "\n\n---\n\n".join(results)

    except httpx.TimeoutException:
        return "⚠️ **arXiv 搜索超时**，请稍后重试。"
    except Exception as e:
        logger.exception(f"arXiv 搜索失败: {e}")
        return f"⚠️ **arXiv 搜索失败**: {type(e).__name__}: {e}"


# ============================================================
# Phase 4.3: 本地 PDF intake
# ============================================================

def _extract_local_pdf_metadata(pdf_path: str) -> dict:
    """从本地 PDF 提取元数据：fitz metadata + 头页文本启发式。

    返回字段：title / authors (list) / year / arxiv_id / venue / file_hash / abstract_snippet
    缺失字段为空（""/[]/None）。
    """
    from src.core.pdf_parser import PDFParser
    from src.core.paper_registry import _compute_file_hash, _extract_arxiv_id_from_filename
    from pathlib import Path as _P
    import re as _re

    result = {
        "title": "", "authors": [], "year": None, "arxiv_id": None,
        "venue": "", "file_hash": "", "abstract_snippet": "",
    }
    p = _P(pdf_path)
    if not p.exists():
        return result

    # 1) file hash + 文件名 arxiv id
    result["file_hash"] = _compute_file_hash(p)
    result["arxiv_id"] = _extract_arxiv_id_from_filename(p.name)

    try:
        with PDFParser(pdf_path) as parser:
            meta = parser.get_metadata()
            if meta.title and len(meta.title) > 3:
                result["title"] = meta.title.strip()
            if meta.author:
                # PDF metadata author 字段可能逗号或分号分隔
                authors = _re.split(r"[,;]\s*", meta.author)
                result["authors"] = [a.strip() for a in authors if a.strip()][:10]

            # 2) 从第一页文本启发式提元数据
            head_text = parser.extract_page_text(0, fix_columns=True)

            # 找 abstract 片段（前 500 char 给 KG 用）
            m = _re.search(r"(?:abstract|摘\s*要)[\.:：\s]+", head_text, _re.IGNORECASE)
            if m:
                snippet = head_text[m.end(): m.end() + 800]
                # 截到下个章节标题
                end_m = _re.search(r"\n\s*(?:1\.|I\.|introduction|引言|##\s*1)", snippet, _re.IGNORECASE)
                if end_m:
                    snippet = snippet[:end_m.start()]
                result["abstract_snippet"] = snippet.strip()[:500]

            # 头页前几行常常含标题（如果 PDF metadata title 缺失/低质）
            # 启发：跳过页眉/期刊/日期；接受全大写（很多论文标题就是全大写）；
            # 跳过含 email 的作者行；最多合并 2 行长标题（如 "REACT: ...\nLANGUAGE MODELS"）
            _HEADER_NOISE = _re.compile(
                r"^(?:published|preprint|arxiv:?|under\s+review|technical\s+report|"
                r"workshop\s+paper|nips|iclr|icml|aaai|emnlp|acl|under\s+submission|"
                r"to\s+appear|accepted|copyright|©|abstract\b|provided\s+proper\s+attribution|"
                # 期刊页眉模式: "Front. Comput. Sci., 2025, 0(0): 1-42" / "IEEE Trans..."
                r"front\.|ieee\s+(?:trans|conf|proc)|j\.\s+[A-Z]|"
                r"\d{4}\b)",
                _re.IGNORECASE,
            )
            if not result["title"]:
                first_lines = [l.strip() for l in head_text.split("\n")[:20] if l.strip()]
                # 找到第一行有效候选
                first_idx = None
                for i, line in enumerate(first_lines):
                    if not (5 <= len(line) <= 200):
                        continue
                    if _HEADER_NOISE.match(line):
                        continue
                    if "@" in line and "." in line.split("@")[-1]:
                        continue
                    first_idx = i
                    break

                if first_idx is not None:
                    first_line = first_lines[first_idx]
                    # 全大写 → 多半是 multi-line 标题，尝试 join 下一行（如果也全大写）
                    if first_line.isupper() and first_idx + 1 < len(first_lines):
                        nxt = first_lines[first_idx + 1]
                        if (nxt.isupper() and 5 <= len(nxt) <= 100
                                and not _HEADER_NOISE.match(nxt)
                                and "@" not in nxt):
                            first_line = first_line + " " + nxt
                    result["title"] = first_line.strip()

            # 年份：搜寻 19xx/20xx 模式（取最高频）
            years = _re.findall(r"\b(19[89]\d|20[0-2]\d)\b", head_text[:3000])
            if years:
                # 取出现次数最多的（论文通常自我引用发表年）
                from collections import Counter
                result["year"] = int(Counter(years).most_common(1)[0][0])

            # arXiv ID 进一步搜索（如果文件名没拿到）
            if not result["arxiv_id"]:
                m = _re.search(r"arXiv:?\s*(\d{4}\.\d{4,5})", head_text[:2000])
                if m:
                    result["arxiv_id"] = m.group(1)

    except Exception as e:  # noqa: BLE001
        logger.warning(f"PDF metadata 提取部分失败: {e}")

    return result


@mcp.tool()
async def ingest_local_pdf(
    pdf_path: str,
    hint_title: str = "",
    hint_arxiv_id: str = "",
    domain: str = "",
    add_to_kg: bool = False,
    first_seen_date: str = "",
) -> str:
    """
    一站式吃下一篇本地 PDF：去重 → 抽元数据 → 注册到 paper_registry → 可选入图谱。

    适合使用的场景：
    - 用户把 IEEE/ACM/Springer 等闭源期刊论文手动下载到 data/papers/ 后
    - 用户说"把这篇加入"或"刚下了一篇新论文"
    - 批量整理已经下载的 PDF 时（更适合用 batch_ingest_directory）

    流程：
      1. 计算 file_hash → check_duplicate（已存在直接返回老记录）
      2. 从 fitz metadata + 头页文本启发式提取 title/authors/year/arxiv_id
      3. paper_registry.register_paper
      4. 若 add_to_kg=True：触发 add_paper_to_graph(domain, first_seen_year=first_seen_date)

    Args:
        pdf_path: 本地 PDF 路径（绝对或相对项目根）
        hint_title: 覆写抽到的 title（提取不准时可手动给）
        hint_arxiv_id: 覆写抽到的 arxiv_id
        domain: KG 目标域，仅 add_to_kg=True 时用
        add_to_kg: 注册后是否自动入 KG（默认 False，让用户先看元数据）
        first_seen_date: 你首次看到这篇论文的 YYYY 或 YYYY-MM-DD

    Returns:
        intake 报告（已注册/新增 + 元数据 + KG 状态）
    """
    from src.core.paper_registry import PaperRegistry, PaperRecord
    from pathlib import Path as _P
    logger.info(f"ingest_local_pdf: {pdf_path}")

    p = _P(pdf_path)
    if not p.exists():
        return f"⚠️ **文件不存在**: `{pdf_path}`"
    if not p.suffix.lower() == ".pdf":
        return f"⚠️ **不是 PDF 文件**: `{p.name}`"

    registry = PaperRegistry()
    meta = _extract_local_pdf_metadata(str(p))
    if hint_title:
        meta["title"] = hint_title
    if hint_arxiv_id:
        meta["arxiv_id"] = hint_arxiv_id

    # 去重
    dup = registry.check_duplicate(
        arxiv_id=meta.get("arxiv_id"),
        title=meta.get("title"),
        file_hash=meta.get("file_hash"),
    )
    if dup:
        return (
            f"✅ **已注册（无需重复入库）**\n\n"
            + registry.format_duplicate_warning(dup)
            + '\n\n如需重新入图谱，请明确说 "重新抽取并入图"。'
        )

    # 注册
    record = PaperRecord(
        title=meta["title"] or p.stem,
        local_path=str(p.as_posix()),
        arxiv_id=meta.get("arxiv_id"),
        authors=meta.get("authors", []),
        year=meta.get("year"),
        file_hash=meta.get("file_hash", ""),
        tags=["local-intake"],
        first_seen_date=first_seen_date or "",
    )
    registry.register_paper(record)

    lines = [
        "✅ **本地 PDF intake 完成**",
        "",
        f"| 字段 | 值 |",
        f"|:---|:---|",
        f"| 标题 | {record.title[:80]} |",
        f"| arXiv ID | {record.arxiv_id or '—'} |",
        f"| 年份 | {record.year or '—'} |",
        f"| 作者 | {', '.join(record.authors[:3]) or '—'} |",
        f"| 文件哈希 | `{record.file_hash}` |",
        f"| 本地路径 | `{record.local_path}` |",
        f"| 首次看到 | {record.first_seen_date or '—'} |",
    ]
    if meta.get("abstract_snippet"):
        lines.append("")
        lines.append(f"**摘要片段**: {meta['abstract_snippet'][:300]}...")

    # 可选入图谱
    if add_to_kg:
        try:
            from src.mcp_servers.knowledge_graph import add_paper_to_graph
            lines.append("")
            lines.append("--- KG 入图 ---")
            kg_result = await add_paper_to_graph(
                pdf_path=str(p),
                paper_title=record.title,
                paper_year=str(record.year) if record.year else "",
                domain=domain,
                first_seen_year=first_seen_date,
            )
            lines.append(kg_result[:1500])
        except Exception as e:  # noqa: BLE001
            lines.append(f"⚠️ KG 入图失败（不阻塞 intake）：{e}")

    return "\n".join(lines)


@mcp.tool()
async def batch_ingest_directory(
    dir_path: str = "data/papers",
    domain: str = "",
    add_to_kg: bool = False,
    limit: int = 0,
) -> str:
    """
    扫描目录里所有 PDF 批量 intake。已注册的跳过，新论文按 ingest_local_pdf 流程处理。

    适合使用的场景：
    - 你下载了一批 IEEE/ACM 论文到 data/papers/，想一次性整理
    - 重新初始化 paper_registry 时

    Args:
        dir_path: 扫描目录（默认 data/papers）
        domain: 入图时的域
        add_to_kg: 是否每篇都 add_paper_to_graph（慢，按需开启）
        limit: 0=全部；>0 仅处理前 N 个

    Returns:
        逐文件 intake 报告（行数 = PDF 数 + 摘要表头）
    """
    from src.core.paper_registry import PaperRegistry
    from pathlib import Path as _P
    p = _P(dir_path)
    if not p.exists() or not p.is_dir():
        return f"⚠️ **目录不存在**: `{dir_path}`"

    pdfs = sorted(p.glob("*.pdf"))
    if limit > 0:
        pdfs = pdfs[:limit]
    if not pdfs:
        return f"📁 目录 `{dir_path}` 下没有 PDF 文件"

    logger.info(f"batch_ingest_directory: {len(pdfs)} PDFs, add_to_kg={add_to_kg}")
    registry = PaperRegistry()

    lines = [
        f"## 📥 批量 intake: `{dir_path}` ({len(pdfs)} files)",
        "",
        "| # | 文件 | 状态 | 标题/arXiv ID |",
        "|---:|:---|:---|:---|",
    ]
    n_new, n_dup, n_err = 0, 0, 0
    for i, pdf in enumerate(pdfs, 1):
        try:
            meta = _extract_local_pdf_metadata(str(pdf))
            dup = registry.check_duplicate(
                arxiv_id=meta.get("arxiv_id"),
                title=meta.get("title"),
                file_hash=meta.get("file_hash"),
            )
            if dup:
                n_dup += 1
                lines.append(
                    f"| {i} | `{pdf.name[:30]}` | 🔁 已存在 | {dup.title[:50]} |"
                )
                continue

            # 直接调 ingest_local_pdf 复用所有逻辑
            await ingest_local_pdf(
                pdf_path=str(pdf),
                domain=domain,
                add_to_kg=add_to_kg,
            )
            n_new += 1
            label = meta.get("arxiv_id") or (meta.get("title") or "")[:40]
            lines.append(f"| {i} | `{pdf.name[:30]}` | ✅ 新增 | {label} |")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            lines.append(f"| {i} | `{pdf.name[:30]}` | ❌ 失败 | {str(e)[:50]} |")
            logger.exception(f"intake {pdf.name} 失败")

    lines += [
        "",
        f"**汇总**：新增 {n_new} · 已存在 {n_dup} · 失败 {n_err} · 总计 {len(pdfs)}",
    ]
    if add_to_kg and n_new > 0:
        lines.append(f"\n⚠️ 已对新论文执行 add_paper_to_graph（domain={domain or 'default'}）")
    return "\n".join(lines)


# ============================================================
# Phase 4.2: 接入 OpenReview / CrossRef / DBLP（公开 API 无 key）
# ============================================================

@mcp.tool()
async def search_crossref(query: str, limit: int = 5) -> str:
    """
    通过 CrossRef 公开 API 搜索学术论文（无需 API Key，DOI 元数据强）。

    适合使用的场景：
    - 已知 DOI 想拿元数据
    - Semantic Scholar 拿不到结果时的备选
    - 找一篇有 DOI 的论文的标题/作者/年份

    Args:
        query: 关键词或标题（英文最佳）
        limit: 返回数量，1-20，默认 5

    Returns:
        论文列表（标题/作者/年份/DOI/journal）
    """
    logger.info(f"CrossRef 搜索: query='{query}', limit={limit}")
    limit = max(1, min(limit, 20))

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(
                "https://api.crossref.org/works",
                params={"query": query, "rows": limit, "select": "DOI,title,author,issued,container-title,abstract"},
                headers={"User-Agent": "ScholarMind/0.1 (academic; mailto:noreply@example.com)"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        return "⚠️ **CrossRef 搜索超时**，请稍后重试。"
    except httpx.HTTPStatusError as e:
        return f"⚠️ **CrossRef 错误** (HTTP {e.response.status_code})"
    except Exception as e:  # noqa: BLE001
        return f"⚠️ **CrossRef 失败**: {type(e).__name__}: {e}"

    items = data.get("message", {}).get("items", [])
    if not items:
        return f"## CrossRef 未找到结果\n\n关键词: \"{query}\""

    results = []
    for i, w in enumerate(items, 1):
        title = (w.get("title") or [""])[0]
        doi = w.get("DOI", "")
        authors_list = w.get("author") or []
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in authors_list[:3]
        )
        if len(authors_list) > 3:
            authors += " et al."
        year = ""
        issued = w.get("issued", {}).get("date-parts") or []
        if issued and issued[0]:
            year = str(issued[0][0])
        venue = (w.get("container-title") or [""])[0]
        abstract = _truncate(w.get("abstract", ""), 200)

        results.append(
            f"### [{i}] {title or '(无标题)'}\n"
            f"- **作者**: {authors or '—'}\n"
            f"- **年份**: {year or '—'} | **期刊**: {venue or '—'}\n"
            f"- **DOI**: [{doi}](https://doi.org/{doi})\n"
            + (f"- **摘要**: {abstract}\n" if abstract else "")
        )

    return f"## 🔍 CrossRef 搜索: \"{query}\"\n\n" + "\n---\n".join(results)


@mcp.tool()
async def search_openreview(query: str, venue: str = "", year: int = 0, limit: int = 5) -> str:
    """
    通过 OpenReview API 搜索 ICLR/NeurIPS 等公开 review 的论文。

    适合使用的场景：
    - 找 ICLR/NeurIPS 论文（含 review 内容）
    - 关心论文的 reviewer 评分
    - arXiv 上拿不到 venue 信息时

    Args:
        query: 关键词
        venue: 可选会议代号（ICLR.cc, NeurIPS.cc, 留空不限）
        year: 可选年份过滤
        limit: 返回数量

    Returns:
        论文列表（标题/作者/venue/链接）
    """
    logger.info(f"OpenReview 搜索: query='{query}', venue='{venue}', year={year}")
    limit = max(1, min(limit, 20))
    # OpenReview v2 API: GET /notes/search?term=
    params = {"term": query, "limit": limit, "type": "all"}
    if venue:
        params["group"] = venue
    if year > 0:
        params["term"] = f"{query} {year}"

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(
                "https://api2.openreview.net/notes/search",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        return "⚠️ **OpenReview 搜索超时**。"
    except httpx.HTTPStatusError as e:
        return f"⚠️ **OpenReview 错误** (HTTP {e.response.status_code})"
    except Exception as e:  # noqa: BLE001
        return f"⚠️ **OpenReview 失败**: {type(e).__name__}: {e}"

    notes = data.get("notes", [])
    if not notes:
        return f"## OpenReview 未找到结果\n\n关键词: \"{query}\""

    results = []
    for i, n in enumerate(notes[:limit], 1):
        content = n.get("content", {})
        title = content.get("title", {}).get("value", "(无标题)") if isinstance(content.get("title"), dict) else content.get("title", "")
        venue_str = content.get("venue", {}).get("value", "") if isinstance(content.get("venue"), dict) else content.get("venue", "")
        authors_val = content.get("authors", {})
        authors_list = authors_val.get("value", []) if isinstance(authors_val, dict) else (authors_val or [])
        authors = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors += " et al."
        forum = n.get("forum", "")
        results.append(
            f"### [{i}] {title}\n"
            f"- **作者**: {authors or '—'}\n"
            f"- **Venue**: {venue_str or '—'}\n"
            f"- **链接**: https://openreview.net/forum?id={forum}\n"
        )
    return f"## 🔍 OpenReview 搜索: \"{query}\"\n\n" + "\n---\n".join(results)


@mcp.tool()
async def search_dblp(author_or_query: str, limit: int = 10) -> str:
    """
    通过 DBLP 公开 API 查找作者及其论文（无 key）。

    适合使用的场景：
    - 想看某个研究者的发表列表
    - arXiv 没有完整论文列表的领域权威

    Args:
        author_or_query: 作者名或关键词（DBLP 通用搜索）
        limit: 返回数量

    Returns:
        论文列表（标题/作者/venue/年份）
    """
    logger.info(f"DBLP 搜索: q='{author_or_query}', limit={limit}")
    limit = max(1, min(limit, 30))
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(
                "https://dblp.org/search/publ/api",
                params={"q": author_or_query, "format": "json", "h": limit},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        return "⚠️ **DBLP 搜索超时**。"
    except Exception as e:  # noqa: BLE001
        return f"⚠️ **DBLP 失败**: {type(e).__name__}: {e}"

    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    if not hits:
        return f"## DBLP 未找到结果\n\n查询: \"{author_or_query}\""

    results = []
    for i, h in enumerate(hits, 1):
        info = h.get("info", {})
        title = info.get("title", "(无标题)")
        year = info.get("year", "")
        venue = info.get("venue", "")
        authors_info = info.get("authors", {}).get("author", [])
        if isinstance(authors_info, dict):
            authors_info = [authors_info]
        authors = ", ".join(a.get("text", "") for a in authors_info[:3])
        if len(authors_info) > 3:
            authors += " et al."
        doi = info.get("doi", "")
        url = info.get("url", "")
        results.append(
            f"**[{i}]** {title}\n"
            f"  - {authors or '—'} | {venue} | {year}\n"
            f"  - DBLP: {url}" + (f" | DOI: {doi}" if doi else "")
        )
    return f"## 🔍 DBLP 搜索: \"{author_or_query}\"\n\n" + "\n\n".join(results)


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    logger.info("ScholarMind Paper Search MCP Server 启动中...")
    mcp.run()
