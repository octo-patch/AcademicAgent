from __future__ import annotations

"""
ScholarMind - LLM 驱动的知识抽取模块
========================================

从论文文本中自动抽取实体和关系，写入知识图谱。

核心设计：Schema-Guided Extraction
  - 将知识图谱的 Schema（节点类型 + 关系类型）嵌入到 Prompt 中
  - LLM 被约束为只能输出 Schema 中定义的类型
  - 输出为结构化 JSON → 解析后生成 KGNode + KGEdge

工程要点：
  1. Prompt 中包含 Schema 定义 + JSON 示例 → 约束 LLM 输出格式
  2. 复用 multimodal.py 中的 Robust JSON 提取器 → 抗 LLM 输出不规范
  3. 支持批量抽取（对论文分段抽取后合并）→ 控制单次 token 量
  4. 置信度标注 → 低质量抽取结果不入图

【工程思考】为什么用 LLM 抽取而不是 NER + RE 管线？
  - 传统 NER 模型（spaCy/HuggingFace）难以稳定理解 Agent 论文中的架构、模块、机制和 benchmark 语义
  - 训练领域 NER 需要大量标注数据（我们没有）
  - LLM 天然理解 "ReAct", "MemGPT", "tool use", "memory retrieval" 等术语
  - Schema-guided prompt = 零标注数据的领域知识抽取
"""

import json
import logging
import os
import re
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

from .schema import (
    KGNode,
    KGEdge,
    ExtractionResult,
    ExtractionOutput,
    SCHEMA_DESCRIPTION,
    SCHEMA_JSON_EXAMPLE,
    coerce_node_type,
    coerce_relation_type,
)
from .entity_linker import AliasMap, EntityLinker
from src.core.llm_response import strip_reasoning_artifacts


class ExtractionError(Exception):
    """LLM 抽取严重失败（双层 fallback 也无法解析）。

    上层（add_paper_to_graph 等）应捕获并向用户报错，而不是当成"成功但没抽到"。
    """

# 加载环境变量
load_dotenv()

logger = logging.getLogger("ScholarMind.Extractor")


# ============================================================
# 抽取 Prompt 模板
# ============================================================

EXTRACTION_PROMPT = """你是一名学术论文知识图谱抽取专家，覆盖 LLM Agent (规划/记忆/反思/工具/多 Agent/RAG) 与通信/SLAM/ISAC 两大研究域。

你的任务：根据下方的 Schema，从给定的文本中抽取实体（节点）和关系（边）。

{schema}

## 通用规则
1. 只能抽取上方 Schema 中定义过的实体和关系类型。
2. 实体标签强制使用 **英文**（即使原文中文，也用标准英文/缩写）。
3. 规范化实体名称：使用最常见/标准形式（ReAct / MemGPT / Toolformer / Long-term Memory / mmWave / SLAM 等）。
4. 每个实体必须具备 Schema 中明确定义的 `node_type`。
5. 每条关系必须包含 `source_label`, `source_type`, `target_label`, `target_type`, `relation_type`。
6. 必须精确：只能抽取文本中明确陈述或强烈暗示的内容。
7. **不要伪造或幻觉**文本不支持的实体或关系。
8. 当不确定时，**不抽取**好过**乱抽取**。

## 实体粒度（atomic concept）规则 —— v1.1 新增

**核心原则：抽 atomic concept，不抽长 phrase。** 每个实体应该是论文领域中可独立检索的最小术语单元。

✅ 好的抽取（atomic, 可独立检索）:
- `ReAct` (method), `Reasoning Trace` (concept), `Chain-of-Thought` (method)
- `RHS-aided SLAM` (method), `Holographic Pattern Optimization` (method)
- `Multipath Propagation` (concept), `Virtual Anchor` (concept)

❌ 不好的抽取（合成长 phrase，难检索）:
- `ReAct framework using tool calls` → 应拆为 `ReAct` (method) + `Tool Use` (concept) + 关系 uses
- `Reasoning and Acting in Language Models` → 拆为 `Reasoning` (concept) + `Acting` (concept) + `Language Model` (concept)
- `multipath-aided localization method based on tensor decomposition` → 拆为 `Multipath Localization` + `Tensor Decomposition` + 关系 uses

**单数 / 复数 / 大小写规范**（v1.1 强约束）:
- 概念用**单数**形式: `Reasoning Trace` 而非 `Reasoning Traces`；`Phase Shifter` 而非 `Phase Shifters`
- 缩写保留原大小写: `CoT`, `RHS`, `ISAC`, `DOA` —— **不要**写 `Cot`, `Rhs`
- 多词术语用 Title Case 或 hyphen 连接: `Chain-of-Thought`（连字符）、`Reasoning Trace`（首字母大写）
- 已是 acronym 的不再展开混写: 用 `RHS` 而不是 `RHS (Reconfigurable Holographic Surface)`，让它们作为独立节点

**论文标题不应作为概念节点**:
- 论文标题（含冒号和长说明）应只出现为 `node_type=paper` 的节点
- ❌ 不要把 `"ReAct: Synergizing Reasoning and Acting in Language Models"` 作为 method 节点
- ✅ 应抽 method=`ReAct` + paper=`ReAct: Synergizing Reasoning and Acting in Language Models`，二者通过 proposes 关系连接

## 关系选择指南（关键：避免 related_to 滥用）

每条关系必须先尝试匹配下面的具体类型，再考虑 related_to。**示例对照**：

- **proposes** (paper → method/framework/system):
  正例: paper(ReAct paper) -> proposes -> method(ReAct)
  负例: ❌ 不要用 paper -> proposes -> concept

- **uses** (paper/method/framework → concept/method/dataset/tool):
  正例: method(ReAct) -> uses -> concept(Tool Use)
  正例: paper -> uses -> dataset(HotpotQA)

- **improves / extends / supersedes** (method → method): 三者递进强度
  - improves: 在同一问题上做了更好 (Flash Attention improves Attention)
  - extends:  扩展能力或场景 (GPT-4V extends GPT-4)
  - supersedes: 完全取代 (Transformer supersedes RNN for NLP)

- **enables / requires** (concept/method → concept/method): 因果/依赖
  - enables: A 使 B 可能 (Tool Use enables Agent Planning)
  - requires: A 需要 B 才能工作 (ReAct requires Tool Use)

- **pretrained_on** (method/system → dataset): 模型/方法在 dataset 上预训练
  正例: system(LLaMA) -> pretrained_on -> dataset(CommonCrawl)

- **tested_on / evaluated_against**:
  - tested_on: method 在 dataset/benchmark 上测试 (ReAct tested_on HotpotQA)
  - evaluated_against: method 与 baseline 对比 (ReAct evaluated_against CoT)

- **compares_with / cites** (paper → paper)
- **authored_by** (paper → author)
- **belongs_to** (concept → concept): 层级 (ReAct belongs_to Agent Planning Methods)
- **evaluated_by** (method → metric)

## ⛔ related_to 使用约束

**related_to 是"最后兜底"类型，不是默认选项。**
- 只有在以上 14 种关系**都明显不适合**时才用 related_to
- 使用 related_to 时**必须**在 properties.reason 字段写明为什么不能归入其他类型
- 大量使用 related_to 会被视为"未理解领域语义"——错误示范应避免

## 论文上下文
- **标题**: {paper_title}
- **年份**: {paper_year}

## 待抽取的文本
{text}

## 输出结构（必须严格遵守）
返回**一个 JSON 对象**，恰好包含两个顶层字段：`nodes`（数组）+ `edges`（数组）。

正确格式：
```json
{{
  "nodes": [
    {{"label": "ReAct", "node_type": "method"}},
    {{"label": "Tool Use", "node_type": "concept"}}
  ],
  "edges": [
    {{"source_label": "ReAct", "source_type": "method",
      "target_label": "Tool Use", "target_type": "concept",
      "relation_type": "uses"}}
  ]
}}
```

❌ 不要返回扁平数组 `[{{...}}, {{...}}]`，必须是带 `nodes` 和 `edges` 字段的对象。
❌ 不要在 JSON 外加任何解释性文本。
"""

MERGE_PROMPT = """你已经从同一篇论文的不同章节中提取了部分知识。
请将以下零散的抽取结果合并为一个统一的、去重后的最终结果。

规则：
1. 合并重复的实体（如果 label 和 type 相同，则视为同一实体）。
2. 合并重复的关系（边）。
3. 为每个实体保留信息量最丰富的属性。
4. 必须以相同的 JSON 格式输出合并后的结果。

部分抽取结果：
{partial_results}

请务必仅返回合法的 JSON 格式数据：
"""


class KnowledgeExtractor:
    """
    LLM 驱动的知识抽取器（MiniMax API，OpenAI 兼容）

    Usage:
        extractor = KnowledgeExtractor()
        result = await extractor.extract_from_text(
            text="We propose a tool-augmented agent with long-term memory...",
            paper_title="Memory-Augmented LLM Agents",
            paper_year="2024",
        )
        print(result.to_summary())
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 16000,
        min_confidence: float = 0.5,
        max_chunk_chars: int = 4000,
        alias_map: AliasMap | None = None,
        linker: "EntityLinker | None" = None,
    ):
        """
        Args:
            model: LLM 模型名称（默认从环境变量 MINIMAX_MODEL 读取）
            max_tokens: LLM 最大输出 token 数（reasoning 模型如 MiniMax-M2 思考块本身可消耗
                数千 token，留 16000 才能保证 JSON 输出不被截断）
            min_confidence: 最低置信度阈值（低于此值的实体/关系不入图）
            max_chunk_chars: 单次抽取的最大文本长度（超过则分段）。默认 4000：
                与 GraphRAG (Microsoft, 1200 tokens ~5000 chars) / LlamaIndex (1024 tokens
                ~4000 chars) 业界推荐一致。

                v1.2 实验发现 (docs/handoff/2026-05-22_chunk_model_experiment.md)：
                - chunk_size=4000 是甜区：rhs_slam 抽出 84 节点 / 112 边
                - chunk_size=30000 抽取量骤降到 40%（34 节点）—— MiniMax 在单次塞
                  ~15K+ 字符时实体覆盖能力严重下降
                - chunk_size=2000 略低于 4000 (79 节点)，跨 chunk 实体重复后被 dedup 损耗

                历史值：早期 3000 为 abab6.5 设计；v1.1 默认 30000 是错误估计。
            alias_map: 实体别名词典（默认懒加载 data/aliases.yaml）。
                       抽取出的 label 会经它规范化，避免 "SLAM" / "Simultaneous Localization
                       and Mapping" 分裂成多个节点 (Phase 2.1)
            linker:    可选的 EntityLinker（需要 store 才能跨论文 dedup）。
                       传入后 _parse_extraction 会在 alias 归并后再做 embedding 兜底，
                       捕捉 "Chain of Thought Reasoning" / "Chain-of-Thought" 这类
                       字面差异较大但语义同一的实体。
        """
        self.model = model or os.getenv("MINIMAX_MODEL", "MiniMax-M2")
        self.max_tokens = max_tokens
        self.min_confidence = min_confidence
        self.max_chunk_chars = max_chunk_chars
        self._client = None
        # 别名表懒加载（首次访问时读 yaml；多 extractor 实例共享单例可手动注入）
        self._alias_map = alias_map
        self._linker = linker

    @property
    def alias_map(self) -> AliasMap:
        if self._alias_map is None:
            self._alias_map = AliasMap()
        return self._alias_map

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI-compatible client (MiniMax)"""
        if self._client is None:
            api_key = os.getenv("MINIMAX_API_KEY")
            base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
            if not api_key:
                raise ValueError(
                    "MINIMAX_API_KEY 未设置。请在 .env 中配置。"
                )
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    async def extract_from_text(
        self,
        text: str,
        paper_title: str = "",
        paper_year: str = "",
        first_seen_year_override: int | None = None,
        linker: "EntityLinker | None" = None,
    ) -> ExtractionResult:
        """
        从文本中抽取实体和关系

        如果文本过长，自动分段抽取后合并。

        Args:
            text: 论文文本（全文或章节）
            paper_title: 论文标题
            paper_year: 论文年份
            first_seen_year_override: 用户手动指定的"首次看到这篇论文的年份"。
                设置后所有抽取出的节点 first_seen_year 都用此值；用于"虽然论文是 2017
                年发表的，但我 2026 年才学到，记录为 2026"这种语义。
                None 时由 LLM 自行从论文上下文推断（通常等于 paper_year）。

        Returns:
            ExtractionResult: 抽取出的节点和边
        """
        # 默认覆盖优先级：first_seen_year_override > paper_year（如能 parse）> LLM 自推断
        effective_year = first_seen_year_override
        if effective_year is None and paper_year:
            try:
                # paper_year 可能是 "2024" 或 "2024-05-20" 或 "(unknown)"
                year_str = paper_year.split("-")[0].strip()
                if year_str.isdigit():
                    effective_year = int(year_str)
            except (ValueError, AttributeError):
                pass

        # 分段
        chunks = self._split_text(text)
        logger.info(
            "知识抽取: paper='%s', text_len=%d, chunks=%d, year_override=%s",
            paper_title, len(text), len(chunks), effective_year,
        )

        if len(chunks) == 1:
            single = await self._extract_single(
                chunks[0], paper_title, paper_year, effective_year,
                linker=linker,
            )
            return self._apply_semantic_dedup(single, linker)

        # 多段分别抽取后合并
        partial_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"抽取第 {i + 1}/{len(chunks)} 段...")
            result = await self._extract_single(
                chunk, paper_title, paper_year, effective_year,
                linker=linker,
            )
            if result.nodes or result.edges:
                partial_results.append(result)

        if not partial_results:
            return ExtractionResult(
                paper_title=paper_title,
                extraction_confidence=0.0,
            )

        merged = self._merge_results(partial_results, paper_title)
        return self._apply_semantic_dedup(merged, linker)

    async def _extract_single(
        self,
        text: str,
        paper_title: str,
        paper_year: str,
        first_seen_year_override: int | None = None,
        linker: "EntityLinker | None" = None,
    ) -> ExtractionResult:
        """对单段文本执行知识抽取"""
        prompt = EXTRACTION_PROMPT.format(
            schema=SCHEMA_DESCRIPTION,
            paper_title=paper_title or "(unknown)",
            paper_year=paper_year or "(unknown)",
            text=text,
        )

        try:
            raw_output = self._call_llm(prompt)
            return self._parse_extraction(
                raw_output, paper_title,
                first_seen_year_override=first_seen_year_override,
                linker=linker,
            )
        except ExtractionError:
            # 显式解析失败 → 让上层处理（不再静默吞）
            raise
        except Exception as e:
            logger.error(f"知识抽取失败: {e}")
            return ExtractionResult(
                paper_title=paper_title,
                raw_llm_output=str(e),
                extraction_confidence=0.0,
            )

    def _call_llm(self, prompt: str) -> str:
        """
        调用 MiniMax API (OpenAI 兼容)

        用 response_format=json_schema 在 API 层面约束输出格式。MiniMax-M2 系列是
        reasoning 模型，响应会被 <think>...</think> 块包裹，调用方需要靠 strip
        清理后再解析。截断的思考块（<think> 开但未闭）会抛 ResponseTruncatedError，
        交给上层 _extract_single 的异常处理（不静默吞）。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_output",
                    "schema": ExtractionOutput.model_json_schema(),
                },
            },
        )
        from src.core.llm_logging import log_model_once
        log_model_once("knowledge.extractor", self.model, response)
        raw = response.choices[0].message.content or ""
        return strip_reasoning_artifacts(raw)

    def _parse_extraction(
        self, raw_output: str, paper_title: str,
        first_seen_year_override: int | None = None,
        linker: "EntityLinker | None" = None,
    ) -> ExtractionResult:
        """
        解析 LLM 输出为 ExtractionResult。

        【2026-05-20 重构】Phase 2.6 + 2.2:
          - 严格 Pydantic 解析失败：logger.warning（不是 debug）+ 标记 had_parse_fallback=True
          - 双层 fallback 全失败：raise ExtractionError，让上层显式处理（而不是静默返回空）
          - 节点类型转换用 coerce_node_type（容错），不再 NodeType() 抛 ValueError
          - 时态字段真正落地：把 LLM 输出的 n.first_seen_year / n.superseded_by /
            e_data.established_year 传给 KGNode/KGEdge 构造函数（之前是死代码）
          - first_seen_year_override: 用户/上游手动覆盖该论文所有节点的 first_seen_year

        Args:
            raw_output: LLM 原始字符串输出
            paper_title: 论文标题（用于溯源 + node.source_paper）
            first_seen_year_override: 覆盖该批所有节点的 first_seen_year
        """
        parse_warnings: list[str] = []
        had_parse_fallback = False
        raw_nodes: list[dict] = []
        raw_edges: list[dict] = []

        # ---- 阶段 1: 严格 Pydantic 解析 ----
        try:
            output = ExtractionOutput.model_validate_json(raw_output)
            raw_nodes = [n.model_dump() for n in output.nodes]
            raw_edges = [e.model_dump() for e in output.edges]
        except Exception as pydantic_err:
            had_parse_fallback = True
            msg = f"strict Pydantic parse failed: {pydantic_err}"
            logger.warning("%s, fallback to lenient JSON + coerce", msg)
            parse_warnings.append(msg)

            # ---- 阶段 2: 宽松 JSON 提取（绕过 Pydantic Literal，让 coerce 处理未知类型）----
            data = self._extract_json(raw_output)
            if not data:
                logger.error("both strict + lenient JSON extraction failed; raw head=%s",
                             raw_output[:200])
                raise ExtractionError(
                    f"无法解析 LLM 输出 (paper={paper_title!r}): "
                    f"严格 Pydantic + 宽松 JSON 提取均失败"
                ) from pydantic_err
            if not isinstance(data, dict):
                raise ExtractionError(
                    f"LLM 输出顶层应为 object，实为 {type(data).__name__} "
                    f"(paper={paper_title!r})"
                ) from pydantic_err
            raw_nodes = list(data.get("nodes", []))
            raw_edges = list(data.get("edges", []))

        # ---- 阶段 3: dict → KGNode/KGEdge（容错路径 + 时态字段落地 + Layer 1 alias 归一化）----
        # 注：语义级 dedup（embedding 兜底）已从 per-mention 调用移到 _merge_results 末端做
        # 一次性 batch（O(N) 代价、跨 chunk 也覆盖、不重新 encode 已见 label）。
        alias = self.alias_map
        # label_remap: 原始 label -> alias 归并后的 label。给 edge 端点回查用，
        # 避免 LLM 在 node 表 / edge 表里写不同形态的同一 label 导致端点对不上。
        label_remap: dict[str, str] = {}

        nodes: list[KGNode] = []
        for n in raw_nodes:
            raw_label = str(n.get("label", "")).strip()
            if not raw_label:
                parse_warnings.append("node skip: 空 label")
                continue
            ntype, is_fb = coerce_node_type(str(n.get("node_type", "")))
            # Layer 1: alias 归一化（含 Fix A 的剥括号）
            label = alias.normalize(raw_label)
            if label != raw_label:
                parse_warnings.append(f"alias: {raw_label!r} -> {label!r}")
            label_remap[raw_label] = label
            if is_fb:
                parse_warnings.append(f"node_type 未知 → OTHER: {n.get('node_type')!r} ({label})")
            props = n.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            try:
                fy = n.get("first_seen_year")
                fy_int = int(fy) if isinstance(fy, (int, str)) and str(fy).strip().isdigit() else None
                node = KGNode(
                    label=label,
                    node_type=ntype,
                    properties=dict(props),
                    source_paper=paper_title,
                    first_seen_year=first_seen_year_override or fy_int,
                    superseded_by=n.get("superseded_by"),
                )
                nodes.append(node)
            except Exception as e:  # noqa: BLE001
                parse_warnings.append(f"node 构造失败 ({label}): {e}")
                logger.warning("node skip: %s / %s", e, n)

        edges: list[KGEdge] = []
        for e_data in raw_edges:
            try:
                # Entity Linking: 同样规范化 edge 两端的 label，保证与 node 一致
                raw_src = str(e_data.get("source_label", "")).strip()
                raw_tgt = str(e_data.get("target_label", "")).strip()
                src_label = label_remap.get(raw_src) or alias.normalize(raw_src)
                tgt_label = label_remap.get(raw_tgt) or alias.normalize(raw_tgt)
                if not src_label or not tgt_label:
                    parse_warnings.append("edge skip: 空 source/target label")
                    continue
                src_type, src_fb = coerce_node_type(str(e_data.get("source_type", "")))
                tgt_type, tgt_fb = coerce_node_type(str(e_data.get("target_type", "")))
                # 注：edge 端点的语义级 dedup 由 _merge_results 末端的 batch dedup
                # 与 edge_remap 统一处理，这里只做 alias 归一化（已通过 label_remap 实现）。
                rel_type, rel_fb = coerce_relation_type(str(e_data.get("relation_type", "")))
                if src_fb or tgt_fb or rel_fb:
                    parse_warnings.append(
                        f"edge 含 fallback 类型: {src_label}-{e_data.get('relation_type')}->{tgt_label}"
                    )
                source_node = KGNode(label=src_label, node_type=src_type)
                target_node = KGNode(label=tgt_label, node_type=tgt_type)
                ey = e_data.get("established_year")
                ey_int = int(ey) if isinstance(ey, (int, str)) and str(ey).strip().isdigit() else None
                edge = KGEdge(
                    source_id=source_node.node_id,
                    target_id=target_node.node_id,
                    relation_type=rel_type,
                    confidence=0.8,
                    source_paper=paper_title,
                    established_year=first_seen_year_override or ey_int,
                )
                edges.append(edge)
            except Exception as e:  # noqa: BLE001
                parse_warnings.append(f"edge 构造失败: {e}")
                logger.warning("edge skip: %s / %s", e, e_data)

        confidence = 0.85 if (nodes and edges) else 0.5 if nodes else 0.0
        # 有 fallback 时降权
        if had_parse_fallback:
            confidence *= 0.8

        logger.info(
            "抽取完成: %d 节点, %d 边, confidence=%.2f, fallback=%s, warnings=%d",
            len(nodes), len(edges), confidence, had_parse_fallback, len(parse_warnings),
        )

        return ExtractionResult(
            nodes=nodes,
            edges=edges,
            raw_llm_output=raw_output,
            paper_title=paper_title,
            extraction_confidence=confidence,
            had_parse_fallback=had_parse_fallback,
            parse_warnings=parse_warnings,
        )

    def _merge_results(
        self,
        partial_results: list[ExtractionResult],
        paper_title: str,
    ) -> ExtractionResult:
        """
        合并多段抽取结果（本地去重）

        【工程思考】为什么本地合并而不用 LLM merge？
        1. 节省 token 成本（merge prompt 也要消耗 token）
        2. 本地去重逻辑确定性高（node_id 相同则合并）
        3. 避免 LLM 在 merge 时引入新的错误
        """
        seen_node_ids = {}  # node_id → KGNode
        seen_edge_ids = set()
        all_edges = []

        for result in partial_results:
            for node in result.nodes:
                nid = node.node_id
                if nid in seen_node_ids:
                    seen_node_ids[nid].merge_properties(node.properties)
                else:
                    seen_node_ids[nid] = node

            for edge in result.edges:
                eid = edge.edge_id
                if eid not in seen_edge_ids:
                    seen_edge_ids.add(eid)
                    all_edges.append(edge)

        merged = ExtractionResult(
            nodes=list(seen_node_ids.values()),
            edges=all_edges,
            paper_title=paper_title,
            extraction_confidence=max(
                r.extraction_confidence for r in partial_results
            ),
        )

        logger.info(
            f"合并完成: {len(partial_results)} 段 → "
            f"{merged.node_count} 节点, {merged.edge_count} 边"
        )
        return merged

    def _apply_semantic_dedup(
        self,
        result: "ExtractionResult",
        linker: "EntityLinker | None",
    ) -> "ExtractionResult":
        """末端一次性 batch 语义去重。

        - linker 为 None / 未配置 encoder → 直接返回原 result
        - 否则按 node_type 分桶 batch encode 计算 cos_sim，合并 ≥0.85 的对
          （含 store 已有的同 type 节点，实现跨论文 dedup），重写 edge 端点。
        """
        eff_linker = linker or self._linker
        if eff_linker is None or not result.nodes:
            return result
        try:
            dedup_nodes, dedup_edges, label_remap, _ = eff_linker.batch_semantic_dedup(
                list(result.nodes), list(result.edges),
                include_store=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch_semantic_dedup 失败（非致命，跳过去重）: %s", exc)
            return result
        if not label_remap:
            return result
        # 用同 ExtractionResult 类型重新打包（保留 paper_title / confidence 等元数据）
        result.nodes = dedup_nodes
        result.edges = dedup_edges
        # 把 dedup 日志加进 parse_warnings 便于事后审计
        if hasattr(result, "parse_warnings") and isinstance(result.parse_warnings, list):
            for old, new in list(label_remap.items())[:20]:
                result.parse_warnings.append(f"batch dedup: {old!r} -> {new!r}")
            if len(label_remap) > 20:
                result.parse_warnings.append(f"... (+{len(label_remap) - 20} more dedup merges)")
        return result

    def _split_text(self, text: str) -> list[str]:
        """
        将长文本分段

        按段落边界分割，每段不超过 max_chunk_chars 字符。 当前默认每段大约 500 个英文词左右。
        """
        if len(text) <= self.max_chunk_chars: # 如果全文不超过 3000 字符，就不切
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > self.max_chunk_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        从 LLM 输出中提取 JSON（3 策略 fallback）

        与 multimodal.py 中的 FigureAnalyzer._extract_json 完全一致。
        """
        # 策略 1: 直接 parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 策略 2: 提取 ```json ``` 块
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 策略 3: 找 { } 对
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.error(f"JSON 提取失败: {text[:200]}")
        return {}
