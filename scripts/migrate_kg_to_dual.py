"""
KG 双域拆分迁移脚本（一次性）
=============================

把 data/knowledge_graph.json（混合通信/SLAM/LLM Agent 的 561 节点）
按 MiniMax 分类器拆为 data/kg_comms.json + data/kg_llm_agent.json。

用法：
    # 干跑：只产报告，不写最终 KG 文件
    python scripts/migrate_kg_to_dual.py --dry-run

    # 限量验证（前 30 节点）
    python scripts/migrate_kg_to_dual.py --dry-run --limit 30

    # 真正执行：生成 kg_comms.json + kg_llm_agent.json
    python scripts/migrate_kg_to_dual.py --apply

输出：
    - data/kg_migration_report.md  （红/黄/绿三档抽查清单）
    - data/kg_migration_classifications.json  （分类缓存，重跑时复用）
    - --apply 时：data/kg_comms.json + data/kg_llm_agent.json

设计准则：
    - 不删 data/knowledge_graph.json，保留为历史快照
    - 分类结果缓存到 classifications.json，断点续跑零成本
    - 共享节点（shared）和跨域边出现在双方 KG 中
    - confidence < 0.7 标红，用户人工复核后调整
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# 把项目根加进 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.minimax_models import (  # noqa: E402
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_OPENAI_BASE_URL,
)
from src.knowledge.schema import (  # noqa: E402
    DOMAIN_COMMS,
    DOMAIN_LLM,
    DOMAIN_SHARED,
    KGEdge,
    KGNode,
    NodeType,
)
from src.recorder import RefactorRecorder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("KGMigrate")

# ============================================================
# Pydantic 输出
# ============================================================

class _Classification(BaseModel):
    """单个节点的分类结果"""
    idx: int = Field(description="输入项 idx（与请求对应）")
    domain: str = Field(description="llm_agent | comms | shared")
    confidence: float = Field(description="0.0-1.0")
    reason: str = Field(description="一句话理由")


class _BatchClassifyOutput(BaseModel):
    results: list[_Classification]


# ============================================================
# 分类器
# ============================================================

CLASSIFY_PROMPT = """你是学术 KG 节点分类器。给定一批知识节点，把每个划归到三个研究域之一：

- llm_agent：LLM Agent / RAG / Memory / Tool Use / Reflection / Multi-Agent / Planning / Reasoning
  示例：ReAct, MemGPT, Toolformer, AutoGen, LangGraph, Generative Agents, ChatGPT, Long-term Memory, Chain-of-Thought
- comms：5G/6G 通信 / ISAC / SLAM / 定位 / 信道 / mmWave / 阵列
  示例：5G NR, mmWave, ISAC, Phased Array, Beam Steering, Channel Estimation, Belief Propagation,
       Kalman Filter, Cramer-Rao Lower Bound, Phased Array, RHS, Multipath Propagation
- shared：两个域共享或更基础的通用概念
  示例：Transformer, Attention, Neural Network, Reinforcement Learning, Optimization, Bayesian Inference

规则：
1. 如果节点明显属于某一域，给 confidence ≥ 0.85
2. 如果有歧义但有倾向性，给 0.7-0.85
3. 如果明显跨域或无法判断，输出 shared 且 confidence ~ 0.7
4. 完全无法判断时输出 shared, confidence ≤ 0.5
5. 论文标题（paper 节点）按主题域归属
6. 作者节点 (author) 默认 shared（除非明显某域专家）

待分类节点：
{nodes_json}

返回 JSON 对象，结构为 {{"results": [{{"idx": 0, "domain": "...", "confidence": 0.x, "reason": "..."}}, ...]}}
每个输入节点必须有一行对应输出。
"""


@dataclass
class NodeClassification:
    node_id: str
    label: str
    node_type: str
    domain: str
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class Classifier:
    def __init__(self, model: Optional[str] = None, batch_size: int = 20):
        self.model = model or os.getenv("MINIMAX_MODEL", DEFAULT_MINIMAX_MODEL)
        self.batch_size = batch_size
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("MINIMAX_API_KEY")
            base_url = os.getenv(
                "MINIMAX_BASE_URL", DEFAULT_MINIMAX_OPENAI_BASE_URL
            )
            if not api_key:
                raise ValueError("MINIMAX_API_KEY 未设置")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def classify_batch(self, nodes: list[KGNode]) -> list[NodeClassification]:
        items = [
            {
                "idx": i,
                "label": n.label,
                "type": n.node_type.value,
                "source_paper": (n.source_paper or "")[:80],
            }
            for i, n in enumerate(nodes)
        ]
        prompt = CLASSIFY_PROMPT.format(
            nodes_json=json.dumps(items, ensure_ascii=False, indent=2),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=16000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "batch_classify",
                        "schema": _BatchClassifyOutput.model_json_schema(),
                    },
                },
            )
            raw = response.choices[0].message.content or ""
            from src.core.llm_response import strip_reasoning_artifacts
            cleaned = strip_reasoning_artifacts(raw)
            output = _BatchClassifyOutput.model_validate_json(cleaned)
        except Exception as exc:  # noqa: BLE001
            logger.error("classify_batch 失败 (%d 节点): %s", len(nodes), exc)
            # 失败时统一标 shared / 0.3，让后续人工抽查
            return [
                NodeClassification(
                    node_id=n.node_id, label=n.label, node_type=n.node_type.value,
                    domain=DOMAIN_SHARED, confidence=0.3, reason=f"LLM classify failed: {exc}",
                )
                for n in nodes
            ]

        by_idx = {c.idx: c for c in output.results}
        results: list[NodeClassification] = []
        for i, n in enumerate(nodes):
            c = by_idx.get(i)
            if c is None:
                results.append(NodeClassification(
                    node_id=n.node_id, label=n.label, node_type=n.node_type.value,
                    domain=DOMAIN_SHARED, confidence=0.3, reason="missing in LLM output",
                ))
            else:
                domain = c.domain if c.domain in (DOMAIN_LLM, DOMAIN_COMMS, DOMAIN_SHARED) else DOMAIN_SHARED
                results.append(NodeClassification(
                    node_id=n.node_id, label=n.label, node_type=n.node_type.value,
                    domain=domain,
                    confidence=max(0.0, min(1.0, float(c.confidence))),
                    reason=c.reason,
                ))
        return results


# ============================================================
# 主流程
# ============================================================

def load_legacy_kg(path: Path) -> tuple[list[KGNode], list[KGEdge]]:
    """加载旧 KG，并把 edges 的 source_id/target_id 从旧 id 重映射到 coerce 后的新 id。

    关键修复（2026-05-20）：
      旧 KG 中 node_id = `<old_type>_<normalized_label>`（如 channel_model_xxx）。
      schema.coerce_node_type 把 channel_model 归并为 method 后，节点的 node_id
      变为 method_xxx。但旧 edge 的 source_id/target_id 仍指向 channel_model_xxx，
      导致 edge 找不到节点被丢弃。
      → 构造 old_id → new_id 映射，重映射 edge 的两端。
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    # 1. 加载节点 + 构造 old_id -> new_id 映射
    nodes: list[KGNode] = []
    id_map: dict[str, str] = {}
    for raw in data["nodes"]:
        old_id = raw.get("node_id", "")
        node = KGNode.from_dict(raw)
        new_id = node.node_id
        if old_id and old_id != new_id:
            id_map[old_id] = new_id
        nodes.append(node)

    # 2. 加载边并应用映射
    edges: list[KGEdge] = []
    for raw in data["edges"]:
        edge = KGEdge.from_dict(raw)
        if edge.source_id in id_map:
            edge.source_id = id_map[edge.source_id]
        if edge.target_id in id_map:
            edge.target_id = id_map[edge.target_id]
        edges.append(edge)

    # 3. 节点 dedup（同 new_id 合并属性，与 graph_store 行为一致）
    by_id: dict[str, KGNode] = {}
    for n in nodes:
        if n.node_id in by_id:
            by_id[n.node_id].merge_properties(n.properties)
        else:
            by_id[n.node_id] = n

    return list(by_id.values()), edges


def build_report(classifications: list[NodeClassification]) -> str:
    red = [c for c in classifications if c.confidence < 0.7]
    yellow = [c for c in classifications if 0.7 <= c.confidence < 0.85]
    green = [c for c in classifications if c.confidence >= 0.85]

    domain_counts: dict[str, int] = {}
    for c in classifications:
        domain_counts[c.domain] = domain_counts.get(c.domain, 0) + 1

    lines = [
        "# KG 双域拆分迁移报告",
        "",
        f"**总节点数**: {len(classifications)}",
        "",
        "## 域分布",
        "",
        "| 域 | 节点数 |",
        "|:---|---:|",
    ]
    for d, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {d} | {n} |")
    lines += [
        "",
        "## 三档抽查清单",
        "",
        f"- 🔴 红 (confidence < 0.7): **{len(red)}** 个（需要人工复核）",
        f"- 🟡 黄 (0.7 ≤ conf < 0.85): **{len(yellow)}** 个（可选抽查）",
        f"- 🟢 绿 (conf ≥ 0.85): **{len(green)}** 个（高置信）",
        "",
        "---",
        "",
        "## 🔴 红色清单（confidence < 0.7，必须复核）",
        "",
        "格式：每行 `node_id | label | type | 当前域 | conf | reason | **你的判定**`",
        '把 "你的判定" 填为以下之一：`llm_agent` / `comms` / `shared` / `drop`',
        "完成后用 `python scripts/apply_migration_review.py data/kg_migration_report.md` 套用。",
        "",
        "| node_id | label | type | 当前域 | conf | reason | 你的判定 |",
        "|:---|:---|:---|:---|---:|:---|:---|",
    ]
    for c in sorted(red, key=lambda x: x.confidence):
        lines.append(
            f"| `{c.node_id}` | {c.label[:50]} | {c.node_type} | {c.domain} "
            f"| {c.confidence:.2f} | {c.reason[:60]} |  |"
        )

    lines += [
        "",
        "## 🟡 黄色清单（0.7-0.85，可选复核，前 30）",
        "",
        "| node_id | label | type | 当前域 | conf |",
        "|:---|:---|:---|:---|---:|",
    ]
    for c in sorted(yellow, key=lambda x: x.confidence)[:30]:
        lines.append(
            f"| `{c.node_id}` | {c.label[:50]} | {c.node_type} | {c.domain} | {c.confidence:.2f} |"
        )

    lines += [
        "",
        "## 🟢 绿色（≥0.85，自动通过，前 10 举例）",
        "",
        "| label | type | 域 | conf |",
        "|:---|:---|:---|---:|",
    ]
    for c in sorted(green, key=lambda x: -x.confidence)[:10]:
        lines.append(f"| {c.label[:50]} | {c.node_type} | {c.domain} | {c.confidence:.2f} |")

    return "\n".join(lines) + "\n"


def write_dual_kg(
    nodes: list[KGNode],
    edges: list[KGEdge],
    classifications: list[NodeClassification],
    out_dir: Path,
) -> dict[str, dict]:
    cls_by_id = {c.node_id: c for c in classifications}

    # 给每个节点贴 domain
    for n in nodes:
        c = cls_by_id.get(n.node_id)
        if c:
            n.domain = c.domain

    # 节点按域分配（shared 出现在所有域里）
    def visible_in(n_domain: str, target: str) -> bool:
        return n_domain == target or n_domain == DOMAIN_SHARED

    by_id = {n.node_id: n for n in nodes}

    summaries: dict[str, dict] = {}
    for target in (DOMAIN_LLM, DOMAIN_COMMS):
        sel_nodes = [n for n in nodes if visible_in(n.domain, target)]
        node_id_set = {n.node_id for n in sel_nodes}
        sel_edges = [
            e for e in edges
            if e.source_id in node_id_set and e.target_id in node_id_set
        ]
        path = out_dir / f"kg_{target}.json"
        data = {
            "metadata": {
                "node_count": len(sel_nodes),
                "edge_count": len(sel_edges),
                "domain": target,
                "version": "1.1",
                "source": "migrate_kg_to_dual.py",
            },
            "nodes": [n.to_dict() for n in sel_nodes],
            "edges": [e.to_dict() for e in sel_edges],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[target] = {
            "path": str(path),
            "nodes": len(sel_nodes),
            "edges": len(sel_edges),
        }
        logger.info("[apply] %s -> %d 节点 / %d 边", path, len(sel_nodes), len(sel_edges))
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="KG dual-domain migration")
    parser.add_argument("--source", default="data/knowledge_graph.json",
                        help="待拆分的源 KG JSON")
    parser.add_argument("--out-dir", default="data", help="输出目录")
    parser.add_argument("--cache", default="data/kg_migration_classifications.json",
                        help="分类缓存文件（断点续跑）")
    parser.add_argument("--report", default="data/kg_migration_report.md",
                        help="抽查报告输出路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑：只产报告，不写 kg_*.json")
    parser.add_argument("--apply", action="store_true",
                        help="实际写出 kg_<domain>.json（与 --dry-run 互斥）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只分类前 N 个节点（smoke test）")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--record", action="store_true", default=True,
                        help="同步到 RefactorRecorder area=identity-and-kg-split")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        parser.error("必须指定 --dry-run 或 --apply")
    if args.dry_run and args.apply:
        parser.error("--dry-run 与 --apply 互斥")

    src = Path(args.source)
    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache)
    report_path = Path(args.report)

    nodes, edges = load_legacy_kg(src)
    logger.info("loaded legacy KG: %d 唯一节点 / %d 边", len(nodes), len(edges))

    if args.limit:
        nodes = nodes[: args.limit]
        logger.info("limited to first %d nodes", args.limit)

    # 缓存命中
    cached: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cached = {c["node_id"]: c for c in json.loads(cache_path.read_text(encoding="utf-8"))}
            logger.info("cache hit: %d nodes", len(cached))
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache load failed: %s", exc)

    # 分批分类
    classifier = Classifier(batch_size=args.batch_size)
    classifications: list[NodeClassification] = []
    to_classify: list[KGNode] = []
    for n in nodes:
        if n.node_id in cached:
            classifications.append(NodeClassification(**cached[n.node_id]))
        else:
            to_classify.append(n)

    logger.info("to classify (fresh): %d / cached: %d", len(to_classify), len(classifications))

    for i in range(0, len(to_classify), args.batch_size):
        batch = to_classify[i: i + args.batch_size]
        logger.info("classifying batch %d (%d nodes)", i // args.batch_size + 1, len(batch))
        batch_results = classifier.classify_batch(batch)
        classifications.extend(batch_results)
        # 增量写缓存（防中断丢进度）
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps([c.to_dict() for c in classifications], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 报告
    report = build_report(classifications)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info("report written: %s", report_path)

    summaries: dict[str, dict] = {}
    if args.apply:
        summaries = write_dual_kg(nodes, edges, classifications, out_dir)

    # 记录到 Recorder
    if args.record:
        try:
            r = RefactorRecorder(area="identity-and-kg-split")
            domain_counts: dict[str, int] = {}
            for c in classifications:
                domain_counts[c.domain] = domain_counts.get(c.domain, 0) + 1
            mode = "dry-run" if args.dry_run else "apply"
            r.log_validation(
                step=f"kg-migration-{mode}",
                command=" ".join(sys.argv),
                output=(
                    f"input: {len(nodes)} 节点 / {len(edges)} 边\n"
                    f"classified: {len(classifications)} (cached {len(cached)}, fresh {len(to_classify)})\n"
                    f"domain distribution: {domain_counts}\n"
                    f"report: {report_path}\n"
                    f"apply: {summaries if summaries else '(dry-run, no kg_*.json written)'}"
                ),
                status="ok",
                notes=f"mode={mode} limit={args.limit}",
                artifacts=[str(report_path), str(cache_path)] + [s["path"] for s in summaries.values()],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("recorder log failed: %s", exc)

    print("\n=== Migration done ===")
    print(f"  report : {report_path}")
    print(f"  cache  : {cache_path}")
    if summaries:
        for d, s in summaries.items():
            print(f"  {d}: {s['nodes']} nodes / {s['edges']} edges -> {s['path']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
