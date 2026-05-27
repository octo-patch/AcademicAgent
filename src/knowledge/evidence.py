"""
ScholarMind - 知识图谱证据累积 + 多源冲突消解
============================================

为什么需要：
  原 graph_store.add_edge 的合并策略是 confidence = max(existing, new)，
  这等于"只看最自信的那一次"，丢失了"被多少篇论文支持过"的关键信息。
  graph_store.py 注释自己都写了 TODO（"需引入证据积累模型或多源冲突消解机制"）。

本模块提供：
  1. Evidence dataclass：单条证据的结构（来源论文、章节、置信度、抽取时间）
  2. accumulate_evidence(): 把新证据追加到 edge.properties.evidence 列表
  3. compute_aggregate_confidence(): 综合多源置信度（多源加成）
  4. detect_conflicts(): 检测同 source-target 但 relation_type 不同的冲突边
  5. mark_conflict(): 给冲突边打标记（不删除，由分析层报告）

设计准则：
  - 证据以 list[dict] 形式存在 KGEdge.properties["evidence"]，JSON 友好
  - confidence 公式：max(individual) + 0.05 * (n_sources - 1)，上限 1.0
  - 冲突不消解（保留所有证据），只标记 + 报告
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import KGEdge


# ============================================================
# Evidence 数据类
# ============================================================

@dataclass
class Evidence:
    """单条证据：某一篇论文里出现的一个关系实例。

    Attributes:
        source_paper:    来源论文标题
        source_section:  来源章节（method/experiment/...，未知时空字符串）
        confidence:      该次抽取的本地置信度 (0.0-1.0)
        extracted_at:    抽取时间 (ISO 8601 UTC)
        note:            可选备注（如"在 Table 2 中明确对比"）
    """
    source_paper: str
    source_section: str = ""
    confidence: float = 0.8
    extracted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        return cls(
            source_paper=data.get("source_paper", ""),
            source_section=data.get("source_section", ""),
            confidence=float(data.get("confidence", 0.8)),
            extracted_at=data.get(
                "extracted_at",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
            note=data.get("note", ""),
        )


# ============================================================
# 证据累积
# ============================================================

# 多源加成系数（每多一个来源加 0.05，上限 1.0）
MULTI_SOURCE_BOOST = 0.05
MAX_CONFIDENCE = 1.0


def get_evidence_list(edge: "KGEdge") -> list[dict]:
    """从 edge.properties 取证据列表（自动初始化为空 list）。"""
    ev = edge.properties.get("evidence")
    if not isinstance(ev, list):
        return []
    return ev


def accumulate_evidence(edge: "KGEdge", new_ev: Evidence) -> int:
    """把新证据追加到 edge.evidence；返回累积后的证据数。

    去重规则：同 source_paper + source_section 已存在 → 不重复追加（取较高 confidence）。
    """
    ev_list = get_evidence_list(edge)
    # 去重 key
    new_key = (new_ev.source_paper, new_ev.source_section)
    for existing in ev_list:
        if (existing.get("source_paper", ""), existing.get("source_section", "")) == new_key:
            # 已存在 → 取较高 confidence
            if new_ev.confidence > float(existing.get("confidence", 0.0)):
                existing["confidence"] = new_ev.confidence
                existing["note"] = new_ev.note or existing.get("note", "")
            return len(ev_list)
    ev_list.append(new_ev.to_dict())
    edge.properties["evidence"] = ev_list
    return len(ev_list)


def compute_aggregate_confidence(edge: "KGEdge") -> float:
    """根据 evidence 列表计算综合 confidence。

    公式：max(individual) + 0.05 × (n_sources - 1)，上限 1.0
    无 evidence 时退回 edge.confidence 原值。
    """
    ev_list = get_evidence_list(edge)
    if not ev_list:
        return edge.confidence
    max_local = max(float(e.get("confidence", 0.0)) for e in ev_list)
    boost = MULTI_SOURCE_BOOST * max(0, len(ev_list) - 1)
    return min(MAX_CONFIDENCE, max_local + boost)


def merge_evidence(edge: "KGEdge", new_edge: "KGEdge") -> int:
    """把 new_edge 的证据合并到 edge。返回累积后的证据数。

    通常用于 graph_store.add_edge 命中已存在 edge_id 的情况：
    把 new_edge 的 source_paper/confidence 作为一条新证据追加。
    """
    new_ev = Evidence(
        source_paper=new_edge.source_paper or "(unknown)",
        source_section=str(new_edge.properties.get("source_section", "")),
        confidence=new_edge.confidence,
        note=str(new_edge.properties.get("note", "")),
    )
    return accumulate_evidence(edge, new_ev)


# ============================================================
# 冲突检测
# ============================================================

@dataclass
class Conflict:
    """两条 edge 之间的冲突描述。"""
    source_id: str
    target_id: str
    relation_a: str
    relation_b: str
    edge_a_id: str
    edge_b_id: str

    def to_dict(self) -> dict:
        return asdict(self)


# 跨语义冲突分组 —— 同组内的关系类型互斥（之一为真意味着另一个为假）
# 如：A improves B 与 A is_superseded_by B 在同向上不自洽
_INCOMPATIBLE_PAIRS: list[set[str]] = [
    {"improves", "supersedes"},     # A 改进 B vs A 取代 B（不同程度）
    {"compares_with", "evaluated_against"},  # paper 级 vs method 级
]


def _relations_incompatible(rel_a: str, rel_b: str) -> bool:
    """两个 relation 是否在 _INCOMPATIBLE_PAIRS 内同组。"""
    if rel_a == rel_b:
        return False
    for grp in _INCOMPATIBLE_PAIRS:
        if rel_a in grp and rel_b in grp:
            return True
    return False


def find_sibling_edges(edges: dict, source_id: str, target_id: str) -> list:
    """返回所有 source/target 相同的边。

    Args:
        edges: graph_store._edges dict（edge_id -> KGEdge）
        source_id, target_id: 待查 pair
    """
    return [e for e in edges.values()
            if e.source_id == source_id and e.target_id == target_id]


def detect_conflict_for(new_edge: "KGEdge", existing_edges: dict) -> Conflict | None:
    """对新加入的 edge 检查是否与已有 sibling edge 不兼容。

    若有冲突，返回 Conflict；否则 None。
    """
    siblings = find_sibling_edges(existing_edges, new_edge.source_id, new_edge.target_id)
    new_rel = new_edge.relation_type.value
    for sib in siblings:
        if sib.edge_id == new_edge.edge_id:
            continue  # 自己
        sib_rel = sib.relation_type.value
        if _relations_incompatible(new_rel, sib_rel):
            return Conflict(
                source_id=new_edge.source_id,
                target_id=new_edge.target_id,
                relation_a=sib_rel,
                relation_b=new_rel,
                edge_a_id=sib.edge_id,
                edge_b_id=new_edge.edge_id,
            )
    return None


def mark_conflict(edge: "KGEdge", conflict_with: str) -> None:
    """在 edge.properties 标记冲突 + 对端 edge_id。"""
    edge.properties["conflict"] = True
    existing = edge.properties.get("conflict_with", [])
    if not isinstance(existing, list):
        existing = []
    if conflict_with not in existing:
        existing.append(conflict_with)
    edge.properties["conflict_with"] = existing
