from __future__ import annotations

"""
KG 抽取的语义匹配指标（v1.0 gold dataset 专用）
==================================================

与 ``kg.py`` 的 strict F1 互补，承担「主信号」角色：

- ``kg_node_f1_semantic``：节点匹配从「严格归一化键相等」放宽到「精确 → alias → embedding」
  三档命中。差距大说明 prompt/抽取还需要工程化。
- ``kg_edge_f1_semantic``：边的端点也走语义匹配（relation_type 仍要求 strict 相等，避免漂移）。
- ``must_have_recall``：在 gold 标 ``must_have`` 的核心实体上的召回率，
  这是「能不能用」的硬指标。
- ``precision_on_known``：抽取项里跟 gold（must/should/may 任一级）能对上的比例 ——
  把"抽到了但 gold 未列"算成"未知"而不是"错"，避免惩罚正常的抽取扩展。

匹配三档：
  Tier 1（精确）   : 归一化 label 完全相等
  Tier 2（alias）  : gold[entity].aliases 含归一化后的抽取 label（或反之）
  Tier 3（embed）  : cos_sim(extracted_label, gold_label) >= EMBED_THRESHOLD
                     —— 当 sentence-transformers 不可用或抽取/gold 任一为空时跳过

embedding 复用 ``src.knowledge.semantic_index._SentenceTransformerEncoder``
（paraphrase-multilingual-MiniLM-L12-v2），不引入新依赖。
"""

import json
import logging
from functools import lru_cache
from typing import Iterable

from ..schema import MetricResult
from .kg import _label_from_node_id, _normalize_label

logger = logging.getLogger("AcademicAgent.Eval.KGSemantic")

EMBED_THRESHOLD = 0.78


# ============================================================
# Embedding 兜底（lazy + 单例）
# ============================================================


@lru_cache(maxsize=1)
def _get_encoder():
    """懒加载 sentence-transformer。失败返回 None，匹配阶段自动跳过 embedding 档。"""
    try:
        from src.knowledge.semantic_index import _SentenceTransformerEncoder
        return _SentenceTransformerEncoder()
    except Exception as exc:  # noqa: BLE001
        logger.info("embedding 兜底不可用 (%s)，semantic F1 降级到 alias-only", exc)
        return None


def _embed_match(extracted: str, gold_labels: list[str], threshold: float) -> str | None:
    """对 extracted label 找 gold_labels 中 cos_sim 最高且超阈值的项，返回归一化 gold key。"""
    enc = _get_encoder()
    if enc is None or not extracted or not gold_labels:
        return None
    try:
        import numpy as np
        vecs = enc.encode([extracted] + list(gold_labels))
        q = vecs[0]
        cands = vecs[1:]
        sims = (cands @ q.T).flatten()
        idx = int(np.argmax(sims))
        if float(sims[idx]) >= threshold:
            return _normalize_label(gold_labels[idx])
    except Exception as exc:  # noqa: BLE001
        logger.debug("embed match failed: %s", exc)
    return None


# ============================================================
# Gold 数据结构辅助
# ============================================================


def _gold_entity_list(gold: dict, tiers: Iterable[str]) -> list[dict]:
    """从 gold 取指定 tier 的实体列表（must_have / should_have / may_have）。"""
    out: list[dict] = []
    for t in tiers:
        for e in gold.get(t, []) or []:
            if isinstance(e, dict) and e.get("label"):
                out.append(e)
    return out


def _aliases_for(entity: dict) -> list[str]:
    """gold entity 的 aliases 列表（含自身 label）。"""
    aliases = list(entity.get("aliases", []) or [])
    label = entity.get("label", "")
    if label and label not in aliases:
        aliases.append(label)
    return aliases


def _build_gold_index(gold_entities: list[dict]) -> dict[str, dict]:
    """归一化 alias -> gold entity 映射。用于 alias 匹配档。"""
    index: dict[str, dict] = {}
    for e in gold_entities:
        for a in _aliases_for(e):
            key = _normalize_label(a)
            if key and key not in index:
                index[key] = e
    return index


# ============================================================
# 核心匹配
# ============================================================


def _match_extracted_to_gold(
    extracted_label: str,
    gold_entities: list[dict],
    alias_index: dict[str, dict],
    use_embedding: bool,
) -> tuple[dict | None, str]:
    """返回 (匹配到的 gold entity 或 None, 匹配档位)。

    档位：'exact' / 'alias' / 'embed' / 'miss'
    exact 和 alias 其实是同一档（alias_index 包含 label 自身），但区分以便诊断。
    """
    norm = _normalize_label(extracted_label)
    if not norm:
        return None, "miss"

    # Tier 1: 精确（label 自身归一化相等）
    for e in gold_entities:
        if _normalize_label(e.get("label", "")) == norm:
            return e, "exact"

    # Tier 2: alias 命中
    hit = alias_index.get(norm)
    if hit is not None:
        return hit, "alias"

    # Tier 3: embedding 兜底
    if use_embedding:
        labels = [e.get("label", "") for e in gold_entities if e.get("label")]
        matched_norm = _embed_match(extracted_label, labels, EMBED_THRESHOLD)
        if matched_norm is not None:
            for e in gold_entities:
                if _normalize_label(e.get("label", "")) == matched_norm:
                    return e, "embed"
    return None, "miss"


# ============================================================
# 指标 1: kg_node_f1_semantic
# ============================================================


def kg_node_f1_semantic(
    extracted_nodes: list[dict],
    gold: dict,
    *,
    use_embedding: bool = True,
    task_id: str | None = None,
) -> MetricResult:
    """节点 F1，但匹配走 精确 → alias → embedding 三档。

    gold schema（v1.0）：
      {
        "must_have":  [{"label", "type", "aliases":[...], "evidence_section"}, ...],
        "should_have":[...],
        "may_have":   [...]
      }

    Recall 分母 = must_have ∪ should_have ∪ may_have 的实体数。
    Precision 分母 = 抽取项数。匹配命中即算 TP（不区分档位）。
    """
    gold_entities = _gold_entity_list(gold, ("must_have", "should_have", "may_have"))
    alias_index = _build_gold_index(gold_entities)

    matched_gold_keys: set[str] = set()
    tp = 0
    archetype = {"exact": 0, "alias": 0, "embed": 0, "miss": 0}

    for n in extracted_nodes:
        label = n.get("label", "")
        hit, tier = _match_extracted_to_gold(
            label, gold_entities, alias_index, use_embedding
        )
        archetype[tier] += 1
        if hit is not None:
            key = _normalize_label(hit.get("label", ""))
            if key and key not in matched_gold_keys:
                matched_gold_keys.add(key)
                tp += 1

    n_pred = len(extracted_nodes)
    n_gold = len(gold_entities)

    if n_pred == 0 and n_gold == 0:
        return MetricResult(
            metric="kg_node_f1_semantic", value=1.0, task_id=task_id,
            numerator=0, denominator=0,
            notes='{"precision":1.0,"recall":1.0,"empty":true}',
        )

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return MetricResult(
        metric="kg_node_f1_semantic", value=f1, task_id=task_id,
        numerator=float(tp), denominator=float(n_gold),
        notes=json.dumps({
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "match_tiers": archetype,
            "embedding": use_embedding,
        }),
    )


# ============================================================
# 指标 2: kg_edge_f1_semantic
# ============================================================


def kg_edge_f1_semantic(
    extracted_edges: list[dict],
    gold: dict,
    *,
    use_embedding: bool = True,
    task_id: str | None = None,
) -> MetricResult:
    """边 F1：端点走语义匹配，relation_type 仍要求 strict 相等。

    gold.expected_edges:
      [{"src": "...", "rel": "...", "dst": "...", "tier": "must|should|may",
        "evidence_section": "..."}]

    端点匹配复用 must/should/may 实体的 aliases 索引；找不到端点对应实体时回退到
    label 字符串直接比对（避免边端点是临时实体却没注册 aliases 时全部漏判）。
    """
    gold_entities = _gold_entity_list(gold, ("must_have", "should_have", "may_have"))
    alias_index = _build_gold_index(gold_entities)
    gold_edges = list(gold.get("expected_edges", []) or [])

    def _endpoint_key(label: str) -> str:
        hit, _ = _match_extracted_to_gold(label, gold_entities, alias_index, use_embedding)
        if hit is not None:
            return _normalize_label(hit.get("label", ""))
        return _normalize_label(label)

    ext_triples: set[tuple[str, str, str]] = set()
    for e in extracted_edges:
        src_lbl = e.get("source_label") or _label_from_node_id(e.get("source_id", ""))
        tgt_lbl = e.get("target_label") or _label_from_node_id(e.get("target_id", ""))
        rel = str(e.get("relation_type", "")).strip().lower()
        ext_triples.add((_endpoint_key(src_lbl), rel, _endpoint_key(tgt_lbl)))

    gold_triples: set[tuple[str, str, str]] = set()
    for g in gold_edges:
        gold_triples.add((
            _endpoint_key(g.get("src", "")),
            str(g.get("rel", "")).strip().lower(),
            _endpoint_key(g.get("dst", "")),
        ))

    if not ext_triples and not gold_triples:
        return MetricResult(
            metric="kg_edge_f1_semantic", value=1.0, task_id=task_id,
            numerator=0, denominator=0,
            notes='{"precision":1.0,"recall":1.0,"empty":true}',
        )

    tp = len(ext_triples & gold_triples)
    precision = tp / len(ext_triples) if ext_triples else 0.0
    recall = tp / len(gold_triples) if gold_triples else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return MetricResult(
        metric="kg_edge_f1_semantic", value=f1, task_id=task_id,
        numerator=float(tp), denominator=float(len(gold_triples)),
        notes=json.dumps({
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "embedding": use_embedding,
        }),
    )


# ============================================================
# 指标 3: must_have_recall
# ============================================================


def must_have_recall(
    extracted_nodes: list[dict],
    gold: dict,
    *,
    use_embedding: bool = True,
    task_id: str | None = None,
) -> MetricResult:
    """只在 gold.must_have 上计算 recall —— 论文不可能不抽到的核心贡献。

    分母 = must_have 实体数；命中走与 kg_node_f1_semantic 相同的三档匹配。
    """
    musts = _gold_entity_list(gold, ("must_have",))
    if not musts:
        return MetricResult(
            metric="must_have_recall", value=1.0, task_id=task_id,
            numerator=0, denominator=0, notes="gold has no must_have entities",
        )
    alias_index = _build_gold_index(musts)
    hit_keys: set[str] = set()
    miss_labels: list[str] = []
    for n in extracted_nodes:
        hit, _tier = _match_extracted_to_gold(
            n.get("label", ""), musts, alias_index, use_embedding
        )
        if hit is not None:
            hit_keys.add(_normalize_label(hit.get("label", "")))

    must_keys = {_normalize_label(e.get("label", "")) for e in musts}
    for k in must_keys:
        if k not in hit_keys:
            miss_labels.append(k)

    value = len(hit_keys & must_keys) / len(must_keys)
    return MetricResult(
        metric="must_have_recall", value=value, task_id=task_id,
        numerator=float(len(hit_keys & must_keys)),
        denominator=float(len(must_keys)),
        notes=json.dumps({"missed": miss_labels[:10], "n_missed": len(miss_labels)}),
    )


# ============================================================
# 指标 4: precision_on_known
# ============================================================


def precision_on_known(
    extracted_nodes: list[dict],
    gold: dict,
    *,
    use_embedding: bool = True,
    task_id: str | None = None,
) -> MetricResult:
    """抽取项中能对上 gold 任一档（must/should/may）的比例。

    抽到 gold 未列的实体不算「错」（gold 必然不完备），因此分母是抽取项中
    能被判定为「已知实体」的数量；剩下的归为 unknown，从指标里剔除。

    适合衡量「抽取是否在领域内」，独立于 recall。
    """
    gold_entities = _gold_entity_list(gold, ("must_have", "should_have", "may_have"))
    if not extracted_nodes:
        return MetricResult(
            metric="precision_on_known", value=1.0, task_id=task_id,
            numerator=0, denominator=0, notes="extraction empty",
        )
    alias_index = _build_gold_index(gold_entities)

    known = 0
    unknown = 0
    for n in extracted_nodes:
        hit, _ = _match_extracted_to_gold(
            n.get("label", ""), gold_entities, alias_index, use_embedding
        )
        if hit is not None:
            known += 1
        else:
            unknown += 1

    if known + unknown == 0:
        value = 1.0
    elif known == 0:
        value = 0.0
    else:
        value = known / (known + unknown)
    return MetricResult(
        metric="precision_on_known", value=value, task_id=task_id,
        numerator=float(known), denominator=float(known + unknown),
        notes=json.dumps({"known": known, "unknown": unknown}),
    )
