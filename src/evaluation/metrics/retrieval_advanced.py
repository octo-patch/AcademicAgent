from __future__ import annotations

"""
检索高级指标（v1.0 gold dataset 专用）
=======================================

承担「按 query 维度看 retrieval 健康度」的职责，与基础 retrieval.py 的
recall@k / precision@k / mrr / ndcg 并列报告。

新增三个指标：

- ``match_must_have_rate``：top-k 中是否命中了 gold.must_match 里的强必中项。
  v1.0 retrieval gold 每条 query 会标至少 1 个 must_match —— 这个对不上
  说明 retrieval 在「真该命中的」上就崩了。
- ``synonym_robustness``：同义词维度 query 在 top-k 上的 recall 平均值。
  query 的 ``dimension == "synonym"`` 时计入；其它维度不影响。
- ``cross_domain_leakage``：top-k 检索结果里，落到 query 标的 ``expected_domain``
  之外的占比。需要 task_result.raw 提供 ``retrieved_domains``（每条结果一个 domain 字符串）。

这些都是「按一组 query 聚合」的全局指标，不是 per-task 指标 ——
runner / aggregate 层应在 retrieval 任务跑完后调用 ``aggregate_retrieval_advanced``
一次性算完，结果进 ``retrieval_report.md`` 的维度 breakdown。
"""

import json
from collections import defaultdict
from typing import Iterable

from ..schema import MetricResult, TaskResult


# ============================================================
# 单 query 指标（wrapper 形式，可注册到 METRIC_REGISTRY）
# ============================================================


def _normalize(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def match_must_have_rate(
    retrieved: list[str],
    must_match: list[str],
    *,
    k: int = 5,
    task_id: str | None = None,
) -> MetricResult:
    """top-k 中 must_match 项命中的比例（gold.must_match 必须存在）。"""
    if not must_match:
        return MetricResult(
            metric="match_must_have_rate", value=1.0, task_id=task_id,
            numerator=0, denominator=0, notes="empty must_match",
        )
    top = {_normalize(x) for x in retrieved[:k]}
    must = {_normalize(x) for x in must_match if x}
    hit = top & must
    return MetricResult(
        metric="match_must_have_rate", value=len(hit) / len(must),
        task_id=task_id,
        numerator=float(len(hit)), denominator=float(len(must)),
        notes=json.dumps({"missed": sorted(must - hit)[:8]}),
    )


def cross_domain_leakage(
    retrieved_domains: list[str],
    expected_domain: str,
    *,
    k: int = 10,
    task_id: str | None = None,
) -> MetricResult:
    """top-k 中域漂移条目占比（越低越好）。

    retrieved_domains: 与 retrieved 一一对应的 domain 字符串列表。
    expected_domain : query 标注的目标域（如 "llm_agent" / "comms"）。
    """
    if not expected_domain:
        return MetricResult(
            metric="cross_domain_leakage", value=0.0, task_id=task_id,
            numerator=0, denominator=0, notes="no expected_domain",
        )
    top = retrieved_domains[:k]
    if not top:
        return MetricResult(
            metric="cross_domain_leakage", value=0.0, task_id=task_id,
            numerator=0, denominator=0, notes="no retrieval results",
        )
    expected = _normalize(expected_domain)
    leaked = sum(1 for d in top if _normalize(d) != expected)
    return MetricResult(
        metric="cross_domain_leakage", value=leaked / len(top), task_id=task_id,
        numerator=float(leaked), denominator=float(len(top)),
        notes=json.dumps({"expected": expected_domain, "n_leaked": leaked}),
    )


# ============================================================
# 跨 query 聚合（写报告时调用）
# ============================================================


def synonym_robustness(
    task_results: Iterable[TaskResult],
    golds: dict[str, dict],
    *,
    k: int = 5,
) -> MetricResult:
    """同义词维度 query 在 recall@k 上的平均值。

    task_results : retrieval 任务结果（raw 中含 "retrieved" / "retrieved_ids"）
    golds        : task_id -> gold dict（要求 gold.dimension == "synonym" 的入选）
    """
    relevant: list[float] = []
    for tr in task_results:
        gold = golds.get(tr.task_id) or {}
        if gold.get("dimension") != "synonym":
            continue
        gold_items = gold.get("expected_node_labels", []) or gold.get("gold_titles", []) or []
        if not gold_items:
            continue
        retrieved = tr.raw.get("retrieved_ids") or tr.raw.get("retrieved") or []
        top = {_normalize(x) for x in retrieved[:k]}
        golds_norm = {_normalize(x) for x in gold_items}
        if not golds_norm:
            continue
        relevant.append(len(top & golds_norm) / len(golds_norm))

    if not relevant:
        return MetricResult(
            metric="synonym_robustness", value=0.0,
            numerator=0, denominator=0,
            notes="no synonym-dimension queries in run",
        )
    avg = sum(relevant) / len(relevant)
    return MetricResult(
        metric="synonym_robustness", value=avg,
        numerator=float(sum(relevant)), denominator=float(len(relevant)),
        notes=json.dumps({"n_queries": len(relevant), "k": k}),
    )


def aggregate_by_dimension(
    task_results: Iterable[TaskResult],
    golds: dict[str, dict],
    *,
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """按 gold.dimension 分桶聚合 recall@k 与样本数。

    输出形如 ``{"synonym": {"recall_at_5": 0.62, "n": 4}, ...}``，
    供 retrieval_report.md 渲染维度 breakdown。
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for tr in task_results:
        gold = golds.get(tr.task_id) or {}
        dim = gold.get("dimension", "unspecified")
        gold_items = gold.get("expected_node_labels", []) or gold.get("gold_titles", []) or []
        if not gold_items:
            continue
        retrieved = tr.raw.get("retrieved_ids") or tr.raw.get("retrieved") or []
        top = {_normalize(x) for x in retrieved[:k]}
        golds_norm = {_normalize(x) for x in gold_items}
        if not golds_norm:
            continue
        buckets[dim].append(len(top & golds_norm) / len(golds_norm))

    out: dict[str, dict[str, float]] = {}
    for dim, vals in sorted(buckets.items()):
        out[dim] = {
            f"recall_at_{k}": round(sum(vals) / len(vals), 4),
            "n": len(vals),
        }
    return out
