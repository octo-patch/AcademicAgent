"""Build personal_v1 dataset wiring files:

- tasks.jsonl           : 30 KG + 25 retrieval (+ optional 2 workflow + edge cases) TaskSpec
- kg_gold.jsonl         : flattened from gold/<slug>.json (one row per paper, keyed by task_id)
- retrieval_gold.jsonl  : copy of gold/retrieval.jsonl (already in TaskSpec-keyed format)
- dataset_version.json  : sha256 + layer counts

Idempotent. Runs from repo root.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DS = REPO_ROOT / "datasets" / "personal_v1"
GOLD_DIR = DS / "gold"
FIX_DIR = DS / "fixtures" / "papers_text"
TASKS_FILE = DS / "tasks.jsonl"
KG_GOLD_FILE = DS / "kg_gold.jsonl"
RETRIEVAL_GOLD_FILE = DS / "retrieval_gold.jsonl"
VERSION_FILE = DS / "dataset_version.json"


# ----------- 共用 metric 配置（gold.metrics）------------ #

KG_METRICS = [
    "kg_node_f1",
    "kg_node_f1_semantic",
    "kg_edge_f1",
    "kg_edge_f1_semantic",
    "must_have_recall",
    "precision_on_known",
    "schema_validity_rate",
    "extraction_nonempty_rate",
]

RETRIEVAL_METRICS = [
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "mrr",
    "ndcg_at_10",
    "match_must_have_rate",
    "cross_domain_leakage",
]


# ----------- 主流程 ------------ #


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_kg_tasks() -> tuple[list[dict], list[dict]]:
    """读 gold/*.json，输出 (kg tasks, kg gold rows)。

    任务规格：layer1_component / kg_extraction / smoke / add_paper_to_graph，
    args.text_file 引向 fixtures/papers_text/<slug>.txt。
    """
    tasks: list[dict] = []
    rows: list[dict] = []

    gold_files = sorted(p for p in GOLD_DIR.glob("*.json") if p.name != "retrieval.jsonl")
    for gf in gold_files:
        gold = json.loads(gf.read_text(encoding="utf-8"))
        slug = gf.stem
        task_id = f"kg_extract_{slug}"
        text_file = (FIX_DIR / f"{slug}.txt").relative_to(REPO_ROOT).as_posix()
        if not (REPO_ROOT / text_file).exists():
            raise FileNotFoundError(f"missing text fixture for {slug}: {text_file}")

        task = {
            "task_id": task_id,
            "layer": "layer1_component",
            "capability": "kg_extraction",
            "tier": "smoke",
            "target": {
                "tool": "add_paper_to_graph",
                "args": {
                    "text_file": text_file,
                    "paper_title": gold.get("paper_title", ""),
                    "paper_year": str(gold.get("year", "")),
                },
            },
            "gold": {
                "gold_file": "kg_gold.jsonl",
                "gold_key": task_id,
                "metrics": KG_METRICS,
            },
            "timeout_s": 180,
            "requires_api": False,
            "requires_llm": True,
            "tags": ["kg", gold.get("domain", "?"), slug],
            "notes": f"v1.0 personal gold: {gold.get('archetype', '')}",
        }
        tasks.append(task)

        row = dict(gold)
        # dataset.gold_for() 取 key=task_id 的 row；用 task_id 替换原 task_id 字段
        row["task_id"] = task_id
        # 标记原 slug 便于回溯
        row["paper_slug"] = slug
        # 兼容历史 strict 指标使用 expected_nodes / expected_edges：合并三档 entity
        expected_nodes_strict = []
        for tier in ("must_have", "should_have", "may_have"):
            for e in gold.get(tier, []) or []:
                expected_nodes_strict.append(
                    {"label": e["label"], "type": e.get("type", "concept")}
                )
        row["expected_nodes"] = expected_nodes_strict
        rows.append(row)

    return tasks, rows


def build_retrieval_tasks() -> tuple[list[dict], list[dict]]:
    """读 gold/retrieval.jsonl，输出 retrieval tasks + retrieval gold rows。"""
    src = GOLD_DIR / "retrieval.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"missing retrieval gold: {src}")
    tasks: list[dict] = []
    rows: list[dict] = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gold = json.loads(line)
            qid = gold["query_id"]
            task_id = f"retrieval_{qid}"
            task = {
                "task_id": task_id,
                "layer": "layer1_component",
                "capability": "kg_query",
                "tier": "smoke",
                "target": {
                    "tool": "query_knowledge",
                    "args": {"query": gold["query"]},
                },
                "gold": {
                    "gold_file": "retrieval_gold.jsonl",
                    "gold_key": task_id,
                    "metrics": RETRIEVAL_METRICS,
                },
                "timeout_s": 30,
                "requires_api": False,
                "requires_llm": False,
                "tags": ["retrieval", gold.get("dimension", "?"), gold.get("expected_domain", "?")],
                "notes": f"v1.0 retrieval gold: {gold.get('notes', '')}",
            }
            tasks.append(task)
            row = dict(gold)
            row["task_id"] = task_id
            # 让 sidecar seed 合并 KG，否则 query_knowledge 在空 KG 上查回 []
            row["seed_graph"] = "fixtures/seed_kg_combined.json"
            rows.append(row)
    return tasks, rows


def main() -> int:
    kg_tasks, kg_rows = build_kg_tasks()
    print(f"[info] KG tasks: {len(kg_tasks)}")

    retr_tasks, retr_rows = build_retrieval_tasks()
    print(f"[info] retrieval tasks: {len(retr_tasks)}")

    all_tasks = kg_tasks + retr_tasks
    print(f"[info] total tasks: {len(all_tasks)}")

    # 写 tasks.jsonl
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {TASKS_FILE.relative_to(REPO_ROOT)}")

    # 写 kg_gold.jsonl
    with open(KG_GOLD_FILE, "w", encoding="utf-8") as f:
        for r in kg_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {KG_GOLD_FILE.relative_to(REPO_ROOT)}")

    # 写 retrieval_gold.jsonl
    with open(RETRIEVAL_GOLD_FILE, "w", encoding="utf-8") as f:
        for r in retr_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {RETRIEVAL_GOLD_FILE.relative_to(REPO_ROOT)}")

    # 写 dataset_version.json
    text = TASKS_FILE.read_text(encoding="utf-8")
    version = {
        "version": "1.0.0",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_count_by_layer": {
            "layer1_component": len(all_tasks),
        },
        "tasks_sha256": _sha256_text(text),
    }
    VERSION_FILE.write_text(json.dumps(version, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {VERSION_FILE.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
