"""装完 sentence-transformers 后重建双域 KG 的 embedding 索引。

用法：
    PYTHONPATH=. python scripts/rebuild_semantic_index.py
    PYTHONPATH=. python scripts/rebuild_semantic_index.py --domain llm_agent
    PYTHONPATH=. python scripts/rebuild_semantic_index.py --smoke

效果：
    - 删除旧 TF-IDF cache（避免维度不一致）
    - 用 sentence-transformers 对每个 domain 全量 encode
    - 写盘新 cache (data/semantic_index_<domain>.npz)
    - 可选 --smoke 跑几个 query 对比新旧召回质量
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 项目根加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.knowledge.graph_store import KnowledgeGraphStore
from src.knowledge.semantic_index import SemanticIndex
from src.recorder import RefactorRecorder

SMOKE_QUERIES = {
    "llm_agent": ["reasoning agent", "memory augmented", "tool calling", "reflection"],
    "comms": ["SLAM positioning", "beamforming", "CRLB", "channel model"],
    "shared": ["neural network", "optimization", "Bayesian inference"],
}


def rebuild_one(domain: str, smoke: bool = False) -> dict:
    """对单个 domain 重建索引并返回统计。"""
    kg_path = _ROOT / "data" / f"kg_{domain}.json"
    if not kg_path.exists():
        print(f"  skip [{domain}]: {kg_path.name} 不存在")
        return {"domain": domain, "status": "skipped", "reason": "kg file not found"}

    store = KnowledgeGraphStore(graph_path=kg_path, domain=domain)
    if store.node_count == 0:
        print(f"  skip [{domain}]: empty store")
        return {"domain": domain, "status": "skipped", "reason": "empty"}

    # 删旧 cache
    old_cache = _ROOT / "data" / f"semantic_index_{domain}.npz"
    if old_cache.exists():
        old_cache.unlink()
        print(f"  removed old cache: {old_cache.name}")

    # 新建索引（sentence-transformers backend）
    t0 = time.time()
    idx = SemanticIndex(domain=domain)
    print(f"  backend: {idx.backend}")
    idx.build_from_store(store)
    elapsed = time.time() - t0

    idx.save_cache()
    summary = {
        "domain": domain,
        "status": "rebuilt",
        "backend": idx.backend,
        "nodes": idx.size,
        "dim": int(idx._embeddings.shape[1]),
        "elapsed_s": round(elapsed, 2),
    }
    print(f"  [{domain}] {idx.size} nodes × {summary['dim']} dim, {summary['elapsed_s']}s, backend={idx.backend}")

    if smoke:
        queries = SMOKE_QUERIES.get(domain, [])
        smoke_results = {}
        for q in queries:
            hits = idx.search(q, top_k=3)
            smoke_results[q] = [
                (h["label"][:50], round(h["similarity"], 3))
                for h in hits
            ]
            print(f"  query {q!r:30s} -> {smoke_results[q]}")
        summary["smoke"] = smoke_results

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["llm_agent", "comms", "shared"],
                        help="只重建某一域；不指定则全部")
    parser.add_argument("--smoke", action="store_true",
                        help="重建后跑 smoke queries 看召回质量")
    parser.add_argument("--no-record", action="store_true",
                        help="不同步到 RefactorRecorder")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else ["llm_agent", "comms"]

    print("===== Rebuilding semantic indexes =====")
    summaries = []
    for d in domains:
        print(f"\n[{d}]")
        summaries.append(rebuild_one(d, smoke=args.smoke))

    if not args.no_record:
        try:
            r = RefactorRecorder(area="kg-quality")
            lines = []
            for s in summaries:
                if s["status"] == "rebuilt":
                    lines.append(
                        f"{s['domain']}: {s['nodes']} nodes × {s['dim']} dim, "
                        f"{s['elapsed_s']}s, backend={s['backend']}"
                    )
                    if "smoke" in s:
                        for q, hits in s["smoke"].items():
                            lines.append(f"  {q!r:30s} -> {hits[0]}")
                else:
                    lines.append(f"{s['domain']}: {s['status']} ({s.get('reason','')})")
            r.log_validation(
                step="semantic-index-rebuild-with-sentence-transformers",
                command=" ".join(sys.argv),
                output="\n".join(lines),
                status="ok",
                notes="装完 sentence-transformers 后双域 KG 重建嵌入；TF-IDF fallback 路径自动让位",
                artifacts=[
                    str(_ROOT / "data" / f"semantic_index_{d}.npz")
                    for d in domains
                ],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"recorder log failed (non-fatal): {exc}")

    print("\n===== Done =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
