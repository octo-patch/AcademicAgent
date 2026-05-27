"""Merge codex/claude/gpt review JSONs back into gold/ with majority vote.

Reads:
  datasets/personal_v1/gold/<slug>.json            (原始 gold)
  datasets/personal_v1/gold_reviews/<slug>.*.json  (一篇可有多 reviewer)

Writes:
  datasets/personal_v1/gold_reviews/<slug>.merged.json   (合并候选，不覆盖原 gold)
  docs/gold_diff_v1_to_v2.md                              (人工 diff 报告)

可选：
  --apply   把 merged 直接覆盖到 gold/<slug>.json (危险操作)
  --slug X  只处理一篇

合并规则（majority vote）：
  KEEP        所有 reviewer 都 KEEP                    -> 不动
  PROMOTE     ≥ 半数 reviewer 同向 PROMOTE 到同一档    -> 接受 (单 reviewer 时直接采纳)
  DEMOTE      同上
  DROP        ≥ 半数 reviewer DROP                      -> 接受
  ADD_ALIAS   合并所有 reviewer 提议的 aliases (去重)
  ADD_NEW     ≥ 半数 reviewer ADD_NEW 且 label 相近    -> 接受
              (单 reviewer 时若 evidence 充分，作为 should 接受)
  edges       同上
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = REPO_ROOT / "datasets" / "personal_v1" / "gold"
REV_DIR = REPO_ROOT / "datasets" / "personal_v1" / "gold_reviews"
DIFF_REPORT = REPO_ROOT / "docs" / "gold_diff_v1_to_v2.md"


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _load_reviews(slug: str) -> list[dict]:
    """同一篇的所有 reviewer JSON，文件名形如 <slug>.<reviewer>.json。"""
    out = []
    for p in REV_DIR.glob(f"{slug}.*.json"):
        if p.name.endswith(".merged.json"):
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to parse {p.name}: {exc}")
    return out


def _entity_diff_index(reviews: list[dict], tier_key: str) -> dict[str, list[dict]]:
    """把 reviewers 在 must_have_diff/should_have_diff/may_have_diff 上的判断按 label 聚合。"""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in reviews:
        for item in r.get(tier_key, []) or []:
            by_label[item.get("label", "")].append({
                "reviewer": r.get("reviewer", "?"),
                **item,
            })
    return by_label


def _majority_verdict(entries: list[dict]) -> tuple[str, str, list[dict]]:
    """对一个 label 的多 reviewer 判断投票。返回 (verdict, target_tier_or_empty, supporting_entries)."""
    if not entries:
        return ("KEEP", "", [])

    verdict_counts: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        verdict_counts[e.get("verdict", "KEEP")].append(e)

    # 单 reviewer：直接采纳
    if len(entries) == 1:
        v = entries[0].get("verdict", "KEEP")
        return (v, entries[0].get("target_tier", ""), entries)

    # 多 reviewer：选 ≥ 半数票的 verdict；平票时 KEEP 优先（保守）
    threshold = (len(entries) + 1) // 2  # 半数向上取整
    for v, supporters in verdict_counts.items():
        if v != "KEEP" and len(supporters) >= threshold:
            tts = [s.get("target_tier", "") for s in supporters if s.get("target_tier")]
            # 多数支持同一档
            tt = max(set(tts), key=tts.count) if tts else ""
            return (v, tt, supporters)
    return ("KEEP", "", entries)


def _merge_aliases(existing: list[str], add_alias_entries: list[dict]) -> list[str]:
    """合并 ADD_ALIAS 提议的 aliases，按归一化去重，保留原顺序。"""
    out = list(existing)
    seen = {_norm(a) for a in existing}
    for entry in add_alias_entries:
        for a in entry.get("new_aliases", []) or []:
            n = _norm(a)
            if n and n not in seen:
                out.append(a)
                seen.add(n)
    return out


def _apply_to_tier_list(
    original: list[dict],
    diff_by_label: dict[str, list[dict]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """对单个 tier (must_have / should_have / may_have) 的实体应用 diff。

    返回 (kept_or_modified, demoted_to_other_tier, promoted_to_other_tier)
    drop 的项不在任何返回里。
    """
    kept: list[dict] = []
    demoted: list[dict] = []  # 元素加 _target_tier 字段
    promoted: list[dict] = []
    for ent in original:
        label = ent.get("label", "")
        entries = diff_by_label.get(label, [])
        verdict, target_tier, _ = _majority_verdict(entries)

        if verdict == "DROP":
            continue
        if verdict == "ADD_ALIAS":
            merged = dict(ent)
            merged["aliases"] = _merge_aliases(ent.get("aliases", []) or [], entries)
            kept.append(merged)
        elif verdict == "DEMOTE":
            new_ent = dict(ent)
            new_ent["_target_tier"] = target_tier or "may"
            demoted.append(new_ent)
        elif verdict == "PROMOTE":
            new_ent = dict(ent)
            new_ent["_target_tier"] = target_tier or "must"
            promoted.append(new_ent)
        else:  # KEEP
            kept.append(ent)
    return kept, demoted, promoted


def _apply_additions(reviews: list[dict]) -> tuple[list[dict], list[dict]]:
    """收集所有 reviewer 的 additions，按 label 投票后插入对应 tier。"""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in reviews:
        for a in r.get("additions", []) or []:
            by_label[a.get("label", "")].append({
                "reviewer": r.get("reviewer", "?"),
                **a,
            })
    add_must: list[dict] = []
    add_should: list[dict] = []
    threshold = (len(reviews) + 1) // 2 if len(reviews) > 1 else 1
    for label, entries in by_label.items():
        if len(entries) < threshold:
            continue
        # 取第一个 entry 作 base
        base = entries[0]
        tier = base.get("tier", "should")
        new_ent = {
            "label": label,
            "type": base.get("type", "concept"),
            "aliases": base.get("aliases", []),
            "evidence_section": base.get("evidence", ""),
            "rationale": f"[v1.2 added by {','.join(e['reviewer'] for e in entries)}] {base.get('reason', '')}",
        }
        if tier == "must":
            add_must.append(new_ent)
        else:
            add_should.append(new_ent)
    return add_must, add_should


def _apply_edges(
    original_edges: list[dict],
    reviews: list[dict],
) -> tuple[list[dict], list[dict]]:
    """edges 处理：drop 的删除，relation 改的更新；返回 (final_edges, dropped_for_report)."""
    # 索引 reviewer 对每条 edge 的判断
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in reviews:
        for e in r.get("edges_diff", []) or []:
            key = (_norm(e.get("src", "")), _norm(e.get("rel", "")), _norm(e.get("dst", "")))
            by_key[key].append({"reviewer": r.get("reviewer", "?"), **e})

    final: list[dict] = []
    dropped: list[dict] = []
    for edge in original_edges:
        key = (_norm(edge.get("src", "")), _norm(edge.get("rel", "")), _norm(edge.get("dst", "")))
        entries = by_key.get(key, [])
        verdict, _, _ = _majority_verdict(entries)
        if verdict == "DROP":
            dropped.append(edge)
            continue
        # 处理 new_rel 改动
        new_rels = [e.get("new_rel", "") for e in entries if e.get("new_rel")]
        if new_rels:
            new_edge = dict(edge)
            new_edge["rel"] = new_rels[0]  # 取第一个 reviewer 的建议
            final.append(new_edge)
        else:
            final.append(edge)
    return final, dropped


def merge_one(slug: str) -> dict | None:
    gold_path = GOLD_DIR / f"{slug}.json"
    if not gold_path.exists():
        print(f"[warn] {slug}: gold not found")
        return None
    reviews = _load_reviews(slug)
    if not reviews:
        return None  # 没有 review 不动

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    merged = dict(gold)
    merged.setdefault("annotation_meta", {})
    merged["annotation_meta"]["reviewed_by"] = sorted({r.get("reviewer", "?") for r in reviews})
    merged["annotation_meta"]["reviewed_at"] = "2026-05-23"
    merged["annotation_meta"]["v1_2_merged"] = True

    # ---- 实体三档处理 ----
    must_diff = _entity_diff_index(reviews, "must_have_diff")
    should_diff = _entity_diff_index(reviews, "should_have_diff")
    may_diff = _entity_diff_index(reviews, "may_have_diff")

    must_kept, must_demoted, _ = _apply_to_tier_list(gold.get("must_have", []) or [], must_diff)
    should_kept, should_demoted, should_promoted = _apply_to_tier_list(
        gold.get("should_have", []) or [], should_diff)
    may_kept, _, may_promoted = _apply_to_tier_list(gold.get("may_have", []) or [], may_diff)

    add_must, add_should = _apply_additions(reviews)

    # 实际应用 PROMOTE / DEMOTE 的跨档移动
    new_must = must_kept + [{k: v for k, v in e.items() if k != "_target_tier"}
                             for e in should_promoted if e.get("_target_tier") == "must"]
    new_should = (
        should_kept
        + [{k: v for k, v in e.items() if k != "_target_tier"}
           for e in must_demoted if e.get("_target_tier") == "should"]
        + [{k: v for k, v in e.items() if k != "_target_tier"}
           for e in may_promoted if e.get("_target_tier") == "should"]
        + add_should
    )
    new_may = (
        may_kept
        + [{k: v for k, v in e.items() if k != "_target_tier"}
           for e in must_demoted if e.get("_target_tier") == "may"]
        + [{k: v for k, v in e.items() if k != "_target_tier"}
           for e in should_demoted if e.get("_target_tier") == "may"]
    )
    # 单独的 ADD_NEW must
    new_must = new_must + add_must

    merged["must_have"] = new_must
    merged["should_have"] = new_should
    merged["may_have"] = new_may

    # ---- edges ----
    final_edges, dropped_edges = _apply_edges(gold.get("expected_edges", []) or [], reviews)
    merged["expected_edges"] = final_edges

    # ---- diff summary ----
    diff_stats = {
        "demoted_from_must": len(must_demoted),
        "demoted_from_should": len(should_demoted),
        "promoted_from_should": len(should_promoted),
        "promoted_from_may": len(may_promoted),
        "added_must": len(add_must),
        "added_should": len(add_should),
        "dropped_edges": len(dropped_edges),
        "n_reviewers": len(reviews),
    }
    merged["annotation_meta"]["v1_2_diff_stats"] = diff_stats
    return merged


def render_diff_row(slug: str, gold: dict, merged: dict) -> str:
    def _cnt(d, k):
        return len(d.get(k, []) or [])
    rows = []
    rows.append(f"## {slug}")
    rows.append("")
    rows.append("| tier | v1.1 | v1.2 | Δ |")
    rows.append("|---|---|---|---|")
    for k in ("must_have", "should_have", "may_have", "expected_edges"):
        a, b = _cnt(gold, k), _cnt(merged, k)
        d = b - a
        sign = f"+{d}" if d > 0 else str(d)
        rows.append(f"| {k} | {a} | {b} | {sign} |")
    rows.append("")
    stats = merged.get("annotation_meta", {}).get("v1_2_diff_stats", {})
    if stats:
        rows.append(f"diff stats: {stats}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", help="只处理一篇")
    parser.add_argument("--apply", action="store_true",
                        help="把 merged 直接覆盖到 gold/<slug>.json (危险)")
    args = parser.parse_args(argv)

    if args.slug:
        slugs = [args.slug]
    else:
        # 找所有有 review 的 slug
        slugs = sorted({p.name.split(".")[0]
                        for p in REV_DIR.glob("*.json")
                        if not p.name.endswith(".merged.json")})

    if not slugs:
        print("nothing to merge. put review JSONs in datasets/personal_v1/gold_reviews/")
        return 0

    diff_lines = [
        "# Gold v1.1 → v1.2 Diff Report",
        "",
        "本报告由 `scripts/apply_gold_reviews.py` 生成。",
        "每篇按 majority vote 合并 reviewer 建议，**输出为 `<slug>.merged.json`，不直接覆盖原 gold**。",
        "用 `--apply` 才会覆盖。",
        "",
    ]

    n_merged = 0
    n_unchanged = 0
    for slug in slugs:
        merged = merge_one(slug)
        if merged is None:
            print(f"[skip] {slug}: no reviews or no gold")
            continue
        gold = json.loads((GOLD_DIR / f"{slug}.json").read_text(encoding="utf-8"))
        # 比较节点/边数变化
        is_changed = any(
            len(merged.get(k, []) or []) != len(gold.get(k, []) or [])
            for k in ("must_have", "should_have", "may_have", "expected_edges")
        )
        out_path = REV_DIR / f"{slug}.merged.json"
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        diff_lines.append(render_diff_row(slug, gold, merged))
        diff_lines.append("")
        if is_changed:
            n_merged += 1
            print(f"[merged]   {slug}: -> {out_path.relative_to(REPO_ROOT)}")
        else:
            n_unchanged += 1
            print(f"[no-diff]  {slug} (review 全 KEEP)")

        if args.apply:
            (GOLD_DIR / f"{slug}.json").write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"           applied to {(GOLD_DIR / f'{slug}.json').relative_to(REPO_ROOT)}")

    DIFF_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DIFF_REPORT.write_text("\n".join(diff_lines), encoding="utf-8")
    print(f"\n=== {n_merged} merged / {n_unchanged} unchanged, diff report: {DIFF_REPORT.relative_to(REPO_ROOT)} ===")
    if not args.apply:
        print("[tip] 检查 *.merged.json 与 diff 报告确认后，跑 --apply 把 merged 写回 gold/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
