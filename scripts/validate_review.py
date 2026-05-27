"""Validate a single gold-review JSON returned by Codex/Claude/GPT.

Usage:
  python scripts/validate_review.py datasets/personal_v1/gold_reviews/react.codex.json
  python scripts/validate_review.py --all   # 校验 gold_reviews/ 下所有 *.json

校验项：
  - JSON 合法
  - 顶层字段齐全 (paper_slug / reviewer / verdict_summary / *_diff / additions / global_notes)
  - 每个 diff item 的 verdict 在合法枚举内
  - 关系词 / type 在词表内
  - evidence 非空
  - 引用的 label 与主 gold 对得上（除 additions / ADD_NEW）
  - verdict_summary 计数与 diff 数量一致
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = REPO_ROOT / "datasets" / "personal_v1" / "gold"
REV_DIR = REPO_ROOT / "datasets" / "personal_v1" / "gold_reviews"

VALID_VERDICTS = {"KEEP", "PROMOTE", "DEMOTE", "DROP", "ADD_ALIAS"}
VALID_EDGE_VERDICTS = {"KEEP", "PROMOTE", "DEMOTE", "DROP"}
VALID_TIERS = {"must", "should", "may", "drop"}
VALID_TYPES = {"method", "concept", "tool", "dataset", "metric", "paper", "author", "other"}
VALID_RELATIONS = {
    "uses", "improves", "extends", "supersedes", "compares_with", "cites", "authored_by",
    "addresses", "is_a", "applied_in", "proposes", "belongs_to", "enables", "requires",
    "pretrained_on", "evaluated_against", "tested_on", "evaluated_by", "related_to",
}


def _load_gold(slug: str) -> dict | None:
    p = GOLD_DIR / f"{slug}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _gold_labels(gold: dict) -> set[str]:
    out = set()
    for tier in ("must_have", "should_have", "may_have"):
        for e in gold.get(tier, []) or []:
            if e.get("label"):
                out.add(e["label"])
    return out


def validate_review(review: dict, source: str) -> list[str]:
    """返回问题列表，空列表 = 通过。"""
    problems: list[str] = []
    P = problems.append

    # ---- 顶层 ----
    for key in ("paper_slug", "reviewer", "verdict_summary"):
        if key not in review:
            P(f"missing top-level field: {key}")

    slug = review.get("paper_slug", "")
    gold = _load_gold(slug) if slug else None
    if not gold:
        P(f"paper_slug {slug!r} 不在 datasets/personal_v1/gold/")
        gold_labels: set[str] = set()
    else:
        gold_labels = _gold_labels(gold)

    # ---- *_diff ----
    counts = {"keep": 0, "promote": 0, "demote": 0, "drop": 0, "add_alias": 0, "add_new": 0}
    for tier_key in ("must_have_diff", "should_have_diff", "may_have_diff"):
        for i, item in enumerate(review.get(tier_key, []) or []):
            ctx = f"{tier_key}[{i}] (label={item.get('label','?')!r})"
            if "label" not in item:
                P(f"{ctx}: missing label")
                continue
            verdict = item.get("verdict", "")
            if verdict not in VALID_VERDICTS:
                P(f"{ctx}: invalid verdict {verdict!r}")
                continue
            counts[verdict.lower()] = counts.get(verdict.lower(), 0) + 1
            # label 应当在原 gold 中
            if gold_labels and item["label"] not in gold_labels:
                P(f"{ctx}: label not in original gold ({item['label']!r})")
            # PROMOTE/DEMOTE 必须给 target_tier
            if verdict in ("PROMOTE", "DEMOTE"):
                tt = item.get("target_tier", "")
                if tt not in VALID_TIERS:
                    P(f"{ctx}: PROMOTE/DEMOTE 缺 target_tier 或非法 ({tt!r})")
            # ADD_ALIAS 必须给 new_aliases
            if verdict == "ADD_ALIAS" and not item.get("new_aliases"):
                P(f"{ctx}: ADD_ALIAS 缺 new_aliases")
            # evidence 非空
            if not (item.get("evidence") or "").strip():
                P(f"{ctx}: empty evidence")

    # ---- edges_diff ----
    for i, e in enumerate(review.get("edges_diff", []) or []):
        ctx = f"edges_diff[{i}] ({e.get('src','?')}--{e.get('rel','?')}->{e.get('dst','?')})"
        if e.get("verdict") not in VALID_EDGE_VERDICTS:
            P(f"{ctx}: invalid edge verdict {e.get('verdict')!r}")
        if e.get("new_rel") and e["new_rel"] not in VALID_RELATIONS:
            P(f"{ctx}: new_rel {e['new_rel']!r} 不在关系词词表")
        if not (e.get("evidence") or "").strip():
            P(f"{ctx}: empty evidence")
        counts[e.get("verdict", "").lower()] = counts.get(e.get("verdict", "").lower(), 0) + 1

    # ---- additions ----
    additions = review.get("additions", []) or []
    counts["add_new"] = len(additions)
    if len(additions) > 5:
        P(f"additions: 有 {len(additions)} 项，超过合理上限（建议 ≤ 5），审稿人可能过度发挥")
    for i, a in enumerate(additions):
        ctx = f"additions[{i}] ({a.get('label','?')})"
        if a.get("tier") not in {"must", "should"}:
            P(f"{ctx}: ADD_NEW tier 必须是 must/should，不接受 may")
        if a.get("type") not in VALID_TYPES:
            P(f"{ctx}: invalid type {a.get('type')!r}")
        if not (a.get("aliases") or []):
            P(f"{ctx}: ADD_NEW 缺 aliases")
        if not (a.get("evidence") or "").strip():
            P(f"{ctx}: empty evidence")

    # ---- verdict_summary 计数一致性 ----
    summary = review.get("verdict_summary", {}) or {}
    expected_total = (counts["keep"] + counts["promote"] + counts["demote"]
                      + counts["drop"] + counts["add_alias"] + counts["add_new"])
    if "total_items" in summary and summary["total_items"] != expected_total:
        P(f"verdict_summary.total_items={summary['total_items']} != 实际计 {expected_total}")
    for k in ("keep", "promote", "demote", "drop", "add_alias", "add_new"):
        if k in summary and summary[k] != counts.get(k, 0):
            P(f"verdict_summary.{k}={summary[k]} != 实际计 {counts.get(k, 0)}")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="review JSON file(s)")
    parser.add_argument("--all", action="store_true", help="校验 gold_reviews/ 下所有 *.json (除 _pending/)")
    args = parser.parse_args(argv)

    if args.all:
        files = [p for p in REV_DIR.glob("*.json") if "_pending" not in str(p)]
    else:
        files = [Path(f) for f in args.files]

    if not files:
        print("nothing to validate. give files or use --all")
        return 0

    n_ok = 0
    n_fail = 0
    for fp in files:
        if not fp.exists():
            print(f"[fail] {fp}: file not found")
            n_fail += 1
            continue
        try:
            review = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {fp.name}: JSON parse error: {exc}")
            n_fail += 1
            continue
        problems = validate_review(review, source=str(fp))
        if not problems:
            print(f"[ok]   {fp.name}")
            n_ok += 1
        else:
            print(f"[fail] {fp.name}:")
            for p in problems:
                print(f"        - {p}")
            n_fail += 1

    print(f"\n=== {n_ok} ok / {n_fail} fail ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
