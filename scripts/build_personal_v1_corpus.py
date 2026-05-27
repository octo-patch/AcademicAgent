"""Build datasets/personal_v1/fixtures/papers_text/<slug>.txt.

每个 gold/<slug>.json 对应一篇论文，从其 annotation_meta.evidence_pdf 抽 PDF text，
裁剪到 ~2000 字符（abstract + intro 头部足以触发抽取），写到 fixtures/papers_text/<slug>.txt。

幂等：已存在的 fixture 跳过；--force 覆盖。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GOLD_DIR = REPO_ROOT / "datasets" / "personal_v1" / "gold"
OUT_DIR = REPO_ROOT / "datasets" / "personal_v1" / "fixtures" / "papers_text"
MAX_CHARS = 8000  # v1.2: 8000 chars 平均 2 chunks @ max_chunk_chars=4000，触发分块以验证 chunk 效果

# 一些通信论文标题里有空格 / (1)，PDF 路径在 EE_papers/，需要原样保留
RETRIEVAL = "retrieval.jsonl"  # 跳过 retrieval gold 文件


def _extract_corpus(pdf_path: Path) -> str:
    """抽 PDF 全文，截取前 MAX_CHARS。"""
    from src.core.pdf_parser import PDFParser

    with PDFParser(str(pdf_path)) as parser:
        full = parser.extract_full_text() or ""
    # 简单 normalize：吃掉多余空行
    lines = [l.rstrip() for l in full.splitlines()]
    text = "\n".join(l for l in lines if l).strip()
    if len(text) <= MAX_CHARS:
        return text
    # 截到最近的句号 / 换行避免在词中间切
    cut = text[:MAX_CHARS]
    last_dot = max(cut.rfind(". "), cut.rfind("\n"))
    if last_dot > MAX_CHARS * 0.7:
        cut = cut[: last_dot + 1]
    return cut


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="覆盖已有 fixture")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只处理指定 slug（不带 .json 后缀）",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_files = sorted(p for p in GOLD_DIR.glob("*.json") if p.name != RETRIEVAL)
    print(f"[info] {len(gold_files)} gold files")

    ok = 0
    skipped = 0
    failed: list[str] = []

    for gf in gold_files:
        slug = gf.stem
        if args.only and slug not in args.only:
            continue

        out_path = OUT_DIR / f"{slug}.txt"
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            gold = json.loads(gf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{slug}: gold parse failed: {exc}")
            continue

        pdf_rel = (gold.get("annotation_meta") or {}).get("evidence_pdf", "")
        if not pdf_rel:
            failed.append(f"{slug}: missing annotation_meta.evidence_pdf")
            continue

        pdf_path = (REPO_ROOT / pdf_rel).resolve()
        if not pdf_path.exists():
            # 兼容：可能直接是绝对 / cwd 相对
            alt = Path(pdf_rel)
            if alt.exists():
                pdf_path = alt
            else:
                failed.append(f"{slug}: PDF not found at {pdf_rel}")
                continue

        try:
            text = _extract_corpus(pdf_path)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{slug}: PDF extract failed: {exc}")
            continue

        if len(text) < 200:
            failed.append(f"{slug}: extracted text too short ({len(text)} chars)")
            continue

        out_path.write_text(text, encoding="utf-8")
        print(f"[ok]  {slug:40s} {len(text):5d} chars -> {out_path.relative_to(REPO_ROOT)}")
        ok += 1

    print(f"\n=== summary: ok={ok}, skipped={skipped}, failed={len(failed)} ===")
    for f in failed:
        print(f"[fail] {f}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
