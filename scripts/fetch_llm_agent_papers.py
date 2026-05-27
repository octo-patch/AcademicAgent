"""
下载 v1.0 gold dataset 缺失的 11 篇 LLM Agent 论文 PDF
=====================================================

handoff §3.4 已有 4 篇本地：ReAct / Generative Agents / MemGPT / LLM Agent Survey
本脚本拉剩余 11 篇 arXiv PDF 到 data/papers/。

幂等：目标文件存在则跳过；下载失败的论文返回非零退出码并打印清单。

运行：
    python scripts/fetch_llm_agent_papers.py            # 串行下载
    python scripts/fetch_llm_agent_papers.py --dry-run  # 只打印计划
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"

ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"
USER_AGENT = "ScholarMind/v1.0 academic-research (vefalun@gmail.com)"
TIMEOUT_S = 60.0
RETRIES = 3
RETRY_BACKOFF_S = 5.0


@dataclass(frozen=True)
class PaperRef:
    name: str
    arxiv_id: str
    filename: str  # 落到 data/papers/<filename>


# handoff §3.4 中标 "arXiv ⬇" 的 11 篇
PAPERS: list[PaperRef] = [
    PaperRef("Toolformer (Schick 2023)",            "2302.04761", "Toolformer_arxiv_2302_04761.pdf"),
    PaperRef("Reflexion (Shinn 2023)",              "2303.11366", "Reflexion_arxiv_2303_11366.pdf"),
    PaperRef("Self-Refine (Madaan 2023)",           "2303.17651", "Self_Refine_arxiv_2303_17651.pdf"),
    PaperRef("Voyager (Wang 2023)",                 "2305.16291", "Voyager_arxiv_2305_16291.pdf"),
    PaperRef("Tree of Thoughts (Yao 2023)",         "2305.10601", "Tree_of_Thoughts_arxiv_2305_10601.pdf"),
    PaperRef("AutoGen (Wu 2023)",                   "2308.08155", "AutoGen_arxiv_2308_08155.pdf"),
    PaperRef("AgentBench (Liu 2023)",               "2308.03688", "AgentBench_arxiv_2308_03688.pdf"),
    PaperRef("CoALA (Sumers 2023)",                 "2309.02427", "CoALA_arxiv_2309_02427.pdf"),
    PaperRef("OS-Copilot (Wu 2024)",                "2402.07456", "OS_Copilot_arxiv_2402_07456.pdf"),
    PaperRef("AppAgent (Yang 2023)",                "2312.13771", "AppAgent_arxiv_2312_13771.pdf"),
    PaperRef("SWE-Agent (Yang 2024)",               "2405.15793", "SWE_Agent_arxiv_2405_15793.pdf"),
]


def _download(client: httpx.Client, ref: PaperRef, dst: Path) -> tuple[bool, str]:
    url = ARXIV_PDF.format(arxiv_id=ref.arxiv_id)
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            with client.stream("GET", url, follow_redirects=True, timeout=TIMEOUT_S) as r:
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    if r.status_code in (429, 503):
                        time.sleep(RETRY_BACKOFF_S * attempt)
                        continue
                    return False, last_err
                tmp = dst.with_suffix(dst.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                size = tmp.stat().st_size
                if size < 50_000:
                    tmp.unlink(missing_ok=True)
                    last_err = f"too small ({size} bytes), likely error page"
                    continue
                tmp.replace(dst)
                return True, f"{size/1024:.1f} KB"
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(RETRY_BACKOFF_S * attempt)
    return False, last_err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只打印计划")
    args = parser.parse_args()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'DRY RUN' if args.dry_run else 'EXECUTE'}: 计划拉取 {len(PAPERS)} 篇")
    print(f"目标目录: {PAPERS_DIR}")
    print()

    todo: list[PaperRef] = []
    for ref in PAPERS:
        dst = PAPERS_DIR / ref.filename
        if dst.exists():
            print(f"  [skip] {ref.filename} ({ref.name}) 已存在 {dst.stat().st_size/1024:.1f} KB")
        else:
            todo.append(ref)
            print(f"  [todo] {ref.filename}  <-  arXiv {ref.arxiv_id} ({ref.name})")

    if args.dry_run or not todo:
        print()
        print(f"待下载: {len(todo)} / 已就位: {len(PAPERS)-len(todo)}")
        return 0

    print()
    print(f"开始下载 {len(todo)} 篇 …")
    print()
    failures: list[tuple[PaperRef, str]] = []
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers) as client:
        for i, ref in enumerate(todo, 1):
            dst = PAPERS_DIR / ref.filename
            print(f"  [{i}/{len(todo)}] {ref.name} ... ", end="", flush=True)
            ok, msg = _download(client, ref, dst)
            print("OK " + msg if ok else f"FAIL {msg}")
            if not ok:
                failures.append((ref, msg))
            # 礼貌停顿，arXiv 对快速连击不友好
            if i < len(todo):
                time.sleep(2.0)

    print()
    if failures:
        print(f"失败 {len(failures)} / {len(todo)}：")
        for ref, msg in failures:
            print(f"  - {ref.filename} (arXiv {ref.arxiv_id}): {msg}")
        return 1
    print(f"全部 {len(todo)} 篇下载完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
