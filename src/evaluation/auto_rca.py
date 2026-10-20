"""
LLM 自动 RCA（Phase 6.4）
=========================

输入：data/evaluation/runs/<run_id>/ 全部产物
输出：<run_id>/rca.md  + （可选）通过 RefactorRecorder("eval-runs") 同步到 Notion

流程：
  1. 读 run_summary.json + failure_cards.jsonl + traces.jsonl
  2. 按 category 分桶（timeout / tool_exception / metric_below_threshold / contamination ...）
  3. 每桶：取代表性证据片段 + 让 MiniMax 生成 RCA（根因假设 + 证据 + 修复建议）
     —— 没装 MiniMax / 没 key / use_llm=False 时退化为基于规则表的根因写入
  4. 输出 rca.md + 简单 mermaid pie chart（失败分桶分布）

CLI：python -m src.evaluation.cli rca --run data/evaluation/runs/<run_id>
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.core.minimax_models import (
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_OPENAI_BASE_URL,
)

logger = logging.getLogger("AcademicAgent.Eval.AutoRCA")


# ============================================================
# IO 辅助
# ============================================================


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


# ============================================================
# 分桶
# ============================================================


def _bucket_failures(cards: list[dict]) -> dict[str, list[dict]]:
    """按 category 分桶。"""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in cards:
        cat = c.get("category") or "unclassified"
        buckets[cat].append(c)
    return dict(buckets)


def _bucket_summary(buckets: dict[str, list[dict]]) -> dict[str, int]:
    return {cat: len(items) for cat, items in sorted(buckets.items())}


def _trace_excerpt(traces: list[dict], task_id: str, limit: int = 6) -> list[dict]:
    """从 traces.jsonl 里挑该 task 的最近 limit 条 event。"""
    rel = [t for t in traces if t.get("task_id") == task_id]
    return rel[-limit:]


# ============================================================
# RCA 文本生成
# ============================================================


_RULE_BASED_HINTS = {
    "tool_exception": (
        "工具内部异常或外部依赖不可用。常见：API key 缺失、网络故障、参数类型错。",
        "1) 检查相关 API 凭证；2) 在 adapter 侧加更明确的输入校验；3) 跑 --offline 复现确认是否依赖外部。",
    ),
    "timeout": (
        "工具长时间无响应或 LLM 调用超时。",
        "1) 收紧 REQUEST_TIMEOUT；2) 缩短 prompt；3) 对相关工具加 retry/backoff 上限。",
    ),
    "metric_below_threshold": (
        "指标低于 thresholds.yaml 设定下限。可能是能力退化或阈值过严。",
        "1) 比对 baseline；2) 若是 gold 偏差，更新 gold；3) 否则把退化样本加入回归集。",
    ),
    "extraction_empty": (
        "抽取器返回 nodes=0。常见：文本过短、prompt 出了 LLM 不擅长的格式、模型切错。",
        "1) 看 traces.jsonl 里 extractor 的 stdin/stdout；2) 单独跑 add_paper_to_graph 复现；3) 提示工程加示例。",
    ),
    "schema_validation": (
        "LLM 输出无法通过 Pydantic 严格校验。",
        "1) 看 had_parse_fallback 是否触发；2) 收紧 prompt 要求的字段；3) 切到 reasoning 模型重试。",
    ),
    "contamination": (
        "受保护文件 sha256 在 run 前后变化，说明评测路径意外写到了真实数据。",
        "1) 立即停止所有 run；2) 看 traces.jsonl 找到哪个 adapter 写出去的；3) 修补 isolation.py 的 env override。",
    ),
}


def _rule_based_rca(cat: str, items: list[dict]) -> dict[str, str]:
    hyp, fix = _RULE_BASED_HINTS.get(
        cat,
        ("（未识别 category，建议人工分析 traces）", "看 traces.jsonl 锁定第一条 error event 即可。"),
    )
    sample = items[0] if items else {}
    return {
        "hypothesis": hyp,
        "evidence": sample.get("trace_excerpt", "")[:600]
                     or sample.get("detail", "")[:600]
                     or "(no trace excerpt)",
        "fix": fix,
    }


def _llm_rca(
    cat: str,
    items: list[dict],
    traces_by_task: dict[str, list[dict]],
) -> dict[str, str] | None:
    """调 MiniMax 生成根因假设。失败返回 None，让上层 fallback 到规则表。

    用 OpenAI-compatible client（同 extractor.py 的路径）：MINIMAX_API_KEY +
    MINIMAX_BASE_URL; MINIMAX_MODEL defaults to MiniMax-M3.1-Flash.
    """
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return None

    samples = []
    for c in items[:3]:
        excerpt = c.get("trace_excerpt", "") or ""
        traces_for = traces_by_task.get(c.get("task_id", ""), [])
        traces_text = "\n".join(
            f"  - {e.get('event')}/{e.get('status','')}: {str(e.get('payload',''))[:200]}"
            for e in traces_for[-4:]
        )
        samples.append(
            f"task={c.get('task_id','?')} severity={c.get('severity','?')}\n"
            f"detail={(c.get('detail') or '')[:300]}\n"
            f"trace_excerpt={excerpt[:300]}\n"
            f"recent traces:\n{traces_text}"
        )

    prompt = (
        f"我在跑评测系统，分类为 `{cat}` 的失败有 {len(items)} 个。下面是最多 3 个证据样本：\n\n"
        + "\n---\n".join(samples)
        + "\n\n请用中文给出（保持每节 100 字内）：\n"
          "1. 根因假设（最可能的 root cause）\n"
          "2. 关键证据片段（指明出自哪条 trace）\n"
          "3. 修复建议（具体可执行）"
    )

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "MINIMAX_BASE_URL", DEFAULT_MINIMAX_OPENAI_BASE_URL
            ),
        )
        resp = client.chat.completions.create(
            model=os.getenv("MINIMAX_MODEL", DEFAULT_MINIMAX_MODEL),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("LLM RCA call failed (%s); fallback to rule-based", e)
        return None
    if not text:
        return None

    # 简单解析三段
    def _section(label: str) -> str:
        m = re.search(rf"{label}[:：]?\s*(.+?)(?=(?:\n\s*\d\.|$))", text, re.DOTALL)
        return (m.group(1).strip() if m else "")[:600]

    hyp = _section(r"(?:1\.|根因)")
    evi = _section(r"(?:2\.|证据)")
    fix = _section(r"(?:3\.|修复|建议)")
    if not (hyp or evi or fix):
        # LLM 没按格式输出，全文塞 hypothesis
        return {"hypothesis": text[:600], "evidence": "(see hypothesis)", "fix": "(see hypothesis)"}
    return {"hypothesis": hyp or "(空)", "evidence": evi or "(空)", "fix": fix or "(空)"}


# ============================================================
# 入口
# ============================================================


def generate_rca(
    run_dir: str | Path,
    *,
    use_llm: bool = True,
    sync_to_recorder: bool = True,
) -> dict[str, Any]:
    """生成 RCA 报告。返回 {md_path, bucket_summary, notion_url?}。"""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")

    summary = _read_json(run_dir / "run_summary.json", {})
    cards = _read_jsonl(run_dir / "failure_cards.jsonl")
    traces = _read_jsonl(run_dir / "traces.jsonl")

    rid = summary.get("run_id", run_dir.name)
    buckets = _bucket_failures(cards)
    bucket_summary = _bucket_summary(buckets)

    traces_by_task: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        tid = t.get("task_id")
        if tid:
            traces_by_task[tid].append(t)

    # 写 md
    lines: list[str] = []
    lines.append(f"# 自动 RCA · {rid}")
    lines.append("")
    lines.append("> 自动生成。**这是假设，不是结论** —— 用它来指导你看 traces.jsonl。")
    lines.append("")
    totals = summary.get("totals", {}) or {}
    lines.append(
        f"- 任务总数: {totals.get('total', '?')}  "
        f"ok={totals.get('ok', 0)}  failed={totals.get('failed', 0)}  "
        f"skipped={totals.get('skipped', 0)}"
    )
    contam = bool(totals.get("contaminated"))
    if contam:
        lines.append("- ⚠️ **CONTAMINATED**：受保护文件在 run 前后变化")
    lines.append(f"- 失败卡片: {len(cards)} 张, 分桶: {bucket_summary}")
    lines.append("")

    # mermaid pie
    if bucket_summary:
        lines.append("## 失败分桶")
        lines.append("")
        lines.append("```mermaid")
        lines.append("pie title 失败分桶分布")
        for cat, n in bucket_summary.items():
            lines.append(f'    "{cat}" : {n}')
        lines.append("```")
        lines.append("")

    if not buckets:
        lines.append("无失败任务 — 不生成 RCA 子节。")
    else:
        lines.append("## 各分桶根因假设")
        lines.append("")
        for cat, items in buckets.items():
            lines.append(f"### `{cat}` × {len(items)}")
            lines.append("")
            rca = None
            if use_llm:
                rca = _llm_rca(cat, items, traces_by_task)
            if rca is None:
                rca = _rule_based_rca(cat, items)
                lines.append("> 来源：规则表（未启用 LLM 或 LLM 不可用）")
                lines.append("")
            else:
                lines.append("> 来源：LLM 生成（MiniMax）")
                lines.append("")
            lines.append(f"**根因假设**：{rca['hypothesis']}")
            lines.append("")
            lines.append(f"**证据**：")
            lines.append("")
            lines.append("```")
            lines.append(rca["evidence"])
            lines.append("```")
            lines.append("")
            lines.append(f"**修复建议**：{rca['fix']}")
            lines.append("")
            # 列出该桶下涉及的 task_id
            tasks = sorted({c.get("task_id", "?") for c in items})
            lines.append(f"_涉及任务_: {', '.join(f'`{t}`' for t in tasks[:10])}"
                         + (f" (+{len(tasks)-10})" if len(tasks) > 10 else ""))
            lines.append("")

    md_path = run_dir / "rca.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("RCA written: %s", md_path)

    out: dict[str, Any] = {
        "md_path": str(md_path),
        "bucket_summary": bucket_summary,
        "total_failures": len(cards),
    }

    if sync_to_recorder:
        try:
            from src.recorder import RefactorRecorder
            recorder = RefactorRecorder(area="eval-runs")
            recorder.log_validation(
                step=f"auto-rca {rid}",
                command=f"python -m src.evaluation.cli rca --run data/evaluation/runs/{rid}",
                output=md_path.read_text(encoding="utf-8")[:1800],
                status="warn" if cards else "ok",
                notes=f"buckets={bucket_summary}, llm={'on' if use_llm else 'off'}",
                artifacts=[f"data/evaluation/runs/{rid}/rca.md"],
            )
            state_path = Path("data/recorder_state/eval-runs.json")
            if state_path.exists():
                try:
                    out["notion_url"] = json.loads(state_path.read_text(encoding="utf-8")).get("notion_url", "")
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("recorder sync skipped: %s", exc)

    return out
