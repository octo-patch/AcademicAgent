---
description: 端到端的论文分析 — 搜索、下载、解析、提取知识，并生成学习建议
---

# Paper Analysis (论文分析) Workflow

从零开始把一篇学术论文走完"找到 → 拿到 PDF → 解析 → 入图 → 出报告"的完整闭环。
目标读者是宿主 Agent（Claude / Antigravity）：每一步都要明确"输入 / 操作 / 产出 / 失败策略"，
便于跑完后通过 Lite Reflector 把经验沉淀回 `memory/experiences/`。

---

## 前置准备（Bootstrap）

在动手之前先建立上下文，避免和用户偏好对冲：

1. 读 [memory/USER.md](memory/USER.md) — 用户当前主方向（默认 5G / ISAC / SLAM / 定位 ↔ LLM Agent，按 USER.md 实际为准）。
2. 读 [memory/MEMORY.md](memory/MEMORY.md) — 历史经验索引，挑出 "search_strategies / analysis_patterns / api_quirks" 中与本任务相关的条目。
3. 读 [memory/preferences.json](memory/preferences.json) — 取 `default_domain` / `default_search_source` 等结构化偏好；无字段时用合理默认。
4. 准备一个本次会话的 `run_state` 字典（内存即可），约定字段：
   - `pdf_path` / `arxiv_id` / `title` / `domain` / `year`
   - `paper_text`（提取后的全文，下游多步要复用）
   - `kg_node_count` / `kg_edge_count`（入图后回填，用于校验）
   - `report_path`（生成的 HTML 报告路径）
   - `steps_log: list[dict]`（每步追加 {name, status, attempts, elapsed_ms, error, captured}）

> 状态管理原则：**一篇论文的 run_state 不跨论文复用**。批量分析时为每篇 PDF 单独建一份 run_state，
> 否则 paper_text / kg_node_count 会被串味。

---

## 详细步骤

每一步统一格式：**目的 → 操作 → 产出（写回 run_state）→ 失败策略**。

### 步骤 1. 判定输入类型（Dispatch）

**目的**：决定走"已有 PDF / 已有目录 / 只有关键词"三条主分支，避免把所有论文都强行走搜索。

**操作**：
- 若用户提供本地 `pdf_path` → 标记 `run_state.mode = "local_pdf"`，跳至步骤 5（`ingest_local_pdf` 是 4+5+6 的一站式）。
- 若用户提供目录（如 `data/papers/`） → 标记 `mode = "batch_dir"`，调 `batch_ingest_directory(dir, domain=...)`，跳至步骤 11 做图谱回查。
- 若用户只给关键词 / 主题 → 标记 `mode = "search"`，进入步骤 2。
- 若用户给了 arXiv ID 但本地无文件 → 标记 `mode = "arxiv_id"`，跳过步骤 2 直接到步骤 3 去重检查。

**产出**：`run_state.mode`，后续步骤通过 mode 决定是否跳过。

**失败策略**：输入完全为空时不要继续猜，直接向用户提一个聚焦问题（"你是有 PDF、有 arXiv ID、还是只想搜某个主题？"）。

### 步骤 2. 搜索论文（仅 `mode = "search"`）

**目的**：在合规且免限流的源里精准定位用户想要的论文。

**操作**（按推荐优先级）：
1. **`search_arxiv`** — 公开 API 无限速；Phase 4.1 起支持 `category / author / recent_days` 过滤。第一选择。
2. **`search_crossref`** — DOI / 期刊元数据强、无 key。需要 venue / DOI 时用。
3. **`search_openreview`** — ICLR / NeurIPS 公开 review，看同行评审用。
4. **`search_dblp`** — 按作者抓发表列表。
5. **`search_papers`（Semantic Scholar）** — 仅作降级，无 key 易 429，前几个都搜不到再用。

具体选哪个先看 [memory/experiences/search_strategies.md](memory/experiences/search_strategies.md) 是否已有对应主题的命中记录。
若用户已经指认了某篇 paper（给了标题或 ID），直接 `get_paper_details` 拿全量元数据，避免误搜。

**产出**：`run_state.title / arxiv_id / pdf_url / authors / year / venue`。

**失败策略**：
- 单源 429 / 5xx → 立刻降级到下一个源，**不要原地重试**（runner 已有 backoff，再叠加重试会放大限流）。
- 全部源都空 → 写一条 `failure_patterns.md` 记录"`<keyword>` 在所有源都无结果"，并向用户确认拼写 / 关键词同义词。

### 步骤 3. 去重检查（必做，下载前）

**目的**：避免把同一篇论文重复落盘 + 重复入图，污染 KG 节点计数。

**操作**：
```python
from src.core.paper_registry import PaperRegistry
registry = PaperRegistry()
dup = registry.check_duplicate(arxiv_id=run_state.get("arxiv_id"),
                               title=run_state.get("title"))
if dup:
    print(registry.format_duplicate_warning(dup))
    run_state["pdf_path"] = dup.local_path
    run_state["dedup_hit"] = True
```

也可 CLI：`python -m src.core.paper_registry check --arxiv <id>`。

**产出**：`run_state.pdf_path`（命中时复用本地文件）、`run_state.dedup_hit: bool`。

**失败策略**：
- 命中 → **跳过步骤 4 下载**，直接进入步骤 5；并在最终汇报里告知用户"已存在本地副本，跳过下载"。
- 未命中 → 进入步骤 4。
- registry 文件损坏（很罕见）→ 不要继续覆盖；走 `failure_patterns.md` 记录并向用户求助，否则会把脏数据写实。

### 步骤 4. 下载 PDF（未命中去重时）

**目的**：把远端 PDF 拿到本地，并按规范命名落盘到 `data/papers/`。

**操作**：
```python
from src.core.paper_registry import suggest_filename
filename = suggest_filename(title=run_state["title"], arxiv_id=run_state["arxiv_id"])
# e.g. "ReAct_2210.03629.pdf"
```
```bash
python -c "import httpx; r=httpx.get('<pdf_url>', follow_redirects=True, timeout=60); open('data/papers/<filename>','wb').write(r.content)"
```

**产出**：`run_state.pdf_path = "data/papers/<filename>"`。

**失败策略**：
- 网络超时 / 403 → 重试 1 次（换 `arxiv.org/pdf/<id>` 镜像 URL）。两次仍败 → 记 `api_quirks.md`，请用户手动下载后回到 `mode = "local_pdf"`。
- 文件大小 < 50 KB（常是 HTML 错误页伪装成 PDF）→ 视为失败，删除局部文件并按上行处理，避免把垃圾 PDF 灌进 registry。

### 步骤 5. 注册并 ingest 论文（推荐入口）

**目的**：让 `paper_registry` 持有这条记录，使后续 KG 节点能反向追溯到源 PDF。

**操作**（推荐）：直接调 MCP 工具 `ingest_local_pdf`，它在内部完成 file_hash 去重 + 元数据抽取 + 注册：

```text
ingest_local_pdf(
    pdf_path=run_state.pdf_path,
    hint_title=run_state.get("title", ""),
    hint_arxiv_id=run_state.get("arxiv_id", ""),
    domain=run_state.get("domain") or prefs.default_domain,
    add_to_kg=False,        # 分步走，下面 step 8 再入图，便于观测
    first_seen_date=run_state.get("first_seen_date", ""),
)
```

如需手工注册，仍可用 `PaperRegistry().register_paper(PaperRecord(...))`。

**产出**：`run_state.intake_status / ingest_report_head`（用于失败时贴回上下文）。

**失败策略**：`retry_once`。常见原因是 PDF 损坏 / 元数据抽取空；reflector 会在错误含 `extractionerror / empty.*nodes` 时建议重试。

### 步骤 6. 元数据与扫描版判定

**目的**：识别"扫描版 / 无文本层"的 PDF，避免后续 `add_paper_to_graph` 拿到空文本。

**操作**：
```bash
python skills/paper_reader/scripts/parse_pdf.py --action metadata --pdf "<pdf_path>"
```

**产出**：`run_state.is_scanned: bool`（脚本会在文本占比过低时输出 `is_scanned=True`）。

**失败策略**：扫描版 → 提示用户"这是扫描版，需要 OCR；要继续吗？"，并在 `run_state.skip_extract = True` 时跳过步骤 8（避免空抽取污染 KG）。

### 步骤 7. 提取论文结构

**目的**：让用户先看到 outline，再决定深读哪些章节，节省 token。

**操作**：
```bash
python skills/paper_reader/scripts/parse_pdf.py --action structure --pdf "<pdf_path>"
```

**产出**：`run_state.structure`（章节列表）。

**失败策略**：`skip`。结构抽取失败不阻断主链，只影响"用户选章节"的体验。

### 步骤 8. 与用户对齐重点章节

**目的**：避免对一篇 30 页的论文做"全文无差别灌入"。

**操作**：把 outline 用 1 屏内的列表呈现，问用户"哪几节重点看 / 哪几节跳过"。

**产出**：`run_state.focus_sections: list[str]`，下游 `--action text` 可按需切片。

**失败策略**：用户未回应 → 默认 Abstract + Intro + Method + Experiments + Conclusion 五节。

### 步骤 9. 提取全文（供 KG 使用）

**目的**：给步骤 11 的 `add_paper_to_graph` 提供 LLM 抽取所需的原文。

**操作**：
```bash
python skills/paper_reader/scripts/parse_pdf.py --action text --pdf "<pdf_path>"
```

**产出**：`run_state.paper_text`（一段长字符串；若过长可裁到 ~30K tokens）。

**失败策略**：`retry_once`。文本为空 / < 500 字符 → 视为失败，按"扫描版"分支处理。

### 步骤 10. 关键图表分析（可选，按需）

**目的**：补足"光看正文看不出"的 system diagram / 结果表。

**操作**（仅在用户要求时做）：
- 提取指定页图片：`--action images --page <N>`
- 或整页渲染：`--action render --page <N>`
- 调 MCP `analyze_figure`（Phase 3.3 起默认 `auto_bind_caption=True`，会自动绑 caption）。

**产出**：图片本地路径 + caption + 模型理解的 1 段话。

**失败策略**：`skip`。视觉理解失败不影响主链。

### 步骤 11. 入知识图谱

**目的**：把这篇论文的实体 / 关系并入领域 KG，喂养后续 `/knowledge-build`。

**操作**：调 MCP `add_paper_to_graph(text=run_state.paper_text, paper_title=..., domain=...)`，
跑完再调 `get_graph_stats()` 拿增量。

**产出**：`run_state.kg_node_count / kg_edge_count`（增量值）。

**失败策略**：
- `retry_once`，由 reflector 视错误类型决定（JSON parse fail / empty nodes → 重试；401/403/missing key → 立刻 abort）。
- 节点数 = 0 → 不算 ok，回写 `final_state.kg_node_count = 0` 并归到 failure。

### 步骤 12. 入图后回查（可达性校验）

**目的**：避免"add 成功但实际节点没落库"的静默失败。

**操作**：
```text
query_knowledge(query=run_state.title, domain=run_state.domain)
```

**产出**：`run_state.query_report_head`（前 500 字摘要）。

**失败策略**：`skip` on fail；但若命中数为 0 → 在 reflector 总结里标注"入图后回查未命中"。

### 步骤 13. 生成研究报告（持久化解读）

**目的**：留一份可被复读的论文笔记，独立于 KG。

**操作**：
```bash
# 先把 paper_text 落到临时文件
python -c "open('data/temp_text.txt','w',encoding='utf8').write(open('<pdf_path>','rb').read().decode('utf8','ignore'))"
# 生成报告
python src/report/generator.py --title "<title>" --text-file data/temp_text.txt \
    --year "<year>" --arxiv-id "<arxiv_id>"
```

**产出**：`run_state.report_path`（指向 `data/reports/papers/<slug>.html`）。

**失败策略**：`skip`。报告生成失败不影响知识入图。

### 步骤 14. 下一步建议

跑完后给用户一个分支菜单，而不是空白结束：
- "需要分析更多图表吗？" → 回到步骤 10。
- "要看学习路径建议吗？" → 引导至 `/knowledge-build`。
- "要用仿真验证论文方法吗？" → 引导至 `/simulation`。

---

## 鲁棒性与状态管理 Checklist

- **幂等性**：步骤 3 去重 + 步骤 5 file_hash 双保险，重跑同一 PDF 不会双倍入图。
- **断点续跑**：每完成 ≥1 个外部副作用步骤（下载 / 注册 / 入图 / 报告）就把 `run_state` 序列化进 `data/runtime_runs/paper_analysis_<YYYYMMDD-HHMM>.json`，会话被打断后可从最近 dump 恢复。
- **失败传播**：除 `abort` 类外，把错误以 `{name, error, attempts}` 推入 `run_state.steps_log`，**不要让单步异常吞掉整个 run**。
- **重试上限**：reflector 单步最多建议 1 次重试，runner 内部还有 `max_attempts=2` 兜底，避免死循环。
- **预算控制**：`paper_text` 超过 ~30K tokens 时按章节裁剪，优先保 Abstract / Method / Results。
- **侧效果隔离**：步骤 4（下载）/ 5（注册）/ 11（入图）/ 13（报告）任何一步失败，都不应当让前面已成功的副作用回滚（registry 是 append-only）；要做的是**标记 partial** 让用户决定重跑哪一步。

---

## Lite Reflector & 经验沉淀（必做收尾）

跑完后由宿主 Agent 自己做这一段 —— 这是"下次开新会话能复用本次经验"的关键。
本节不依赖任何外部 runner，只用文件 I/O。

### A. 组装 run_log（自己拼，存成 JSON）

把会话里维护的 `run_state.steps_log` 整理成一份 JSON，落到 `data/runtime_runs/paper_analysis_<YYYYMMDD-HHMM>.json`：

```json
{
  "run_id": "<时间戳 + 6 位随机>",
  "workflow": "paper_analysis",
  "started_at": "<iso8601>",
  "ended_at": "<iso8601>",
  "status": "ok | failed | partial",
  "steps": [
    {"name":"dispatch","status":"ok","attempts":1,"elapsed_ms":...,"captured":{...}},
    {"name":"dedup","status":"ok","attempts":1,"elapsed_ms":...},
    {"name":"ingest","status":"ok","attempts":2,"elapsed_ms":...,"error":""},
    {"name":"extract","status":"ok","attempts":1,"elapsed_ms":...,"captured":{"kg_node_count":17}}
  ],
  "final_state": {
    "paper_title": "...",
    "kg_node_count": 17,
    "kg_edge_count": 24,
    "report_path": "data/reports/papers/<slug>.html"
  },
  "summary": "<一句话总结，≤100 字>"
}
```

写盘是为了**事后可追溯**，不是给某个 runner 消费。

### B. 一句话总结（deterministic 模板，自己产出）

格式：

```
[paper_analysis] 状态=ok; 步骤 4/4 成功; 重试 1 次; 总耗时 18432 ms
```

- 失败时加 `失败: <step>(<error 前 80 字>)`
- 有 skip 时加 `跳过: <names>`
- 有 partial 时加 `部分成功: <step>`

把这一行同时写进 run_log 的 `summary` 字段和下面要追加的经验文件第一行。

### C. 追加到 `memory/experiences/`（直接用 Write / Edit）

按 `status` 分桶：
- `status == "ok"` → append 到 [memory/experiences/success_patterns.md](memory/experiences/success_patterns.md)
- `status == "failed"` → append 到 [memory/experiences/failure_patterns.md](memory/experiences/failure_patterns.md)
- 任意 `attempts > 1` 的步骤 → 同时 append 一条到 [memory/experiences/api_quirks.md](memory/experiences/api_quirks.md)

每条 append 的最小模板：

```markdown
### <YYYY-MM-DD HH:MM> paper_analysis · <run_id>
- 总结: <一句话总结>
- 关键 final_state: paper_title="..."; kg_node_count=17; report_path=...
- 失败步骤（若有）: ingest(extractor empty nodes), retry 1 后成功
```

### D. 旁路更新（人为判断后再写）

只有在本次任务里**明显出现新经验**时才写，避免噪声：
- 搜索源新限流 / 新可用过滤参数 → [memory/experiences/search_strategies.md](memory/experiences/search_strategies.md)
- PDF 解析的新坑（特定出版社版面、字符编码等）→ [memory/experiences/analysis_patterns.md](memory/experiences/analysis_patterns.md)
- 用户在本次明确提了新偏好 / 新主方向 → [memory/USER.md](memory/USER.md)
- MEMORY.md > 800 token 时，删除最旧或已被新经验取代的条目；格式 `- [YYYY-MM-DD] 描述。建议：...`。

> 关键守则：**经验文件是 append-only 的事实日志，不是日记**。一条经验要可执行（"下次遇到 X，做 Y"），否则就别写。
