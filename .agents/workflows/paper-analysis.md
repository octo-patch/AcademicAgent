---
description: 端到端的论文分析 — 搜索、下载、解析、提取知识，并生成学习建议
---

# Paper Analysis (论文分析) Workflow

从零开始分析一篇学术论文的完整工作流。

## 前置准备

- 读取 `memory/USER.md` 了解用户研究方向
- 读取 `memory/MEMORY.md` 了解已有搜索经验和已知问题
- 读取 `memory/preferences.json` 了解结构化偏好（默认 KG 域、搜索源、是否开反思器等）

## 一键路径（Phase 5 起，推荐）

如果用户只给了一个本地 PDF，可直接调 Python Runtime 一键走完 precheck → ingest → extract → 回查：

```bash
python -m src.runtime run paper_analysis_flow --pdf_path data/papers/<paper>.pdf
```

runtime 会：
- 跑前在 KG 里做语义 precheck，避免重复入库（`skip` on fail）
- ingest_local_pdf 注册元数据（`retry_once`）
- add_paper_to_graph LLM 抽取并入图（`retry_once`，由 reflector 决策）
- query_knowledge 回查验证可达性（`skip` on fail）
- 成功 → 追加到 `memory/experiences/success_patterns.md`
- 失败 → 追加到 `memory/experiences/failure_patterns.md`
- 含重试 → 同时写 `memory/experiences/api_quirks.md`
- run_log 落盘到 `data/runtime_runs/paper_analysis_flow_<run_id>.json`

宿主仍可手动逐步调 MCP（下面的步骤），runtime 不替代灵活路径，只是给"一键"。

## 步骤

1. **判断输入类型**（Phase 4 起的默认分支）:

   - **A. 用户给了本地 PDF 路径** → 直接走 `ingest_local_pdf(pdf_path, domain=...)`，跳到步骤 5
     - 适合 IEEE / ACM / Springer 等闭源期刊论文（已手动下载）
     - 工具自动：file_hash 去重 → 抽元数据 → 注册 paper_registry → 可选入图
   - **B. 用户给了目录路径**（如 `data/papers/`）→ 走 `batch_ingest_directory(dir_path, domain=...)`，跳到步骤 5
   - **C. 用户只给了主题/关键词** → 进入步骤 2 搜索

2. **搜索论文**（仅当 1.C 时）:
   - **首选 `search_arxiv`**（公开 API 无限速；Phase 4.1 新增 category/author/recent_days 过滤）
   - **备选 `search_crossref`**（CrossRef，DOI 元数据强，无 key）
   - **备选 `search_openreview`**（ICLR/NeurIPS 公开 review）
   - **备选 `search_dblp`**（作者发表列表）
   - **降级 `search_papers`**（Semantic Scholar，无 key 易 429；只在前面都搜不到时用）
   - 参考 `memory/experiences/search_strategies.md` 选择最优 API
   - 如果用户需要特定的某篇论文，使用 Paper ID 调用 `get_paper_details`

3. **去重检查** (下载前必做):
   - 使用 Paper Registry 检查论文是否已在本地:
   ```python
   from src.core.paper_registry import PaperRegistry
   registry = PaperRegistry()
   dup = registry.check_duplicate(arxiv_id="<arxiv_id>", title="<title>")
   if dup:
       print(registry.format_duplicate_warning(dup))
       # → 跳过下载，直接使用 dup.local_path
   ```
   - 也可用 CLI: `python -m src.core.paper_registry check --arxiv <id>`
   - 如果已存在 → **告知用户并跳过下载**，直接进入步骤 3

4. **下载 PDF** (仅在去重检查通过后):
   - PDF 统一存储到 `data/papers/` 目录
   - 使用规范化文件名: `{标题关键词}_{arXiv ID}.pdf`
   ```python
   from src.core.paper_registry import suggest_filename
   filename = suggest_filename(title="<title>", arxiv_id="<arxiv_id>")
   # → 例如: "ReAct_2210.03629.pdf"
   ```
   ```bash
   python -c "import httpx; r=httpx.get('<pdf_url>', follow_redirects=True, timeout=60); open('data/papers/<filename>', 'wb').write(r.content)"
   ```

5. **注册论文** (下载后立即执行；推荐用 `ingest_local_pdf` 一站式完成 4+5+5.5):
   ```python
   from src.core.paper_registry import PaperRegistry, PaperRecord
   registry = PaperRegistry()
   record = PaperRecord(
       title="<title>",
       arxiv_id="<arxiv_id>",
       authors=["<author1>", "<author2>"],
       year=<year>,
       local_path="data/papers/<filename>",
       source_url="<pdf_url>",
       venue="<venue>",
   )
   registry.register_paper(record)
   ```

6. **获取论文元数据并检查是否为扫描版**:
   ```bash
   python skills/paper_reader/scripts/parse_pdf.py --action metadata --pdf "<pdf_path>"
   ```

7. **提取论文结构**:
   ```bash
   python skills/paper_reader/scripts/parse_pdf.py --action structure --pdf "<pdf_path>"
   ```

8. **向用户展示结构概览** 并询问需要重点关注哪些章节。

9. **提取全文** 以供知识图谱使用:
   ```bash
   python skills/paper_reader/scripts/parse_pdf.py --action text --pdf "<pdf_path>"
   ```

10. **分析关键图表** (仅在用户明确要求时；Phase 3.3 起 analyze_figure 默认 auto_bind_caption=True):
   - 提取图片: `--action images --page <N>`
   - 或者渲染整页: `--action render --page <N>`
   - 对保存的图片使用 `view_file`，以便调用视觉能力进行分析

11. **添加至知识图谱**:
   - 使用 `knowledge-graph` MCP 工具：将提取的文本作为参数调用 `add_paper_to_graph`
   - 使用 `get_graph_stats` 检查图谱统计信息

12. **生成研究报告** (持久化论文解读):
   - 将步骤 8 提取的全文保存为临时文件（如 `data/temp_text.txt`）
   - 生成结构化报告：
   ```bash
   python src/report/generator.py --title "<paper_title>" --text-file data/temp_text.txt --year "<year>" --arxiv-id "<arxiv_id>"
   ```
   - 报告自动保存到 `data/reports/papers/`（JSON + HTML 双格式）
   - 向用户展示 HTML 报告路径，建议用浏览器打开查看

13. **建议下一步操作**:
    - "需要分析更多图表吗？"
    - "要查看学习路径建议吗？" → 建议运行 `/knowledge-build`
    - "要用仿真验证论文方法吗？" → 建议运行 `/simulation`

14. **经验归档** (自动):
    如果本次任务中遇到了以下情况之一，请更新记忆文件：
    - API 报错或限流 → 追加到 `memory/MEMORY.md` 搜索策略章节，并更新 `memory/experiences/search_strategies.md`
    - 论文解析的特殊处理技巧 → 追加到 `memory/MEMORY.md` 解析经验章节
    - 用户偏好变化 → 更新 `memory/USER.md`
    - **预算约束**: MEMORY.md 超过 800 Token 时，删除最旧条目
    - 格式：`- [YYYY-MM-DD] 描述。建议：...`
