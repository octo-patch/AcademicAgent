---
description: 追踪近期新论文 — 抓取 arXiv 最新论文，LLM 智能筛选并推荐
---

# Paper Watch (论文追踪) Workflow

定期把"用户关心方向上的最新论文"拉到本地，由 LLM 按用户偏好做精读价值排序，
最后只把 Top N 推到用户面前。本 workflow 的核心价值是**降噪**：arXiv 每天上百篇，
能进入用户视野的应该 < 10 篇。

---

## 前置准备（Bootstrap）

1. 读 [memory/USER.md](memory/USER.md) — 主方向 + 关键词列表，这是筛选的核心 ground truth。
2. 读 [memory/MEMORY.md](memory/MEMORY.md) — 历史搜索经验（哪些关键词总产生噪声、哪些 category 高产）。
3. 读 [memory/preferences.json](memory/preferences.json) — `paper_watch.days_window / paper_watch.top_k / paper_watch.exclude_authors` 等。
4. 在内存建 `run_state`，约定字段：
   - `topics: list[str]` — 本轮追踪关键词，默认来自 USER.md，可被用户参数覆盖
   - `days: int` — 时间窗（默认 7）
   - `fetched_count: int` — 抓回多少篇
   - `summary_md: str` — 今日摘要文件的内容
   - `recommended: list[dict]` — Top K 推荐 `[{title, arxiv_id, url, one_line, relevance, why}]`
   - `steps_log: list[dict]` — 喂给 Lite Reflector

> 频率约定：建议每天最多跑 1 次。判定方式：检查 `data/paper_watch/<YYYY-MM-DD>.json` 是否已存在，存在则跳过步骤 1 直接读缓存。

---

## 详细步骤

### 步骤 1. 抓取近期论文（带去重缓存）

**目的**：拉一段时间窗内的新论文到本地，**只抓不筛**。

**操作**：
- 默认：
  ```bash
  python skills/paper_watch/scripts/fetch_papers.py --days <run_state.days>
  ```
- 用户指定主题时：
  ```bash
  python skills/paper_watch/scripts/fetch_papers.py \
      --topics "<topic1>,<topic2>" --days <run_state.days>
  ```
- 当日缓存：如果今日已跑过（`data/paper_watch/<today>.json` 存在），直接 reuse，不发新请求 → **省 arXiv 配额，避免误判限流**。

**产出**：`run_state.fetched_count`，缓存文件路径。

**失败策略**：
- arXiv 429 / 5xx → `retry_once`，仍败则用上一日缓存（提示用户"使用昨日数据"）。
- 网络完全不可达 → abort，记 `failure_patterns.md`。
- `fetched_count == 0` → 不视为失败，但要在步骤 4 报告里说明（避免用户以为筛选丢了论文）。

### 步骤 2. 读取今日摘要

**目的**：把 N 篇论文压成 LLM 可一次性吃下的 markdown 摘要，为步骤 3 做准备。

**操作**：
```bash
python skills/paper_watch/scripts/fetch_papers.py --action summary
```

**产出**：`run_state.summary_md`。

**失败策略**：`retry_once`。若摘要脚本失败但步骤 1 有 raw JSON → 降级直接读 JSON 自己拼摘要（不要让单脚本卡死全流程）。

### 步骤 3. LLM 智能筛选（核心降噪）

**目的**：从 N 篇里挑出 Top K（默认 K=5）相关度最高的，并给出**可被人类一眼判断要不要展开读**的解释。

**操作**：把 `USER.md` 的主方向 + 关键词列表 + 用户当前研究焦点（可临时传入）一起拼成 prompt，
对 `summary_md` 做相关性打分。每篇推荐输出：
- 📌 **一句话概括**（≤ 30 字，用动词开头，避免"本文提出"套话）
- 🔗 **关联度**：高 / 中 / 低（带 1 行理由：和用户的 X 方向相关）
- 📎 **链接**：arXiv abs 页面（不是 PDF 直链，免触发反爬）
- 🏷 **建议动作**：`/paper-analysis` / `/knowledge-build` / 仅收藏

**产出**：`run_state.recommended: list[dict]`，长度 ≤ K。

**失败策略**：
- LLM 输出无法解析（非 JSON / 漏字段）→ `retry_once`，并在 retry prompt 里**明示上次失败原因**（参考 reflector 的 `_RETRY_OK_PATTERNS` 设计）。
- 仍败 → 降级为"按抓取时间倒序前 K 篇"，明确告诉用户"LLM 筛选失败，仅按时间排序"。

### 步骤 4. 推荐展示 + 引导

**目的**：把 Top K 用表格 + 行动菜单呈现，让用户能"一眼挑、一键跳"。

**操作**：渲染如下：

```
| # | 标题 | 关联度 | 一句话 | 建议动作 |
|---|------|--------|--------|----------|
| 1 | ... |  高    | ...    | /paper-analysis |
| 2 | ... |  中    | ...    | 仅收藏 |
...
```

紧跟两个分支问题：
- "要深入分析第 N 篇吗？" → `/paper-analysis` 用 `arxiv_id`
- "要调整追踪关键词吗？" → 进入步骤 5

**产出**：呈现给用户；同时把推荐落 `data/paper_watch/recommend_<YYYY-MM-DD>.json`，便于次日 diff。

**失败策略**：`skip`。展示失败不影响 run_log 沉淀。

### 步骤 5. 关键词调整（可选）

**目的**：让 USER.md 的关键词随用户兴趣漂移而更新。

**操作**：用户给出"加 X / 删 Y"指令时：
- 读 USER.md → 解析 "关键词" / "主方向" 段 → 增删 → 写回。
- 记录变更前后 diff 到 `run_state.user_md_diff`。

**产出**：USER.md 更新；`run_state.user_md_diff`。

**失败策略**：写失败 → 立即回滚（保留原文件备份 `.bak`）；不要让 USER.md 半改不改。

### 步骤 6. 趋势观察（可选，长程价值）

**目的**：连跑 ≥7 天后能产出"这周高频的新概念是 X" 这类洞察。

**操作**：
- 读 `data/paper_watch/recommend_*.json` 最近 7 天，统计高频 keyword / author / institution。
- 输出 1 段"本周观察"，建议写进 `memory/experiences/analysis_patterns.md`（人审后再写，避免噪声）。

**产出**：`run_state.weekly_trend_md`。

**失败策略**：`skip`。统计失败完全不影响主链。

---

## 鲁棒性与状态管理 Checklist

- **每日幂等**：步骤 1 的 daily cache 是关键，避免同一天反复打 arXiv。
- **降级链**：抓取失败 → 用昨日缓存；LLM 筛选失败 → 按时间排序。每一层降级都要在最终呈现里**明示**。
- **USER.md 安全写**：步骤 5 必须先备份 `.bak`，且解析失败时不写。
- **隐私 / 无 PII**：abstract 已是公开数据；不要把 USER.md 整段塞进 LLM prompt，只塞"主方向 + 关键词"即可。
- **配额意识**：avoid 把 `--days` 调过大（>30 容易触发 arXiv 反爬）；超大窗口请用 `search_arxiv` 的 category 过滤代替。

---

## Lite Reflector & 经验沉淀（必做收尾）

跑完后由宿主 Agent 自己做这一段，仅靠文件 I/O，无外部 runner 依赖。

### A. 组装 run_log（自己拼，存成 JSON）

把会话里维护的 `run_state.steps_log` 整理成一份 JSON，落到 `data/runtime_runs/paper_watch_<YYYYMMDD-HHMM>.json`：

```json
{
  "run_id": "<时间戳 + 6 位随机>",
  "workflow": "paper_watch",
  "started_at": "<iso8601>",
  "ended_at": "<iso8601>",
  "status": "ok | failed | partial",
  "steps": [
    {"name":"fetch","status":"ok","attempts":1,"elapsed_ms":...,"captured":{"fetched_count":47,"used_cache":false}},
    {"name":"summary","status":"ok","attempts":1,"elapsed_ms":...},
    {"name":"filter","status":"ok","attempts":2,"elapsed_ms":...,"captured":{"top_k":5}},
    {"name":"present","status":"ok","attempts":1,"elapsed_ms":...}
  ],
  "final_state": {
    "days": 7,
    "topics": ["...", "..."],
    "fetched_count": 47,
    "recommended_count": 5,
    "top_relevance": "高",
    "used_cache": false
  },
  "summary": "..."
}
```

### B. 一句话总结（deterministic 模板，自己产出）

```
[paper_watch] 状态=ok; 步骤 4/4 成功; 抓 47 篇 → 推 5 篇; 高关联 2 篇; 总耗时 9128 ms
```

- 降级时显式标注：`; LLM 筛选失败，回退时间排序` 或 `; 使用昨日缓存`
- 失败时加 `失败: <step>(<error 前 80 字>)`

把这一行同时写进 run_log 的 `summary` 字段，和下面要追加的经验文件第一行。

### C. 追加到 `memory/experiences/`（直接用 Write / Edit）

按 `status` 分桶：
- `ok` → append 到 [memory/experiences/success_patterns.md](memory/experiences/success_patterns.md)
- `failed` → append 到 [memory/experiences/failure_patterns.md](memory/experiences/failure_patterns.md)
- 任意 `attempts > 1` 的步骤 → 同时 append 一条到 [memory/experiences/api_quirks.md](memory/experiences/api_quirks.md)

每条 append 的最小模板：

```markdown
### <YYYY-MM-DD HH:MM> paper_watch · <run_id>
- 总结: <一句话总结>
- 关键 final_state: topics=[...]; fetched=47; recommended=5; used_cache=false
- 降级/失败步骤（若有）: filter 第 1 次 JSON 解析失败，retry 后通过
```

### D. 旁路更新（人为判断）

- 某关键词反复零命中 / 反复噪声 → 进 [memory/experiences/search_strategies.md](memory/experiences/search_strategies.md)（"`<keyword>` 在 arxiv 上更适合用 `<cat>` 过滤"之类）
- 用户调整了主方向 / 关键词 → 已经在步骤 5 写过 USER.md，无需再写 MEMORY.md
- 连续 N 天高频出现的新概念 → 进 [memory/experiences/analysis_patterns.md](memory/experiences/analysis_patterns.md)，标注"建议加进图谱重点关注"
- MEMORY.md > 800 token → 优先删旧的关键词调整记录（这类信息已在 USER.md 里）

> 关键守则：**Paper watch 的经验最容易堆"今天看了什么"流水账，必须强迫自己只写"下次怎么筛得更准"的经验**。
