---
description: 从多篇论文中构建知识图谱，检测知识盲区，并生成个性化学习路径
---

# Knowledge Build (知识构建) Workflow

把"零散读过的若干篇论文"折叠成一张可被 Agent 复用的领域知识图谱，并基于图谱给用户产出
"现在的盲区 → 下一步该读什么"的路径。
本 workflow 是 `/paper-analysis` 的下游聚合层：单篇深读是燃料，知识构建才是发动机。

---

## 前置准备（Bootstrap）

1. 读 [memory/USER.md](memory/USER.md) — 用户主方向，决定后面盲区分析的 `focus_area`。
2. 读 [memory/MEMORY.md](memory/MEMORY.md) — 历史经验索引，关注 `success_patterns / failure_patterns` 中已知的图谱构建坑。
3. 读 [memory/preferences.json](memory/preferences.json) — 取 `default_domain`（决定查哪一张图）/ `learning_path.max_items` 等。
4. 在内存建一个本次会话的 `run_state`，约定字段：
   - `domain: str` — 目标知识图谱名（如 `llm_agent` / `comms`）
   - `papers_input: list[dict]` — `[{title?, arxiv_id?, pdf_path?, text?}]`，每篇一项
   - `stats_before / stats_after: dict` — 图谱节点 / 边 / 概念数变化（用于汇报增量）
   - `gaps_report / path_report / importance_report: str` — 三段 markdown，最后整合
   - `steps_log: list[dict]` — 与 paper-analysis 同结构，喂给 Lite Reflector

> 状态管理原则：**`run_state.domain` 一旦决定后整个 run 不可换**。中途换域会让 stats_before/after 失去对比意义。

---

## 详细步骤

每步统一：**目的 → 操作 → 产出 → 失败策略**。

### 步骤 1. 检查当前图谱状态

**目的**：确认是否"图为空 → 应先去读论文"还是"已有积累 → 直接做盲区分析"。

**操作**：
- MCP 调 `get_graph_stats(domain=run_state.domain)`。
- 把返回的 `nodes / edges / papers / concepts` 计入 `run_state.stats_before`。

**产出**：`run_state.stats_before`。

**失败策略**：
- 图为空（nodes < 5）→ **不要继续后面的盲区分析**（会得到空报告），直接引导用户去 `/paper-analysis` 灌入至少 3-5 篇论文再回来。
- MCP 不可用 → abort，把错误透给用户（这是基础设施问题，不是用户能修的）。

### 步骤 2. 批量添加论文（可选，按需）

**目的**：本次会话内补充新论文到图谱。允许跳过（如果用户只想分析现状）。

**操作**：对 `run_state.papers_input` 里每条记录：
- 有 `pdf_path` → 走 `ingest_local_pdf(..., add_to_kg=True)`，省一次 LLM 抽取调用。
- 有 `text` → 直接 `add_paper_to_graph(text=..., paper_title=..., domain=...)`。
- 只有 `title` / `arxiv_id` → 复用 `/paper-analysis` 步骤 2-5（搜索 + 下载 + 注册）。

每篇成功后更新 `stats_before` 的快照 ≠ 不动；用 `kg_node_count_delta` 累计到 `run_state.added_summary`。

**产出**：`run_state.added_summary: [{paper, delta_nodes, delta_edges}]`。

**失败策略**：
- 单篇失败 → **不阻断整批**，把错误推入 `steps_log` 后继续下一篇。
- 同一篇 retry 仍败 → 标记 `partial`，最终汇报里单列"以下 N 篇未成功入图"。

### 步骤 3. 回顾图谱增长

**目的**：用增量数字（而不是绝对值）给用户成就感，并校验本次 ingest 确实生效。

**操作**：
- 再次 `get_graph_stats(domain=run_state.domain)` → `run_state.stats_after`。
- 计算 `delta = stats_after - stats_before`。
- 对核心概念列表（来自 `USER.md` 主方向关键词）跑 `query_knowledge(concept, domain)`，确认它们在图里。

**产出**：`run_state.delta`、`run_state.core_concept_hits: dict[str, int]`。

**失败策略**：
- `delta.nodes == 0` 但步骤 2 报告"成功" → 说明 LLM 抽取空跑了；记入 `failure_patterns.md` 并提醒用户。
- 核心概念 0 命中 → 不是 bug，但意味着该方向 KG 还很弱，要在最终建议里点出。

### 步骤 4. 知识盲区检测

**目的**：找出"图里看起来薄弱、连通性差、单一来源"的区域 = 学习优先级最高的地方。

**操作**：
```bash
python skills/learning_path/scripts/analyze_knowledge.py \
    --action detect_gaps --domain <run_state.domain> --save
```

`--save` 会落到 `memory/knowledge_export/`，便于跨会话复用。

**产出**：`run_state.gaps_report`（markdown 字符串）+ 落盘文件路径。

**失败策略**：
- `retry_once`。常见失败是 KG 为空（步骤 1 应已拦住）或脚本依赖缺失。
- 脚本退码 != 0 → 抓 stderr 写入 `steps_log.error`，不要继续步骤 5（路径分析依赖盲区结果）。

### 步骤 5. 生成学习路径

**目的**：把盲区翻译成"先读哪几篇 / 哪几个 concept"的有序推荐。

**操作**：
```bash
python skills/learning_path/scripts/analyze_knowledge.py \
    --action learning_path --domain <run_state.domain> \
    --focus_area "<USER.md 主方向或用户当前问题>" \
    --max_items 10 --save
```

**产出**：`run_state.path_report`。

**失败策略**：`retry_once`；若仍失败但盲区分析成功 → 把盲区直接呈现给用户（部分价值 > 零价值）。

### 步骤 6. 概念重要性排名

**目的**：给"应当反复回看"的核心概念一个排序，区别于"盲区"。盲区是缺口；重要性是支柱。

**操作**：
```bash
python skills/learning_path/scripts/analyze_knowledge.py \
    --action importance --domain <run_state.domain> --top 10
```

**产出**：`run_state.importance_report`。

**失败策略**：`skip`。不影响主链。

### 步骤 7. 整合报告呈现

**目的**：把前面三段（盲区 / 路径 / 重要性）拼成"用户可以读完就动手"的一页。

**操作**：模板：

```
## 图谱健康度（本次增量）
- 节点: <stats_before.nodes> → <stats_after.nodes>  (+<delta.nodes>)
- 边:   <stats_before.edges> → <stats_after.edges>  (+<delta.edges>)
- 论文: ...  概念: ...

## 知识盲区 Top（按严重程度）
<gaps_report top 5>

## 推荐学习路径
1. 关键: ...
2. 重要: ...
3. 补充: ...

## 核心概念排名（重要性 Top 10）
<importance_report>

## 下一步建议
- foundation_gaps: 建议阅读综述论文补足
- isolated_concepts: 建议寻找关联论文建立连接
- single_source: 建议从其他论文交叉验证
```

**产出**：渲染到对话；同时把整份 markdown 存到 `data/reports/knowledge_build/<run_id>.md`。

**失败策略**：`skip`（整合失败也至少把已有片段贴出来）。

### 步骤 8. 行动建议（分桶）

按盲区类型派发不同建议，避免一句"多读论文"敷衍：
- **foundation_gaps** → 建议阅读综述 / Tutorial 类论文；可直接调 `/paper-watch` 加该主题。
- **isolated_concepts** → 找连接概念的论文；建议 `search_arxiv` 用 `<conceptA> AND <conceptB>` 组合查询。
- **single_source** → 同一结论只有一篇文献支撑；建议交叉验证，搜替代作者 / 替代时间窗。
- **stale_concepts**（如果脚本支持）→ 该概念两年没新论文 → 建议加进 `/paper-watch`。

每条建议附"如果你要做，下面 1 行命令"，让用户能直接复制。

---

## 鲁棒性与状态管理 Checklist

- **域一致性**：所有步骤共享 `run_state.domain`；切忌中途换域。
- **图为空保护**：步骤 1 已拦截，但步骤 4-6 内部脚本也应能容忍 nodes=0（防御性）。
- **批量 ingest 隔离**：步骤 2 单篇失败不滚整批；最终在 reflector 总结里点名 partial 状况。
- **stats diff 真实性**：`delta` 必须用 same-session 的 before/after，不能跨会话拼。
- **重试上限**：盲区检测 / 学习路径每步最多 1 次重试，避免长链放大错误。
- **报告幂等**：步骤 7 输出落 `data/reports/knowledge_build/<run_id>.md`，重跑会写新 run_id 文件，不覆盖。
- **副作用边界**：步骤 2 的 ingest 是唯一外部副作用；步骤 4-7 只读 + 写报告，重跑成本低。

---

## Lite Reflector & 经验沉淀（必做收尾）

跑完后由宿主 Agent 自己做这一段，仅靠文件 I/O，无外部 runner 依赖。

### A. 组装 run_log（自己拼，存成 JSON）

把会话里维护的 `run_state.steps_log` 整理成一份 JSON，落到 `data/runtime_runs/knowledge_build_<YYYYMMDD-HHMM>.json`：

```json
{
  "run_id": "<时间戳 + 6 位随机>",
  "workflow": "knowledge_build",
  "started_at": "<iso8601>",
  "ended_at": "<iso8601>",
  "status": "ok | failed | partial",
  "steps": [
    {"name":"check_graph","status":"ok","attempts":1,"elapsed_ms":...,"captured":{"nodes_before":120}},
    {"name":"batch_add","status":"partial","attempts":1,"elapsed_ms":...,"captured":{"added":3,"failed":1}},
    {"name":"detect_gaps","status":"ok","attempts":1,"elapsed_ms":...,"captured":{"gap_report_head":"..."}},
    {"name":"learning_path","status":"ok","attempts":1,"elapsed_ms":...,"captured":{"learning_path_head":"..."}}
  ],
  "final_state": {
    "domain": "...",
    "delta_nodes": 12,
    "delta_edges": 18,
    "top_gap_kind": "foundation_gaps",
    "report_path": "data/reports/knowledge_build/<run_id>.md"
  },
  "summary": "..."
}
```

### B. 一句话总结（deterministic 模板，自己产出）

```
[knowledge_build] 状态=ok; 步骤 4/4 成功; 总耗时 7321 ms; 增量 +12 nodes / +18 edges; Top gap: foundation_gaps
```

- 失败时加 `失败: <step>(<error 前 80 字>)`
- 有 skip 时加 `跳过: <names>`
- 有 partial 时加 `部分成功: batch_add(3/4)`

把这一行同时写进 run_log 的 `summary` 字段，和下面要追加的经验文件第一行。

### C. 追加到 `memory/experiences/`（直接用 Write / Edit）

按 `status` 分桶：
- `ok` → append 到 [memory/experiences/success_patterns.md](memory/experiences/success_patterns.md)
- `failed` → append 到 [memory/experiences/failure_patterns.md](memory/experiences/failure_patterns.md)
- 任意 `attempts > 1` 的步骤 → 同时 append 一条到 [memory/experiences/api_quirks.md](memory/experiences/api_quirks.md)

每条 append 的最小模板：

```markdown
### <YYYY-MM-DD HH:MM> knowledge_build · <run_id>
- 总结: <一句话总结>
- 关键 final_state: domain=...; delta_nodes=12; top_gap_kind=foundation_gaps
- 失败/跳过步骤（若有）: batch_add 中 1/4 篇 LLM 抽取空
```

### D. 旁路更新（人为判断）

只在出现新经验时写：
- 某 domain 的 KG 已发现"反复在 X 概念缺连接"模式 → 进 [memory/experiences/analysis_patterns.md](memory/experiences/analysis_patterns.md)
- 用户的研究兴趣发生迁移（从 A 主方向漂到 B） → 更新 [memory/USER.md](memory/USER.md)
- `analyze_knowledge.py` 出现新的失败模式（如某参数组合稳定崩）→ 进 `api_quirks.md`
- MEMORY.md > 800 token 时按"最旧 + 已被取代"优先删。

> 关键守则：**只有可在下次复用的判断才进 experiences**；"这次跑成功了"本身不是经验，"在 KG nodes < 5 时跳过盲区分析能省 30s"才是经验。
