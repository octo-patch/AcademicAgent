---
description: 使用模板或自定义代码运行论文方法验证和科研实验
---

# Simulation (实验验证) Workflow

把"论文里那个方法 / 评测指标 / toy example"翻译成**可重复运行**的 Python 实验，
跑出图表与数值后回归到知识图谱，让"读懂"升级为"复现得了"。
本 workflow 是 `/paper-analysis` 的姊妹环：分析回答 *"它说了什么"*，仿真回答 *"它真的成立吗"*。

---

## 前置准备（Bootstrap）

1. 读 [memory/USER.md](memory/USER.md) — 用户主方向，决定模板优先级（comms / agent / sensing）。
2. 读 [memory/MEMORY.md](memory/MEMORY.md) — 历史经验索引，关注 [memory/experiences/api_quirks.md](memory/experiences/api_quirks.md) 中 sandbox / matplotlib / numpy 的已知坑。
3. 读 [memory/preferences.json](memory/preferences.json) — `simulation.default_timeout / simulation.allow_network / simulation.max_outputs` 等。
4. 在内存建 `run_state`，约定字段：
   - `mode: "template" | "custom"` — 走模板还是自定义代码
   - `template_name: str` — 模板名（mode=template 时）
   - `param_overrides: dict[str, Any]` — 用户改的参数
   - `code: str` — 自定义代码（mode=custom 时）
   - `timeout_s: int` — 超时（默认 60，长跑实验可调）
   - `stdout / stderr / output_files: list[str]` — 运行产物
   - `figure_descriptions: list[str]` — 对生成图的视觉理解（用户要求时）
   - `linked_paper: str | None` — 关联到 KG 中的哪篇论文（用于步骤 7）
   - `steps_log: list[dict]` — 喂 Lite Reflector

> 安全前提：所有代码均在 `src/execution/sandbox.py` 沙箱里跑，遵循 env 白名单 + setrlimit + import 白名单
> （见 [memory/MEMORY.md](memory/MEMORY.md) 的 Sandbox Phase 7 条目）。不要尝试关闭沙箱来"图方便"。

---

## 详细步骤

### 步骤 1. 明确实验目标（先想再跑）

**目的**：避免"上来就 run 模板"，跑完发现做的不是用户想要的。

**操作**：1 轮短对话，澄清以下三点（用户没明说时主动问）：
- **要复现的对象**：论文里的某个公式、某张图、某个 baseline 还是 toy demo？
- **关心的指标**：BER vs SNR？任务成功率？检索 recall@k？
- **比较对象**：要不要叠加自变量扫描或 baseline 对比？

**产出**：`run_state.experiment_goal: str`（一段 ≤ 200 字的目标声明）。

**失败策略**：用户给不出具体目标 → 默认建议先跑 `agent_eval_toy` / `ofdm_basic` 之类的入门模板熟悉接口，不强行让用户立刻精确化。

### 步骤 2. 列出可用模板

**目的**：让用户知道有哪些"开箱即用"的脚手架，避免重写。

**操作**：MCP 调 `list_code_templates()`。当前可用（以 MCP 返回为准）：
- `agent_eval_toy` — Agent 评测的最小可跑闭环
- `rag_retrieval_eval` — RAG 检索 recall / NDCG
- `ofdm_basic` — OFDM 系统的 BER vs SNR
- `mimo_beamforming` — MIMO 波束成形
- `aoa_music` — MUSIC 算法 AoA 估计

**产出**：`run_state.available_templates: list[str]`。

**失败策略**：MCP 不可用 → 退而走步骤 4 的自定义代码模式，提示用户"模板服务暂不可用，本次以自定义代码进行"。

### 步骤 3. 选模板 vs 自定义（分支）

**目的**：节省用户精力。能用模板就别从零写，能复用模板加参数就别 fork。

**操作**：
- **走模板** → `run_state.mode = "template"`；调 `explain_template(name)` 拉模板的参数表、依赖、预期产物，**逐项展示给用户后再让他确认是否修改**。
- **走自定义** → `run_state.mode = "custom"`；进入步骤 4。
- **介于两者之间**（模板能覆盖 80%，但要小改）→ 仍走 template，把改动通过 `param_overrides` 表达；只有当模板核心逻辑要替换才转 custom。

**产出**：`run_state.mode` + 模板的参数说明。

**失败策略**：`explain_template` 报错 → 给用户看模板源码路径 + README，让他手动看；不要凭空猜参数。

### 步骤 4. 参数 / 代码准备

**目的**：把"用户想跑什么"翻译成"sandbox 接收的入参"。

**操作**：
- **模板**：和用户确认 `param_overrides`，常见格式 `"N_tasks=100, success_threshold=0.7"`。把它解析成 dict 存 `run_state.param_overrides`。
- **自定义**：协助用户写 Python；优先复用 numpy / scipy / matplotlib；避免引入沙箱白名单外的依赖。代码要：
  - 把"配置在顶部"（常量集中），便于复用
  - `print()` 关键中间量（runner 抓 stdout）
  - 显式 `plt.savefig("xxx.png", dpi=120)` 而不是 `plt.show()`（沙箱无 GUI）

**产出**：`run_state.code` 或 `run_state.param_overrides`。

**失败策略**：参数解析失败 → 直接报给用户看到具体哪条解析失败，不要静默丢弃。

### 步骤 5. 运行实验

**目的**：在沙箱里真跑一次，拿到 stdout + 图。

**操作**：
- 模板：`run_template(template_name, parameter_overrides, timeout=run_state.timeout_s)`
- 自定义：`run_code(code, timeout=run_state.timeout_s, description="<experiment_goal 摘要>")`

**产出**：`run_state.stdout / stderr / output_files / elapsed_ms / exit_code`。

**失败策略**：
- 超时 → 不重试相同参数（无意义）。提示用户缩 N / 缩迭代次数 / 关闭画图，参数改完后再跑。
- import 报错 → 检查是否触碰白名单（参考 Sandbox Phase 7）。在 `failure_patterns.md` 里查 "ImportError" 是否有同名问题，复用历史解。
- 矩阵维度类 ValueError → `retry_once` 没用，直接把 stderr 完整贴给用户，配合代码上下文修。
- 退码非 0 但有部分产出 → 标记 `partial`，仍进入步骤 6 分析已有图（部分胜过零）。

### 步骤 6. 结果分析

**目的**：让数字 / 图变成"结论"。

**操作**：
1. **数值结果**：用 stdout 里的关键数（BER / accuracy / recall）回贴用户，配合预期范围给一句判断（"BER 在 SNR=10dB 时为 1e-3，与论文 Fig.4 一致"）。
2. **图表**：对每个 `output_files` 里的 `.png` 调 `view_file` 看图，写 1-2 句视觉描述（趋势、拐点、是否与理论符合）。
3. **可疑结果检测**：曲线异常（如 BER 不单调下降、recall 突然为 0）→ 主动提醒，而不是把图丢给用户自己判断。

**产出**：`run_state.figure_descriptions / numeric_summary`。

**失败策略**：`skip` 的子项（如某张图打不开）不影响整体；写进 `steps_log` 即可。

### 步骤 7. 与知识库 / 论文关联

**目的**：让仿真结果回流到知识图谱，把"我跑过这个实验"变成可被未来检索的事实。

**操作**：
- 若实验对应 KG 中某论文 → MCP `add_insight_to_paper(paper_id=..., insight="<numeric_summary 浓缩>")`（如工具可用）；或在 `data/reports/simulations/<run_id>.md` 落一份带 `linked_paper` 字段的报告。
- 若实验暴露了"论文中没说清楚的边界" → 写进 [memory/experiences/analysis_patterns.md](memory/experiences/analysis_patterns.md)，标"<paper> 在 <param> 取 <range> 时偏离"。

**产出**：`run_state.linked_paper / insight_recorded: bool`。

**失败策略**：`skip`。关联失败不算实验失败。

### 步骤 8. 迭代菜单

跑完一次后，给用户一个**封闭式**菜单（开放问题会让对话发散）：
- "要调参数重跑吗？"（给 1-2 个建议方向）
- "要叠加 baseline / 对比曲线吗？"
- "要把这组数对照论文 Table 来核吗？"
- "要把脚本固化成新模板吗？"（如果是 custom 且效果好，可考虑提交回 `src/execution/templates/`）

---

## 鲁棒性与状态管理 Checklist

- **沙箱不绕**：永远在沙箱里跑；任何"关沙箱就能跑"的请求都要先理解需求再判断（绝大多数情况换个写法即可）。
- **超时分级**：默认 60s；CPU-bound 仿真允许调到 300s；> 300s 的实验建议拆成"小数据先验证、大数据后台跑"。
- **产物路径稳定**：所有图统一相对路径 + `dpi=120`，避免沙箱 cwd 漂移导致路径无法回找。
- **状态隔离**：每次 run 一个独立 `run_state`；同一会话连跑多次时**不要共享 `run_state.code`**（很容易把上次的全局变量带进来）。
- **重试粒度**：实验失败的本质是逻辑问题，不是瞬时问题；除"沙箱启动失败"这类基础设施类错误，**不要自动重试**，把决策权交给用户。
- **观测可追溯**：步骤 5 的 stdout + stderr 全量保留，落 `data/runtime_runs/simulation_<YYYYMMDD-HHMM>.json` 的 `final_state`，便于事后查。
- **预算**：单次 run_log 里 stdout 截断 16K，图描述截断 1K，避免 run_log 爆炸。

---

## Lite Reflector & 经验沉淀（必做收尾）

跑完后由宿主 Agent 自己做这一段，仅靠文件 I/O，无外部 runner 依赖。

### A. 组装 run_log（自己拼，存成 JSON）

把会话里维护的 `run_state.steps_log` 整理成一份 JSON，落到 `data/runtime_runs/simulation_<YYYYMMDD-HHMM>.json`：

```json
{
  "run_id": "<时间戳 + 6 位随机>",
  "workflow": "simulation",
  "started_at": "<iso8601>",
  "ended_at": "<iso8601>",
  "status": "ok | failed | partial",
  "steps": [
    {"name":"select_template","status":"ok","attempts":1,"elapsed_ms":...},
    {"name":"prepare_params","status":"ok","attempts":1,"elapsed_ms":...},
    {"name":"run","status":"ok","attempts":1,"elapsed_ms":...,"captured":{"exit_code":0,"output_files":["ber_snr.png"]}},
    {"name":"analyze","status":"ok","attempts":1,"elapsed_ms":...},
    {"name":"link_to_kg","status":"ok","attempts":1,"elapsed_ms":...}
  ],
  "final_state": {
    "mode": "template",
    "template_name": "ofdm_basic",
    "param_overrides": {"N_subcarriers": 1024, "snr_db": [0,5,10,15,20]},
    "linked_paper": "ofdm_xxx_2024",
    "stdout_head": "...",
    "figure_files": ["ber_snr.png"],
    "verdict": "与论文 Fig.4 一致"
  },
  "summary": "..."
}
```

### B. 一句话总结（deterministic 模板，自己产出）

```
[simulation] 状态=ok; 步骤 5/5 成功; 模板=ofdm_basic; 出图 1 张; 与论文一致; 总耗时 12871 ms
```

- 失败时点名失败步骤 + 错误前 80 字：`失败: run(ImportError: scikit-learn not allowed)`
- partial 时显式：`部分成功: run(exit=1，但生成了 1/3 张图)`

把这一行同时写进 run_log 的 `summary` 字段，和下面要追加的经验文件第一行。

### C. 追加到 `memory/experiences/`（直接用 Write / Edit）

按 `status` 分桶：
- `ok` → append 到 [memory/experiences/success_patterns.md](memory/experiences/success_patterns.md)（沉淀"<模板> 在 <参数组> 下能复现 <论文>"这类高价值经验）
- `failed` → append 到 [memory/experiences/failure_patterns.md](memory/experiences/failure_patterns.md)（沉淀"<模板> 在 <参数> 下会触发 <错误>"）
- 任意 `attempts > 1` 的步骤 → 同时 append 一条到 [memory/experiences/api_quirks.md](memory/experiences/api_quirks.md)（仿真很少自动重试，命中时往往是沙箱启动抖动）

每条 append 的最小模板：

```markdown
### <YYYY-MM-DD HH:MM> simulation · <run_id>
- 总结: <一句话总结>
- 关键 final_state: template=ofdm_basic; params={...}; linked_paper=...; verdict=与论文一致
- 失败/跳过步骤（若有）: link_to_kg 跳过（未关联到具体论文）
```

### D. 旁路更新（人为判断）

- 发现新的 sandbox 限制 / 白名单坑 → 进 [memory/experiences/api_quirks.md](memory/experiences/api_quirks.md)，建议同时在 `src/execution/sandbox.py` 提 issue
- 自定义代码跑出"足够通用"的实验 → 建议固化成模板：把代码挪到 `src/execution/templates/<name>.py`，并更新 MEMORY.md 索引
- 与论文严重不符的复现结果 → 进 [memory/experiences/analysis_patterns.md](memory/experiences/analysis_patterns.md)，标 "<paper> 在 <param> 时与原文不符，怀疑 <原因>"
- 用户研究主方向从 A 漂到 B（仿真模板偏好变化）→ 更新 [memory/USER.md](memory/USER.md)
- MEMORY.md > 800 token → 优先清掉过时模板的经验

> 关键守则：**仿真经验最值钱的是"复现/不复现 + 边界条件"**。"代码能跑通"不是经验；"在 SNR > 15dB 时模板 X 与论文 Y 不一致，需要把 noise 模型从 AWGN 换成 Rician" 才是经验。
