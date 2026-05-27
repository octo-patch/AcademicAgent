"""
ScholarMind - 知识图谱交互式可视化
=================================

【从 graph_store.py 拆出】2026-05-20:
  原 graph_store.py 1011 行里有 ~393 行是 HTML/CSS/JS 嵌入字符串，
  违反"数据层不掺 UI"的单一职责。本模块独立化后：
    - graph_store.py 砍到 ~700 行，只做 CRUD + 持久化 + 查询
    - visualizer.py 集中所有 pyvis / HTML 拼接

用法：
    from src.knowledge.graph_store import KnowledgeGraphStore
    from src.knowledge.visualizer import visualize

    store = KnowledgeGraphStore("data/kg_llm_agent.json", domain="llm_agent")
    visualize(store, output_path="data/kg_visualization.html")

设计准则：
  - 接受 KnowledgeGraphStore 实例，纯函数风格
  - 不修改 store 的任何状态，只读
  - 失败优雅降级（pyvis 未装 / 图空 → 返回 ""）
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path

from .graph_store import KnowledgeGraphStore

logger = logging.getLogger("ScholarMind.Visualizer")


# 节点类型颜色 + 中文标签（pyvis 图例 + 自定义 tooltip 用）
_COLOR_MAP = {
    "paper":             "#2563eb",  # 蓝色
    "concept":           "#dc2626",  # 红色
    "method":            "#0891b2",  # 青色
    "author":            "#6b7280",  # 灰色
    "metric":            "#d97706",  # 琥珀
    "dataset":           "#7c3aed",  # 紫色
    "tool":              "#059669",  # 绿色
    "framework":         "#84cc16",  # 黄绿
    "benchmark":         "#f97316",  # 橙色
    "hardware_platform": "#a855f7",  # 紫蓝
    "system":            "#0ea5e9",  # 天蓝
    "other":             "#94a3b8",  # 中性灰
}

_TYPE_LABELS = {
    "paper":             "📄 论文",
    "concept":           "💡 概念",
    "method":            "⚙️ 方法",
    "author":            "👤 作者",
    "metric":            "📏 指标",
    "dataset":           "📊 数据集",
    "tool":              "🔧 工具",
    "framework":         "🧩 框架",
    "benchmark":         "🏁 基准",
    "hardware_platform": "🖥️ 硬件",
    "system":            "🛰️ 系统",
    "other":             "❓ 其它",
}


def visualize(
    store: KnowledgeGraphStore,
    output_path: str | Path = "data/kg_visualization.html",
) -> str:
    """
    生成交互式知识图谱 HTML（pyvis 力导向图 + 图例 + 字号滑块 + 类型过滤）。

    Args:
        store: 知识图谱 store 实例
        output_path: HTML 输出路径

    Returns:
        输出文件的绝对路径；失败返回 ""。
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.error("pyvis 未安装。请运行: pip install pyvis")
        return ""

    if store.node_count == 0:
        logger.warning("知识图谱为空，无法生成可视化")
        return ""

    # 内部访问：可视化需要 store 的私有图结构，这里集中读取
    nodes = store._nodes
    edges = store._edges
    nx_graph = store._graph

    net = Network(
        height="650px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#f8fafc",
        font_color="#1e293b",
    )

    # 统计各类型数量
    type_counts: dict[str, int] = {}
    for node in nodes.values():
        t = node.node_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    # 构建每个类型的节点列表（供图例下拉用）
    type_node_lists: dict[str, list[dict]] = {t: [] for t in _TYPE_LABELS}

    # 添加节点
    for node_id, node in nodes.items():
        node_type_str = node.node_type.value
        color = _COLOR_MAP.get(node_type_str, "#94a3b8")
        degree = nx_graph.degree(node_id) if node_id in nx_graph else 1

        # tooltip 数据（用 JSON 存储，自定义 JS 解析渲染）
        tooltip_data = {
            "name": node.label,
            "type": _TYPE_LABELS.get(node_type_str, node_type_str),
            "color": color,
            "domain": getattr(node, "domain", "shared"),
        }
        if node.first_seen_year:
            tooltip_data["year"] = node.first_seen_year
        if node.created_at:
            tooltip_data["added"] = node.created_at[:10]
        if node.source_paper:
            tooltip_data["source"] = node.source_paper[:60]
        for key in ("description", "definition", "category"):
            val = node.properties.get(key)
            if val and isinstance(val, str):
                tooltip_data["desc"] = val[:100] + ("..." if len(val) > 100 else "")
                break
        tooltip_data["degree"] = degree
        tooltip_data["ntype"] = node_type_str

        title_json = _json.dumps(tooltip_data, ensure_ascii=False)

        # 节点大小
        size = max(12, min(55, 8 + degree * 4))
        if node_type_str == "author":
            size = max(10, min(20, 8 + degree * 2))

        short_label = node.label if len(node.label) <= 22 else node.label[:19] + "..."
        font_size = max(12, min(30, 10 + degree * 2))

        net.add_node(
            node_id,
            label=short_label,
            color=color,
            title=title_json,
            size=size,
            font={"size": font_size, "color": "#1e293b",
                  "strokeWidth": 3, "strokeColor": "#ffffff"},
            borderWidth=2,
            borderWidthSelected=4,
        )

        if node_type_str in type_node_lists:
            type_node_lists[node_type_str].append({
                "id": node_id,
                "label": node.label,
                "degree": degree,
            })

    # 按度数排序每个类型的列表
    for t in type_node_lists:
        type_node_lists[t].sort(key=lambda x: x["degree"], reverse=True)

    # 添加边
    for edge in edges.values():
        if edge.source_id in nodes and edge.target_id in nodes:
            rel_label = edge.relation_type.value.replace("_", " ")
            net.add_edge(
                edge.source_id,
                edge.target_id,
                title=rel_label,
                label="",
                color="#94a3b888",
                arrows="to",
                width=1.5,
            )

    # 物理引擎配置
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.008,
                "springLength": 160,
                "springConstant": 0.04,
                "avoidOverlap": 0.8
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 200}
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "font": {"size": 0}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 9999999,
            "zoomView": true,
            "dragView": true
        },
        "nodes": {
            "shape": "dot",
            "font": {"face": "Segoe UI, Microsoft YaHei, sans-serif"}
        }
    }
    """)

    # 生成 pyvis 原始 HTML
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output))

    # 读取并注入自定义包装
    raw_html = output.read_text(encoding="utf-8")

    legend_items = _build_legend_items(type_counts, type_node_lists)

    # 年份分布
    years: dict[str, int] = {}
    for n in nodes.values():
        if n.first_seen_year:
            y = str(n.first_seen_year)
            years[y] = years.get(y, 0) + 1
    year_info = " · ".join(f"{y}年: {c}个" for y, c in sorted(years.items()))

    header_html = _build_header(
        store.node_count, store.edge_count, type_counts, year_info,
        domain=store.domain,
    ).replace("{LEGEND_ITEMS}", legend_items)
    footer_html = _build_footer(store.node_count, store.edge_count)
    custom_js = _build_custom_js()

    enhanced = raw_html.replace("<body>", f"<body>\n{header_html}\n", 1)
    enhanced = enhanced.replace("</body>", f"\n{footer_html}\n{custom_js}\n</body>", 1)

    output.write_text(enhanced, encoding="utf-8")
    logger.info(
        f"知识图谱可视化已保存 [{store.domain}]: {output} "
        f"({store.node_count} 节点, {store.edge_count} 边)"
    )
    return str(output.resolve())


# ============================================================
# 内部构造函数
# ============================================================

def _build_legend_items(
    type_counts: dict[str, int],
    type_node_lists: dict[str, list[dict]],
) -> str:
    items = []
    for t, label in _TYPE_LABELS.items():
        c = _COLOR_MAP.get(t, "#94a3b8")
        count = type_counts.get(t, 0)
        if count <= 0:
            continue
        nodes_in_type = type_node_lists.get(t, [])
        dropdown_items = "".join(
            f'<div class="dropdown-node" data-node-id="{n["id"]}">'
            f'{n["label"]} <span class="dd-degree">度={n["degree"]}</span></div>'
            for n in nodes_in_type
        )
        items.append(
            f'<div class="legend-group" data-type="{t}">'
            f'<label class="legend-item">'
            f'<input type="checkbox" class="type-toggle" data-type="{t}" checked '
            f'style="margin:0 2px 0 0;cursor:pointer;">'
            f'<span class="legend-dot" style="background:{c}"></span>'
            f'{label} ({count})</label>'
            f'<div class="legend-dropdown">{dropdown_items}</div>'
            f'</div>'
        )
    return "".join(items)


def _build_header(
    node_count: int,
    edge_count: int,
    type_counts: dict[str, int],
    year_info: str,
    domain: str = "shared",
) -> str:
    """构造 header HTML 模板。{LEGEND_ITEMS} 占位由调用方注入。"""
    css = _CUSTOM_CSS
    domain_badge = {
        "llm_agent": "🤖 LLM Agent",
        "comms":     "📡 Comms",
        "shared":    "🌐 Shared",
    }.get(domain, f"📚 {domain}")
    return f"""
        {css}
        <div class="kg-header">
          <a class="nav-back" href="dashboard.html">← 返回 Dashboard</a>
          <h1>🕸️ 知识图谱交互式可视化 · <span class="domain-badge">{domain_badge}</span></h1>
          <div class="meta">{node_count} 个节点 · {edge_count} 条关系 · {len(type_counts)} 种类型</div>
        </div>
        <div class="kg-bar">
          <span style="font-weight:600;color:#2563eb;margin-right:4px;">图例：</span>
          {{LEGEND_ITEMS}}
          <span style="margin-left:auto;color:#6b7280;font-size:0.78rem;">年份: {year_info}</span>
        </div>
        <div class="kg-tip">
          💡 <b>操作提示</b>：滚轮缩放 · 拖拽画布 · 拖拽节点 · 悬停节点查看详情 · 悬停图例展开列表 · 点击列表项定位高亮
          <span style="margin-left:auto;display:inline-flex;align-items:center;gap:6px;">
            🔤 字号：<input type="range" id="fontSlider" min="6" max="40" value="22" style="width:100px;cursor:pointer;"><span id="fontVal" style="min-width:28px">22</span>
          </span>
        </div>
        <div id="kg-tooltip"></div>
    """


def _build_footer(node_count: int, edge_count: int) -> str:
    return f"""
        <div class="kg-footer">
          ScholarMind 知识图谱 · {node_count} nodes / {edge_count} edges · 节点大小=连接度
        </div>
    """


_CUSTOM_CSS = """
<style>
  body { margin: 0; font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f5f7fb; }
  .kg-header { background: linear-gradient(135deg, #2563eb, #0891b2); color: white; padding: 1.2rem 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
  .kg-header h1 { font-size: 1.3rem; margin: 0 0 0.3rem; }
  .kg-header .meta { opacity: 0.85; font-size: 0.85rem; }
  .kg-header .nav-back { color: white; text-decoration: none; font-size: 0.85rem; opacity: 0.8; }
  .kg-header .nav-back:hover { opacity: 1; text-decoration: underline; }
  .kg-header .domain-badge { background: rgba(255,255,255,0.2); padding: 2px 10px; border-radius: 12px; font-size: 0.85rem; margin-left: 8px; }
  .kg-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem; background: white; padding: 0.6rem 2rem; border-bottom: 1px solid #e2e8f0; font-size: 0.85rem; }
  .legend-group { position: relative; display: inline-block; }
  .legend-item { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 6px; color: #374151; cursor: pointer; transition: background 0.15s; }
  .legend-item:hover { background: #f0f4f8; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .legend-dropdown { display: none; position: absolute; top: 100%; left: 0; z-index: 999; background: white; border: 1px solid #e2e8f0; border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,0.12); min-width: 240px; max-height: 280px; overflow-y: auto; padding: 6px 0; }
  .legend-group:hover .legend-dropdown { display: block; }
  .dropdown-node { padding: 6px 14px; cursor: pointer; font-size: 0.82rem; transition: background 0.1s; display: flex; justify-content: space-between; align-items: center; }
  .dropdown-node:hover { background: #eff6ff; color: #2563eb; }
  .dd-degree { font-size: 0.7rem; color: #94a3b8; }
  .kg-tip { background: #fffbeb; border-top: 1px solid #fde68a; padding: 0.5rem 2rem; font-size: 0.78rem; color: #92400e; }
  .kg-footer { text-align: center; padding: 0.6rem; font-size: 0.72rem; color: #94a3b8; border-top: 1px solid #e2e8f0; background: white; }
  #mynetwork { border: none !important; }
  #kg-tooltip {
    display: none; position: fixed; z-index: 10000;
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12); padding: 14px 18px;
    min-width: 220px; max-width: 340px; pointer-events: none;
    font-size: 0.85rem; line-height: 1.6;
  }
  #kg-tooltip .tt-name { font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; }
  #kg-tooltip .tt-type { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; margin-bottom: 8px; }
  #kg-tooltip .tt-row { display: flex; align-items: center; gap: 6px; color: #475569; font-size: 0.82rem; margin: 3px 0; }
  #kg-tooltip .tt-row .tt-icon { width: 16px; text-align: center; flex-shrink: 0; }
  #kg-tooltip .tt-desc { margin-top: 8px; padding-top: 8px; border-top: 1px solid #f1f5f9; color: #64748b; font-size: 0.8rem; }
  #kg-tooltip .tt-bar { height: 3px; border-radius: 2px; margin-top: 8px; }
</style>
"""


def _build_custom_js() -> str:
    # 注意：花括号用 {{ }} 转义，因为返回的是模板片段会经 f-string 路径
    # 但这里我们直接 return 字符串，不再二次 format，所以单括号即可
    return """
        <script>
        (function() {
          var tooltip = document.getElementById('kg-tooltip');
          var net = network;  // pyvis 暴露的全局变量

          // --- 1. 自定义 Tooltip ---
          net.on('hoverNode', function(params) {
            var nodeId = params.node;
            var nodeData = net.body.data.nodes.get(nodeId);
            if (!nodeData || !nodeData.title) return;
            try {
              var d = JSON.parse(nodeData.title);
              var html = '<div class="tt-name" style="color:' + (d.color||'#333') + '">' + d.name + '</div>';
              html += '<span class="tt-type" style="background:' + (d.color||'#333') + '22;color:' + (d.color||'#333') + '">' + (d.type||'') + '</span>';
              if (d.domain) html += '<div class="tt-row"><span class="tt-icon">🌐</span> 域: ' + d.domain + '</div>';
              if (d.year) html += '<div class="tt-row"><span class="tt-icon">📅</span> 首次出现: ' + d.year + ' 年</div>';
              if (d.added) html += '<div class="tt-row"><span class="tt-icon">🕐</span> 录入时间: ' + d.added + '</div>';
              if (d.source) html += '<div class="tt-row"><span class="tt-icon">📎</span> 来源: ' + d.source + '</div>';
              html += '<div class="tt-row"><span class="tt-icon">🔗</span> 连接数: ' + (d.degree||0) + '</div>';
              if (d.desc) html += '<div class="tt-desc">📝 ' + d.desc + '</div>';
              html += '<div class="tt-bar" style="background:' + (d.color||'#333') + '"></div>';
              tooltip.innerHTML = html;
              tooltip.style.display = 'block';
            } catch(e) {}
          });

          net.on('blurNode', function() {
            tooltip.style.display = 'none';
          });

          document.getElementById('mynetwork').addEventListener('mousemove', function(e) {
            if (tooltip.style.display === 'block') {
              var x = e.clientX + 15;
              var y = e.clientY + 15;
              if (x + 350 > window.innerWidth) x = e.clientX - 260;
              if (y + 200 > window.innerHeight) y = e.clientY - 180;
              tooltip.style.left = x + 'px';
              tooltip.style.top = y + 'px';
            }
          });

          // --- 2. 图例下拉：点击节点项 → 高亮并聚焦 ---
          document.querySelectorAll('.dropdown-node').forEach(function(el) {
            el.addEventListener('click', function(e) {
              e.stopPropagation();
              var nodeId = this.getAttribute('data-node-id');
              net.selectNodes([nodeId]);
              net.focus(nodeId, {
                scale: 1.5,
                animation: {duration: 600, easingFunction: 'easeInOutQuad'}
              });
            });
          });

          // --- 3. 字号滑块 ---
          var slider = document.getElementById('fontSlider');
          var fontVal = document.getElementById('fontVal');
          var origFonts = {};
          var nodeTypes = {};
          var hiddenTypes = {};
          net.body.data.nodes.forEach(function(node) {
            origFonts[node.id] = (node.font && node.font.size) || 14;
            try { nodeTypes[node.id] = JSON.parse(node.title).ntype || ''; } catch(e) { nodeTypes[node.id] = ''; }
          });

          if (slider) {
            slider.addEventListener('input', function() {
              var scale = parseInt(this.value) / 22.0;
              fontVal.textContent = parseInt(this.value);
              var allNodes = net.body.data.nodes;
              var updates = [];
              allNodes.forEach(function(node) {
                var hidden = hiddenTypes[nodeTypes[node.id]];
                var newSize = hidden ? 0 : Math.round(origFonts[node.id] * scale);
                updates.push({id: node.id, font: {size: newSize}});
              });
              allNodes.update(updates);
            });
          }

          // --- 4. Checkbox 切换类型标签显示/隐藏 ---
          document.querySelectorAll('.type-toggle').forEach(function(cb) {
            cb.addEventListener('change', function() {
              var type = this.getAttribute('data-type');
              var show = this.checked;
              hiddenTypes[type] = !show;
              var scale = slider ? parseInt(slider.value) / 22.0 : 1.0;
              var allNodes = net.body.data.nodes;
              var updates = [];
              allNodes.forEach(function(node) {
                if (nodeTypes[node.id] === type) {
                  var newSize = show ? Math.round(origFonts[node.id] * scale) : 0;
                  updates.push({id: node.id, font: {size: newSize}});
                }
              });
              allNodes.update(updates);
            });
          });
        })();
        </script>
    """
