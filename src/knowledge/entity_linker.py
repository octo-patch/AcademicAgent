"""
ScholarMind - Entity Linking（实体规范化）
==========================================

为什么需要：
  没有 entity linking 时，"SLAM" / "Simultaneous Localization and Mapping" / "slam"
  会被识别为 3 个不同的节点。Phase 1 后 KG 已经看到这种分裂（见 data/kg_comms.json）。

本模块的三层（轻量 + 可演进）：

  Layer 1 — Mention Normalization:
    现有 KGNode._normalize_label + 本模块加载 data/aliases.yaml 同义词典。
    "Reasoning and Acting" → "ReAct"。

  Layer 2 — Candidate Generation:
    在 store 中找 node_id 完全一致的候选；
    若候选 0 个，可选 fallback 到 SemanticIndex（Phase 2.4）的 top-K 近邻。

  Layer 3 — Disambiguation (LLM-as-judge, 可选):
    候选 ≥2 时让 MiniMax 选一个；prompt 给候选的 label/type/source/properties。
    本阶段先实现 layer 1+2，Layer 3 留接口（is_ambiguous() + disambiguate_with_llm()）。

设计准则：
  - 不强制：调用方可以选择只做 normalize（不写 KG）或 link（命中已有节点）
  - 别名表懒加载，缺 yaml 包时优雅退化
  - 大小写不敏感，加宽字符、连字符差异
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import KGNode, NodeType
    from .graph_store import KnowledgeGraphStore

logger = logging.getLogger("ScholarMind.EntityLinker")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ALIASES = _PROJECT_ROOT / "data" / "aliases.yaml"


# ============================================================
# Layer 1 — AliasMap
# ============================================================

def _key(label: str) -> str:
    """归一化 alias 查找 key：小写 + 剥括号注释 + 折叠连字符/下划线/空格。

    剥括号："Chain-of-Thought (CoT)" / "ReAct (Reasoning and Acting)" 这种
    注释式后缀很常见，括号里的内容跟前面是同义体，应当合并到同一 canonical。
    """
    s = label.lower().strip()
    # NEW: 剥掉括号及其内容（含中文全角括号），避免 "X (Y)" 被当作新实体
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"（[^）]*）", " ", s)
    s = re.sub(r"[-_/]+", " ", s)  # hyphen/underscore -> space
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s一-鿿]", "", s)  # keep ASCII + Chinese
    return s.strip()


class AliasMap:
    """加载并查询 data/aliases.yaml 的同义词典。"""

    def __init__(self, yaml_path: Path | None = None):
        self._path = yaml_path or _DEFAULT_ALIASES
        self._alias_to_canonical: dict[str, str] = {}
        self._canonical_keys: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("aliases.yaml 不存在 (%s)，AliasMap 为空", self._path)
            return
        try:
            import yaml  # PyYAML
        except ImportError:
            logger.warning("PyYAML 未安装，跳过 aliases.yaml 加载")
            return
        try:
            data = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("aliases.yaml 解析失败: %s", exc)
            return
        if not isinstance(data, dict):
            logger.warning("aliases.yaml 顶层应为 dict，实为 %s", type(data).__name__)
            return

        n_canonical, n_alias = 0, 0
        for canonical, aliases in data.items():
            canonical = str(canonical).strip()
            if not canonical:
                continue
            ck = _key(canonical)
            self._canonical_keys.add(ck)
            self._alias_to_canonical[ck] = canonical  # 自身映射
            n_canonical += 1
            for a in aliases or []:
                ak = _key(str(a))
                if ak and ak != ck:
                    self._alias_to_canonical[ak] = canonical
                    n_alias += 1
        logger.info("AliasMap 加载完成: %d canonical / %d aliases (%s)",
                    n_canonical, n_alias, self._path.name)

    @property
    def size(self) -> int:
        return len(self._alias_to_canonical)

    def normalize(self, label: str) -> str:
        """返回标准化标签。无命中则返回原 label（去首尾空白）。"""
        if not label:
            return label
        ck = _key(label)
        if ck in self._alias_to_canonical:
            canonical = self._alias_to_canonical[ck]
            if canonical != label.strip():
                logger.debug("alias hit: %r -> %r", label, canonical)
            return canonical
        return label.strip()

    def is_known(self, label: str) -> bool:
        return _key(label) in self._alias_to_canonical


# ============================================================
# Layer 2 — Candidate Generation + EntityLinker
# ============================================================

class EntityLinker:
    """实体规范化器。

    使用：
        linker = EntityLinker(store)            # store 可选；不传时只做 normalize
        canonical_label = linker.normalize_label("Reasoning and Acting", "method")
        # → "ReAct"
        matched_node = linker.link("ReAct", "method")
        # → 已存在的同 id 节点，或 None
    """

    def __init__(
        self,
        store: "KnowledgeGraphStore | None" = None,
        alias_map: AliasMap | None = None,
        *,
        semantic_threshold: float = 0.85,
        enable_semantic: bool = True,
    ):
        self.store = store
        self.alias_map = alias_map or AliasMap()
        self.semantic_threshold = semantic_threshold
        self.enable_semantic = enable_semantic
        # 懒加载：encoder 是 sentence-transformer 实例或 False（哨兵：尝试过但失败）
        self._encoder = None
        # node_id -> 归一化后的 embedding，避免重复 encode 已见 label
        self._label_emb_cache: dict[str, "object"] = {}

    # ---------------- encoder 懒加载 ----------------

    def _get_encoder(self):
        """复用 semantic_index 的 sentence-transformer；不可用时返回 None。"""
        if self._encoder is False:
            return None
        if self._encoder is not None:
            return self._encoder
        if not self.enable_semantic:
            self._encoder = False
            return None
        try:
            from .semantic_index import _SentenceTransformerEncoder
            self._encoder = _SentenceTransformerEncoder()
            return self._encoder
        except Exception as exc:  # noqa: BLE001
            logger.info("EntityLinker 语义兜底不可用 (%s)，仅走 Layer 1+2", exc)
            self._encoder = False
            return None

    def _encode(self, text: str):
        """单条 encode。caller 保证已检查 encoder 存在。"""
        enc = self._get_encoder()
        if enc is None:
            return None
        return enc.encode([text])[0]

    # ---------------- Layer 1 ----------------

    def normalize_label(self, mention: str, node_type_hint: str | None = None) -> str:
        """Layer 1: 用 AliasMap 把 mention 规范化为 canonical 形式。

        不查询 store；纯字符串变换。
        """
        return self.alias_map.normalize(mention)

    # ---------------- Layer 2: id 严格匹配 + Layer 2b: embedding 兜底 ----------------

    def link(
        self,
        mention: str,
        node_type: "NodeType | str",
        *,
        use_semantic_fallback: bool = True,
    ) -> "KGNode | None":
        """Layer 2: 把 mention 规范化后在 store 里找 node。

        优先严格 id 命中；失败时若 ``use_semantic_fallback=True``，调用
        ``find_semantic_match`` 在同 node_type 的现有节点里找 cos_sim ≥
        semantic_threshold (默认 0.85) 的近邻。

        返回命中的 KGNode 或 None（无 store 也返回 None）。
        """
        if self.store is None:
            return None
        from .schema import KGNode, NodeType as _NT

        if isinstance(node_type, str):
            try:
                node_type_enum = _NT(node_type)
            except ValueError:
                from .schema import coerce_node_type
                node_type_enum, _ = coerce_node_type(node_type)
        else:
            node_type_enum = node_type

        canonical = self.normalize_label(mention)
        probe = KGNode(label=canonical, node_type=node_type_enum)
        hit = self.store.get_node(probe.node_id)
        if hit is not None:
            return hit

        if not use_semantic_fallback:
            return None

        # Layer 2b: embedding 兜底
        match_label = self.find_semantic_match(
            mention, node_type_enum, source="store"
        )
        if match_label is None:
            return None
        probe2 = KGNode(label=match_label, node_type=node_type_enum)
        return self.store.get_node(probe2.node_id)

    def find_semantic_match(
        self,
        mention: str,
        node_type: "NodeType | str",
        *,
        source: str = "store",
        extra_candidates: list[str] | None = None,
        threshold: float | None = None,
    ) -> str | None:
        """在候选 label 集合里找 cos_sim ≥ threshold 的最相似项。

        Args:
            mention: 待查询的 label 文本（已规范化或原始均可）
            node_type: 候选限定的 node_type（同型号才比对）
            source: "store" 走 self.store 同 type 节点；"extra" 只看
                ``extra_candidates``；"both" 合并两者。
            extra_candidates: 额外候选 label 列表（如同批刚抽到的节点）
            threshold: 覆盖实例默认 ``semantic_threshold``

        Returns:
            匹配到的候选 label（不归一化），或 None。
        """
        enc = self._get_encoder()
        if enc is None or not mention:
            return None
        from .schema import NodeType as _NT
        try:
            ntype = _NT(node_type) if isinstance(node_type, str) else node_type
        except Exception:  # noqa: BLE001
            return None

        thr = self.semantic_threshold if threshold is None else threshold
        candidates: list[str] = []
        if source in ("store", "both") and self.store is not None:
            for n in self.store._nodes.values():
                if n.node_type == ntype and n.label and n.label != mention:
                    candidates.append(n.label)
        if source in ("extra", "both") and extra_candidates:
            for lbl in extra_candidates:
                if lbl and lbl != mention and lbl not in candidates:
                    candidates.append(lbl)
        if not candidates:
            return None

        try:
            import numpy as np
            vecs = enc.encode([mention] + candidates)  # 已 L2-normalized
            q = vecs[0]
            cands = vecs[1:]
            sims = cands @ q.T
            idx = int(np.argmax(sims))
            best = float(sims[idx])
            if best >= thr:
                logger.debug(
                    "semantic match: %r -> %r (sim=%.3f, type=%s)",
                    mention, candidates[idx], best, ntype.value,
                )
                return candidates[idx]
        except Exception as exc:  # noqa: BLE001
            logger.debug("semantic match failed: %s", exc)
        return None

    def link_or_create(
        self,
        mention: str,
        node_type: "NodeType | str",
        **kwargs,
    ) -> "KGNode":
        """便捷：命中已存在节点 → 返回；否则构造新 KGNode（不写 store）。"""
        from .schema import KGNode, NodeType as _NT

        existing = self.link(mention, node_type)
        if existing:
            return existing

        if isinstance(node_type, str):
            try:
                node_type_enum = _NT(node_type)
            except ValueError:
                from .schema import coerce_node_type
                node_type_enum, _ = coerce_node_type(node_type)
        else:
            node_type_enum = node_type

        canonical = self.normalize_label(mention)
        return KGNode(label=canonical, node_type=node_type_enum, **kwargs)

    # ============================================================
    # Batch 末端去重 —— 抽取完成后一次性 batch 跑 embedding 合并
    # ============================================================

    def batch_semantic_dedup(
        self,
        nodes: list,
        edges: list,
        *,
        threshold: float | None = None,
        include_store: bool = True,
    ) -> tuple[list, list, dict[str, str], dict[str, str]]:
        """对一批 KGNode/KGEdge 做一次性语义去重。

        策略：按 node_type 分桶 → 桶内 encode 全部 label（含 store 同 type 节点）
        → 计算 cos_sim 矩阵 → 贪心合并：cos_sim ≥ threshold 的对，把"批内后出现"
        或"批内不在 store 里"的 label 合并到"先出现 / 在 store 里"的那个。

        Args:
            nodes:  抽取得到的 KGNode 列表（_merge_results 后的产物）
            edges:  抽取得到的 KGEdge 列表，端点 id 来自上述 nodes
            threshold: cos_sim 阈值（默认 self.semantic_threshold = 0.85）
            include_store: 是否把 self.store 已有的同 type 节点也纳入比对池
                （为 True 时实现跨论文 dedup）

        Returns:
            (deduped_nodes, remapped_edges, node_label_remap, edge_id_remap)
              - node_label_remap: 旧 label -> canonical label
              - edge_id_remap:   旧 edge_id -> 新 edge_id
        """
        from .schema import KGNode, KGEdge

        if not nodes:
            return list(nodes), list(edges), {}, {}

        enc = self._get_encoder()
        thr = self.semantic_threshold if threshold is None else threshold
        if enc is None:
            # encoder 不可用：直接返回原物（不做 dedup，但接口契约一致）
            return list(nodes), list(edges), {}, {}

        # ---- 1. 按 type 分桶（batch 内）----
        type_to_indices: dict[str, list[int]] = {}
        for i, n in enumerate(nodes):
            t = n.node_type.value
            type_to_indices.setdefault(t, []).append(i)

        node_label_remap: dict[str, str] = {}  # 旧 label -> canonical label

        for ntype, indices in type_to_indices.items():
            # 候选池 = batch 内同 type 的 label + store 同 type 的 label
            batch_labels = [nodes[i].label for i in indices]
            store_labels: list[str] = []
            if include_store and self.store is not None:
                try:
                    from .schema import NodeType as _NT
                    nt_enum = _NT(ntype)
                except Exception:  # noqa: BLE001
                    nt_enum = None
                if nt_enum is not None:
                    for sn in self.store._nodes.values():
                        if sn.node_type == nt_enum and sn.label and sn.label not in batch_labels:
                            store_labels.append(sn.label)

            all_labels = batch_labels + store_labels
            if len(all_labels) < 2:
                continue

            try:
                import numpy as np
                vecs = enc.encode(all_labels)  # 已 L2-normalized
                sim = vecs @ vecs.T
                np.fill_diagonal(sim, 0.0)
            except Exception as exc:  # noqa: BLE001
                logger.debug("batch encode failed (%s), skip dedup for type=%s", exc, ntype)
                continue

            n_batch = len(batch_labels)
            # 贪心：扫描 batch 中的每个 label j，找候选池里 (i < j 或 store_label) 的最高相似项
            # 优先级：store 已有的优先（i in [n_batch, end)），然后 batch 内先出现的
            for j in range(n_batch):
                # 当前 batch label
                lbl_j = batch_labels[j]
                if lbl_j in node_label_remap:
                    continue  # 已合并过
                # 先看 store 端候选（如果有相似项，合并到 store label）
                best_i, best_s = -1, -1.0
                for i in range(n_batch, len(all_labels)):
                    if float(sim[j, i]) > best_s:
                        best_s = float(sim[j, i])
                        best_i = i
                # 再看 batch 内更早出现的 label
                for i in range(j):
                    s = float(sim[j, i])
                    # 同 batch 同 label 时 sim=1.0 但 fill_diagonal=0；不同 label 才走这里
                    if all_labels[i] in node_label_remap:
                        continue
                    if s > best_s:
                        best_s = s
                        best_i = i

                if best_i >= 0 and best_s >= thr:
                    canonical = all_labels[best_i]
                    if canonical != lbl_j:
                        node_label_remap[lbl_j] = canonical
                        logger.debug(
                            "batch dedup [%s]: %r -> %r (sim=%.3f)",
                            ntype, lbl_j, canonical, best_s,
                        )

        # ---- 2. 应用 remap 到 nodes（去掉被合并的，保留剩下的）----
        deduped_nodes = []
        seen_ids: set[str] = set()
        for n in nodes:
            if n.label in node_label_remap:
                # 这个节点已被合并到 canonical → 跳过；
                # 注：properties 合并由后续 store.add_node 处理（同 id merge）。
                continue
            if n.node_id in seen_ids:
                continue
            seen_ids.add(n.node_id)
            deduped_nodes.append(n)

        # ---- 3. 应用 remap 到 edges 端点（重建 source_id / target_id）----
        edge_id_remap: dict[str, str] = {}
        remapped_edges = []
        for e in edges:
            new_e = e
            changed = False
            # 端点 id 形如 "{type}_{normalized_label}"；要 remap 需要先反解 label
            # 但 KGEdge 没存 label，只有 id。我们用 store 或 batch 反查。
            def _resolve_label(node_id: str) -> str | None:
                # 优先在 batch 找
                for nn in nodes:
                    if nn.node_id == node_id:
                        return nn.label
                if self.store is not None:
                    snode = self.store.get_node(node_id)
                    if snode is not None:
                        return snode.label
                return None

            new_src_label = None
            new_tgt_label = None
            src_lbl = _resolve_label(e.source_id)
            tgt_lbl = _resolve_label(e.target_id)
            if src_lbl and src_lbl in node_label_remap:
                new_src_label = node_label_remap[src_lbl]
                changed = True
            if tgt_lbl and tgt_lbl in node_label_remap:
                new_tgt_label = node_label_remap[tgt_lbl]
                changed = True

            if changed:
                # 找到 canonical label 对应的 type → 重建 id
                from copy import copy
                from .schema import KGNode as _KGNode

                def _type_of(label: str, fallback_id: str):
                    for nn in nodes:
                        if nn.label == label:
                            return nn.node_type
                    if self.store is not None:
                        for sn in self.store._nodes.values():
                            if sn.label == label:
                                return sn.node_type
                    # fallback: 从原 id 解出 type
                    parts = fallback_id.split("_", 1)
                    if len(parts) == 2:
                        from .schema import NodeType as _NT
                        try:
                            return _NT(parts[0])
                        except Exception:  # noqa: BLE001
                            return None
                    return None

                new_e = copy(e)
                if new_src_label is not None:
                    src_t = _type_of(new_src_label, e.source_id)
                    if src_t is not None:
                        new_e.source_id = _KGNode(label=new_src_label, node_type=src_t).node_id
                if new_tgt_label is not None:
                    tgt_t = _type_of(new_tgt_label, e.target_id)
                    if tgt_t is not None:
                        new_e.target_id = _KGNode(label=new_tgt_label, node_type=tgt_t).node_id

                # 重新计算 edge_id（如果 schema 是 source_id|rel|target_id 派生）
                try:
                    new_e.edge_id = type(e)(
                        source_id=new_e.source_id,
                        target_id=new_e.target_id,
                        relation_type=e.relation_type,
                    ).edge_id
                except Exception:  # noqa: BLE001
                    pass

                if new_e.edge_id != e.edge_id:
                    edge_id_remap[e.edge_id] = new_e.edge_id
            remapped_edges.append(new_e)

        # ---- 4. 去掉 edge_id 重复 ----
        seen_edges: set[str] = set()
        unique_edges = []
        for e in remapped_edges:
            if e.edge_id in seen_edges:
                continue
            seen_edges.add(e.edge_id)
            unique_edges.append(e)

        if node_label_remap:
            logger.info(
                "batch_semantic_dedup: %d label 合并; nodes %d -> %d, edges %d -> %d",
                len(node_label_remap),
                len(nodes), len(deduped_nodes),
                len(edges), len(unique_edges),
            )
        return deduped_nodes, unique_edges, node_label_remap, edge_id_remap

    # ============================================================
    # Layer 3 — Disambiguation (placeholder)
    # ============================================================

    def is_ambiguous(self, mention: str, node_type: "NodeType | str") -> bool:
        """是否有多个候选需要 LLM 消歧。当前实现：alias map + exact id 命中
        是 1:1 的，本层无歧义。Phase 2.4 接入 embedding 召回后可能产生候选 >1。"""
        return False

    def disambiguate_with_llm(self, mention: str, candidates: list) -> str:
        """Layer 3 占位：让 LLM 在候选中选一个 node_id。
        Phase 2.4 之后接入。"""
        if not candidates:
            return ""
        return candidates[0].node_id
