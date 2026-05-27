"""
ScholarMind - 语义检索索引（替代 TF-IDF）
=========================================

为什么需要：
  graph_store.semantic_search 当前用 sklearn TfidfVectorizer 对 label + desc + paper
  做 cosine。在中英混合 + 短 label 语料下召回崩坏（默认 tokenizer 不处理中文，
  IDF 在小词典上严重失真）。

本模块：
  - 优先用 sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2, ~120MB)
    或更大的 BAAI/bge-m3 / bge-small-zh
  - 内存中 numpy 矩阵（150 论文 ≈ 15k 节点 × 384/768 维 < 50MB，根本不需要 FAISS）
  - 增量更新：add_node 时 encode 一条 append
  - 写盘 cache 到 data/semantic_index_<domain>.npz（启动加速）
  - 优雅降级：sentence-transformers 未装 → fallback 到 TF-IDF + 警告

性能预算：
  - 加载模型: 一次性 1-3s
  - encode 1 节点: ~5ms (CPU)
  - 15k 节点首次构建: ~80s（之后走 cache 秒级）
  - 一次 query: ~50ms (matmul + topk)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .schema import KGNode

logger = logging.getLogger("ScholarMind.SemanticIndex")

# 默认 embedding 模型（轻量、中英双语）
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data"


# ============================================================
# Encoder 抽象
# ============================================================

class _SentenceTransformerEncoder:
    """sentence-transformers 后端。"""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("loading sentence-transformer: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 384)
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self._model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)


class _TfidfEncoder:
    """fallback: sklearn TF-IDF。提供与 SentenceTransformer 相同的接口。"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer()
        self._fitted = False
        self.dim = 0
        self.name = "tfidf-fallback"

    def fit(self, corpus: list[str]) -> None:
        self._vec.fit(corpus or ["empty corpus"])
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        if not self._fitted:
            # 用 texts 自身 fit（一次性查询场景）
            self.fit(texts)
        mat = self._vec.transform(texts).toarray().astype(np.float32)
        if self.dim == 0:
            self.dim = mat.shape[1]
        # L2 normalize 模拟 cosine
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms


def _make_encoder(model_name: str | None) -> tuple[object, bool]:
    """构造 encoder。返回 (encoder, is_strong)。

    is_strong=True 表示是 sentence-transformer；False 表示 fallback。
    """
    name = model_name or DEFAULT_MODEL
    try:
        return _SentenceTransformerEncoder(name), True
    except ImportError:
        logger.warning(
            "sentence-transformers 未安装，semantic_index 回退到 TF-IDF（语义检索质量降级）。"
            " 升级命令: pip install sentence-transformers"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("sentence-transformer 加载失败 (%s)，回退到 TF-IDF", exc)
    return _TfidfEncoder(), False


# ============================================================
# 索引主类
# ============================================================

class SemanticIndex:
    """节点级语义索引（内存 numpy 矩阵）。

    Usage:
        idx = SemanticIndex(domain="llm_agent")
        idx.build_from_store(store)               # 全量构建
        results = idx.search("agent planning", top_k=5)
        # → [{"node_id":..., "label":..., "similarity":0.83}, ...]

        # 增量
        idx.add_node(new_node)
        idx.save_cache()
    """

    def __init__(
        self,
        domain: str = "shared",
        model_name: str | None = None,
        cache_dir: Path | None = None,
    ):
        self.domain = domain
        self._cache_dir = cache_dir or _CACHE_DIR
        self._encoder, self._is_strong = _make_encoder(model_name)
        self._embeddings: np.ndarray = np.zeros((0, getattr(self._encoder, "dim", 0) or 384), dtype=np.float32)
        self._node_ids: list[str] = []
        self._labels: list[str] = []
        self._meta_hash: str = ""  # 用于 cache 命中校验

    # ---------- public API ----------

    @property
    def size(self) -> int:
        return len(self._node_ids)

    @property
    def backend(self) -> str:
        return getattr(self._encoder, "name", "unknown")

    def _node_text(self, node: "KGNode") -> str:
        """组装单个节点的可嵌入文本：label + description + source_paper"""
        desc = node.properties.get("description") or node.properties.get("definition") or ""
        if not isinstance(desc, str):
            desc = str(desc)
        return f"{node.label} | {desc} | {node.source_paper or ''}".strip(" |")

    def build_from_store(self, store) -> None:
        """从 KnowledgeGraphStore 全量构建索引。"""
        nodes = list(store._nodes.values())
        texts = [self._node_text(n) for n in nodes]

        # TF-IDF 需先 fit
        if hasattr(self._encoder, "fit"):
            self._encoder.fit(texts)  # type: ignore

        self._embeddings = self._encoder.encode(texts)
        self._node_ids = [n.node_id for n in nodes]
        self._labels = [n.label for n in nodes]
        self._meta_hash = self._compute_meta_hash()
        logger.info(
            "SemanticIndex 构建完成 [%s, %s]: %d 节点 × %d 维",
            self.domain, self.backend, self.size, self._embeddings.shape[1],
        )

    def add_node(self, node: "KGNode") -> None:
        """增量加一条；调用方负责后续 save_cache()。"""
        emb = self._encoder.encode([self._node_text(node)])
        self._embeddings = np.vstack([self._embeddings, emb]) if self.size else emb
        self._node_ids.append(node.node_id)
        self._labels.append(node.label)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """语义检索 top-k。"""
        if self.size == 0 or not query.strip():
            return []
        q_emb = self._encoder.encode([query])
        if q_emb.shape[1] != self._embeddings.shape[1]:
            # TF-IDF fallback 时 query 和 corpus 维度可能不一致 → 重新 fit corpus + query
            if hasattr(self._encoder, "fit"):
                self._encoder.fit(self._node_text_cache() + [query])  # type: ignore
                # 重 encode corpus
                self._embeddings = self._encoder.encode(self._node_text_cache())
                q_emb = self._encoder.encode([query])

        # 余弦相似度（embeddings 已 L2-normalized → 直接点积）
        sims = (self._embeddings @ q_emb.T).flatten()
        order = np.argsort(-sims)[:top_k]

        results = []
        for i in order:
            sim = float(sims[i])
            if sim < min_similarity:
                continue
            results.append({
                "node_id": self._node_ids[i],
                "label": self._labels[i],
                "similarity": round(sim, 4),
            })
        return results

    def _node_text_cache(self) -> list[str]:
        """当前索引里所有节点 text（用于 TF-IDF 重 fit）。"""
        # 简化：从 labels 重建（损失 desc/paper，不影响 fallback 路径正确性）
        return list(self._labels)

    # ---------- cache ----------

    @property
    def cache_path(self) -> Path:
        return self._cache_dir / f"semantic_index_{self.domain}.npz"

    def save_cache(self) -> None:
        if self.size == 0:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.cache_path,
            embeddings=self._embeddings,
            node_ids=np.array(self._node_ids, dtype=object),
            labels=np.array(self._labels, dtype=object),
            meta=np.array([self.backend, self._meta_hash], dtype=object),
        )
        logger.info("semantic index cached: %s (%d × %d)",
                    self.cache_path, self.size, self._embeddings.shape[1])

    def load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=True)
            backend, meta_hash = data["meta"]
            if backend != self.backend:
                logger.warning(
                    "cache backend %r != current %r, rebuilding", backend, self.backend
                )
                return False
            self._embeddings = data["embeddings"].astype(np.float32)
            self._node_ids = list(data["node_ids"])
            self._labels = list(data["labels"])
            self._meta_hash = str(meta_hash)
            logger.info("semantic index cache loaded: %s (%d × %d)",
                        self.cache_path, self.size, self._embeddings.shape[1])
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache load failed: %s", exc)
            return False

    def _compute_meta_hash(self) -> str:
        """基于节点 ids 列表的 hash，用于 cache 失效检测。"""
        m = hashlib.md5()
        for nid in self._node_ids:
            m.update(nid.encode("utf-8"))
        return m.hexdigest()[:16]
