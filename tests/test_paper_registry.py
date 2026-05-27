"""tests/test_paper_registry.py — Phase 4 新增

验证：
  - 三级去重（arxiv / doi / file_hash）
  - **改进的标题模糊匹配**：避免把 "Attention Is All You Need" 与
    "Attention Is All You Need For Sign Language" 误判为同一篇
  - first_seen_date 字段往返序列化
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.core.paper_registry import PaperRegistry, PaperRecord


@pytest.fixture
def tmp_registry(tmp_path):
    """每个测试一个隔离 registry。"""
    return PaperRegistry(
        registry_path=tmp_path / "r.json",
        papers_dir=tmp_path / "papers",
    )


class TestFirstSeenDate:
    def test_round_trip(self, tmp_registry):
        rec = PaperRecord(title="X", local_path="p.pdf", first_seen_date="2026-01-20")
        tmp_registry.register_paper(rec)
        rec2 = PaperRecord.from_dict(rec.to_dict())
        assert rec2.first_seen_date == "2026-01-20"


class TestFuzzyMatchPhase4:
    """Phase 4 修复：token Jaccard 替代 80% 子串"""

    def test_extended_title_does_not_falsely_match(self, tmp_registry):
        """关键 case：原 80% 子串实现会误判，新 Jaccard ≥ 0.85 应不命中。"""
        tmp_registry.register_paper(
            PaperRecord(title="Attention Is All You Need", local_path="a.pdf")
        )
        hit = tmp_registry.check_duplicate(
            title="Attention Is All You Need For Sign Language Recognition"
        )
        assert hit is None, "extended title 不应被误判为同一篇"

    def test_exact_title_matches(self, tmp_registry):
        tmp_registry.register_paper(
            PaperRecord(title="ReAct: Synergizing Reasoning and Acting", local_path="r.pdf")
        )
        hit = tmp_registry.check_duplicate(
            title="ReAct: Synergizing Reasoning and Acting"
        )
        assert hit is not None

    def test_case_and_punctuation_invariant(self, tmp_registry):
        tmp_registry.register_paper(
            PaperRecord(title="Attention Is All You Need", local_path="a.pdf")
        )
        for q in [
            "Attention Is All You Need.",
            "ATTENTION IS ALL YOU NEED",
            "attention is all you need",
        ]:
            hit = tmp_registry.check_duplicate(title=q)
            assert hit is not None, f"case/punct invariant failed for {q!r}"

    def test_too_few_tokens_falls_back_to_exact(self, tmp_registry):
        """少于 3 token 时只看精确匹配，避免误判。"""
        tmp_registry.register_paper(
            PaperRecord(title="Deep Learning", local_path="dl.pdf")
        )
        assert tmp_registry.check_duplicate(title="Deep Learning") is not None
        assert tmp_registry.check_duplicate(title="Deep Reinforcement Learning") is None


class TestThreeLevelDedup:
    def test_arxiv_id_match(self, tmp_registry):
        tmp_registry.register_paper(
            PaperRecord(title="X", local_path="x.pdf", arxiv_id="2210.03629")
        )
        # versioned arxiv id 应该 strip 后命中
        assert tmp_registry.check_duplicate(arxiv_id="2210.03629v3") is not None

    def test_doi_match(self, tmp_registry):
        tmp_registry.register_paper(
            PaperRecord(title="X", local_path="x.pdf", doi="10.1234/abc")
        )
        # 大小写不敏感
        assert tmp_registry.check_duplicate(doi="10.1234/ABC") is not None

    def test_file_hash_match(self, tmp_registry):
        tmp_registry.register_paper(
            PaperRecord(title="X", local_path="x.pdf", file_hash="abc123")
        )
        assert tmp_registry.check_duplicate(file_hash="abc123") is not None
