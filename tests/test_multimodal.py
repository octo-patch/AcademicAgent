"""
多模态分析模块 + 论文结构解析器 测试
=====================================

测试策略:
- FigureType / FigureAnalysisResult 数据类: 离线
- PaperStructureParser: 离线（纯文本操作）
- _extract_json: 离线（核心鲁棒性测试）
- FigureAnalyzer.analyze: 需要 API Key → integration

Usage:
  pytest tests/test_multimodal.py -v
  pytest tests/test_multimodal.py -v -m integration  (with API)
"""

import pytest

# Guard: multimodal.py depends on openai SDK (OpenAI 兼容的 MiniMax 调用)
openai = pytest.importorskip("openai", reason="openai SDK not installed")

from src.core.multimodal import (
    FigureAnalysisResult,
    FigureAnalyzer,
    FigureType,
    PaperStructureParser,
)


class TestFigureType:
    """图表类型枚举测试"""

    def test_all_types_exist(self):
        assert FigureType.BLOCK_DIAGRAM == "block_diagram"
        assert FigureType.PERFORMANCE_CURVE == "performance_curve"
        assert FigureType.TABLE == "table"
        assert FigureType.EQUATION == "equation"
        assert FigureType.UNKNOWN == "unknown"

    def test_from_string(self):
        assert FigureType("block_diagram") == FigureType.BLOCK_DIAGRAM
        assert FigureType("table") == FigureType.TABLE


class TestFigureAnalysisResult:
    """分析结果数据类测试"""

    def test_to_markdown(self):
        result = FigureAnalysisResult(
            figure_type=FigureType.BLOCK_DIAGRAM,
            description="ReAct agent workflow diagram",
            key_findings=["Planner calls tools", "Reflection updates memory"],
            entities=[
                {"name": "Planner", "type": "module"},
                {"name": "Memory Store", "type": "module"},
            ],
        )
        md = result.to_markdown()
        assert "block_diagram" in md
        assert "ReAct agent" in md
        assert "Planner calls tools" in md
        assert "Memory Store" in md

    def test_empty_result(self):
        result = FigureAnalysisResult(
            figure_type=FigureType.UNKNOWN,
            description="",
            key_findings=[],
            entities=[],
        )
        md = result.to_markdown()
        assert "（无）" in md


class TestExtractJSON:
    """JSON 提取鲁棒性测试 - 核心防 Silent Regression"""

    def setup_method(self):
        self.analyzer = FigureAnalyzer()

    def test_pure_json(self):
        text = '{"figure_type": "table", "description": "Results"}'
        result = self.analyzer._extract_json(text)
        assert result["figure_type"] == "table"

    def test_json_in_code_block(self):
        text = 'Here is the result:\n```json\n{"type": "ok"}\n```'
        result = self.analyzer._extract_json(text)
        assert result["type"] == "ok"

    def test_json_with_surrounding_text(self):
        text = 'I found: {"answer": 42} and more text'
        result = self.analyzer._extract_json(text)
        assert result["answer"] == 42

    def test_invalid_json_returns_empty(self):
        text = "This is not JSON at all"
        result = self.analyzer._extract_json(text)
        assert result == {}

    def test_nested_json(self):
        text = '{"entities": [{"name": "A", "type": "B"}], "count": 1}'
        result = self.analyzer._extract_json(text)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "A"


class TestPaperStructureParser:
    """论文结构解析测试"""

    def setup_method(self):
        self.parser = PaperStructureParser()

    def test_basic_sections(self):
        text = """
Abstract
This paper presents a novel approach...

1. Introduction
In recent years, LLM Agents with tool use and memory have gained attention...

3. Proposed Method
We propose a deep learning based...

4. Experiments
We evaluate on two datasets...

5. Conclusion
In this paper, we proposed...

References
[1] Zhang et al...
"""
        sections = self.parser.parse_sections(text)
        assert "abstract" in sections
        assert "introduction" in sections
        assert "method" in sections
        assert "experiments" in sections
        assert "conclusion" in sections

    def test_abstract_content(self):
        text = "Abstract\nThis is the abstract content.\n\n1. Introduction\nIntro text."
        sections = self.parser.parse_sections(text)
        assert "abstract content" in sections["abstract"]

    def test_no_sections(self):
        text = "Just some random text without section headers"
        sections = self.parser.parse_sections(text)
        assert "preamble" in sections

    def test_section_summary(self):
        sections = {"abstract": "word " * 50, "method": "word " * 100}
        summary = self.parser.get_section_summary(sections)
        assert "abstract" in summary
        assert "method" in summary

    def test_roman_numeral_sections(self):
        """Agent 领域论文常用数字编号"""
        text = """
Abstract
Test abstract.

I. Introduction
Intro content.

II. System Model
Model content.

III. Proposed Algorithm
Algorithm content.

IV. Simulation Results
Results content.

V. Conclusion
Conclusion content.
"""
        sections = self.parser.parse_sections(text)
        # Should identify at least abstract and conclusion
        assert "abstract" in sections
        assert "conclusion" in sections



# ============================================================
# Phase 3 新功能测试
# ============================================================


class TestPhase3ChineseAndMarkdownSections:
    """Phase 3.4: 章节正则扩展（中文 + markdown）"""

    def setup_method(self):
        from src.core.multimodal import PaperStructureParser
        self.parser = PaperStructureParser()

    def test_chinese_sections(self):
        text = "\n".join([
            "摘要", "ABS",
            "引言", "INT",
            "相关工作", "RW",
            "方法", "MET",
            "实验", "EXP",
            "结论", "CONC",
            "参考文献", "REF",
        ])
        sections = self.parser.parse_sections(text)
        for k in ("abstract", "introduction", "related_work",
                  "method", "experiments", "conclusion", "references"):
            assert k in sections, f"missing {k}"

    def test_chinese_with_numeric_prefix(self):
        text = "\n".join([
            "1. 引言", "A",
            "2. 相关工作", "B",
            "3. 方法", "C",
            "4. 实验", "D",
            "5. 结论", "E",
        ])
        sections = self.parser.parse_sections(text)
        for k in ("introduction", "related_work", "method", "experiments", "conclusion"):
            assert k in sections, f"missing {k}"

    def test_neurips_markdown(self):
        text = "\n".join([
            "## Abstract", "ABS",
            "## 1 Introduction", "INT",
            "## 3 Methodology", "MET",
            "## 4 Experiments", "EXP",
            "## 5 Conclusion", "CONC",
        ])
        sections = self.parser.parse_sections(text)
        for k in ("abstract", "introduction", "method", "experiments", "conclusion"):
            assert k in sections


class TestPhase3VisionConfig:
    """Phase 3.2: vision 模型 env 隔离 + cache + prompt safety"""

    def test_vision_uses_separate_env(self, monkeypatch):
        from src.core.multimodal import FigureAnalyzer
        monkeypatch.setenv("MINIMAX_VISION_MODEL", "MiniMax-M3")
        monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M2.7")
        fa = FigureAnalyzer()
        assert fa.model == "MiniMax-M3",               "vision should use MINIMAX_VISION_MODEL not MINIMAX_MODEL"

    def test_vision_falls_back_to_default(self, monkeypatch):
        from src.core.multimodal import FigureAnalyzer
        monkeypatch.delenv("MINIMAX_VISION_MODEL", raising=False)
        monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M2.7")
        fa = FigureAnalyzer()
        # The vision default must not inherit a text-only MINIMAX_MODEL.
        assert fa.model == "MiniMax-M3"

    def test_cache_toggle(self):
        from src.core.multimodal import FigureAnalyzer
        fa_on = FigureAnalyzer(enable_cache=True)
        fa_off = FigureAnalyzer(enable_cache=False)
        assert fa_on._cache is not None
        assert fa_off._cache is None


class TestPhase3CaptionExtraction:
    """Phase 3.3: caption 检测 + 正文引用"""

    def test_caption_regex_requires_punctuation(self):
        """Figure X without :/. is NOT a caption (avoids body false positives)."""
        from src.core.pdf_parser import PDFParser
        # 用 _CAPTION_PATTERNS 直接测，不依赖 PDF 文件
        ok_cases = [
            "Figure 1: Comparison of methods",
            "Fig. 2. The architecture",
            "Table 3: Results",
            "图 1: 系统框图",
        ]
        bad_cases = [
            "Figure 5 in Section 4",  # no punctuation
            "Table 1 shows accuracy",  # no punctuation
            "see Figure 2 for details",  # body reference
        ]
        for line in ok_cases:
            matched = any(p.match(line) for p in PDFParser._CAPTION_PATTERNS)
            assert matched, f"should match: {line!r}"
        for line in bad_cases:
            matched = any(p.match(line) for p in PDFParser._CAPTION_PATTERNS)
            assert not matched, f"should NOT match: {line!r}"
