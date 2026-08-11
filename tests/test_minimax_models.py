from src.core.minimax_models import (
    DEFAULT_MINIMAX_ANTHROPIC_BASE_URL,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_OPENAI_BASE_URL,
    MINIMAX_MODEL_IDS,
    MINIMAX_MODELS,
    MINIMAX_PROVIDER_NAME,
    MINIMAX_REGIONAL_ENDPOINTS,
)
from src.evaluation.cost import price_for


def test_minimax_model_metadata_matches_supported_models():
    assert MINIMAX_PROVIDER_NAME == "MiniMax"
    assert DEFAULT_MINIMAX_MODEL == "MiniMax-M3"
    assert MINIMAX_MODEL_IDS == ("MiniMax-M3", "MiniMax-M2.7")

    m3 = MINIMAX_MODELS["MiniMax-M3"]
    assert m3.context_window == 1_000_000
    assert m3.pricing_usd_per_million_tokens == {
        "input": 0.6,
        "output": 2.4,
        "cache_read": 0.12,
        "cache_write": None,
    }
    assert m3.input_modalities == ("text", "image", "video")
    assert m3.thinking == ("adaptive", "disabled")
    assert m3.to_dict()["input_modalities"] == ["text", "image", "video"]

    m27 = MINIMAX_MODELS["MiniMax-M2.7"]
    assert m27.context_window == 204_800
    assert m27.pricing_usd_per_million_tokens == {
        "input": 0.3,
        "output": 1.2,
        "cache_read": 0.06,
        "cache_write": 0.375,
    }
    assert m27.input_modalities == ("text",)
    assert m27.thinking == ("always_on",)


def test_minimax_regional_endpoints_are_exposed():
    assert DEFAULT_MINIMAX_OPENAI_BASE_URL == "https://api.minimax.io/v1"
    assert DEFAULT_MINIMAX_ANTHROPIC_BASE_URL == "https://api.minimax.io/anthropic"
    assert MINIMAX_REGIONAL_ENDPOINTS == [
        {
            "region": "global_en",
            "openai_base_url": "https://api.minimax.io/v1",
            "anthropic_base_url": "https://api.minimax.io/anthropic",
            "docs_root": "https://platform.minimax.io/docs",
        },
        {
            "region": "cn_zh",
            "openai_base_url": "https://api.minimaxi.com/v1",
            "anthropic_base_url": "https://api.minimaxi.com/anthropic",
            "docs_root": "https://platform.minimaxi.com/docs",
        },
    ]


def test_model_metadata_reaches_active_integrations(monkeypatch):
    from src.core.multimodal import FigureAnalyzer
    from src.knowledge.extractor import KnowledgeExtractor
    from src.report.generator import ReportGenerator

    monkeypatch.delenv("MINIMAX_MODEL", raising=False)
    monkeypatch.delenv("MINIMAX_VISION_MODEL", raising=False)

    extractor = KnowledgeExtractor()
    report = ReportGenerator()
    vision = FigureAnalyzer()
    extractor_m27 = KnowledgeExtractor(model="MiniMax-M2.7")
    report_m27 = ReportGenerator(model="MiniMax-M2.7")

    assert extractor.model_metadata is MINIMAX_MODELS["MiniMax-M3"]
    assert report.model_metadata is MINIMAX_MODELS["MiniMax-M3"]
    assert vision.model_metadata is MINIMAX_MODELS["MiniMax-M3"]
    assert "image" in vision.model_metadata.input_modalities
    assert extractor_m27.model_metadata is MINIMAX_MODELS["MiniMax-M2.7"]
    assert report_m27.model_metadata is MINIMAX_MODELS["MiniMax-M2.7"]


def test_cost_table_uses_model_metadata_pricing():
    assert price_for("MiniMax-M3") == {"input": 0.0006, "output": 0.0024}
    assert price_for("MiniMax-M2.7") == {"input": 0.0003, "output": 0.0012}
