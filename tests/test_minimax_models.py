from src.core.minimax_models import (
    DEFAULT_MINIMAX_ANTHROPIC_BASE_URL,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_OPENAI_BASE_URL,
    LEGACY_MINIMAX_MODELS,
    MINIMAX_MODEL_IDS,
    MINIMAX_MODELS,
    MINIMAX_PROVIDER_NAME,
    MINIMAX_REGIONAL_ENDPOINTS,
    get_minimax_model_metadata,
)
from src.evaluation.cost import price_for


def test_minimax_model_metadata_matches_supported_models():
    assert MINIMAX_PROVIDER_NAME == "MiniMax"
    assert DEFAULT_MINIMAX_MODEL == "MiniMax-M3.1-Flash"
    assert MINIMAX_MODEL_IDS == ("MiniMax-M3.1-Flash", "MiniMax-M3")

    flash = MINIMAX_MODELS[DEFAULT_MINIMAX_MODEL]
    assert flash.context_window == 1_000_000
    assert flash.max_output_tokens == 524_288
    assert flash.pricing_usd_per_million_tokens == {
        "input": 0.15,
        "output": 0.60,
        "cache_read": 0.015,
        "cache_write": None,
    }
    assert flash.thinking == ("adaptive",)
    assert flash.reasoning_efforts == ("low", "medium", "high", "xhigh", "max")
    assert flash.to_dict()["max_output_tokens"] == 524_288

    m3 = MINIMAX_MODELS["MiniMax-M3"]
    assert m3.context_window == 1_000_000
    assert m3.pricing_usd_per_million_tokens == {
        "input": 0.3,
        "output": 1.2,
        "cache_read": 0.06,
        "cache_write": None,
    }
    assert m3.input_modalities == ("text", "image", "video")
    assert m3.thinking == ("adaptive", "disabled")
    assert m3.to_dict()["input_modalities"] == ["text", "image", "video"]


def test_legacy_metadata_remains_available_without_becoming_a_choice():
    m27 = get_minimax_model_metadata("MiniMax-M2.7")
    assert m27 is LEGACY_MINIMAX_MODELS["MiniMax-M2.7"]
    assert m27.context_window == 204_800
    assert m27.pricing_usd_per_million_tokens == {
        "input": 0.3,
        "output": 1.2,
        "cache_read": 0.06,
        "cache_write": 0.375,
    }
    assert m27.input_modalities == ("text",)
    assert m27.thinking == ("always_on",)
    assert "MiniMax-M2.7" not in MINIMAX_MODEL_IDS
    assert get_minimax_model_metadata("custom-model") is None


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

    assert extractor.model_metadata is MINIMAX_MODELS[DEFAULT_MINIMAX_MODEL]
    assert report.model_metadata is MINIMAX_MODELS[DEFAULT_MINIMAX_MODEL]
    assert vision.model_metadata is MINIMAX_MODELS["MiniMax-M3"]
    assert "image" in vision.model_metadata.input_modalities
    assert extractor_m27.model_metadata is LEGACY_MINIMAX_MODELS["MiniMax-M2.7"]
    assert report_m27.model_metadata is LEGACY_MINIMAX_MODELS["MiniMax-M2.7"]


def test_cost_table_uses_model_metadata_pricing():
    assert price_for("MiniMax-M3.1-Flash") == {"input": 0.00015, "output": 0.0006}
    assert price_for("MiniMax-M3") == {"input": 0.0003, "output": 0.0012}
    assert price_for("MiniMax-M2.7") == {"input": 0.0003, "output": 0.0012}
    assert price_for("MiniMax-M2") == {"input": 0.0003, "output": 0.0014}
    assert price_for("MiniMax-Text-01") == {"input": 0.0002, "output": 0.0011}
    assert price_for("MiniMax-VL-01") == {"input": 0.0002, "output": 0.0011}
    assert price_for("custom-model") == price_for("_default")
