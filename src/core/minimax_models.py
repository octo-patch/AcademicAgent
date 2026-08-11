"""Canonical MiniMax model metadata and regional API endpoints."""

from __future__ import annotations

from dataclasses import dataclass

MINIMAX_PROVIDER_NAME = "MiniMax"
DEFAULT_MINIMAX_MODEL = "MiniMax-M3"
DEFAULT_MINIMAX_VISION_MODEL = "MiniMax-M3"


@dataclass(frozen=True)
class MiniMaxModelMetadata:
    model_id: str
    context_window: int
    pricing_usd_per_million_tokens: dict[str, float | None]
    input_modalities: tuple[str, ...]
    thinking: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "context_window": self.context_window,
            "pricing_usd_per_million_tokens": dict(
                self.pricing_usd_per_million_tokens
            ),
            "input_modalities": list(self.input_modalities),
            "thinking": list(self.thinking),
        }


MINIMAX_MODELS: dict[str, MiniMaxModelMetadata] = {
    "MiniMax-M3": MiniMaxModelMetadata(
        model_id="MiniMax-M3",
        context_window=1_000_000,
        pricing_usd_per_million_tokens={
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
        input_modalities=("text", "image", "video"),
        thinking=("adaptive", "disabled"),
    ),
    "MiniMax-M2.7": MiniMaxModelMetadata(
        model_id="MiniMax-M2.7",
        context_window=204_800,
        pricing_usd_per_million_tokens={
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": 0.375,
        },
        input_modalities=("text",),
        thinking=("always_on",),
    ),
}
MINIMAX_MODEL_IDS = tuple(MINIMAX_MODELS)

MINIMAX_REGIONAL_ENDPOINTS = [
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
MINIMAX_REGIONAL_ENDPOINTS_BY_REGION = {
    endpoint["region"]: endpoint for endpoint in MINIMAX_REGIONAL_ENDPOINTS
}

DEFAULT_MINIMAX_OPENAI_BASE_URL = MINIMAX_REGIONAL_ENDPOINTS_BY_REGION[
    "global_en"
]["openai_base_url"]
DEFAULT_MINIMAX_ANTHROPIC_BASE_URL = MINIMAX_REGIONAL_ENDPOINTS_BY_REGION[
    "global_en"
]["anthropic_base_url"]


def get_minimax_model_metadata(model_id: str) -> MiniMaxModelMetadata | None:
    """Return metadata for a supported model without rejecting custom IDs."""
    return MINIMAX_MODELS.get(model_id)
