"""Exercise real SDK serialization at the HTTP transport, without network calls."""

import asyncio
import json
import socket

import httpx
import pytest
from openai import OpenAI

from scripts.migrate_kg_to_dual import Classifier
from src.core.multimodal import FigureAnalyzer
from src.evaluation.auto_rca import _llm_rca
from src.knowledge.extractor import KnowledgeExtractor
from src.report.generator import ReportGenerator


@pytest.fixture
def requests(monkeypatch):
    captured = []
    monkeypatch.setenv("MINIMAX_API_KEY", "offline-test")
    monkeypatch.delenv("MINIMAX_MODEL", raising=False)
    monkeypatch.delenv("MINIMAX_VISION_MODEL", raising=False)
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    def handle_request(request):
        payload = json.loads(request.content)
        captured.append((str(request.url), payload))
        schema_name = payload.get("response_format", {}).get("json_schema", {}).get("name")
        content = {
            "extraction_output": {"nodes": [], "edges": []},
            "batch_classify": {"results": []},
            "paper_report": {
                "meta": {"paper_title": "Offline paper"},
                "summary": {
                    "one_sentence": "Summary",
                    "problem": "Problem",
                    "contributions": [],
                    "methodology": "Method",
                    "key_results": "Results",
                },
            },
        }.get(schema_name, {"description": "Offline response"})
        return httpx.Response(200, json={
            "id": "offline-completion",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(content)},
            }],
        })

    def forbid_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in request tests")

    original_init = OpenAI.__init__
    clients = []

    def init_with_transport(self, *args, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handle_request))
        clients.append(client)
        kwargs["http_client"] = client
        kwargs["max_retries"] = 0
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    monkeypatch.setattr(OpenAI, "__init__", init_with_transport)
    yield captured
    for client in clients:
        client.close()


def invoke_text(caller, model=None):
    if caller == "extraction":
        agent = KnowledgeExtractor(model=model)
        try:
            assert json.loads(agent._call_llm("Extract entities")) == {"nodes": [], "edges": []}
        finally:
            agent.client.close()
    elif caller == "report":
        agent = ReportGenerator(model=model)
        try:
            report = agent.generate("Paper text", paper_title="Offline paper")
            assert report.summary.one_sentence == "Summary"
        finally:
            agent.client.close()
    elif caller == "classification":
        agent = Classifier(model=model)
        try:
            assert agent.classify_batch([]) == []
        finally:
            agent.client.close()
    else:
        assert _llm_rca("extraction_empty", [], {}) is not None


@pytest.mark.parametrize("caller", ["extraction", "report", "classification", "rca"])
@pytest.mark.parametrize("base_url", [None, "https://api.minimaxi.com/v1", "https://custom.invalid/v1"])
@pytest.mark.parametrize("saved_model", [None, "MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2", "custom-model"])
def test_text_default_and_saved_models_reach_http(requests, monkeypatch, caller, base_url, saved_model):
    if base_url:
        monkeypatch.setenv("MINIMAX_BASE_URL", base_url)
    if saved_model:
        monkeypatch.setenv("MINIMAX_MODEL", saved_model)

    invoke_text(caller)

    assert len(requests) == 1
    url, payload = requests[0]
    assert url == (base_url or "https://api.minimax.io/v1") + "/chat/completions"
    assert payload["model"] == (saved_model or "MiniMax-M3.1-Flash")
    assert payload["messages"][0]["role"] == "user"
    assert payload["max_tokens"] == (600 if caller == "rca" else 16000)
    # Omission uses the API's adaptive/max defaults for Flash. Never send disabled/none.
    assert all(key not in payload for key in (
        "thinking", "reasoning_effort", "reasoning_split", "service_tier",
    ))
    if caller != "rca":
        assert payload["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize("caller", ["extraction", "report", "classification"])
@pytest.mark.parametrize("model", ["MiniMax-M3", "MiniMax-M2.7", "custom-model"])
def test_explicit_model_overrides_saved_model(requests, monkeypatch, caller, model):
    monkeypatch.setenv("MINIMAX_MODEL", "saved-model")
    invoke_text(caller, model=model)
    assert len(requests) == 1
    assert requests[0][1]["model"] == model


@pytest.mark.parametrize("model", [None, "abab6.5s-chat", "custom-vision"])
@pytest.mark.parametrize("base_url", ["https://api.minimax.io/v1", "https://api.minimaxi.com/v1"])
def test_vision_model_and_image_payload_are_preserved(requests, monkeypatch, model, base_url):
    monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M3.1-Flash")
    monkeypatch.setenv("MINIMAX_BASE_URL", base_url)
    if model:
        monkeypatch.setenv("MINIMAX_VISION_MODEL", model)
    agent = FigureAnalyzer(enable_cache=False)
    try:
        asyncio.run(agent._call_vision("AA==", "Describe this figure", "image/png"))
    finally:
        agent.client.close()
    assert len(requests) == 1
    url, payload = requests[0]
    assert url == base_url + "/chat/completions"
    assert payload["model"] == (model or "MiniMax-M3")
    assert payload["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "text", "text": "Describe this figure"},
    ]
