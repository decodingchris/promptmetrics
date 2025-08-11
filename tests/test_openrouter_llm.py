import json
import types
import importlib
from pathlib import Path
import pytest
from pytest_mock import MockerFixture

import promptmetrics.llm_providers.openrouter as orouter
from promptmetrics.llm_providers.openrouter import OpenRouterLLM
from promptmetrics.benchmarks.hle import OfficialHLEEvaluation


# --- Core Value: Robust LLM client with caching, vision/reasoning detection, and structured parsing ---

def test_get_model_details_caches_and_reads(monkeypatch, tmp_path):
    # Redirect cache directory to tmp
    monkeypatch.setattr(orouter, "Path", orouter.Path)  # ensure correct binding
    monkeypatch.setattr(orouter.Path, "home", lambda: tmp_path)
    # First call, simulate network response
    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"data": [{"id": "m1", "architecture": {"input_modalities": ["text"]}, "supported_parameters": []}]}
    monkeypatch.setattr(orouter, "httpx", types.SimpleNamespace(get=lambda url: Resp()))
    models = orouter.get_model_details()
    assert "m1" in models
    # Second call within cache TTL should read from cache
    models2 = orouter.get_model_details()
    assert models2 == models
    cache_file = tmp_path / ".cache" / "promptmetrics" / "openrouter_models.json"
    assert cache_file.exists()
    data = json.loads(cache_file.read_text())
    assert "timestamp" in data


def test_openrouterllm_supports_vision_and_reasoning(monkeypatch):
    # Inject model map
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {
        "unit-test-model": {
            "id": "unit-test-model",
            "architecture": {"input_modalities": ["image", "text"]},
            "supported_parameters": ["reasoning"],
        }
    })
    client = OpenRouterLLM("unit-test-model")
    assert client.supports_vision is True
    assert client.supports_reasoning is True


@pytest.mark.asyncio
async def test_generate_success(monkeypatch):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    c = OpenRouterLLM("anything")

    class FakeMessage:
        def __init__(self):
            self.content = "OK"
            self.reasoning = {"chain": ["a", "b"]}

    class FakeResponse:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=FakeMessage())]

    async def fake_create(**kwargs):
        return FakeResponse()

    # patch underlying client method
    fake_chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    c.client = types.SimpleNamespace(chat=fake_chat)

    out = await c.generate([{"role": "user", "content": "Hello"}], temperature=0.1, max_tokens=32)
    assert out["content"] == "OK"
    assert out["reasoning"] == {"chain": ["a", "b"]}


@pytest.mark.asyncio
async def test_generate_handles_exception(monkeypatch, capsys):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    c = OpenRouterLLM("anything")

    async def fake_create(**kwargs):
        raise RuntimeError("boom")

    fake_chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    c.client = types.SimpleNamespace(chat=fake_chat)

    out = await c.generate([{"role": "user", "content": "Hi"}])
    assert out["content"].startswith("API_ERROR: ")
    assert out["reasoning"] is None


@pytest.mark.asyncio
async def test_generate_structured_success(monkeypatch):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    c = OpenRouterLLM("anything")

    verdict = OfficialHLEEvaluation(
        extracted_final_answer="4",
        reasoning="Matches exactly",
        correct="yes",
        confidence=92,
    )

    class FakeMessage:
        def __init__(self):
            self.refusal = None
            self.parsed = verdict
            self.content = None

    class FakeCompletion:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=FakeMessage())]

    async def fake_parse(**kwargs):
        return FakeCompletion()

    c.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(parse=fake_parse))
    )

    out = await c.generate_structured("prompt", response_model=OfficialHLEEvaluation, max_tokens=256)
    assert isinstance(out, OfficialHLEEvaluation)
    assert out.correct == "yes"
    assert out.confidence == 92


@pytest.mark.asyncio
async def test_generate_structured_refusal(monkeypatch):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    c = OpenRouterLLM("anything")

    class FakeMessage:
        def __init__(self):
            self.refusal = "Refused by policy"
            self.parsed = None
            self.content = None

    class FakeCompletion:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=FakeMessage())]

    async def fake_parse(**kwargs):
        return FakeCompletion()

    c.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(parse=fake_parse))
    )

    out = await c.generate_structured("prompt", response_model=OfficialHLEEvaluation)
    assert isinstance(out, OfficialHLEEvaluation)
    assert out.reasoning.startswith("Model refused to respond")


@pytest.mark.asyncio
async def test_generate_structured_parsing_failed(monkeypatch):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    c = OpenRouterLLM("anything")

    class FakeMessage:
        def __init__(self):
            self.refusal = None
            self.parsed = None
            self.content = "RAW"

    class FakeCompletion:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=FakeMessage())]

    async def fake_parse(**kwargs):
        return FakeCompletion()

    c.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(parse=fake_parse))
    )

    out = await c.generate_structured("prompt", response_model=OfficialHLEEvaluation)
    assert isinstance(out, OfficialHLEEvaluation)
    assert out.reasoning.startswith("Failed to parse model response.")


@pytest.mark.asyncio
async def test_generate_structured_exception(monkeypatch, capsys):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    c = OpenRouterLLM("anything")

    async def fake_parse(**kwargs):
        raise ValueError("no structured outputs")

    c.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(parse=fake_parse))
    )

    out = await c.generate_structured("prompt", response_model=OfficialHLEEvaluation)
    assert isinstance(out, OfficialHLEEvaluation)
    assert "Evaluator API call failed" in out.reasoning


def test_openrouterllm_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
        OpenRouterLLM("any")