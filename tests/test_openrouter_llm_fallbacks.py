import json
import types
import pytest
from pydantic import ValidationError

from promptmetrics.llm_providers.openrouter import OpenRouterLLM
from promptmetrics.benchmarks.hle import OfficialHLEEvaluation


@pytest.mark.asyncio
async def test_generate_structured_fallback_json_schema_success(monkeypatch):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    # Force primary parse path to raise, triggering fallbacks
    c = OpenRouterLLM("any")

    async def fake_parse(**kwargs):
        raise RuntimeError("no native parse")

    # Fallback 1: return valid JSON that matches the model
    class FakeMsg:
        def __init__(self):
            self.content = json.dumps(
                {
                    "extracted_final_answer": "4",
                    "reasoning": "ok",
                    "correct": "yes",
                    "confidence": 91,
                }
            )

    class FakeCreateCompletion:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=FakeMsg())]

    async def fake_create(**kwargs):
        return FakeCreateCompletion()

    c.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(parse=fake_parse, create=fake_create)
        )
    )

    out = await c.generate_structured("p", OfficialHLEEvaluation, max_tokens=64)
    assert isinstance(out, OfficialHLEEvaluation)
    assert out.correct == "yes"
    assert out.confidence == 91


@pytest.mark.asyncio
async def test_generate_structured_fallback_json_schema_success_object_path(
    monkeypatch,
):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    # Same as above, but make model_validate_json raise to hit the object-validate branch
    c = OpenRouterLLM("any")

    async def fake_parse(**kwargs):
        raise RuntimeError("no native parse")

    # Valid JSON content
    content_dict = {
        "extracted_final_answer": "cat",
        "reasoning": "ok",
        "correct": "yes",
        "confidence": 88,
    }

    class FakeMsg:
        def __init__(self):
            self.content = json.dumps(content_dict)

    class FakeCreateCompletion:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=FakeMsg())]

    async def fake_create(**kwargs):
        return FakeCreateCompletion()

    # Patch the method on the model to raise, forcing the dict path
    def raise_val_err(_json: str):
        raise ValidationError.from_exception_data("OfficialHLEEvaluation", [])

    monkeypatch.setattr(
        OfficialHLEEvaluation, "model_validate_json", staticmethod(raise_val_err)
    )

    c.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(parse=fake_parse, create=fake_create)
        )
    )

    out = await c.generate_structured("p", OfficialHLEEvaluation)
    assert isinstance(out, OfficialHLEEvaluation)
    assert out.extracted_final_answer == "cat"
    assert out.confidence == 88


@pytest.mark.asyncio
async def test_generate_structured_fallback_json_mode_success(monkeypatch):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    # Make json_schema fallback throw so we hit json_object mode
    c = OpenRouterLLM("any")

    async def fake_parse(**kwargs):
        raise RuntimeError("no native parse")

    class FakeMsg:
        def __init__(self):
            self.content = json.dumps(
                {
                    "extracted_final_answer": "dog",
                    "reasoning": "ok",
                    "correct": "no",
                    "confidence": 73,
                }
            )

    class FakeCreateJsonMode:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=FakeMsg())]

    # We’ll switch create() behavior depending on response_format
    def dynamic_create(**kwargs):
        rf = kwargs.get("response_format")
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            raise RuntimeError("schema failed")
        return FakeCreateJsonMode()

    c.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(parse=fake_parse, create=dynamic_create)
        )
    )

    out = await c.generate_structured("p", OfficialHLEEvaluation, max_tokens=64)
    assert isinstance(out, OfficialHLEEvaluation)
    assert out.extracted_final_answer == "dog"
    assert out.correct == "no"
    assert out.confidence == 73


@pytest.mark.asyncio
async def test_generate_structured_all_fallbacks_fail(monkeypatch, capsys):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    # Ensure we end on the helpful error message path
    c = OpenRouterLLM("any")

    async def fake_parse(**kwargs):
        raise ValueError("boom-parse")

    async def fake_create(**kwargs):
        # Fail both schema and json mode paths
        raise RuntimeError("boom-create")

    c.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(parse=fake_parse, create=fake_create)
        )
    )
    out = await c.generate_structured("p", OfficialHLEEvaluation)
    assert isinstance(out, OfficialHLEEvaluation)
    assert "Evaluator API call failed" in out.reasoning


def test_build_json_schema_response_format_smoke(monkeypatch):
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})
    c = OpenRouterLLM("any")
    rf = c._build_json_schema_response_format(OfficialHLEEvaluation)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "OfficialHLEEvaluation"
    assert rf["json_schema"]["schema"]["type"] == "object"
    assert rf["json_schema"]["strict"] is True
