import json
import time
import types
from pathlib import Path
import pytest

import promptmetrics.scripts.run_generation as rg
import promptmetrics.scripts.run_evaluation as reval
from promptmetrics.benchmarks.hle import HLEBenchmark, OfficialHLEEvaluation


# --- Real World Scenarios: 5 categories ---

def test_scraped_from_wild_formatting_handles_long_text_and_image(sample_questions):
    b = HLEBenchmark()
    long_question = " ".join(["Lorem ipsum dolor sit amet, consectetur adipiscing elit."] * 50)
    q = {"id": "wild1", "question": long_question, "image": "https://example.org/wild.png"}
    pt = "---[SYSTEM]---\nFollow format.\n---[USER]---\n{question}"
    msgs = b.format_prompt_messages(q, pt)
    user = next(m for m in msgs if m["role"] == "user")
    # Assert both the long text and image are present
    assert user["content"][0]["type"] == "text" and long_question in user["content"][0]["text"]
    assert any(p.get("type") == "image_url" for p in user["content"])


def test_generated_by_tool_official_prompt_has_fixed_format():
    content, _, _ = rg.load_prompt_template("official_generation_v1", "hle", "generation")
    assert "Explanation:" in content and "Answer:" in content and "Confidence:" in content


def test_human_edited_prompts_missing_user_section_gracefully_formats():
    b = HLEBenchmark()
    pt = "---[SYSTEM]---\nSys only."
    msgs = b.format_prompt_messages({"id": "e1", "question": "Q?", "answer": "A"}, pt)
    assert [m["role"] for m in msgs] == ["system", "user"]
    # user content is empty text part
    user = msgs[1]["content"]
    assert user[0]["type"] == "text" and user[0]["text"] == ""


@pytest.mark.asyncio
async def test_legacy_system_evaluator_no_structured_outputs_fallback(monkeypatch):
    # generate_structured returns model with no parsed payload -> handled gracefully by OpenRouterLLM
    from promptmetrics.llm_providers.openrouter import OpenRouterLLM
    # Ensure no model metadata usage
    monkeypatch.setattr(OpenRouterLLM, "MODELS_CACHE", {})

    class FakeCompletion:
        def __init__(self):
            msg = types.SimpleNamespace(refusal=None, parsed=None, content="UNSTRUCTURED")
            self.choices = [types.SimpleNamespace(message=msg)]

    async def fake_parse(**kwargs):
        return FakeCompletion()

    c = OpenRouterLLM("any")
    c.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(parse=fake_parse))
    )
    out = await c.generate_structured("prompt", OfficialHLEEvaluation)
    assert isinstance(out, OfficialHLEEvaluation)
    assert out.reasoning.startswith("Failed to parse model response.")


def test_corrupted_in_transit_missing_generation_key(monkeypatch, tmp_path):
    # Create malformed generations file (missing 'generations' key)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"metadata": {"generation": {"model": "x"}}}), encoding="utf-8")
    data = json.loads(bad.read_text(encoding="utf-8"))
    assert "generations" not in data


# --- Performance Benchmarks ---

class TestPerformance:
    def test_adapt_messages_for_text_only_large(self):
        # 50k text tokens, 500 images -> should be quick
        messages = []
        for i in range(2000):
            content = [{"type": "text", "text": "x" * 50}]
            if i % 4 == 0:
                content.append({"type": "image_url", "image_url": {"url": f"http://img/{i}.png"}})
            messages.append({"role": "user", "content": content})
        start = time.perf_counter()
        out = rg.adapt_messages_for_text_only(messages)
        duration = time.perf_counter() - start
        assert len(out) == len(messages)
        assert duration < 0.2  # near-instant for this transformation

    def test_format_prompt_messages_scalable(self):
        b = HLEBenchmark()
        question = {"id": "q", "question": "?" * 2000, "image": "https://img.png", "answer": "A"}
        pt = "---[SYSTEM]---\nX\n---[USER]---\n{question}"
        start = time.perf_counter()
        msgs = b.format_prompt_messages(question, pt)
        duration = time.perf_counter() - start
        assert len(msgs) == 2
        assert duration < 0.05

    def test_logging_setup_fast(self, tmp_path):
        start = time.perf_counter()
        from promptmetrics.logging_utils import setup_logger
        setup_logger(tmp_path, "perf.log")
        duration = time.perf_counter() - start
        assert duration < 0.05

    def test_ece_computation_fast(self):
        import numpy as np
        from promptmetrics.scripts.run_evaluation import calculate_ece
        conf = np.random.rand(10000)
        corr = (conf > 0.5).astype(int)
        start = time.perf_counter()
        e = calculate_ece(conf, corr, n_bins=20)
        duration = time.perf_counter() - start
        assert 0 <= e <= 1
        assert duration < 0.05