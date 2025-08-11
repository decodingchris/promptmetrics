import json
import types
from pathlib import Path
import pytest

import promptmetrics.scripts.run_generation as rg
from promptmetrics.benchmarks.hle import HLEBenchmark


# --- Core Value: Reproducible generation artifacts and multimodal handling ---


def test_adapt_messages_for_text_only_variants():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Q"},
                {"type": "image_url", "image_url": {"url": "http://x"}},
            ],
        },
    ]
    out = rg.adapt_messages_for_text_only(messages)
    assert isinstance(out[0]["content"], str)
    assert out[0]["content"] == "sys"
    assert isinstance(out[1]["content"], str)
    assert "[NOTE: An image was part of this question" in out[1]["content"]


@pytest.mark.asyncio
async def test_pm_generate_creates_artifact(
    tmp_output_dir, monkeypatch, sample_questions
):
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: HLEBenchmark())

    monkeypatch.setattr(
        HLEBenchmark, "load_data", lambda self, max_samples=None: sample_questions
    )

    async def fake_generate(messages, temperature, max_tokens):
        return {
            "content": "Explanation: ...\nAnswer: a cat\nConfidence: 95",
            "reasoning": None,
        }

    llm_mock = types.SimpleNamespace(
        supports_vision=True,
        supports_reasoning=False,
        generate=fake_generate,
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm_mock)

    def fake_parse_args():
        class A:
            pass

        a = A()
        a.model = "openai/gpt-4o"
        a.benchmark = "hle"
        a.generation_prompt_source = "official"
        a.output_dir = Path(tmp_output_dir)
        a.temperature = 0.0
        a.max_tokens = 128
        a.max_samples = 2
        a.num_workers = 1
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        rg.argparse.ArgumentParser, "parse_args", lambda self: fake_parse_args()
    )
    monkeypatch.setattr(
        rg,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *coros: rg.asyncio.gather(*coros)),
    )

    await rg.main_async()

    results_dir = (
        tmp_output_dir
        / "results"
        / "hle"
        / "openai_gpt-4o"
        / "public-official_generation_v1"
        / "generations"
    )
    assert results_dir.exists()
    files = list(results_dir.glob("*_generations.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert "metadata" in data and "generations" in data
    assert set(data["generations"].keys()) == {"q1", "q2"}


@pytest.mark.asyncio
async def test_pm_generate_text_only_fallback_when_model_no_vision(
    tmp_output_dir, monkeypatch, sample_questions
):
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: HLEBenchmark())
    monkeypatch.setattr(
        HLEBenchmark, "load_data", lambda self, max_samples=None: sample_questions
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_vision = False
            self.supports_reasoning = False

        async def generate(self, messages, temperature, max_tokens):
            return {"content": "Answer: test\nConfidence: 80", "reasoning": None}

    monkeypatch.setattr(rg, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")
    monkeypatch.setattr(
        rg,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *coros: rg.asyncio.gather(*coros)),
    )

    def fake_parse_args():
        class A:
            pass

        a = A()
        a.model = "any"
        a.benchmark = "hle"
        a.generation_prompt_source = "official"
        a.output_dir = Path(tmp_output_dir)
        a.temperature = 0.0
        a.max_tokens = 64
        a.max_samples = 1
        a.num_workers = 1
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        rg.argparse.ArgumentParser, "parse_args", lambda self: fake_parse_args()
    )
    await rg.main_async()

    results_dir = (
        tmp_output_dir
        / "results"
        / "hle"
        / "any"
        / "public-official_generation_v1"
        / "generations"
    )
    files = list(results_dir.glob("*_generations.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert "modality_handling" in data["metadata"]["generation"]
    assert (
        data["metadata"]["generation"]["modality_handling"]["status"]
        == "text_only_fallback"
    )


@pytest.mark.asyncio
async def test_pm_generate_full_run_decline(tmp_output_dir, monkeypatch):
    b = HLEBenchmark()
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: b)

    fake_infos = {
        b.dataset_config: types.SimpleNamespace(
            splits={b.dataset_split: types.SimpleNamespace(num_examples=123)}
        )
    }
    monkeypatch.setattr(
        "promptmetrics.benchmarks.hle.get_dataset_infos",
        lambda *a, **k: fake_infos,
    )

    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")

    def fake_parse_args():
        class A:
            pass

        a = A()
        a.model = "openai/gpt-4o"
        a.benchmark = "hle"
        a.generation_prompt_source = "official"
        a.output_dir = Path(tmp_output_dir)
        a.temperature = 0.0
        a.max_tokens = 128
        a.max_samples = None
        a.num_workers = 1
        a.allow_full_run = False
        return a

    monkeypatch.setattr(
        rg.argparse.ArgumentParser, "parse_args", lambda self: fake_parse_args()
    )

    await rg.main_async()
    assert not (tmp_output_dir / "results").exists()


@pytest.mark.asyncio
async def test_pm_generate_full_run_accepts(monkeypatch, tmp_output_dir):
    b = HLEBenchmark()
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: b)

    fake_infos = {
        b.dataset_config: types.SimpleNamespace(
            splits={b.dataset_split: types.SimpleNamespace(num_examples=123)}
        )
    }
    monkeypatch.setattr(
        "promptmetrics.benchmarks.hle.get_dataset_infos",
        lambda *a, **k: fake_infos,
    )

    monkeypatch.setattr(
        HLEBenchmark,
        "load_data",
        lambda self, max_samples=None: [{"id": "q", "question": "?", "answer": "a"}],
    )
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")

    async def fake_generate(messages, temperature, max_tokens):
        return {"content": "Answer: a\nConfidence: 100", "reasoning": None}

    llm = types.SimpleNamespace(
        supports_vision=True, supports_reasoning=True, generate=fake_generate
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm)

    def args():
        class A:
            pass

        a = A()
        a.model = "m"
        a.benchmark = "hle"
        a.generation_prompt_source = "official"
        a.output_dir = tmp_output_dir
        a.temperature = 0.0
        a.max_tokens = 16
        a.max_samples = None
        a.num_workers = 1
        a.allow_full_run = False
        return a

    monkeypatch.setattr(rg.argparse.ArgumentParser, "parse_args", lambda self: args())
    monkeypatch.setattr(
        rg,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *coros: rg.asyncio.gather(*coros)),
    )
    await rg.main_async()

    results_dir = (
        tmp_output_dir
        / "results"
        / "hle"
        / "m"
        / "public-official_generation_v1"
        / "generations"
    )
    files = list(results_dir.glob("*_generations.json"))
    assert len(files) == 1
