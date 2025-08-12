import json
import types
from pathlib import Path

import pytest

import promptmetrics.scripts.run_generation as rg
from promptmetrics.benchmarks.aime import AIMEBenchmark


# --- Core Value: Reproducible AIME generation artifacts with official-fallback behavior ---


@pytest.mark.asyncio
async def test_pm_generate_aime_fallback_creates_artifact(tmp_path, monkeypatch):
    # Use AIME benchmark
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: AIMEBenchmark())

    # Provide text-only questions
    questions = [
        {"id": "q1", "question": "What is 2+2?", "answer": "4"},
        {"id": "q2", "question": "Compute 5-3.", "answer": "2"},
    ]
    monkeypatch.setattr(
        AIMEBenchmark, "load_data", lambda self, max_samples=None: questions
    )

    # LLM mock: return AIME-style answer string following public non_official_generation_v1 format
    async def fake_generate(messages, temperature, max_tokens):
        return {
            "content": "Explanation: ...\nAnswer: 4\nConfidence: 85",
            "reasoning": None,
        }

    llm_mock = types.SimpleNamespace(
        supports_vision=False, supports_reasoning=False, generate=fake_generate
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm_mock)

    def fake_args():
        class A:
            pass

        a = A()
        a.model = "my/model:v1"
        a.benchmark = "aime_2025"
        a.generation_prompt_source = (
            "official"  # will fallback to non_official_generation_v1
        )
        a.output_dir = Path(tmp_path)
        a.temperature = 0.0
        a.max_tokens = 128
        a.max_samples = 2
        a.num_workers = 1
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        rg.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    monkeypatch.setattr(
        rg,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *coros: rg.asyncio.gather(*coros)),
    )

    await rg.main_async()

    results_dir = (
        Path(tmp_path)
        / "results"
        / "aime_2025"
        / "my_model-v1"
        / "public-non_official_generation_v1"
        / "generations"
    )
    files = list(results_dir.glob("*_generations.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert "metadata" in data and "generations" in data
    assert set(data["generations"].keys()) == {"q1", "q2"}
    # Ensure the prompt source recorded is the resolved fallback name
    assert (
        data["metadata"]["generation"]["prompt_source"] == "non_official_generation_v1"
    )
    assert data["metadata"]["generation"]["prompt_source_type"] == "public"


@pytest.mark.asyncio
async def test_pm_generate_aime_external_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: AIMEBenchmark())
    # minimal questions
    monkeypatch.setattr(
        AIMEBenchmark,
        "load_data",
        lambda self, max_samples=None: [
            {"id": "q", "question": "What is 1+3?", "answer": "4"}
        ],
    )

    ext = tmp_path / "gen.txt"
    ext.write_text(">>> {question}", encoding="utf-8")

    async def fake_generate(messages, temperature, max_tokens):
        return {
            "content": "Explanation: ...\nAnswer: 4\nConfidence: 99",
            "reasoning": None,
        }

    llm = types.SimpleNamespace(
        supports_vision=False, supports_reasoning=False, generate=fake_generate
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm)

    def args():
        class A:
            pass

        a = A()
        a.model = "m"
        a.benchmark = "aime_2025"
        a.generation_prompt_source = str(ext)
        a.output_dir = Path(tmp_path)
        a.temperature = 0.0
        a.max_tokens = 64
        a.max_samples = 1
        a.num_workers = 1
        a.allow_full_run = True
        return a

    monkeypatch.setattr(rg.argparse.ArgumentParser, "parse_args", lambda self: args())
    monkeypatch.setattr(
        rg,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *c: rg.asyncio.gather(*c)),
    )

    await rg.main_async()

    results_dir = (
        Path(tmp_path) / "results" / "aime_2025" / "m" / ext.stem / "generations"
    )
    files = list(results_dir.glob("*_generations.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["metadata"]["generation"]["prompt_source_type"] == "external"
