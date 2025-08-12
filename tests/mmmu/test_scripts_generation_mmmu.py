# tests/mmmu/test_scripts_generation_mmmu.py

import json
import types
from pathlib import Path

import pytest

import promptmetrics.scripts.run_generation as rg
from promptmetrics.benchmarks.mmmu import MMMUAllBenchmark


@pytest.mark.asyncio
async def test_pm_generate_mmmu_fallback_creates_artifact(tmp_path, monkeypatch):
    # Avoid network on init
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Art", "Physics"],
    )
    # Use MMMU benchmark
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: MMMUAllBenchmark())

    # Provide text-only questions (no images) to keep simple
    questions = [
        {
            "id": "q1",
            "question": "Which is correct?",
            "parsed_choices": {"A": "alpha", "B": "beta"},
            "answer": "B",
        },
        {
            "id": "q2",
            "question": "Pick one.",
            "parsed_choices": {"A": "x", "B": "y", "C": "z"},
            "answer": "A",
        },
    ]
    monkeypatch.setattr(
        MMMUAllBenchmark, "load_data", lambda self, max_samples=None: questions
    )

    async def fake_generate(messages, temperature, max_tokens):
        return {
            "content": "Explanation: ...\nAnswer: B\nConfidence: 88",
            "reasoning": None,
        }

    llm_mock = types.SimpleNamespace(
        supports_vision=False, supports_reasoning=False, generate=fake_generate
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm_mock)

    # Confirm text-only fallback prompt
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")

    def fake_args():
        class A:
            pass

        a = A()
        a.model = "my/model:v1"
        a.benchmark = "mmmu"
        a.generation_prompt_source = (
            "official"  # fallback -> non_official_generation_v1
        )
        a.output_dir = Path(tmp_path)
        a.temperature = 0.0
        a.max_tokens = 256
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
        types.SimpleNamespace(gather=lambda *c: rg.asyncio.gather(*c)),
    )

    await rg.main_async()

    results_dir = (
        Path(tmp_path)
        / "results"
        / "mmmu"
        / "my_model-v1"
        / "public-non_official_generation_v1"
        / "generations"
    )
    files = list(results_dir.glob("*_generations.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert "metadata" in data and "generations" in data
    assert set(data["generations"].keys()) == {"q1", "q2"}
    assert (
        data["metadata"]["generation"]["prompt_source"] == "non_official_generation_v1"
    )
    assert data["metadata"]["generation"]["prompt_source_type"] == "public"


@pytest.mark.asyncio
async def test_pm_generate_mmmu_text_only_fallback_when_model_no_vision(
    tmp_path, monkeypatch
):
    # Avoid network on init
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Art"],
    )
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: MMMUAllBenchmark())
    monkeypatch.setattr(
        MMMUAllBenchmark,
        "load_data",
        lambda self, max_samples=None: [
            {
                "id": "q1",
                "question": "Look?",
                "parsed_choices": {"A": "a", "B": "b"},
                "answer": "A",
                # Not providing images here; just exercising modality path
            }
        ],
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_vision = False
            self.supports_reasoning = False

        async def generate(self, messages, temperature, max_tokens):
            return {"content": "Answer: A\nConfidence: 77", "reasoning": None}

    monkeypatch.setattr(rg, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")
    monkeypatch.setattr(
        rg,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *c: rg.asyncio.gather(*c)),
    )

    def fake_args():
        class A:
            pass

        a = A()
        a.model = "no-vision"
        a.benchmark = "mmmu"
        a.generation_prompt_source = "official"
        a.output_dir = Path(tmp_path)
        a.temperature = 0.0
        a.max_tokens = 64
        a.max_samples = 1
        a.num_workers = 1
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        rg.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    await rg.main_async()

    results_dir = (
        Path(tmp_path)
        / "results"
        / "mmmu"
        / "no-vision"
        / "public-non_official_generation_v1"
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
async def test_pm_generate_mmmu_external_prompt(tmp_path, monkeypatch):
    # Avoid network on init
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Physics"],
    )
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: MMMUAllBenchmark())
    monkeypatch.setattr(
        MMMUAllBenchmark,
        "load_data",
        lambda self, max_samples=None: [
            {
                "id": "q",
                "question": "Pick:",
                "parsed_choices": {"A": "x", "B": "y"},
                "answer": "B",
            }
        ],
    )

    ext = tmp_path / "gen.txt"
    ext.write_text(
        "---[USER]---\n{question}\n\nChoices:\n{choices_block}",
        encoding="utf-8",
    )

    async def fake_generate(messages, temperature, max_tokens):
        return {
            "content": "Explanation: ...\nAnswer: B\nConfidence: 99",
            "reasoning": None,
        }

    llm = types.SimpleNamespace(
        supports_vision=True, supports_reasoning=False, generate=fake_generate
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm)

    def args():
        class A:
            pass

        a = A()
        a.model = "m"
        a.benchmark = "mmmu"
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

    results_dir = Path(tmp_path) / "results" / "mmmu" / "m" / ext.stem / "generations"
    files = list(results_dir.glob("*_generations.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["metadata"]["generation"]["prompt_source_type"] == "external"
