import json
import types
from pathlib import Path

import pytest

import promptmetrics.scripts.run_generation as rg
from promptmetrics.benchmarks.facts import FACTSBenchmark


@pytest.mark.asyncio
async def test_pm_generate_facts_official_creates_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: FACTSBenchmark())

    questions = [
        {
            "id": "q1",
            "system_instruction": "Use doc only.",
            "user_request": "Who wrote A?",
            "context_document": "Alice wrote A.",
        },
        {
            "id": "q2",
            "system_instruction": "Use doc only.",
            "user_request": "When was B founded?",
            "context_document": "B was founded in 2000.",
        },
    ]
    monkeypatch.setattr(
        FACTSBenchmark, "load_data", lambda self, max_samples=None: questions
    )

    async def fake_generate(messages, temperature, max_tokens):
        return {"content": "Some answer\nConfidence: 90", "reasoning": None}

    llm_mock = types.SimpleNamespace(
        supports_vision=False, supports_reasoning=False, generate=fake_generate
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm_mock)

    def fake_args():
        class A:
            pass

        a = A()
        a.model = "my/model:v1"
        a.benchmark = "facts"
        a.generation_prompt_source = "official"
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
        / "facts"
        / "my_model-v1"
        / "public-official_generation_v1"
        / "generations"
    )
    files = list(results_dir.glob("*_generations.json"))
    assert len(files) == 1

    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert "metadata" in data and "generations" in data
    assert set(data["generations"].keys()) == {"q1", "q2"}
    assert data["metadata"]["generation"]["prompt_source"] == "official_generation_v1"
    assert data["metadata"]["generation"]["prompt_source_type"] == "public"


@pytest.mark.asyncio
async def test_pm_generate_facts_external_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: FACTSBenchmark())
    monkeypatch.setattr(
        FACTSBenchmark,
        "load_data",
        lambda self, max_samples=None: [
            {
                "id": "q",
                "system_instruction": "Only doc",
                "user_request": "What is X?",
                "context_document": "X is 42.",
            }
        ],
    )

    ext = tmp_path / "gen.txt"
    ext.write_text(
        "---[SYSTEM]---\n{system_instruction}\n---[USER]---\n"
        "Request:\n{user_request}\n\nDoc:\n{context_document}",
        encoding="utf-8",
    )

    async def fake_generate(messages, temperature, max_tokens):
        return {"content": "Answer: 42\nConfidence: 99", "reasoning": None}

    llm = types.SimpleNamespace(
        supports_vision=False, supports_reasoning=False, generate=fake_generate
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm)

    def args():
        class A:
            pass

        a = A()
        a.model = "m"
        a.benchmark = "facts"
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

    results_dir = Path(tmp_path) / "results" / "facts" / "m" / ext.stem / "generations"
    files = list(results_dir.glob("*_generations.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["metadata"]["generation"]["prompt_source_type"] == "external"
