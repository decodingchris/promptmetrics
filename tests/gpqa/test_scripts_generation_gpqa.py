import json
import types
from pathlib import Path
import pytest

import promptmetrics.scripts.run_generation as rg
from promptmetrics.benchmarks.gpqa import GPQADiamondBenchmark


# --- Core Value: Reproducible GPQA generation artifacts ---


@pytest.mark.asyncio
async def test_pm_generate_gpqa_creates_artifact(tmp_path, monkeypatch):
    # Use GPQA benchmark
    monkeypatch.setattr(
        rg, "get_benchmark_instance", lambda name: GPQADiamondBenchmark()
    )

    # Provide two pre-built GPQA items (already have shuffled choices)
    questions = [
        {
            "id": "q1",
            "Question": "Which gas is most abundant in Earth's atmosphere?",
            "shuffled_choices": {
                "A": "Nitrogen",
                "B": "Oxygen",
                "C": "Argon",
                "D": "CO2",
            },
            "correct_answer_letter": "A",
        },
        {
            "id": "q2",
            "Question": "What is the powerhouse of the cell?",
            "shuffled_choices": {
                "A": "Nucleus",
                "B": "Ribosome",
                "C": "Chloroplast",
                "D": "Mitochondria",
            },
            "correct_answer_letter": "D",
        },
    ]
    monkeypatch.setattr(
        GPQADiamondBenchmark, "load_data", lambda self, max_samples=None: questions
    )

    # LLM mock: return GPQA-style answer string
    async def fake_generate(messages, temperature, max_tokens):
        return {"content": "The correct answer is (B).", "reasoning": None}

    llm_mock = types.SimpleNamespace(
        supports_vision=False, supports_reasoning=False, generate=fake_generate
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm_mock)

    def fake_args():
        class A:
            pass

        a = A()
        a.model = "test-m"
        a.benchmark = "gpqa_diamond"
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
        / "gpqa_diamond"
        / "test-m"
        / "public-official_generation_zeroshot_v1"
        / "generations"
    )
    files = list(results_dir.glob("*_generations.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert "metadata" in data and "generations" in data
    assert set(data["generations"].keys()) == {"q1", "q2"}
    # Ensure the prompt source recorded is the resolved official name
    assert (
        data["metadata"]["generation"]["prompt_source"]
        == "official_generation_zeroshot_v1"
    )
