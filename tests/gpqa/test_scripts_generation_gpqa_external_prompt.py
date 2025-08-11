import json
import types
from pathlib import Path
import pytest
import promptmetrics.scripts.run_generation as rg
from promptmetrics.benchmarks.gpqa import GPQADiamondBenchmark


@pytest.mark.asyncio
async def test_pm_generate_gpqa_external_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rg, "get_benchmark_instance", lambda name: GPQADiamondBenchmark()
    )
    # minimal questions
    monkeypatch.setattr(
        GPQADiamondBenchmark,
        "load_data",
        lambda self, max_samples=None: [
            {
                "id": "q",
                "Question": "Q",
                "shuffled_choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "correct_answer_letter": "A",
            }
        ],
    )

    ext = tmp_path / "gen.txt"
    ext.write_text(
        ">>> {question}\nA){choice_A}\nB){choice_B}\nC){choice_C}\nD){choice_D}",
        encoding="utf-8",
    )

    async def fake_generate(messages, temperature, max_tokens):
        return {"content": "The correct answer is (A)", "reasoning": None}

    llm = types.SimpleNamespace(
        supports_vision=False, supports_reasoning=False, generate=fake_generate
    )
    monkeypatch.setattr(rg, "OpenRouterLLM", lambda model_name: llm)

    def args():
        class A:
            pass

        a = A()
        a.model = "m"
        a.benchmark = "gpqa_diamond"
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
        Path(tmp_path) / "results" / "gpqa_diamond" / "m" / ext.stem / "generations"
    )
    files = list(results_dir.glob("*_generations.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["metadata"]["generation"]["prompt_source_type"] == "external"
