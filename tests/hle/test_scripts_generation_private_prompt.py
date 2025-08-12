# tests/hle/test_scripts_generation_private_prompt.py

import json
import types
from pathlib import Path

import pytest

import promptmetrics.scripts.run_generation as rg
from promptmetrics.benchmarks.hle import HLEBenchmark


@pytest.mark.asyncio
async def test_pm_generate_with_private_prompt(tmp_path, monkeypatch):
    # Create a private prompt file in prompts/private to exercise 'private' path
    private_dir = Path("prompts") / "private" / "hle" / "generation"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_file = private_dir / "my_private_prompt.txt"
    private_file.write_text(
        "---[SYSTEM]---\nS\n---[USER]---\n{question}",
        encoding="utf-8",
    )

    try:
        monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: HLEBenchmark())
        # Provide minimal questions
        monkeypatch.setattr(
            HLEBenchmark,
            "load_data",
            lambda self, max_samples=None: [
                {"id": "q", "question": "?", "answer": "a"}
            ],
        )

        async def fake_generate(messages, temperature, max_tokens):
            return {
                "content": "Explanation: ...\nAnswer: a\nConfidence: 99",
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
            a.benchmark = "hle"
            a.generation_prompt_source = "my_private_prompt"
            a.output_dir = Path(tmp_path)
            a.temperature = 0.0
            a.max_tokens = 32
            a.max_samples = 1
            a.num_workers = 1
            a.allow_full_run = True
            return a

        monkeypatch.setattr(
            rg.argparse.ArgumentParser, "parse_args", lambda self: args()
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
            / "hle"
            / "m"
            / "private-my_private_prompt"
            / "generations"
        )
        files = list(results_dir.glob("*_generations.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["metadata"]["generation"]["prompt_source_type"] == "private"
    finally:
        private_file.unlink(missing_ok=True)
