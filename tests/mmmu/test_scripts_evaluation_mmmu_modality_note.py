# tests/mmmu/test_scripts_evaluation_mmmu_modality_note.py

import json
import types

import pytest

import promptmetrics.scripts.run_evaluation as reval
from promptmetrics.benchmarks.mmmu import MMMUAllBenchmark


@pytest.mark.asyncio
async def test_pm_evaluate_mmmu_with_modality_note_zero_items(tmp_path, monkeypatch):
    # Create a generations artifact that includes modality_handling but no generations
    base = (
        tmp_path
        / "results"
        / "mmmu"
        / "m"
        / "public-non_official_generation_v1"
        / "generations"
    )
    base.mkdir(parents=True, exist_ok=True)
    art = base / "20250101T000000Z_generations.json"
    artifact = {
        "metadata": {
            "generation": {
                "model": "m",
                "reasoning_model": False,
                "benchmark": "mmmu",
                "prompt_source": "non_official_generation_v1",
                "prompt_source_type": "public",
                "prompt_file": "prompts/public/mmmu/generation/non_official_generation_v1.txt",
                "temperature": 0.0,
                "max_tokens": 1,
                "generated_at_utc": "t",
                "modality_handling": {
                    "status": "text_only_fallback",
                    "note": "Images omitted",
                },
            }
        },
        "generations": {},
    }
    art.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # Avoid network configs in MMMUAllBenchmark()
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Art"],
    )
    monkeypatch.setattr(
        reval, "get_benchmark_instance", lambda name: MMMUAllBenchmark()
    )
    # No items needed
    monkeypatch.setattr(
        MMMUAllBenchmark,
        "load_data",
        lambda self, ids_to_load=None, max_samples=None: [],
    )
    # Evaluator stub
    monkeypatch.setattr(
        reval,
        "OpenRouterLLM",
        lambda model_name: types.SimpleNamespace(supports_reasoning=False),
    )
    monkeypatch.setattr(
        reval,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *c: reval.asyncio.gather(*c)),
    )

    def args():
        class A:
            pass

        a = A()
        a.input_file = art
        a.evaluator_model = "e"
        a.evaluation_prompt_source = "official"
        a.num_workers = 1
        a.evaluator_max_tokens = 8
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: args()
    )
    await reval.main_async()

    eval_dir = base.parent / "evaluations"
    files = list(
        eval_dir.glob("*_evaluations_by_*_with_non_official_evaluation_v1.json")
    )
    assert len(files) == 1
