import json
import math
import types
from pathlib import Path
import pytest

import promptmetrics.scripts.run_evaluation as reval
from promptmetrics.benchmarks.hle import HLEBenchmark, OfficialHLEEvaluation


# --- Core Value: Accurate, structured evaluation with calibration metrics ---

@pytest.mark.parametrize(
    "conf, corr, expected_ece",
    [
        ([0.95], [1], 0.05),  # |0.95 - 1.0| in its bin
        ([0.55], [0], 0.55),  # |0.55 - 0.0|
        ([0.95, 0.55], [1, 0], 0.30),
        ([0.85, 0.85], [1, 1], 0.15),  # |0.85 - 1.0|
        ([0.15, 0.85], [0, 1], 0.15),  # average of |0.15-0| and |0.85-1|
    ],
    ids=["single-correct", "single-incorrect", "two-mixed", "both-certain-correct", "symmetric"],
)
def test_calculate_ece(conf, corr, expected_ece):
    import numpy as np
    res = reval.calculate_ece(np.array(conf), np.array(corr), n_bins=10)
    assert round(res, 2) == expected_ece


@pytest.mark.asyncio
async def test_pm_evaluate_creates_evaluation_artifact(tmp_output_dir, generations_sample_artifact, monkeypatch):
    # configure parser to accept our input file
    def fake_parse_args():
        class A:
            pass
        a = A()
        a.input_file = Path(generations_sample_artifact)
        a.evaluator_model = "any-evaluator"
        a.evaluation_prompt_source = "official"
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(reval.argparse.ArgumentParser, "parse_args", lambda self: fake_parse_args())

    # benchmark and data
    monkeypatch.setattr(reval, "get_benchmark_instance", lambda name: HLEBenchmark())

    # load only the needed ids from the generations artifact
    def fake_load_data(self, max_samples=None, ids_to_load=None):
        q_map = {
            "q1": {"id": "q1", "question": "What is shown in the image?", "answer": "a cat"},
            "q2": {"id": "q2", "question": "Compute 2 + 2.", "answer": "4"},
        }
        return [q_map[i] for i in ids_to_load]

    monkeypatch.setattr(HLEBenchmark, "load_data", fake_load_data)

    # OpenRouterLLM.generate_structured returning OfficialHLEEvaluation verdicts
    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False
        async def generate_structured(self, prompt, response_model, max_tokens):
            if "'q1'" in prompt or "a cat" in prompt:
                return OfficialHLEEvaluation(
                    extracted_final_answer="a cat",
                    reasoning="Matches exactly.",
                    correct="yes",
                    confidence=95,
                )
            else:
                return OfficialHLEEvaluation(
                    extracted_final_answer="5",
                    reasoning="Incorrect.",
                    correct="no",
                    confidence=55,
                )

    monkeypatch.setattr(reval, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr(reval, "tqdm_asyncio", types.SimpleNamespace(gather=lambda *coros: reval.asyncio.gather(*coros)))

    await reval.main_async()

    # Verify evaluations artifact
    base = Path(generations_sample_artifact).parent.parent
    eval_dir = base / "evaluations"
    files = list(eval_dir.glob("*_evaluations_by_*_with_official_evaluation_v1.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]

    # 1 correct out of 2 = 50% +/- 69.3% and ECE 30.0%
    assert summary["accuracy"] == 50.0
    assert summary["correct_count"] == 1
    assert summary["total_evaluated"] == 2
    assert summary["accuracy_ci_95"] == 69.3
    assert summary["expected_calibration_error"] == 30.0


@pytest.mark.asyncio
async def test_pm_evaluate_prompts_for_confirmation_when_not_allowed(tmp_output_dir, generations_sample_artifact, monkeypatch):
    # same as above but do not pass --allow-full-run
    def fake_parse_args():
        class A:
            pass
        a = A()
        a.input_file = Path(generations_sample_artifact)
        a.evaluator_model = "any-evaluator"
        a.evaluation_prompt_source = "official"
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = False
        return a

    monkeypatch.setattr(reval.argparse.ArgumentParser, "parse_args", lambda self: fake_parse_args())
    # Make the prompt decline
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")

    await reval.main_async()

    # Should not create evaluations dir
    base = Path(generations_sample_artifact).parent.parent
    eval_dir = base / "evaluations"
    assert not eval_dir.exists()


@pytest.mark.asyncio
async def test_pm_evaluate_with_non_official_prompt_uses_default_verdict(tmp_output_dir, generations_sample_artifact, monkeypatch, tmp_path):
    # Use a custom evaluation prompt (non-official) -> default EvaluationVerdict path, no ECE fields
    custom_prompt = tmp_path / "custom_eval.txt"
    custom_prompt.write_text("Evaluate informally: {question} :: {model_response} vs {correct_answer}", encoding="utf-8")

    def fake_parse_args():
        class A:
            pass
        a = A()
        a.input_file = Path(generations_sample_artifact)
        a.evaluator_model = "any-evaluator"
        a.evaluation_prompt_source = str(custom_prompt)
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(reval.argparse.ArgumentParser, "parse_args", lambda self: fake_parse_args())
    monkeypatch.setattr(reval, "get_benchmark_instance", lambda name: HLEBenchmark())

    def fake_load_data(self, max_samples=None, ids_to_load=None):
        q_map = {
            "q1": {"id": "q1", "question": "What is shown in the image?", "answer": "a cat"},
            "q2": {"id": "q2", "question": "Compute 2 + 2.", "answer": "4"},
        }
        return [q_map[i] for i in ids_to_load]

    monkeypatch.setattr(HLEBenchmark, "load_data", fake_load_data)

    # Default EvaluationVerdict path
    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False
        async def generate_structured(self, prompt, response_model, max_tokens):
            # Construct default verdicts (is_correct True for q1, False for q2)
            if "a cat" in prompt:
                return reval.EvaluationVerdict(is_correct=True, extracted_answer="a cat", reasoning="ok")
            return reval.EvaluationVerdict(is_correct=False, extracted_answer="5", reasoning="bad")

    monkeypatch.setattr(reval, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr(reval, "tqdm_asyncio", types.SimpleNamespace(gather=lambda *coros: reval.asyncio.gather(*coros)))

    await reval.main_async()

    base = Path(generations_sample_artifact).parent.parent
    eval_dir = base / "evaluations"
    files = list(eval_dir.glob("*_evaluations_by_*_with_custom_eval.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]
    assert summary["accuracy"] == 50.0
    assert summary["correct_count"] == 1
    assert summary["total_evaluated"] == 2
    # Non-official: no CI or ECE
    assert "accuracy_ci_95" not in summary
    assert "expected_calibration_error" not in summary


@pytest.mark.asyncio
async def test_pm_evaluate_with_zero_items_yields_zero_metrics(tmp_output_dir, monkeypatch, tmp_path):
    # Create an empty generations artifact
    artifact_dir = tmp_output_dir / "results" / "hle" / "m" / "public-official_generation_v1" / "generations"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = artifact_dir / "20250101T000000Z_generations.json"
    artifact = {
        "metadata": {
            "generation": {
                "model": "m",
                "reasoning_model": False,
                "benchmark": "hle",
                "prompt_source": "official_generation_v1",
                "prompt_source_type": "public",
                "prompt_file": "prompts/public/hle/generation/official_generation_v1.txt",
                "temperature": 0.0,
                "max_tokens": 128,
                "generated_at_utc": "20250101T000000Z",
            }
        },
        "generations": {}
    }
    artifact_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    def fake_parse_args():
        class A:
            pass
        a = A()
        a.input_file = artifact_file
        a.evaluator_model = "eval-model"
        a.evaluation_prompt_source = "official"
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(reval.argparse.ArgumentParser, "parse_args", lambda self: fake_parse_args())
    monkeypatch.setattr(reval, "get_benchmark_instance", lambda name: HLEBenchmark())
    # No questions to load
    monkeypatch.setattr(HLEBenchmark, "load_data", lambda self, max_samples=None, ids_to_load=None: [])
    # Ensure no real LLM used; it won't be called anyway
    monkeypatch.setattr(reval, "OpenRouterLLM", lambda model_name: types.SimpleNamespace(supports_reasoning=False))
    monkeypatch.setattr(reval, "tqdm_asyncio", types.SimpleNamespace(gather=lambda *coros: reval.asyncio.gather(*coros)))

    await reval.main_async()

    # evaluations directory sits alongside generations inside the prompt version folder
    eval_dir = artifact_dir.parent / "evaluations"
    files = list(eval_dir.glob("*_evaluations_by_*_with_official_evaluation_v1.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]
    assert summary["accuracy"] == 0
    assert summary["correct_count"] == 0
    assert summary["total_evaluated"] == 0
    # No CI/ECE computed for zero items even for official
    assert "accuracy_ci_95" not in summary
    assert "expected_calibration_error" not in summary