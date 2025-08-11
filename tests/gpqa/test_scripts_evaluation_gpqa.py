import json
import types
from pathlib import Path
import pytest

import promptmetrics.scripts.run_evaluation as reval
from promptmetrics.benchmarks.gpqa import GPQADiamondBenchmark, OfficialGPQAEvaluation


# --- Core Value: GPQA advanced evaluation without an official evaluation prompt ---


@pytest.fixture
def gpqa_questions():
    # Pre-shuffled, pre-annotated questions (simulate output of load_data)
    return [
        {
            "id": "q1",
            "Question": "Which planet is known as the Red Planet?",
            "shuffled_choices": {
                "A": "Earth",
                "B": "Mars",
                "C": "Venus",
                "D": "Jupiter",
            },
            "correct_answer_letter": "B",
        },
        {
            "id": "q2",
            "Question": "What is H2O commonly known as?",
            "shuffled_choices": {
                "A": "Salt",
                "B": "Oxygen",
                "C": "Water",
                "D": "Hydrogen",
            },
            "correct_answer_letter": "C",
        },
    ]


@pytest.fixture
def gpqa_generations_artifact(tmp_path, gpqa_questions):
    base = (
        tmp_path
        / "results"
        / "gpqa_diamond"
        / "test-model"
        / "public-official_generation_zeroshot_v1"
        / "generations"
    )
    base.mkdir(parents=True, exist_ok=True)
    fpath = base / "20250101T000000Z_generations.json"
    data = {
        "metadata": {
            "generation": {
                "model": "test-model",
                "reasoning_model": False,
                "benchmark": "gpqa_diamond",
                "prompt_source": "official_generation_zeroshot_v1",
                "prompt_source_type": "public",
                "prompt_file": "prompts/public/gpqa/generation/official_generation_zeroshot_v1.txt",
                "temperature": 0.0,
                "max_tokens": 128,
                "generated_at_utc": "20250101T000000Z",
            }
        },
        "generations": {
            gpqa_questions[0]["id"]: {"response": "The correct answer is (B)"},
            gpqa_questions[1]["id"]: {"response": "The correct answer is (D)"},
        },
    }
    fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return fpath


@pytest.mark.asyncio
async def test_pm_evaluate_gpqa_official_fallback_still_advanced(
    gpqa_generations_artifact, monkeypatch, gpqa_questions
):
    # CLI args
    def fake_args():
        class A:
            pass

        a = A()
        a.input_file = Path(gpqa_generations_artifact)
        a.evaluator_model = "eval/m"
        a.evaluation_prompt_source = "official"  # GPQA has no official eval -> fallback to non_official_evaluation_v1
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    # Use GPQA benchmark
    monkeypatch.setattr(
        reval, "get_benchmark_instance", lambda name: GPQADiamondBenchmark()
    )
    # Provide only required IDs
    monkeypatch.setattr(
        GPQADiamondBenchmark,
        "load_data",
        lambda self, ids_to_load=None, max_samples=None: [
            q for q in gpqa_questions if q["id"] in ids_to_load
        ],
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False

        async def generate_structured(self, prompt, response_model, max_tokens):
            if "Mars" in prompt:
                return OfficialGPQAEvaluation(
                    extracted_answer_choice="B",
                    reasoning="Match",
                    correct="yes",
                    confidence=95,
                )
            return OfficialGPQAEvaluation(
                extracted_answer_choice="D",
                reasoning="Mismatch",
                correct="no",
                confidence=55,
            )

    monkeypatch.setattr(reval, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr(
        reval,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *coros: reval.asyncio.gather(*coros)),
    )

    await reval.main_async()

    eval_dir = Path(gpqa_generations_artifact).parent.parent / "evaluations"
    files = list(
        eval_dir.glob("*_evaluations_by_*_with_non_official_evaluation_v1.json")
    )
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]
    assert summary["accuracy"] == 50.0
    assert summary["correct_count"] == 1
    assert summary["total_evaluated"] == 2
    assert "accuracy_ci_95" in summary
    assert summary["expected_calibration_error"] == 30.0


@pytest.mark.asyncio
async def test_pm_evaluate_gpqa_custom_prompt_still_advanced(
    gpqa_generations_artifact, monkeypatch, gpqa_questions, tmp_path
):
    custom_prompt = tmp_path / "custom_eval.txt"
    custom_prompt.write_text(
        "Evaluate: {question} :: {model_response} :: {correct_answer_letter}",
        encoding="utf-8",
    )

    def fake_args():
        class A:
            pass

        a = A()
        a.input_file = Path(gpqa_generations_artifact)
        a.evaluator_model = "eval/m"
        a.evaluation_prompt_source = str(custom_prompt)
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    monkeypatch.setattr(
        reval, "get_benchmark_instance", lambda name: GPQADiamondBenchmark()
    )
    monkeypatch.setattr(
        GPQADiamondBenchmark,
        "load_data",
        lambda self, ids_to_load=None, max_samples=None: [
            q for q in gpqa_questions if q["id"] in ids_to_load
        ],
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False

        async def generate_structured(self, prompt, response_model, max_tokens):
            if " :: B" in prompt or "Mars" in prompt:
                return OfficialGPQAEvaluation(
                    extracted_answer_choice="B",
                    reasoning="OK",
                    correct="yes",
                    confidence=90,
                )
            return OfficialGPQAEvaluation(
                extracted_answer_choice="D",
                reasoning="Wrong",
                correct="no",
                confidence=60,
            )

    monkeypatch.setattr(reval, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr(
        reval,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *coros: reval.asyncio.gather(*coros)),
    )

    await reval.main_async()

    eval_dir = Path(gpqa_generations_artifact).parent.parent / "evaluations"
    files = list(eval_dir.glob("*_evaluations_by_*_with_custom_eval.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]
    assert summary["accuracy"] == 50.0
    assert "accuracy_ci_95" in summary
    assert "expected_calibration_error" in summary
