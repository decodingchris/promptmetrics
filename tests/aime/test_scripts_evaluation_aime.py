import json
import types
from pathlib import Path

import pytest

import promptmetrics.scripts.run_evaluation as reval
from promptmetrics.benchmarks.aime import AIMEBenchmark, OfficialAIMEEvaluation


# --- Core Value: AIME advanced evaluation always-on (no official prompt name, but structured model present) ---


@pytest.fixture
def aime_questions():
    # Adapted AIME-like questions
    return [
        {"id": "q1", "question": "What is 2+2?", "answer": "4"},
        {"id": "q2", "question": "What is 5-2?", "answer": "3"},
    ]


@pytest.fixture
def aime_generations_artifact(tmp_path, aime_questions):
    base = (
        tmp_path
        / "results"
        / "aime_2025"
        / "test-model"
        / "public-non_official_generation_v1"
        / "generations"
    )
    base.mkdir(parents=True, exist_ok=True)
    fpath = base / "20250101T000000Z_generations.json"
    data = {
        "metadata": {
            "generation": {
                "model": "test-model",
                "reasoning_model": False,
                "benchmark": "aime_2025",
                "prompt_source": "non_official_generation_v1",
                "prompt_source_type": "public",
                "prompt_file": "prompts/public/aime/generation/non_official_generation_v1.txt",
                "temperature": 0.0,
                "max_tokens": 128,
                "generated_at_utc": "20250101T000000Z",
            }
        },
        "generations": {
            aime_questions[0]["id"]: {
                "response": "Explanation: ... Answer: 4\nConfidence: 95"
            },
            aime_questions[1]["id"]: {
                "response": "Explanation: ... Answer: 4\nConfidence: 55"
            },
        },
    }
    fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return fpath


@pytest.mark.asyncio
async def test_pm_evaluate_aime_official_fallback_is_advanced(
    aime_generations_artifact, monkeypatch, aime_questions
):
    def fake_args():
        class A:
            pass

        a = A()
        a.input_file = Path(aime_generations_artifact)
        a.evaluator_model = "eval/m"
        a.evaluation_prompt_source = "official"  # AIME has no official eval -> fallback to non_official_evaluation_v1
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    monkeypatch.setattr(reval, "get_benchmark_instance", lambda name: AIMEBenchmark())
    # Only required IDs
    monkeypatch.setattr(
        AIMEBenchmark,
        "load_data",
        lambda self, ids_to_load=None, max_samples=None: [
            q for q in aime_questions if q["id"] in ids_to_load
        ],
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False

        async def generate_structured(self, prompt, response_model, max_tokens):
            if "2+2" in prompt:
                return OfficialAIMEEvaluation(
                    extracted_final_answer="4",
                    reasoning="Matches",
                    correct="yes",
                    confidence=95,
                )
            # Intentional wrong
            return OfficialAIMEEvaluation(
                extracted_final_answer="4",
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

    eval_dir = Path(aime_generations_artifact).parent.parent / "evaluations"
    files = list(
        eval_dir.glob("*_evaluations_by_*_with_non_official_evaluation_v1.json")
    )
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]
    # 1 correct out of 2 -> 50% +/- (CI present)
    assert summary["accuracy"] == 50.0
    assert summary["correct_count"] == 1
    assert summary["total_evaluated"] == 2
    assert "accuracy_ci_95" in summary
    # ECE with [0.95 correct, 0.55 incorrect] -> 30.0%
    assert summary["expected_calibration_error"] == 30.0


@pytest.mark.asyncio
async def test_pm_evaluate_aime_custom_prompt_uses_default_verdict(
    aime_generations_artifact, monkeypatch, aime_questions, tmp_path
):
    custom_prompt = tmp_path / "custom_eval.txt"
    custom_prompt.write_text(
        "Evaluate: {question} // {model_response} // {correct_answer}",
        encoding="utf-8",
    )

    def fake_args():
        class A:
            pass

        a = A()
        a.input_file = Path(aime_generations_artifact)
        a.evaluator_model = "eval/m"
        a.evaluation_prompt_source = str(
            custom_prompt
        )  # Custom prompt -> default verdict path
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    monkeypatch.setattr(reval, "get_benchmark_instance", lambda name: AIMEBenchmark())
    monkeypatch.setattr(
        AIMEBenchmark,
        "load_data",
        lambda self, ids_to_load=None, max_samples=None: [
            q for q in aime_questions if q["id"] in ids_to_load
        ],
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False

        async def generate_structured(self, prompt, response_model, max_tokens):
            if "2+2" in prompt:
                return reval.EvaluationVerdict(
                    is_correct=True, extracted_answer="4", reasoning="OK"
                )
            return reval.EvaluationVerdict(
                is_correct=False, extracted_answer="4", reasoning="Wrong"
            )

    monkeypatch.setattr(reval, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr(
        reval,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *coros: reval.asyncio.gather(*coros)),
    )

    await reval.main_async()

    eval_dir = Path(aime_generations_artifact).parent.parent / "evaluations"
    files = list(eval_dir.glob("*_evaluations_by_*_with_custom_eval.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]
    # Default verdict path: accuracy computed, but no CI or ECE present
    assert summary["accuracy"] == 50.0
    assert summary["correct_count"] == 1
    assert summary["total_evaluated"] == 2
    assert "accuracy_ci_95" not in summary
    assert "expected_calibration_error" not in summary
