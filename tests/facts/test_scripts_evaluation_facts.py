import json
import types
from pathlib import Path

import pytest

import promptmetrics.scripts.run_evaluation as reval
from promptmetrics.benchmarks.facts import FACTSBenchmark, OfficialFACTSEvaluation


@pytest.fixture
def facts_questions():
    return [
        {
            "id": "q1",
            "user_request": "Who wrote the book?",
            "context_document": "Plato wrote the book.",
        },
        {
            "id": "q2",
            "user_request": "Where is the capital of France?",
            "context_document": "Paris is the capital of France.",
        },
    ]


@pytest.fixture
def facts_generations_artifact(tmp_path, facts_questions):
    base = (
        tmp_path
        / "results"
        / "facts"
        / "test-model"
        / "public-official_generation_v1"
        / "generations"
    )
    base.mkdir(parents=True, exist_ok=True)
    fpath = base / "20250101T000000Z_generations.json"
    data = {
        "metadata": {
            "generation": {
                "model": "test-model",
                "benchmark": "facts",
                "prompt_source": "official_generation_v1",
            }
        },
        "generations": {
            facts_questions[0]["id"]: {"response": "Answer: Plato wrote the book."},
            facts_questions[1]["id"]: {"response": "Answer: London"},
        },
    }
    fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return fpath


@pytest.mark.asyncio
async def test_pm_evaluate_facts_official_fallback_is_advanced(
    facts_generations_artifact, monkeypatch, facts_questions
):
    def fake_args():
        class A:
            pass

        a = A()
        a.input_file = Path(facts_generations_artifact)
        a.evaluator_model = "eval/m"
        a.evaluation_prompt_source = "official"
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    monkeypatch.setattr(reval, "get_benchmark_instance", lambda name: FACTSBenchmark())
    monkeypatch.setattr(
        FACTSBenchmark,
        "load_data",
        lambda self, ids_to_load=None, max_samples=None: [
            q for q in facts_questions if q["id"] in ids_to_load
        ],
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False

        async def generate_structured(self, prompt, response_model, max_tokens):
            if "Who wrote the book?" in prompt:
                return OfficialFACTSEvaluation(
                    reasoning="Matches document.", correct="yes", confidence=95
                )
            return OfficialFACTSEvaluation(
                reasoning="Contradicts document.", correct="no", confidence=55
            )

    monkeypatch.setattr(reval, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr(
        reval,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *coros: reval.asyncio.gather(*coros)),
    )

    await reval.main_async()

    eval_dir = Path(facts_generations_artifact).parent.parent / "evaluations"
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
async def test_pm_evaluate_facts_custom_prompt_still_advanced(
    facts_generations_artifact, monkeypatch, facts_questions, tmp_path
):
    custom_prompt = tmp_path / "facts_custom_eval.txt"
    custom_prompt.write_text(
        "Evaluate: {user_request} // {context_document} // {model_response}",
        encoding="utf-8",
    )

    def fake_args():
        class A:
            pass

        a = A()
        a.input_file = Path(facts_generations_artifact)
        a.evaluator_model = "eval/m"
        a.evaluation_prompt_source = str(custom_prompt)
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    monkeypatch.setattr(reval, "get_benchmark_instance", lambda name: FACTSBenchmark())
    monkeypatch.setattr(
        FACTSBenchmark,
        "load_data",
        lambda self, ids_to_load=None, max_samples=None: [
            q for q in facts_questions if q["id"] in ids_to_load
        ],
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False

        async def generate_structured(self, prompt, response_model, max_tokens):
            if "Who wrote the book?" in prompt:
                return OfficialFACTSEvaluation(
                    reasoning="OK", correct="yes", confidence=90
                )
            return OfficialFACTSEvaluation(
                reasoning="Wrong", correct="no", confidence=60
            )

    monkeypatch.setattr(reval, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr(
        reval,
        "tqdm_asyncio",
        types.SimpleNamespace(gather=lambda *coros: reval.asyncio.gather(*coros)),
    )

    await reval.main_async()

    eval_dir = Path(facts_generations_artifact).parent.parent / "evaluations"
    files = list(eval_dir.glob("*_evaluations_by_*_with_facts_custom_eval.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]
    assert summary["accuracy"] == 50.0
    assert "accuracy_ci_95" in summary
    assert "expected_calibration_error" in summary


@pytest.mark.asyncio
async def test_pm_evaluate_facts_official_fallback_missing_raises(
    monkeypatch, tmp_path
):
    base = (
        tmp_path
        / "results"
        / "facts"
        / "m"
        / "public-official_generation_v1"
        / "generations"
    )
    base.mkdir(parents=True, exist_ok=True)
    art = base / "20250101T000000Z_generations.json"
    art.write_text(
        '{"metadata":{"generation":{"benchmark":"facts"}},"generations":{}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(reval, "get_benchmark_instance", lambda name: FACTSBenchmark())

    def fake_load_prompt_template(source, bench, ptype):
        raise FileNotFoundError("missing fallback")

    monkeypatch.setattr(reval, "load_prompt_template", fake_load_prompt_template)

    def fake_args():
        class A:
            pass

        a = A()
        a.input_file = art
        a.evaluator_model = "eval"
        a.evaluation_prompt_source = "official"
        a.num_workers = 1
        a.evaluator_max_tokens = 1
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    with pytest.raises(
        ValueError,
        match="does not have an official evaluation prompt.*could not be found",
    ):
        await reval.main_async()
