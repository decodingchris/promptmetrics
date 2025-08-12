# tests/mmmu/test_scripts_evaluation_mmmu.py

import json
import types
from pathlib import Path

import pytest

import promptmetrics.scripts.run_evaluation as reval
from promptmetrics.benchmarks.mmmu import MMMUAllBenchmark, OfficialMMMU_V1Evaluation


@pytest.fixture
def mmmu_questions():
    # Parsed choices not strictly required for evaluation, but realistic
    return [
        {
            "id": "q1",
            "question": "Which animal?",
            "parsed_choices": {"A": "dog", "B": "cat", "C": "bird"},
            "answer": "B",
        },
        {
            "id": "q2",
            "question": "Which number?",
            "parsed_choices": {"A": "1", "B": "2", "C": "3"},
            "answer": "A",
        },
    ]


@pytest.fixture
def mmmu_generations_artifact(tmp_path, mmmu_questions):
    base = (
        tmp_path
        / "results"
        / "mmmu"
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
                "benchmark": "mmmu",
                "prompt_source": "non_official_generation_v1",
                "prompt_source_type": "public",
                "prompt_file": "prompts/public/mmmu/generation/non_official_generation_v1.txt",
                "temperature": 0.0,
                "max_tokens": 128,
                "generated_at_utc": "20250101T000000Z",
            }
        },
        "generations": {
            mmmu_questions[0]["id"]: {
                "response": "Explanation: ... Answer: B\nConfidence: 95"
            },
            mmmu_questions[1]["id"]: {
                "response": "Explanation: ... Answer: C\nConfidence: 55"
            },
        },
    }
    fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return fpath


@pytest.mark.asyncio
async def test_pm_evaluate_mmmu_official_fallback_is_advanced(
    mmmu_generations_artifact, monkeypatch, mmmu_questions
):
    # Avoid network config load in MMMUAllBenchmark() constructor
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Art"],
    )

    def fake_args():
        class A:
            pass

        a = A()
        a.input_file = Path(mmmu_generations_artifact)
        a.evaluator_model = "eval/m"
        a.evaluation_prompt_source = "official"  # MMMU has no official eval -> fallback
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    monkeypatch.setattr(
        reval, "get_benchmark_instance", lambda name: MMMUAllBenchmark()
    )
    # Only required IDs
    monkeypatch.setattr(
        MMMUAllBenchmark,
        "load_data",
        lambda self, ids_to_load=None, max_samples=None: [
            q for q in mmmu_questions if q["id"] in ids_to_load
        ],
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False

        async def generate_structured(self, prompt, response_model, max_tokens):
            if "Which animal?" in prompt:
                return OfficialMMMU_V1Evaluation(
                    extracted_answer_choice="B",
                    reasoning="Match",
                    correct="yes",
                    confidence=95,
                )
            return OfficialMMMU_V1Evaluation(
                extracted_answer_choice="C",
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

    eval_dir = Path(mmmu_generations_artifact).parent.parent / "evaluations"
    files = list(
        eval_dir.glob("*_evaluations_by_*_with_non_official_evaluation_v1.json")
    )
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]
    # 1 correct out of 2 -> 50% +/- (CI present) and ECE computed
    assert summary["accuracy"] == 50.0
    assert summary["correct_count"] == 1
    assert summary["total_evaluated"] == 2
    assert "accuracy_ci_95" in summary
    assert summary["expected_calibration_error"] == 30.0


@pytest.mark.asyncio
async def test_pm_evaluate_mmmu_custom_prompt_still_advanced(
    mmmu_generations_artifact, monkeypatch, mmmu_questions, tmp_path
):
    # Avoid network config load
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Physics"],
    )
    custom_prompt = tmp_path / "custom_eval.txt"
    custom_prompt.write_text(
        "Evaluate: {question} // {model_response} // {answer}",
        encoding="utf-8",
    )

    def fake_args():
        class A:
            pass

        a = A()
        a.input_file = Path(mmmu_generations_artifact)
        a.evaluator_model = "eval/m"
        a.evaluation_prompt_source = str(custom_prompt)  # Still advanced for MMMU
        a.num_workers = 1
        a.evaluator_max_tokens = 256
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        reval.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    monkeypatch.setattr(
        reval, "get_benchmark_instance", lambda name: MMMUAllBenchmark()
    )
    monkeypatch.setattr(
        MMMUAllBenchmark,
        "load_data",
        lambda self, ids_to_load=None, max_samples=None: [
            q for q in mmmu_questions if q["id"] in ids_to_load
        ],
    )

    class FakeLLM:
        def __init__(self, model_name):
            self.supports_reasoning = False

        async def generate_structured(self, prompt, response_model, max_tokens):
            if "Which animal?" in prompt:
                return OfficialMMMU_V1Evaluation(
                    extracted_answer_choice="B",
                    reasoning="OK",
                    correct="yes",
                    confidence=90,
                )
            return OfficialMMMU_V1Evaluation(
                extracted_answer_choice="C",
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

    eval_dir = Path(mmmu_generations_artifact).parent.parent / "evaluations"
    files = list(eval_dir.glob("*_evaluations_by_*_with_custom_eval.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["summary_metrics"]
    # Advanced metrics present for MMMU even with custom prompt
    assert summary["accuracy"] == 50.0
    assert "accuracy_ci_95" in summary
    assert "expected_calibration_error" in summary


@pytest.mark.asyncio
async def test_pm_evaluate_mmmu_official_fallback_missing_raises(monkeypatch, tmp_path):
    # Minimal generations artifact
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
    art.write_text(
        '{"metadata":{"generation":{"model":"m","reasoning_model":false,"benchmark":"mmmu","prompt_source":"non_official_generation_v1","prompt_source_type":"public","prompt_file":"x","temperature":0,"max_tokens":1,"generated_at_utc":"t"}},"generations":{}}',
        encoding="utf-8",
    )

    # Avoid network on init
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Art"],
    )
    monkeypatch.setattr(
        reval, "get_benchmark_instance", lambda name: MMMUAllBenchmark()
    )

    # Force fallback "non_official_evaluation_v1" to be missing
    def fake_load_prompt_template(source, bench, ptype):
        if source == "non_official_evaluation_v1":
            raise FileNotFoundError("missing fallback")
        # Allow other loads if needed, though this test path shouldn't need them
        from promptmetrics.utils import load_prompt_template as real_load

        return real_load(source, bench, ptype)

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
        match="does not have an official evaluation prompt.*fallback prompt 'non_official_evaluation_v1' could not be found",
    ):
        await reval.main_async()
