import pytest

import promptmetrics.scripts.run_evaluation as reval
from promptmetrics.benchmarks.aime import AIMEBenchmark


@pytest.mark.asyncio
async def test_pm_evaluate_aime_official_fallback_missing_raises(monkeypatch, tmp_path):
    # Minimal generations artifact
    base = (
        tmp_path
        / "results"
        / "aime_2025"
        / "m"
        / "public-non_official_generation_v1"
        / "generations"
    )
    base.mkdir(parents=True, exist_ok=True)
    art = base / "20250101T000000Z_generations.json"
    art.write_text(
        '{"metadata":{"generation":{"model":"m","reasoning_model":false,"benchmark":"aime_2025","prompt_source":"non_official_generation_v1","prompt_source_type":"public","prompt_file":"x","temperature":0,"max_tokens":1,"generated_at_utc":"t"}},"generations":{}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(reval, "get_benchmark_instance", lambda name: AIMEBenchmark())

    # Force fallback "non_official_evaluation_v1" to be missing
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
        match="does not have an official evaluation prompt.*fallback prompt 'non_official_evaluation_v1' could not be found",
    ):
        await reval.main_async()
