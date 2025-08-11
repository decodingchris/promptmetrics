import pytest
from pathlib import Path
import promptmetrics.scripts.run_generation as rg
from promptmetrics.benchmarks.hle import HLEBenchmark


@pytest.mark.asyncio
async def test_pm_generate_official_fallback_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        HLEBenchmark, "official_generation_prompt_name", property(lambda self: None)
    )
    monkeypatch.setattr(rg, "get_benchmark_instance", lambda name: HLEBenchmark())

    def fake_load_prompt_template(source, bench, ptype):
        raise FileNotFoundError("missing fallback")

    monkeypatch.setattr(rg, "load_prompt_template", fake_load_prompt_template)

    def fake_args():
        class A:
            pass

        a = A()
        a.model = "m"
        a.benchmark = "hle"
        a.generation_prompt_source = "official"
        a.output_dir = Path(tmp_path)
        a.temperature = 0.0
        a.max_tokens = 8
        a.max_samples = 1
        a.num_workers = 1
        a.allow_full_run = True
        return a

    monkeypatch.setattr(
        rg.argparse.ArgumentParser, "parse_args", lambda self: fake_args()
    )
    with pytest.raises(ValueError, match="does not have an official generation prompt"):
        await rg.main_async()
