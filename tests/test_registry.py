import pytest
from promptmetrics.registry import get_benchmark_instance, BENCHMARK_REGISTRY
from promptmetrics.benchmarks.hle import HLEBenchmark


def test_registry_contains_hle():
    assert "hle" in BENCHMARK_REGISTRY
    assert issubclass(BENCHMARK_REGISTRY["hle"], HLEBenchmark)


def test_get_benchmark_instance_success():
    b = get_benchmark_instance("hle")
    assert isinstance(b, HLEBenchmark)


def test_get_benchmark_instance_unknown():
    with pytest.raises(ValueError, match="Unknown benchmark: 'bad'.*Available benchmarks:"):
        get_benchmark_instance("bad")