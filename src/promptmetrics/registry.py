from .benchmarks.base import BaseBenchmark
from .benchmarks.gpqa import GPQADiamondBenchmark
from .benchmarks.hle import HLEBenchmark
from .benchmarks.aime import AIMEBenchmark


_BENCHMARK_CLASSES: tuple[
    type[HLEBenchmark], type[GPQADiamondBenchmark], type[AIMEBenchmark]
] = (HLEBenchmark, GPQADiamondBenchmark, AIMEBenchmark)

BENCHMARK_REGISTRY: dict[str, type[BaseBenchmark]] = {
    cls().name: cls for cls in _BENCHMARK_CLASSES
}


def get_benchmark_instance(name: str) -> BaseBenchmark:
    """
    Instantiates and returns a benchmark instance from the registry.

    Args:
        name: The name of the benchmark to retrieve.

    Returns:
        An instance of the requested benchmark class.

    Raises:
        ValueError: If the benchmark name is not found in the registry.
    """
    benchmark_name = name.lower()
    if benchmark_name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark: '{name}'. "
            f"Available benchmarks: {list(BENCHMARK_REGISTRY.keys())}"
        )
    return BENCHMARK_REGISTRY[benchmark_name]()
