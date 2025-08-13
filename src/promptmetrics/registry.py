import logging
from datasets import get_dataset_infos

from .benchmarks.aime import AIMEBenchmark
from .benchmarks.base import BaseBenchmark
from .benchmarks.facts import FACTSBenchmark
from .benchmarks.gpqa import GPQADiamondBenchmark
from .benchmarks.hle import HLEBenchmark
from .benchmarks.mmmu import MMMUAllBenchmark, MMMUSingleBenchmark

logger = logging.getLogger(__name__)

_MMMU_CONFIGS = None


def get_mmmu_configs():
    global _MMMU_CONFIGS
    if _MMMU_CONFIGS is None:
        try:
            infos = get_dataset_infos("MMMU/MMMU")
            _MMMU_CONFIGS = {name.lower(): name for name in infos.keys()}
        except Exception as e:
            logger.error(f"Could not fetch MMMU configs from Hugging Face Hub: {e}")
            _MMMU_CONFIGS = {}
    return _MMMU_CONFIGS


BENCHMARK_REGISTRY: dict[str, type[BaseBenchmark]] = {
    "hle": HLEBenchmark,
    "gpqa_diamond": GPQADiamondBenchmark,
    "aime_2025": AIMEBenchmark,
    "mmmu": MMMUAllBenchmark,
    "facts": FACTSBenchmark,
}


def get_benchmark_instance(name: str) -> BaseBenchmark:
    benchmark_name = name.lower()

    if benchmark_name.startswith("mmmu_"):
        parts = benchmark_name.split("_", 1)
        if len(parts) == 2 and parts[1]:
            config_name_lower = parts[1].replace("-", "_")

            available_configs = get_mmmu_configs()
            if not available_configs:
                raise RuntimeError(
                    "Could not retrieve MMMU subject list. Check network connection or HF Hub status."
                )

            if config_name_lower in available_configs:
                correct_case_config = available_configs[config_name_lower]
                return MMMUSingleBenchmark(config_name=correct_case_config)
            else:
                raise ValueError(
                    f"Unknown MMMU subject: '{parts[1]}'. "
                    f"Available subjects: {sorted(list(available_configs.keys()))}"
                )
        else:
            raise ValueError(
                "Invalid MMMU subject format. Expected 'mmmu_<subject>', e.g., 'mmmu_art'."
            )

    if benchmark_name in BENCHMARK_REGISTRY:
        return BENCHMARK_REGISTRY[benchmark_name]()

    available_benchmarks = sorted(list(BENCHMARK_REGISTRY.keys()))
    raise ValueError(
        f"Unknown benchmark: '{name}'. Available benchmarks: {available_benchmarks}. "
        "For a single MMMU subject, use the format 'mmmu_<subject>' (e.g., 'mmmu_art')."
    )
