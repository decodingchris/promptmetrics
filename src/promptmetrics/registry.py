# src/promptmetrics/registry.py

import logging
from datasets import get_dataset_infos

from .benchmarks.aime import AIMEBenchmark
from .benchmarks.base import BaseBenchmark
from .benchmarks.gpqa import GPQADiamondBenchmark
from .benchmarks.hle import HLEBenchmark
from .benchmarks.mmmu import MMMUAllBenchmark, MMMUSingleBenchmark

logger = logging.getLogger(__name__)


# A helper to fetch and cache the list of valid MMMU configuration names.
_MMMU_CONFIGS = None


def get_mmmu_configs():
    """Fetches and caches the list of valid MMMU configuration names."""
    global _MMMU_CONFIGS
    if _MMMU_CONFIGS is None:
        try:
            infos = get_dataset_infos("MMMU/MMMU")
            # Store a mapping from lowercase name to the correct cased name
            _MMMU_CONFIGS = {name.lower(): name for name in infos.keys()}
        except Exception as e:
            logger.error(f"Could not fetch MMMU configs from Hugging Face Hub: {e}")
            _MMMU_CONFIGS = {}
    return _MMMU_CONFIGS


# The registry is now defined manually. This prevents any objects from being
# created when the module is imported, fixing the startup bug.
BENCHMARK_REGISTRY: dict[str, type[BaseBenchmark]] = {
    "hle": HLEBenchmark,
    "gpqa_diamond": GPQADiamondBenchmark,
    "aime_2025": AIMEBenchmark,
    "mmmu": MMMUAllBenchmark,
}


def get_benchmark_instance(name: str) -> BaseBenchmark:
    """
    Instantiates and returns a benchmark instance from the registry.

    This function supports simple benchmarks (e.g., 'hle'), the full MMMU
    benchmark ('mmmu'), and single-subject MMMU benchmarks ('mmmu_art').
    """
    benchmark_name = name.lower()

    # Handle single-subject MMMU benchmarks dynamically.
    if benchmark_name.startswith("mmmu_"):
        parts = benchmark_name.split("_", 1)
        if len(parts) == 2 and parts[1]:
            config_name_lower = parts[1].replace("-", "_")

            # Find the correct case-sensitive config name from the Hub.
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

    # Check the main registry for 'mmmu' and all other simple benchmarks.
    if benchmark_name in BENCHMARK_REGISTRY:
        return BENCHMARK_REGISTRY[benchmark_name]()

    # If nothing matches, provide a comprehensive error message (single line to aid regex matching in tests).
    available_benchmarks = sorted(list(BENCHMARK_REGISTRY.keys()))
    raise ValueError(
        f"Unknown benchmark: '{name}'. Available benchmarks: {available_benchmarks}. "
        "For a single MMMU subject, use the format 'mmmu_<subject>' (e.g., 'mmmu_art')."
    )
