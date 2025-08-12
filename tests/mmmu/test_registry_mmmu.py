# tests/mmmu/test_registry_mmmu.py

import pytest

import promptmetrics.registry as registry_mod
from promptmetrics.benchmarks.mmmu import MMMUAllBenchmark, MMMUSingleBenchmark
from promptmetrics.registry import get_benchmark_instance


def test_get_benchmark_instance_mmmu_all(monkeypatch):
    b = get_benchmark_instance("mmmu")
    assert isinstance(b, MMMUAllBenchmark)


def test_get_benchmark_instance_mmmu_single_subject(monkeypatch):
    # Patch registry helper to avoid network: provide mapping from lowercase to proper case
    monkeypatch.setattr(
        "promptmetrics.registry.get_mmmu_configs",
        lambda: {"art": "Art", "computer_science": "Computer_Science"},
    )
    b = get_benchmark_instance("mmmu_art")
    assert isinstance(b, MMMUSingleBenchmark)
    # Ensure the dataset config used correct case from mapping
    assert b.dataset_config == "Art"
    assert b.name == "mmmu_art"


def test_get_benchmark_instance_mmmu_subject_no_configs_raises_runtimeerror(
    monkeypatch,
):
    # Simulate failure to fetch subject list
    monkeypatch.setattr("promptmetrics.registry.get_mmmu_configs", lambda: {})
    with pytest.raises(RuntimeError, match="Could not retrieve MMMU subject list"):
        get_benchmark_instance("mmmu_art")


def test_get_mmmu_configs_success_and_cache(monkeypatch):
    # Reset cache
    registry_mod._MMMU_CONFIGS = None
    # First call: pull from HF infos
    fake_infos = {"Art": object(), "Computer_Science": object()}
    monkeypatch.setattr(registry_mod, "get_dataset_infos", lambda *a, **k: fake_infos)
    mapping = registry_mod.get_mmmu_configs()
    assert mapping == {"art": "Art", "computer_science": "Computer_Science"}

    # Second call should use cache (not call get_dataset_infos)
    monkeypatch.setattr(
        registry_mod,
        "get_dataset_infos",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    mapping2 = registry_mod.get_mmmu_configs()
    assert mapping2 == mapping


def test_get_benchmark_instance_invalid_subject_format(monkeypatch):
    with pytest.raises(ValueError, match="Invalid MMMU subject format.*mmmu_<subject>"):
        get_benchmark_instance("mmmu_")  # missing subject after underscore


def test_get_benchmark_instance_unknown_subject_listed(monkeypatch):
    # Provide an available list that does not contain 'biology'
    monkeypatch.setattr(
        "promptmetrics.registry.get_mmmu_configs",
        lambda: {"art": "Art", "computer_science": "Computer_Science"},
    )
    with pytest.raises(
        ValueError, match="Unknown MMMU subject: 'biology'.*Available subjects:"
    ):
        get_benchmark_instance("mmmu_biology")
