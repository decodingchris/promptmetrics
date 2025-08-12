# tests/mmmu/test_mmmu_configs_cache.py

import json
import time

from promptmetrics.benchmarks import mmmu as mmmu_mod


def test_get_all_mmmu_configs_reads_cache_within_ttl(monkeypatch, tmp_path):
    # Redirect cache directory to tmp
    monkeypatch.setattr(mmmu_mod.Path, "home", lambda: tmp_path)
    cache_dir = tmp_path / ".cache" / "promptmetrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "mmmu_configs.json"
    cache_payload = {"timestamp": time.time(), "configs": ["A", "B"]}
    cache_file.write_text(json.dumps(cache_payload), encoding="utf-8")

    # If cache is used, we should not need get_dataset_infos; force it to raise if called
    monkeypatch.setattr(
        mmmu_mod,
        "get_dataset_infos",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    out = mmmu_mod._get_all_mmmu_configs()
    assert out == ["A", "B"]


def test_get_all_mmmu_configs_cache_expired_refetches(monkeypatch, tmp_path):
    monkeypatch.setattr(mmmu_mod.Path, "home", lambda: tmp_path)
    cache_dir = tmp_path / ".cache" / "promptmetrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "mmmu_configs.json"
    old_ts = time.time() - 90000  # > 24h
    cache_file.write_text(
        json.dumps({"timestamp": old_ts, "configs": ["Old"]}), encoding="utf-8"
    )

    # Provide fresh data via Hub
    fake_infos = {"Art": object(), "Physics": object()}
    monkeypatch.setattr(mmmu_mod, "get_dataset_infos", lambda *a, **k: fake_infos)
    out = mmmu_mod._get_all_mmmu_configs()
    assert sorted(out) == ["Art", "Physics"]

    # Verify cache overwritten
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "timestamp" in data and sorted(data["configs"]) == ["Art", "Physics"]


def test_get_all_mmmu_configs_corrupt_cache_and_hub_error(monkeypatch, tmp_path):
    monkeypatch.setattr(mmmu_mod.Path, "home", lambda: tmp_path)
    cache_dir = tmp_path / ".cache" / "promptmetrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "mmmu_configs.json"
    cache_file.write_text("{not-json", encoding="utf-8")

    # Make fetch fail
    monkeypatch.setattr(
        mmmu_mod,
        "get_dataset_infos",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("hub down")),
    )
    out = mmmu_mod._get_all_mmmu_configs()
    assert out == []
