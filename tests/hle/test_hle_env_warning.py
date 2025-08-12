# tests/hle/test_hle_env_warning.py

import logging

from promptmetrics.benchmarks.hle import HLEBenchmark


def test_hle_load_data_warns_when_hf_token_missing(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    # Ensure HF_TOKEN is not set
    monkeypatch.delenv("HF_TOKEN", raising=False)

    class FakeDataset:
        def __init__(self):
            self._data = [{"id": "x", "question": "q", "answer": "a"}]

        def __getitem__(self, key):
            if key == "id":
                return [r["id"] for r in self._data]
            raise KeyError

        def select(self, indices_or_range):
            return self

        def __iter__(self):
            return iter(self._data)

    monkeypatch.setattr(
        "promptmetrics.benchmarks.hle.load_dataset", lambda *a, **k: FakeDataset()
    )

    b = HLEBenchmark()
    out = b.load_data(max_samples=1)
    assert [r["id"] for r in out] == ["x"]
    # Warning should be logged due to missing HF_TOKEN
    assert any("HF_TOKEN not set" in rec.message for rec in caplog.records)
