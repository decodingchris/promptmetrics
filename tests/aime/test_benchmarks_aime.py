import pytest

from promptmetrics.benchmarks.aime import AIMEBenchmark, OfficialAIMEEvaluation
from promptmetrics.benchmarks.base import BaseBenchmark
from promptmetrics.registry import BENCHMARK_REGISTRY


# --- Core Value: AIME support with correct dataset adaptation and prompt formatting ---


def test_aime_properties_and_officials():
    b = AIMEBenchmark()
    assert isinstance(b, BaseBenchmark)
    assert b.name == "aime_2025"
    assert b.answer_key == "answer"
    # AIME has no official generation/evaluation prompt files
    assert b.official_generation_prompt_name is None
    assert b.official_evaluation_prompt_name is None
    assert b.official_evaluation_model is OfficialAIMEEvaluation
    assert b.is_multimodal is False


def test_registry_contains_aime():
    assert "aime_2025" in BENCHMARK_REGISTRY
    assert issubclass(BENCHMARK_REGISTRY["aime_2025"], AIMEBenchmark)


def test_get_size_uses_load_data(monkeypatch):
    b = AIMEBenchmark()
    monkeypatch.setattr(
        AIMEBenchmark,
        "load_data",
        lambda self, max_samples=None, ids_to_load=None: [
            {"id": "1"},
            {"id": "2"},
            {"id": "3"},
        ],
    )
    assert b.get_size() == 3


@pytest.mark.parametrize(
    "has_system,has_user,expected_roles",
    [
        (True, True, ["system", "user"]),
        (False, True, ["user"]),
        (True, False, ["system", "user"]),  # user becomes empty string
        (False, False, ["user"]),  # default to user-only with empty content
    ],
    ids=[
        "system+user",
        "user-only",
        "system-only",
        "neither-system-nor-user",
    ],
)
def test_format_prompt_messages_variants(has_system, has_user, expected_roles):
    b = AIMEBenchmark()
    parts = []
    if has_system:
        parts.append("---[SYSTEM]---\nBehave like a top math contestant.")
    if has_user:
        parts.append("---[USER]---\nQuestion:\n{question}")
    else:
        # Force an empty user section if none provided
        parts.append("---[USER]---\n")

    prompt_template = "\n".join(parts)

    q = {
        "id": "a1",
        "question": "Compute 1+1.",
        "answer": "2",
    }

    msgs = b.format_prompt_messages(q, prompt_template)
    roles = [m["role"] for m in msgs]
    assert roles == expected_roles

    user_msg = next(m for m in msgs if m["role"] == "user")
    content = user_msg["content"]
    if has_user:
        assert "Compute 1+1." in content
        assert "{question}" not in content
    else:
        # When there is no real user section beyond the empty marker
        assert content == ""


def test_load_data_maps_and_renames_and_limits(monkeypatch):
    b = AIMEBenchmark()

    class FakeDataset:
        def __init__(self, rows):
            self._data = rows

        def map(self, fn, remove_columns=None):
            mapped = [fn(dict(r)) for r in self._data]
            if remove_columns:
                for r in mapped:
                    for col in remove_columns:
                        if col in r:
                            del r[col]
            return FakeDataset(mapped)

        def rename_column(self, old, new):
            renamed = []
            for r in self._data:
                if old in r:
                    r[new] = r.pop(old)
                renamed.append(r)
            return FakeDataset(renamed)

        def __getitem__(self, key):
            if key == "id":
                return [r["id"] for r in self._data]
            raise KeyError(key)

        def select(self, indices_or_range):
            if isinstance(indices_or_range, range):
                idxs = list(indices_or_range)
            else:
                idxs = list(indices_or_range)
            return FakeDataset([self._data[i] for i in idxs])

        def __iter__(self):
            return iter(self._data)

    rows = [
        {"id": 101, "problem": "What is 2+2?", "answer": "4"},
        {"id": 102, "problem": "Compute 3*3.", "answer": "9"},
        {"id": 103, "problem": "Find 10-7.", "answer": "3"},
    ]

    monkeypatch.setattr(
        "promptmetrics.benchmarks.aime.load_dataset", lambda *a, **k: FakeDataset(rows)
    )

    limited = b.load_data(max_samples=2)
    assert len(limited) == 2
    for sample in limited:
        # Adaptation assertions
        assert sample["question"] in {"What is 2+2?", "Compute 3*3."}
        assert "answer" in sample
        # Ids should be strings of original id
        assert isinstance(sample["id"], str)
        assert sample["id"] in {"101", "102", "103"}
        # Original 'problem' and numeric 'id' should have been removed by map(remove_columns=...)
        assert "problem" not in sample


def test_load_data_selects_by_ids(monkeypatch):
    b = AIMEBenchmark()

    class FakeDataset:
        def __init__(self, rows):
            self._data = rows

        def map(self, fn, remove_columns=None):
            mapped = [fn(dict(r)) for r in self._data]
            if remove_columns:
                for r in mapped:
                    for col in remove_columns:
                        r.pop(col, None)
            return FakeDataset(mapped)

        def rename_column(self, old, new):
            renamed = []
            for r in self._data:
                if old in r:
                    r[new] = r.pop(old)
                renamed.append(r)
            return FakeDataset(renamed)

        def __getitem__(self, key):
            if key == "id":
                return [r["id"] for r in self._data]
            raise KeyError(key)

        def select(self, indices):
            return FakeDataset([self._data[i] for i in indices])

        def __iter__(self):
            return iter(self._data)

    rows = [
        {"id": 201, "problem": "Q1", "answer": "A1"},
        {"id": 202, "problem": "Q2", "answer": "A2"},
        {"id": 203, "problem": "Q3", "answer": "A3"},
    ]
    monkeypatch.setattr(
        "promptmetrics.benchmarks.aime.load_dataset", lambda *a, **k: FakeDataset(rows)
    )
    all_samples = b.load_data()
    chosen = [s["id"] for s in all_samples][1:]
    filtered = b.load_data(ids_to_load=chosen)
    assert [s["id"] for s in filtered] == chosen
