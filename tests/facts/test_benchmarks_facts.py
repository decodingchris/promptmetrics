import hashlib
import types

import pytest

from promptmetrics.benchmarks.base import BaseBenchmark
from promptmetrics.benchmarks.facts import FACTSBenchmark, OfficialFACTSEvaluation
from promptmetrics.registry import BENCHMARK_REGISTRY
from promptmetrics.utils import load_prompt_template


def test_facts_properties_and_officials():
    b = FACTSBenchmark()
    assert isinstance(b, BaseBenchmark)
    assert b.name == "facts"
    assert b.answer_key == "context_document"
    assert b.official_generation_prompt_name == "official_generation_v1"
    assert b.official_evaluation_prompt_name is None
    assert b.official_evaluation_model is OfficialFACTSEvaluation
    assert b.is_multimodal is False


def test_registry_contains_facts():
    assert "facts" in BENCHMARK_REGISTRY
    assert issubclass(BENCHMARK_REGISTRY["facts"], FACTSBenchmark)


def test_get_size_quick_path_success(monkeypatch):
    b = FACTSBenchmark()
    fake_infos = {
        b.dataset_config: types.SimpleNamespace(
            splits={b.dataset_split: types.SimpleNamespace(num_examples=321)}
        )
    }
    monkeypatch.setattr(
        "promptmetrics.benchmarks.facts.get_dataset_infos", lambda *a, **k: fake_infos
    )
    assert b.get_size() == 321


def test_get_size_fallback(monkeypatch):
    b = FACTSBenchmark()
    monkeypatch.setattr(
        "promptmetrics.benchmarks.facts.get_dataset_infos",
        lambda *a, **k: (_ for _ in ()).throw(KeyError("x")),
    )
    monkeypatch.setattr(
        FACTSBenchmark,
        "load_data",
        lambda self, max_samples=None, ids_to_load=None: [{"id": "a"}, {"id": "b"}],
    )
    assert b.get_size() == 2


def test_load_data_maps_adds_id_and_limits(monkeypatch):
    b = FACTSBenchmark()

    class FakeDataset:
        def __init__(self, rows):
            self._data = rows

        def map(self, fn):
            mapped = [fn(dict(r)) for r in self._data]
            return FakeDataset(mapped)

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
        {
            "system_instruction": "Only use the document",
            "user_request": "Who wrote 'The Republic'?",
            "context_document": "Plato wrote The Republic.",
        },
        {
            "system_instruction": "Only use the document",
            "user_request": "When was the UN founded?",
            "context_document": "The United Nations was founded in 1945.",
        },
        {
            "system_instruction": "Only use the document",
            "user_request": "Capital of Japan?",
            "context_document": "Tokyo is the capital of Japan.",
        },
    ]
    monkeypatch.setattr(
        "promptmetrics.benchmarks.facts.load_dataset", lambda *a, **k: FakeDataset(rows)
    )
    out = b.load_data(max_samples=2)
    assert len(out) == 2
    for sample in out:
        assert "id" in sample and len(sample["id"]) == 40
        expected_id = hashlib.sha1(sample["user_request"].encode("utf-8")).hexdigest()
        assert sample["id"] == expected_id
        assert "system_instruction" in sample
        assert "user_request" in sample
        assert "context_document" in sample


def test_load_data_selects_by_ids(monkeypatch):
    b = FACTSBenchmark()

    class FakeDataset:
        def __init__(self, rows):
            self._data = rows

        def map(self, fn):
            mapped = [fn(dict(r)) for r in self._data]
            return FakeDataset(mapped)

        def __getitem__(self, key):
            if key == "id":
                return [r["id"] for r in self._data]
            raise KeyError(key)

        def select(self, indices):
            return FakeDataset([self._data[i] for i in indices])

        def __iter__(self):
            return iter(self._data)

    rows = [
        {"user_request": "Q1"},
        {"user_request": "Q2"},
        {"user_request": "Q3"},
    ]
    monkeypatch.setattr(
        "promptmetrics.benchmarks.facts.load_dataset", lambda *a, **k: FakeDataset(rows)
    )
    all_samples = b.load_data()
    chosen = [all_samples[2]["id"], all_samples[0]["id"]]
    filtered = b.load_data(ids_to_load=chosen)
    assert [s["id"] for s in filtered] == chosen


@pytest.mark.parametrize(
    "has_system,has_user,expected_roles,expect_user_empty",
    [
        (True, True, ["system", "user"], False),
        (False, True, ["user"], False),
        (True, False, ["system", "user"], True),
        (False, False, ["user"], True),
    ],
    ids=["system+user", "user-only", "system-only", "neither-system-nor-user"],
)
def test_format_prompt_messages_variants(
    has_system, has_user, expected_roles, expect_user_empty
):
    b = FACTSBenchmark()
    parts = []
    if has_system:
        parts.append("---[SYSTEM]---\nPolicy: {system_instruction}")
    if has_user:
        parts.append("---[USER]---\nRequest: {user_request}\nDoc:\n{context_document}")
    else:
        parts.append("---[USER]---\n")
    prompt_template = "\n".join(parts)

    q = {
        "id": "x",
        "system_instruction": "Use only doc",
        "user_request": "Who wrote 'The Republic'?",
        "context_document": "Plato wrote The Republic.",
    }

    msgs = b.format_prompt_messages(q, prompt_template)
    roles = [m["role"] for m in msgs]
    assert roles == expected_roles

    user_msg = next(m for m in msgs if m["role"] == "user")
    content = user_msg["content"]

    if expect_user_empty:
        assert content == ""
    else:
        assert "Request: Who wrote 'The Republic'?" in content
        assert "Doc:" in content
        assert "Plato wrote The Republic." in content
        assert "{user_request}" not in content
        assert "{context_document}" not in content

    if has_system:
        sys_msg = next(m for m in msgs if m["role"] == "system")
        assert sys_msg["content"] == "Policy: Use only doc"


def test_format_prompt_messages_with_official_generation_template():
    b = FACTSBenchmark()
    content, path, src_type = load_prompt_template(
        "official_generation_v1", "facts", "generation"
    )
    assert src_type == "public"
    assert path.as_posix().endswith(
        "prompts/public/facts/generation/official_generation_v1.txt"
    )

    q = {
        "id": "id1",
        "system_instruction": "Answer strictly from the document.",
        "user_request": "Summarize the doc.",
        "context_document": "The quick brown fox jumps over the lazy dog.",
    }
    msgs = b.format_prompt_messages(q, content)
    assert [m["role"] for m in msgs] == ["system", "user"]

    sys_txt = msgs[0]["content"]
    user_txt = msgs[1]["content"]
    assert sys_txt == "Answer strictly from the document."
    assert "[USER REQUEST]:" in user_txt and "Summarize the doc." in user_txt
    assert "[CONTEXT DOCUMENT]:" in user_txt and "quick brown fox" in user_txt
