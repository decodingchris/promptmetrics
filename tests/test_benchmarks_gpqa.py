import pytest

from promptmetrics.benchmarks.gpqa import (
    GPQADiamondBenchmark,
    OfficialGPQAEvaluation,
)
from promptmetrics.benchmarks.base import BaseBenchmark
from promptmetrics.registry import BENCHMARK_REGISTRY
from promptmetrics.utils import load_prompt_template


# --- Core Value: GPQA support with deterministic shuffling and prompt formatting ---


def test_gpqa_properties_and_officials():
    b = GPQADiamondBenchmark()
    assert isinstance(b, BaseBenchmark)
    assert b.name == "gpqa_diamond"
    assert b.answer_key == "correct_answer_letter"
    assert b.official_generation_prompt_name == "official_generation_zeroshot_v1"
    # GPQA has no official evaluation prompt
    assert b.official_evaluation_prompt_name is None
    assert b.official_evaluation_model is OfficialGPQAEvaluation
    assert b.is_multimodal is False


def test_registry_contains_gpqa():
    assert "gpqa_diamond" in BENCHMARK_REGISTRY
    assert issubclass(BENCHMARK_REGISTRY["gpqa_diamond"], GPQADiamondBenchmark)


def test_format_prompt_messages_official_generation_template():
    b = GPQADiamondBenchmark()
    content, path, src_type = load_prompt_template(
        "official_generation_zeroshot_v1", "gpqa_diamond", "generation"
    )
    assert src_type == "public"
    assert path.as_posix().endswith(
        "prompts/public/gpqa/generation/official_generation_zeroshot_v1.txt"
    )

    q = {
        "id": "id1",
        "Question": "What is the capital of France?",
        "shuffled_choices": {"A": "Paris", "B": "Lyon", "C": "Marseille", "D": "Nice"},
        "correct_answer_letter": "A",
    }
    msgs = b.format_prompt_messages(q, content)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    text = msgs[0]["content"]
    # Ensure core structure with precise tokens present
    assert "What is the correct answer to this question:" in text
    assert "(A) Paris" in text and "(B) Lyon" in text
    assert "(C) Marseille" in text and "(D) Nice" in text
    assert "{question}" not in text  # formatted


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
    b = GPQADiamondBenchmark()
    parts = []
    if has_system:
        parts.append("---[SYSTEM]---\nFollow GPQA format.")
    if has_user:
        parts.append(
            "---[USER]---\nQ: {question}\nA) {choice_A}\nB) {choice_B}\nC) {choice_C}\nD) {choice_D}"
        )
    prompt_template = "\n".join(parts) if parts else "---[USER]---\n"

    q = {
        "id": "x",
        "Question": "Q?",
        "shuffled_choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "correct_answer_letter": "B",
    }

    msgs = b.format_prompt_messages(q, prompt_template)
    roles = [m["role"] for m in msgs]
    assert roles == expected_roles

    user_msg = next(m for m in msgs if m["role"] == "user")
    content = user_msg["content"]
    # If user section present, formatted values must appear; otherwise it's empty
    if has_user:
        assert "Q: Q?" in content
        assert "A) a" in content and "B) b" in content
        assert "C) c" in content and "D) d" in content
    else:
        assert content == ""


def test_load_data_with_fake_dataset_shuffles_and_ids(monkeypatch):
    b = GPQADiamondBenchmark()

    # Build a minimal fake HF dataset with required interface
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
        {
            "Question": "Which gas is most abundant in Earth's atmosphere?",
            "Correct Answer": "Nitrogen",
            "Incorrect Answer 1": "Oxygen",
            "Incorrect Answer 2": "Carbon Dioxide",
            "Incorrect Answer 3": "Argon",
        },
        {
            "Question": "What is the powerhouse of the cell?",
            "Correct Answer": "Mitochondria",
            "Incorrect Answer 1": "Nucleus",
            "Incorrect Answer 2": "Ribosome",
            "Incorrect Answer 3": "Chloroplast",
        },
    ]

    monkeypatch.setattr(
        "promptmetrics.benchmarks.gpqa.load_dataset", lambda *a, **k: FakeDataset(rows)
    )
    out1 = b.load_data()
    # Each sample should have a stable id and shuffled choices that still map to the correct letter
    for sample in out1:
        assert "id" in sample and len(sample["id"]) == 40  # sha1 hex
        assert set(sample["shuffled_choices"].keys()) == {"A", "B", "C", "D"}
        letter = sample["correct_answer_letter"]
        assert letter in {"A", "B", "C", "D"}
        # The value at that letter equals the original correct answer
        assert sample["shuffled_choices"][letter] == sample["Correct Answer"], (
            "Correct answer letter must point to the true correct option"
        )

    # Determinism: repeated load must produce identical shuffles/ids for same questions
    out2 = b.load_data()
    map1 = {s["id"]: s["shuffled_choices"] for s in out1}
    map2 = {s["id"]: s["shuffled_choices"] for s in out2}
    assert map1 == map2


def test_load_data_selects_by_ids(monkeypatch):
    b = GPQADiamondBenchmark()

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
        {
            "Question": "Q1",
            "Correct Answer": "X",
            "Incorrect Answer 1": "Y1",
            "Incorrect Answer 2": "Y2",
            "Incorrect Answer 3": "Y3",
        },
        {
            "Question": "Q2",
            "Correct Answer": "Z",
            "Incorrect Answer 1": "W1",
            "Incorrect Answer 2": "W2",
            "Incorrect Answer 3": "W3",
        },
    ]

    monkeypatch.setattr(
        "promptmetrics.benchmarks.gpqa.load_dataset", lambda *a, **k: FakeDataset(rows)
    )

    all_samples = b.load_data()
    chosen_id = all_samples[1]["id"]
    filtered = b.load_data(ids_to_load=[chosen_id])
    assert [s["id"] for s in filtered] == [chosen_id]


def test_load_data_limits_by_max_samples(monkeypatch):
    b = GPQADiamondBenchmark()

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

    rows = []
    for i in range(10):
        rows.append(
            {
                "Question": f"Q{i}",
                "Correct Answer": "A",
                "Incorrect Answer 1": "B",
                "Incorrect Answer 2": "C",
                "Incorrect Answer 3": "D",
            }
        )

    monkeypatch.setattr(
        "promptmetrics.benchmarks.gpqa.load_dataset", lambda *a, **k: FakeDataset(rows)
    )
    limited = b.load_data(max_samples=3)
    assert len(limited) == 3


def test_get_size_uses_load_data(monkeypatch):
    b = GPQADiamondBenchmark()
    monkeypatch.setattr(
        GPQADiamondBenchmark,
        "load_data",
        lambda self, max_samples=None, ids_to_load=None: [{"id": "1"}, {"id": "2"}],
    )
    assert b.get_size() == 2
