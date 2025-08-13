# tests/mmmu/test_benchmarks_mmmu.py

import types
from PIL import Image

import pytest

from promptmetrics.benchmarks.base import BaseBenchmark
from promptmetrics.benchmarks.mmmu import (
    MMMUAllBenchmark,
    MMMUSingleBenchmark,
    OfficialMMMU_V1Evaluation,
    _adapt_mmmu_sample,
)
from promptmetrics.utils import load_prompt_template


def test_mmmu_all_properties_and_officials(monkeypatch):
    # Ensure __init__ doesn't try to hit network for configs
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Art", "Physics"],
    )
    b = MMMUAllBenchmark()
    assert isinstance(b, BaseBenchmark)
    assert b.name == "mmmu"
    assert b.answer_key == "answer"
    assert b.official_generation_prompt_name is None
    assert b.official_evaluation_prompt_name is None
    assert b.official_evaluation_model is OfficialMMMU_V1Evaluation
    assert b.is_multimodal is True


def test_mmmu_single_properties_and_officials():
    b = MMMUSingleBenchmark("Art")
    assert isinstance(b, BaseBenchmark)
    # Name normalized to lower and underscores -> dashes
    assert b.name == "mmmu_art"
    assert b.answer_key == "answer"
    assert b.official_generation_prompt_name is None
    assert b.official_evaluation_prompt_name is None
    assert b.official_evaluation_model is OfficialMMMU_V1Evaluation
    assert b.is_multimodal is True


def test_format_prompt_messages_builds_choices_and_strips_image_tokens(monkeypatch):
    # Use single to avoid needing config list
    b = MMMUSingleBenchmark("Art")
    content, _, src_type = load_prompt_template(
        "non_official_generation_v1", "mmmu", "generation"
    )
    assert src_type == "public"
    question = {
        "id": "q1",
        "question": "Look at <image 1> and <image 2>. Which is correct?",
        "parsed_choices": {"A": "alpha", "B": "beta", "C": "gamma"},
        "answer": "B",
    }
    msgs = b.format_prompt_messages(question, content)
    # Prompt has a SYSTEM and a USER section
    assert [m["role"] for m in msgs] == ["system", "user"]
    # System formatting text should be present
    assert "Your response should be in the following format:" in msgs[0]["content"]

    user_content = msgs[1]["content"]
    # Text-only since we didn't add image objects
    assert isinstance(user_content, list) and user_content[0]["type"] == "text"
    text = user_content[0]["text"]
    # Ensure tokens removed and choices present
    assert "<image 1>" not in text and "<image 2>" not in text
    assert "(A) alpha" in text and "(B) beta" in text and "(C) gamma" in text


@pytest.mark.parametrize(
    "has_system,has_user,with_images,expected_roles,expected_image_count",
    [
        (True, True, False, ["system", "user"], 0),
        (False, True, False, ["user"], 0),
        (True, False, False, ["system", "user"], 0),
        (True, True, True, ["system", "user"], 2),
        (False, True, True, ["user"], 2),
    ],
    ids=[
        "system+user-no-images",
        "user-only-no-images",
        "system-only-no-images",
        "system+user-with-images",
        "user-only-with-images",
    ],
)
def test_format_prompt_messages_variants(
    has_system, has_user, with_images, expected_roles, expected_image_count
):
    b = MMMUSingleBenchmark("Physics")
    parts = []
    if has_system:
        parts.append("---[SYSTEM]---\nBehave like MMMU.")
    if has_user:
        parts.append("---[USER]---\nQuestion:\n{question}\n\nChoices:\n{choices_block}")
    prompt_template = "\n".join(parts) if parts else "---[USER]---\n"

    q = {
        "id": "x",
        "question": "Q?",
        "parsed_choices": {"A": "a", "B": "b"},
        "answer": "A",
    }
    if with_images:
        # Attach two small images to image_1 and image_3
        img1 = Image.new("RGB", (2, 2), color="red")
        img3 = Image.new("RGB", (3, 3), color="blue")
        q["image_1"] = img1
        q["image_3"] = img3

    msgs = b.format_prompt_messages(q, prompt_template)
    roles = [m["role"] for m in msgs]
    assert roles == expected_roles

    user_msg = next(m for m in msgs if m["role"] == "user")
    content = user_msg["content"]
    assert isinstance(content, list)
    # First part is text
    assert content[0]["type"] == "text"
    # Image count check
    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert len(image_parts) == expected_image_count
    for part in image_parts:
        url = part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")


def test_get_size_all_configs_aggregates(monkeypatch):
    # Ensure we control the subject list
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Art", "Physics", "Math"],
    )

    # Mock get_dataset_infos to return sizes
    class SplitInfo(types.SimpleNamespace):
        pass

    fake_infos = {
        "Art": types.SimpleNamespace(splits={"validation": SplitInfo(num_examples=10)}),
        "Physics": types.SimpleNamespace(
            splits={"validation": SplitInfo(num_examples=20)}
        ),
        "Math": types.SimpleNamespace(splits={"validation": SplitInfo(num_examples=5)}),
    }
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu.get_dataset_infos", lambda *a, **k: fake_infos
    )

    b = MMMUAllBenchmark()
    assert b.get_size() == 35


def test_load_data_ids_to_load_streaming_optimized(monkeypatch):
    # Provide two configs and three total items across them; select two by id
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Art", "CS"],
    )

    class FakeIterableDataset:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            for r in self._rows:
                yield r

    # Rows per config with options as string to test parsing via ast.literal_eval
    art_rows = [
        {"id": "a1", "question": "Q1", "options": "['optA','optB']", "answer": "A"},
        {"id": "a2", "question": "Q2", "options": "['x','y','z']", "answer": "B"},
    ]
    cs_rows = [
        {"id": "c1", "question": "Q3", "options": "['c','d','e','f']", "answer": "D"}
    ]

    def fake_load_dataset(path, name=None, split=None, streaming=False):
        assert streaming is True
        if name == "Art":
            return FakeIterableDataset(art_rows)
        elif name == "CS":
            return FakeIterableDataset(cs_rows)
        raise AssertionError("Unexpected config")

    monkeypatch.setattr("promptmetrics.benchmarks.mmmu.load_dataset", fake_load_dataset)

    b = MMMUAllBenchmark()
    out = b.load_data(ids_to_load=["c1", "a2"])
    # Order is "as found"; both must be present
    returned_ids = {s["id"] for s in out}
    assert returned_ids == {"c1", "a2"}
    # parsed_choices must exist
    for s in out:
        assert "parsed_choices" in s and isinstance(s["parsed_choices"], dict)
        assert all(
            k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for k in s["parsed_choices"].keys()
        )


def test_mmmu_single_load_data_max_samples_streaming(monkeypatch):
    class FakeIterableDataset:
        def __init__(self, rows):
            self._rows = rows

        def map(self, fn):
            # Apply fn to each row and return new mapped iterable
            mapped = [fn(dict(r)) for r in self._rows]
            return FakeIterableDataset(mapped)

        def __iter__(self):
            for r in self._rows:
                yield r

    rows = [
        {"id": "s1", "question": "?", "options": "['a','b']", "answer": "A"},
        {"id": "s2", "question": "??", "options": "['x','y','z']", "answer": "B"},
        {"id": "s3", "question": "???", "options": "['m','n','o','p']", "answer": "C"},
    ]

    def fake_load_dataset(path, name=None, split=None, streaming=False):
        assert streaming is True
        return FakeIterableDataset(rows)

    monkeypatch.setattr("promptmetrics.benchmarks.mmmu.load_dataset", fake_load_dataset)

    b = MMMUSingleBenchmark("Math")
    out = b.load_data(max_samples=2)
    assert len(out) == 2
    for s in out:
        assert "parsed_choices" in s and isinstance(s["parsed_choices"], dict)


def test_mmmu_all_load_data_max_samples_streaming(monkeypatch):
    # Cover the 'max_samples' streaming branch on MMMUAllBenchmark
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["A", "B"],
    )

    class FakeIterableDataset:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            for r in self._rows:
                yield r

    a_rows = [
        {"id": "a1", "question": "Q1", "options": "['a','b']", "answer": "A"},
        {"id": "a2", "question": "Q2", "options": "['x','y','z']", "answer": "B"},
    ]
    b_rows = [
        {"id": "b1", "question": "Q3", "options": "['m','n','o']", "answer": "C"},
    ]

    def fake_load_dataset(path, name=None, split=None, streaming=False):
        assert streaming is True
        if name == "A":
            return FakeIterableDataset(a_rows)
        elif name == "B":
            return FakeIterableDataset(b_rows)
        raise AssertionError("Unexpected config")

    monkeypatch.setattr("promptmetrics.benchmarks.mmmu.load_dataset", fake_load_dataset)

    b = MMMUAllBenchmark()
    out = b.load_data(max_samples=2)
    assert [s["id"] for s in out] == ["a1", "a2"]
    for s in out:
        assert "parsed_choices" in s and isinstance(s["parsed_choices"], dict)


def test_mmmu_single_ids_to_load_selection(monkeypatch):
    # ids_to_load path on MMMUSingleBenchmark
    class FakeIterableDataset:
        def __init__(self, rows):
            self._rows = rows

        def map(self, fn):
            mapped = [fn(dict(r)) for r in self._rows]
            return FakeIterableDataset(mapped)

        def __iter__(self):
            for r in self._rows:
                yield r

    rows = [
        {"id": "s1", "question": "Q", "options": "['a','b']", "answer": "A"},
        {"id": "s2", "question": "Q2", "options": "['x','y']", "answer": "B"},
    ]

    def fake_load_dataset(path, name=None, split=None, streaming=False):
        assert streaming is True
        return FakeIterableDataset(rows)

    monkeypatch.setattr("promptmetrics.benchmarks.mmmu.load_dataset", fake_load_dataset)

    b = MMMUSingleBenchmark("Art")
    out = b.load_data(ids_to_load=["s2"])
    assert len(out) == 1 and out[0]["id"] == "s2"


def test_adapt_mmmu_sample_parsing_failure():
    sample = {"id": "bad", "question": "Q", "options": "not-a-list", "answer": "A"}
    out = _adapt_mmmu_sample(sample)
    assert out["parsed_choices"] == {}


def test_adapt_mmmu_sample_parsing_failure_no_id():
    # Ensure logging code path doesn't crash when id is missing
    sample = {"question": "Q", "options": "not-a-list", "answer": "A"}
    out = _adapt_mmmu_sample(sample)
    assert out["parsed_choices"] == {}


def test_mmmu_single_ids_to_load_early_stop(monkeypatch):
    # Validate that ids_to_load selection stops early once all IDs are found
    iter_count = {"n": 0}

    class FakeIterableDataset:
        def __init__(self, rows):
            self._rows = rows

        def map(self, fn):
            mapped = [fn(dict(r)) for r in self._rows]
            return FakeIterableDataset(mapped)

        def __iter__(self):
            for r in self._rows:
                iter_count["n"] += 1
                yield r

    # Build a larger stream; the first match is near the beginning
    rows = [
        {"id": f"s{i}", "question": "Q", "options": "['a','b']", "answer": "A"}
        for i in range(1000)
    ]

    def fake_load_dataset(path, name=None, split=None, streaming=False):
        assert streaming is True
        return FakeIterableDataset(rows)

    monkeypatch.setattr("promptmetrics.benchmarks.mmmu.load_dataset", fake_load_dataset)

    b = MMMUSingleBenchmark("Art")
    # Request a single early ID; iterator should break soon after finding it
    out = b.load_data(ids_to_load=["s1"])
    assert [s["id"] for s in out] == ["s1"]
    assert iter_count["n"] < 50  # should not consume the whole stream


def test_mmmu_all_full_run_streaming_no_concatenate(monkeypatch):
    # Ensure deterministic subject list
    monkeypatch.setattr(
        "promptmetrics.benchmarks.mmmu._get_all_mmmu_configs",
        lambda: ["Art", "CS"],
    )

    # Make any attempt to call concatenate_datasets fail to prove we avoided it
    import promptmetrics.benchmarks.mmmu as mmmu_mod

    # If present, force it to fail if called; if absent, that already guarantees it won't be used.
    if hasattr(mmmu_mod, "concatenate_datasets"):
        monkeypatch.setattr(
            mmmu_mod,
            "concatenate_datasets",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("should not use concatenate_datasets")
            ),
        )

    class FakeIterableDataset:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            for r in self._rows:
                yield r

    art_rows = [
        {"id": "a1", "question": "Q1", "options": "['A','B']", "answer": "A"},
    ]
    cs_rows = [
        {"id": "c1", "question": "Q2", "options": "['X','Y','Z']", "answer": "B"},
    ]

    def fake_load_dataset(path, name=None, split=None, streaming=False):
        assert streaming is True
        if name == "Art":
            return FakeIterableDataset(art_rows)
        elif name == "CS":
            return FakeIterableDataset(cs_rows)
        raise AssertionError("Unexpected config")

    monkeypatch.setattr("promptmetrics.benchmarks.mmmu.load_dataset", fake_load_dataset)

    b = MMMUAllBenchmark()
    out = b.load_data()  # full run path (no ids_to_load, no max_samples)
    ids = {s["id"] for s in out}
    assert ids == {"a1", "c1"}
    for s in out:
        assert "parsed_choices" in s and isinstance(s["parsed_choices"], dict)
        # Check choice keys are capital letters
        assert all(
            k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for k in s["parsed_choices"].keys()
        )
