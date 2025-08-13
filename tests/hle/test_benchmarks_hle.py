import pytest
import types

from promptmetrics.benchmarks.hle import HLEBenchmark, OfficialHLEEvaluation
from promptmetrics.benchmarks.base import BaseBenchmark


# --- Core Value: Multi-modal formatting and official prompt compatibility ---


def test_hle_properties_and_officials():
    b = HLEBenchmark()
    assert isinstance(b, BaseBenchmark)
    assert b.name == "hle"
    assert b.answer_key == "answer"
    assert b.official_generation_prompt_name == "official_generation_v1"
    assert b.official_evaluation_prompt_name == "official_evaluation_v1"
    assert b.official_evaluation_model is OfficialHLEEvaluation
    assert b.is_multimodal is True


@pytest.mark.parametrize(
    "has_system,has_user,image_url,expected_roles,expected_image",
    [
        (True, True, None, ["system", "user"], False),
        (True, True, "https://example.com/img.png", ["system", "user"], True),
        (False, True, "https://example.com/img2.png", ["user"], True),
        (True, False, None, ["system", "user"], False),  # user becomes empty text
        (False, True, None, ["user"], False),
    ],
    ids=[
        "system+user-text-only",
        "system+user-with-image",
        "user-only-with-image",
        "system-only",
        "user-only-text-only",
    ],
)
def test_format_prompt_messages_variants(
    has_system, has_user, image_url, expected_roles, expected_image
):
    b = HLEBenchmark()
    parts = []
    if has_system:
        parts.append("---[SYSTEM]---\nAct like a grader.")
    if has_user:
        parts.append("---[USER]---\n{question}")
    prompt_template = "\n".join(parts) if parts else "---[USER]---\n{question}"

    q = {"id": "x", "question": "What is 2+2?", "answer": "4"}
    if image_url:
        q["image"] = image_url

    msgs = b.format_prompt_messages(q, prompt_template)
    roles = [m["role"] for m in msgs]
    assert roles == expected_roles

    user_msg = next(m for m in msgs if m["role"] == "user")
    content = user_msg["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    if has_user:
        assert "What is 2+2?" in content[0]["text"]
    else:
        # When there is no ---[USER]--- section, the user content is an empty text part
        assert content[0]["text"] == ""

    if expected_image:
        assert any(
            p.get("type") == "image_url" and p["image_url"]["url"] == image_url
            for p in content
        )
    else:
        assert all(p.get("type") != "image_url" for p in content)


def test_get_size_fallback(monkeypatch):
    b = HLEBenchmark()

    # Force get_dataset_infos to raise so we hit the fallback
    monkeypatch.setattr(
        "promptmetrics.benchmarks.hle.get_dataset_infos",
        lambda *a, **k: (_ for _ in ()).throw(KeyError("x")),
    )
    # Avoid network in load_data fallback
    monkeypatch.setattr(
        HLEBenchmark,
        "load_data",
        lambda self, max_samples=None, ids_to_load=None: [{"id": "1"}, {"id": "2"}],
    )
    assert b.get_size() == 2


def test_load_data_selects_by_ids(monkeypatch):
    b = HLEBenchmark()

    class FakeDataset:
        def __init__(self):
            self._data = [
                {"id": "a", "question": "...", "answer": "x"},
                {"id": "b", "question": "...", "answer": "y"},
                {"id": "c", "question": "...", "answer": "z"},
            ]

        def __getitem__(self, key):
            if key == "id":
                return [r["id"] for r in self._data]
            raise KeyError

        def select(self, indices):
            new = FakeDataset()
            new._data = [self._data[i] for i in indices]
            return new

        def __iter__(self):
            return iter(self._data)

    monkeypatch.setattr(
        "promptmetrics.benchmarks.hle.load_dataset", lambda *a, **k: FakeDataset()
    )
    selected = b.load_data(ids_to_load=["c", "a"])
    assert [r["id"] for r in selected] == ["c", "a"]


def test_load_data_limits_by_max_samples(monkeypatch):
    b = HLEBenchmark()

    class FakeDataset:
        def __init__(self):
            self._data = [{"id": str(i)} for i in range(10)]

        def select(self, indices_or_range):
            new = FakeDataset()
            if isinstance(indices_or_range, range):
                new._data = [self._data[i] for i in indices_or_range]
            else:
                new._data = [self._data[i] for i in indices_or_range]
            return new

        def __iter__(self):
            return iter(self._data)

    monkeypatch.setattr(
        "promptmetrics.benchmarks.hle.load_dataset", lambda *a, **k: FakeDataset()
    )
    limited = b.load_data(max_samples=3)
    assert [r["id"] for r in limited] == ["0", "1", "2"]


def test_get_size_quick_path_success(monkeypatch):
    from promptmetrics.benchmarks.hle import HLEBenchmark

    b = HLEBenchmark()
    # get_dataset_infos returns a dict with config -> splits -> split -> num_examples
    fake_infos = {
        b.dataset_config: types.SimpleNamespace(
            splits={b.dataset_split: types.SimpleNamespace(num_examples=123)}
        )
    }
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setattr(
        "promptmetrics.benchmarks.hle.get_dataset_infos", lambda *a, **k: fake_infos
    )
    assert b.get_size() == 123


def test_format_prompt_messages_accepts_pil_image():
    from PIL import Image

    b = HLEBenchmark()
    prompt_template = "---[USER]---\n{question}"
    q = {"id": "x", "question": "What is shown?", "answer": "a"}
    q["image"] = Image.new("RGB", (2, 2), color="red")
    msgs = b.format_prompt_messages(q, prompt_template)
    user_msg = next(m for m in msgs if m["role"] == "user")
    content = user_msg["content"]
    assert any(
        p.get("type") == "image_url"
        and p["image_url"]["url"].startswith("data:image/png;base64,")
        for p in content
    )
