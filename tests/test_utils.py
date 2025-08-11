import os
from pathlib import Path
import pytest

from promptmetrics.utils import load_prompt_template


# --- Core Value: Reliable prompt loading with precedence and useful errors ---

def test_load_prompt_template_should_load_public_generation_official():
    content, path, source_type = load_prompt_template(
        "official_generation_v1", "hle", "generation"
    )
    assert "Your response should be in the following format:" in content
    assert path.as_posix().endswith("prompts/public/hle/generation/official_generation_v1.txt")
    assert source_type == "public"


def test_load_prompt_template_should_load_public_evaluation_official():
    content, path, source_type = load_prompt_template(
        "official_evaluation_v1", "hle", "evaluation"
    )
    assert "Evaluate whether the following [response] to [question]" in content
    assert path.as_posix().endswith("prompts/public/hle/evaluation/official_evaluation_v1.txt")
    assert source_type == "public"


def test_load_prompt_template_should_prefer_private_over_public(tmp_path, monkeypatch):
    # Create a private prompt with the same name in the private path to ensure precedence
    private_dir = Path("prompts") / "private" / "hle" / "generation"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_file = private_dir / "my_private_prompt.txt"
    private_file.write_text("PRIVATE CONTENT", encoding="utf-8")

    content, path, source_type = load_prompt_template("my_private_prompt", "hle", "generation")
    try:
        assert content == "PRIVATE CONTENT"
        assert path == private_file
        assert source_type == "private"
    finally:
        # Cleanup
        private_file.unlink(missing_ok=True)


def test_load_prompt_template_should_accept_external_file(tmp_path):
    external = tmp_path / "external_prompt.txt"
    external.write_text("EXTERNAL PROMPT", encoding="utf-8")
    content, path, source_type = load_prompt_template(str(external), "hle", "generation")
    assert content == "EXTERNAL PROMPT"
    assert path == external
    assert source_type == "external"


@pytest.mark.parametrize(
    "prompt_source,benchmark_name,prompt_type,expected_fragment",
    [
        ("does_not_exist", "hle", "generation", "Generation prompt 'does_not_exist' not found"),
        ("does_not_exist", "hle", "evaluation", "Evaluation prompt 'does_not_exist' not found"),
    ],
    ids=["missing-generation", "missing-evaluation"],
)
def test_load_prompt_template_should_raise_on_missing(prompt_source, benchmark_name, prompt_type, expected_fragment):
    with pytest.raises(FileNotFoundError) as ei:
        load_prompt_template(prompt_source, benchmark_name, prompt_type)
    assert expected_fragment in str(ei.value)