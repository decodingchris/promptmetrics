import os
import json
import asyncio
import time
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch, tmp_path):
    # Ensure API key exists for any code path that creates OpenRouterLLM
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    # Avoid writing cache to the real home directory
    monkeypatch.chdir(Path.cwd())
    yield


@pytest.fixture
def tmp_output_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def sample_questions():
    # realistic multi-modal and text-only question samples
    return [
        {
            "id": "q1",
            "question": "What is shown in the image?",
            "image": "https://example.com/img1.png",
            "answer": "a cat",
        },
        {
            "id": "q2",
            "question": "Compute 2 + 2.",
            "answer": "4",
        },
    ]


@pytest.fixture
def generations_sample_artifact(tmp_output_dir, sample_questions):
    # Create a small generations file similar to pm-generate output
    artifact_dir = tmp_output_dir / "results" / "hle" / "test-model" / "public-official_generation_v1" / "generations"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "20250101T000000Z_generations.json"

    gens = {
        "metadata": {
            "generation": {
                "model": "test-model",
                "reasoning_model": False,
                "benchmark": "hle",
                "prompt_source": "official_generation_v1",
                "prompt_source_type": "public",
                "prompt_file": "prompts/public/hle/generation/official_generation_v1.txt",
                "temperature": 0.0,
                "max_tokens": 128,
                "generated_at_utc": "20250101T000000Z",
            }
        },
        "generations": {
            sample_questions[0]["id"]: {"response": "Explanation: ... Answer: a cat\nConfidence: 95"},
            sample_questions[1]["id"]: {"response": "Explanation: ... Answer: 5\nConfidence: 55"},
        },
    }
    artifact_path.write_text(json.dumps(gens, indent=2), encoding="utf-8")
    return artifact_path


@pytest.fixture
def restore_cwd():
    cwd = Path.cwd()
    yield
    os.chdir(cwd)


@pytest.fixture(scope="session")
def event_loop():
    # Ensure pytest-asyncio compatibility with session scope
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()