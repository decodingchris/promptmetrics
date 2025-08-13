# src/promptmetrics/utils.py

from pathlib import Path
from typing import Tuple
import base64
from io import BytesIO
from PIL import Image


def pil_to_base64_url(img: Image.Image, format: str = "PNG") -> str:
    """Converts a Pillow image to a base64 data URL."""
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def load_prompt_template(
    prompt_source: str, benchmark_name: str, prompt_type: str
) -> Tuple[str, Path, str]:
    """
    Loads a prompt template from a file, a public, or a private directory.

    Args:
        prompt_source: Name of the prompt file, path to a custom prompt file.
        benchmark_name: The name of the benchmark to scope the search.
        prompt_type: The type of prompt, either 'generation' or 'evaluation'.

    Returns:
        A tuple containing the prompt content, its path, and its source type.

    Raises:
        FileNotFoundError: If the prompt cannot be found in any search location.
    """
    if Path(prompt_source).is_file():
        path = Path(prompt_source)
        return path.read_text(encoding="utf-8"), path, "external"

    benchmark_base_name = benchmark_name
    if benchmark_name.startswith("mmmu_"):
        benchmark_base_name = "mmmu"
    elif benchmark_name == "aime_2025":
        benchmark_base_name = "aime"
    elif benchmark_name == "gpqa_diamond":
        benchmark_base_name = "gpqa"

    prompt_name_with_ext = f"{prompt_source}.txt"
    private_path = (
        Path("prompts")
        / "private"
        / benchmark_base_name
        / prompt_type
        / prompt_name_with_ext
    )
    if private_path.exists():
        return private_path.read_text(encoding="utf-8"), private_path, "private"

    public_path = (
        Path("prompts")
        / "public"
        / benchmark_base_name
        / prompt_type
        / prompt_name_with_ext
    )
    if public_path.exists():
        return public_path.read_text(encoding="utf-8"), public_path, "public"

    raise FileNotFoundError(
        f"{prompt_type.capitalize()} prompt '{prompt_source}' not found as a file or in any of "
        f"the {prompt_type} search paths."
    )
