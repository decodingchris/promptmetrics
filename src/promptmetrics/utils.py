"""Utility functions for image encoding and prompt loading."""

from pathlib import Path
from typing import Tuple
import base64
from io import BytesIO
from PIL import Image


def pil_to_base64_url(img: Image.Image, format: str = "PNG") -> str:
    """Convert a Pillow image to a base64 data URL (data:image/<fmt>;base64,...).

    Args:
        img: PIL Image to encode.
        format: Image format for encoding (e.g., 'PNG', 'JPEG').

    Returns:
        A data URL string suitable for image_url payloads.
    """
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def load_prompt_template(
    prompt_source: str, benchmark_name: str, prompt_type: str
) -> Tuple[str, Path, str]:
    """
    Load a prompt template from one of: external file path, prompts/private, prompts/public.

    Args:
        prompt_source: File stem (without .txt) or full file path.
        benchmark_name: Benchmark name to scope the search (handles mmmu_* variants).
        prompt_type: 'generation' or 'evaluation'.

    Returns:
        (prompt_content, resolved_path, source_type) where source_type is
        one of {'private','public','external'}.

    Raises:
        FileNotFoundError: If no prompt is found in any search location.
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
