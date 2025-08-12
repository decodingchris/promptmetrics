# src/promptmetrics/benchmarks/mmmu.py

import logging
import re
import ast
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Type
from pydantic import BaseModel, Field
from datasets import load_dataset, get_dataset_infos, concatenate_datasets
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image

from .base import BaseBenchmark, MessageContentType
from ..utils import pil_to_base64_url

logger = logging.getLogger(__name__)


class OfficialMMMU_V1Evaluation(BaseModel):
    """
    Pydantic model for the structured output of the MMMU evaluation prompt.
    Supports up to 10 choices (A-J).
    """

    extracted_answer_choice: (
        Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] | None
    ) = None
    reasoning: str
    correct: Literal["yes", "no"] | None = None
    confidence: int = Field(ge=0, le=100, default=100)


def _get_all_mmmu_configs() -> List[str]:
    """Helper to get all valid MMMU configuration names from the Hub, with local caching."""
    cache_dir = Path.home() / ".cache" / "promptmetrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "mmmu_configs.json"

    if cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text())
            if time.time() - cache_data.get("timestamp", 0) < 86400:  # 24-hour TTL
                return cache_data.get("configs", [])
        except json.JSONDecodeError:
            logger.warning("Could not decode MMMU configs cache. Refetching.")

    try:
        infos = get_dataset_infos("MMMU/MMMU")
        configs = list(infos.keys())
        cache_payload = {"timestamp": time.time(), "configs": configs}
        cache_file.write_text(json.dumps(cache_payload, indent=2))
        return configs
    except Exception as e:
        logger.error(f"Could not fetch MMMU configs from Hugging Face Hub: {e}")
        return []


def _adapt_mmmu_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """Shared function to parse and adapt a single MMMU sample."""
    if isinstance(example.get("options"), str):
        try:
            options_list = ast.literal_eval(example["options"])
            example["parsed_choices"] = {
                chr(65 + i): opt for i, opt in enumerate(options_list)
            }
        except (ValueError, SyntaxError):
            logger.warning(f"Could not parse options for sample {example['id']}")
            example["parsed_choices"] = {}
    return example


class MMMUBaseBenchmark(BaseBenchmark):
    """
    A base class for MMMU benchmarks containing shared logic for formatting and properties.
    """

    dataset_name = "MMMU/MMMU"
    dataset_split = "validation"  # Per official guidelines for public evaluation

    @property
    def answer_key(self) -> str:
        return "answer"

    @property
    def official_generation_prompt_name(self) -> str | None:
        return None  # No official prompt file

    @property
    def official_evaluation_prompt_name(self) -> str | None:
        return None  # No official LLM evaluation prompt

    @property
    def official_evaluation_model(self) -> Type[BaseModel] | None:
        return OfficialMMMU_V1Evaluation

    @property
    def is_multimodal(self) -> bool:
        return True

    def format_prompt_messages(
        self, question: Dict[str, Any], prompt_template: str
    ) -> List[Dict[str, MessageContentType]]:
        # Dynamically build the choices block for the prompt.
        choices_list = []
        if "parsed_choices" in question:
            for letter, text in sorted(question["parsed_choices"].items()):
                choices_list.append(f"({letter}) {text}")
        choices_block = "\n".join(choices_list)

        format_dict: Dict[str, Any] = {
            "question": re.sub(r"<image \d+>", "", question["question"]).strip(),
            "choices_block": choices_block,
        }
        # Also add individual choices for evaluation prompts that might need them.
        if "parsed_choices" in question:
            for key, value in question["parsed_choices"].items():
                format_dict[f"choice_{key}"] = value

        messages: List[Dict[str, Any]] = []
        system_content = None
        user_template = prompt_template

        if "---[SYSTEM]---" in prompt_template:
            parts = prompt_template.split("---[SYSTEM]---", 1)[1].split(
                "---[USER]---", 1
            )
            system_content = parts[0].strip()
            user_template = parts[1].strip() if len(parts) > 1 else ""
        elif "---[USER]---" in prompt_template:
            user_template = prompt_template.split("---[USER]---", 1)[1].strip()

        if system_content:
            messages.append({"role": "system", "content": system_content})

        formatted_user_text = user_template.format_map(format_dict)
        user_content_list: List[Dict[str, Any]] = [
            {"type": "text", "text": formatted_user_text}
        ]

        for i in range(1, 11):
            image_field = f"image_{i}"
            if image_field in question and isinstance(
                question[image_field], Image.Image
            ):
                image_data_url = pil_to_base64_url(question[image_field])
                user_content_list.append(
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                )

        messages.append({"role": "user", "content": user_content_list})
        return messages


class MMMUAllBenchmark(MMMUBaseBenchmark):
    """
    Implementation for the full MMMU benchmark, combining all 30 subjects.
    This is triggered by using the benchmark name 'mmmu'.
    """

    def __init__(self):
        self._name = "mmmu"
        self.all_configs = _get_all_mmmu_configs()

    @property
    def name(self) -> str:
        return self._name

    def get_size(self) -> int:
        """Efficiently gets the total number of samples across all subjects."""
        try:
            infos = get_dataset_infos(self.dataset_name)
            return sum(
                infos[config].splits[self.dataset_split].num_examples
                for config in self.all_configs
            )
        except (HfHubHTTPError, KeyError, AttributeError) as e:
            logger.warning(
                "Could not quickly fetch total dataset size due to a '%s' error. "
                "Falling back to a full data load to determine the size.",
                type(e).__name__,
            )
            return len(self.load_data())

    def load_data(
        self,
        max_samples: int | None = None,
        ids_to_load: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.all_configs:
            raise RuntimeError(
                "Could not retrieve MMMU subject list. Check network connection."
            )

        # --- OPTIMIZATION for ids_to_load (used by pm-evaluate) ---
        if ids_to_load:
            logger.info(
                f"Lazily loading {len(ids_to_load)} specific samples for MMMU..."
            )
            samples = []
            ids_set = set(ids_to_load)
            for config in self.all_configs:
                if not ids_set:
                    break  # Early exit if all IDs have been found
                dataset_slice = load_dataset(
                    self.dataset_name,
                    name=config,
                    split=self.dataset_split,
                    streaming=True,
                )
                for sample in dataset_slice:
                    if sample["id"] in ids_set:
                        samples.append(_adapt_mmmu_sample(sample))
                        ids_set.remove(sample["id"])
                        if not ids_set:
                            break
            return samples
        # --- END OPTIMIZATION ---

        # --- OPTIMIZATION for max_samples (used by pm-generate) ---
        if max_samples:
            logger.info(
                f"Lazily loading up to {max_samples} samples for MMMU benchmark..."
            )
            samples = []
            for config in self.all_configs:
                if len(samples) >= max_samples:
                    break
                dataset_slice = load_dataset(
                    self.dataset_name,
                    name=config,
                    split=self.dataset_split,
                    streaming=True,
                )
                for sample in dataset_slice:
                    if len(samples) >= max_samples:
                        break
                    samples.append(_adapt_mmmu_sample(sample))
            return samples
        # --- END OPTIMIZATION ---

        # Original logic for a full run
        logger.info(
            f"Streaming all {len(self.all_configs)} subjects for a full MMMU run..."
        )
        all_datasets = [
            load_dataset(
                self.dataset_name, name=config, split=self.dataset_split, streaming=True
            )
            for config in self.all_configs
        ]
        dataset = concatenate_datasets(all_datasets)
        dataset = dataset.map(_adapt_mmmu_sample)
        return list(dataset)


class MMMUSingleBenchmark(MMMUBaseBenchmark):
    """
    Implementation for a single subject of the MMMU benchmark.
    This is triggered by using a benchmark name like 'mmmu_art'.
    """

    def __init__(self, config_name: str):
        self.dataset_config = config_name
        self._name = f"mmmu_{self.dataset_config.lower().replace('_', '-')}"

    @property
    def name(self) -> str:
        return self._name

    def get_size(self) -> int:
        """Efficiently gets the total number of samples in the dataset split."""
        try:
            infos = get_dataset_infos(self.dataset_name)
            return infos[self.dataset_config].splits[self.dataset_split].num_examples
        except (HfHubHTTPError, KeyError, AttributeError) as e:
            logger.warning(
                "Could not quickly fetch dataset size due to a '%s' error. "
                "Falling back to a full data load to determine the size.",
                type(e).__name__,
            )
            return len(self.load_data())

    def load_data(
        self,
        max_samples: int | None = None,
        ids_to_load: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        logger.info(f"Streaming subject '{self.dataset_config}' for MMMU benchmark...")
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.dataset_split,
            streaming=True,
        )
        dataset = dataset.map(_adapt_mmmu_sample)

        # ids_to_load is much less common here but supported for consistency
        if ids_to_load:
            samples = []
            ids_set = set(ids_to_load)
            for sample in dataset:
                if sample["id"] in ids_set:
                    samples.append(sample)
            return samples

        if max_samples:
            return list(itertools.islice(dataset, max_samples))

        return list(dataset)
