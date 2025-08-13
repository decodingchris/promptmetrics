import hashlib
import logging
from typing import Any, Dict, List, Literal, Type

from datasets import get_dataset_infos, load_dataset
from pydantic import BaseModel, Field

from .base import BaseBenchmark, MessageContentType

logger = logging.getLogger(__name__)


class OfficialFACTSEvaluation(BaseModel):
    reasoning: str
    correct: Literal["yes", "no"] | None = None
    confidence: int = Field(ge=0, le=100, default=100)


class FACTSBenchmark(BaseBenchmark):
    def __init__(self):
        self._name = "facts"
        self.dataset_name = "google/FACTS-grounding-public"
        self.dataset_config = "examples"
        self.dataset_split = "public"

    @property
    def name(self) -> str:
        return self._name

    @property
    def answer_key(self) -> str:
        return "context_document"

    @property
    def official_generation_prompt_name(self) -> str | None:
        return "official_generation_v1"

    @property
    def official_evaluation_prompt_name(self) -> str | None:
        return None

    @property
    def official_evaluation_model(self) -> Type[BaseModel] | None:
        return OfficialFACTSEvaluation

    @property
    def is_multimodal(self) -> bool:
        return False

    def get_size(self) -> int:
        try:
            infos = get_dataset_infos(self.dataset_name)
            return infos[self.dataset_config].splits[self.dataset_split].num_examples
        except (KeyError, AttributeError) as e:
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
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.dataset_split,
        )

        def add_id(example: Dict[str, Any]) -> Dict[str, Any]:
            identifier_string = example["user_request"]
            example["id"] = hashlib.sha1(identifier_string.encode("utf-8")).hexdigest()
            return example

        dataset = dataset.map(add_id)

        if ids_to_load:
            logger.info(
                "Optimized load: Loading %d specific samples from the benchmark.",
                len(ids_to_load),
            )
            id_to_index = {id_val: i for i, id_val in enumerate(dataset["id"])}
            indices_to_load = [
                id_to_index[id_val] for id_val in ids_to_load if id_val in id_to_index
            ]
            dataset = dataset.select(indices_to_load)
        elif max_samples:
            dataset = dataset.select(range(max_samples))

        return [sample for sample in dataset]

    def format_prompt_messages(
        self, question: Dict[str, Any], prompt_template: str
    ) -> List[Dict[str, MessageContentType]]:
        format_dict = {
            "system_instruction": question["system_instruction"],
            "user_request": question["user_request"],
            "context_document": question["context_document"],
        }

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
            formatted_system_text = system_content.format_map(format_dict)
            messages.append({"role": "system", "content": formatted_system_text})

        formatted_user_text = user_template.format_map(format_dict)
        messages.append({"role": "user", "content": formatted_user_text})

        return messages
