"""AIME 2025 benchmark integration."""

import logging
from typing import Any, Dict, List, Literal, Type

from datasets import load_dataset
from pydantic import BaseModel, Field

from .base import BaseBenchmark, MessageContentType

logger = logging.getLogger(__name__)


class OfficialAIMEEvaluation(BaseModel):
    """Structured evaluation verdict for AIME.

    Fields mirror the expected, constrained output of the evaluation prompt.
    """

    extracted_final_answer: str | None = None
    reasoning: str
    correct: Literal["yes", "no"] | None = None
    confidence: int = Field(ge=0, le=100, default=100)


class AIMEBenchmark(BaseBenchmark):
    """Implementation for the AIME 2025 benchmark (yentinglin/aime_2025)."""

    def __init__(self):
        self._name = "aime_2025"
        self.dataset_name = "yentinglin/aime_2025"
        self.dataset_config = "default"
        self.dataset_split = "train"  # This dataset only has a 'train' split

    @property
    def name(self) -> str:
        return self._name

    @property
    def answer_key(self) -> str:
        return "answer"

    @property
    def official_generation_prompt_name(self) -> str | None:
        # This benchmark does not have an author-provided "official" prompt.
        return None

    @property
    def official_evaluation_prompt_name(self) -> str | None:
        # This benchmark does not have an author-provided "official" evaluation prompt.
        return None

    @property
    def official_evaluation_model(self) -> Type[BaseModel] | None:
        return OfficialAIMEEvaluation

    @property
    def is_multimodal(self) -> bool:
        return False

    def get_size(self) -> int:
        """
        Return total number of samples.

        This dataset is small (30 rows), so loading and counting is inexpensive.
        """
        return len(self.load_data())

    def load_data(
        self,
        max_samples: int | None = None,
        ids_to_load: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Load and adapt AIME samples.

        Adds:
          - question: copied from 'problem'
          - id: stringified original 'id'

        Respects ids_to_load over max_samples.
        """
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.dataset_split,
        )

        def adapt_sample(example: Dict[str, Any]) -> Dict[str, Any]:
            # Create new columns for 'question' and a string-based ID.
            example["question"] = example["problem"]
            example["str_id"] = str(example["id"])
            return example

        # Create 'question' and 'str_id', and drop 'id'/'problem' to avoid confusion.
        dataset = dataset.map(adapt_sample, remove_columns=["id", "problem"])
        # Standardize ID column name.
        dataset = dataset.rename_column("str_id", "id")

        if ids_to_load:
            logger.info(
                "Optimized load: Loading %d specific samples from the benchmark.",
                len(ids_to_load),
            )
            # dataset['id'] is the string ID column created above.
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
        """Render a sample into chat messages using the given template.

        Supports optional ---[SYSTEM]--- and ---[USER]--- sections.
        If no user section is present, an empty user message is still produced.
        """
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

        formatted_user_text = user_template.format(question=question["question"])
        messages.append({"role": "user", "content": formatted_user_text})

        return messages
