import hashlib
import logging
import random
from typing import Any, Dict, List, Literal, Type

from datasets import load_dataset
from pydantic import BaseModel, Field

from .base import BaseBenchmark, MessageContentType

logger = logging.getLogger(__name__)


class OfficialGPQAEvaluation(BaseModel):
    """Pydantic model for the structured output of the GPQA evaluation prompt."""

    extracted_answer_choice: Literal["A", "B", "C", "D"] | None = None
    reasoning: str
    correct: Literal["yes", "no"] | None = None
    confidence: int = Field(ge=0, le=100, default=100)


class GPQADiamondBenchmark(BaseBenchmark):
    """Implementation for the GPQA Diamond Benchmark."""

    def __init__(self):
        self._name = "gpqa_diamond"
        self.dataset_name = "idavidrein/gpqa"
        self.dataset_config = "gpqa_diamond"
        self.dataset_split = "train"  # GPQA diamond only has a 'train' split

    @property
    def name(self) -> str:
        return self._name

    @property
    def answer_key(self) -> str:
        return "correct_answer_letter"

    @property
    def official_generation_prompt_name(self) -> str | None:
        return "official_generation_zeroshot_v1"

    @property
    def official_evaluation_prompt_name(self) -> str | None:
        # Per our discussion, there is no official evaluation prompt from the authors.
        return None

    @property
    def official_evaluation_model(self) -> Type[BaseModel] | None:
        # We define a model to signal that this benchmark supports a specific
        # structured evaluation format, which triggers advanced metrics.
        return OfficialGPQAEvaluation

    @property
    def is_multimodal(self) -> bool:
        return False

    def get_size(self) -> int:
        """
        Gets the total number of samples in the dataset split.

        NOTE: For this dataset, the fast metadata lookup via get_dataset_infos
        does not provide the number of examples, so we must load the dataset
        to determine its size.
        """
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

        def add_id_and_shuffle(example: Dict[str, Any]) -> Dict[str, Any]:
            # Create a stable ID by hashing the question
            question_text = example["Question"]
            example["id"] = hashlib.sha1(question_text.encode("utf-8")).hexdigest()

            # Reproducibly shuffle choices based on the question ID
            choices_list = [
                example["Correct Answer"],
                example["Incorrect Answer 1"],
                example["Incorrect Answer 2"],
                example["Incorrect Answer 3"],
            ]
            rng = random.Random(example["id"])
            rng.shuffle(choices_list)

            correct_index = choices_list.index(example["Correct Answer"])

            example["shuffled_choices"] = {
                "A": choices_list[0],
                "B": choices_list[1],
                "C": choices_list[2],
                "D": choices_list[3],
            }
            example["correct_answer_letter"] = ["A", "B", "C", "D"][correct_index]
            return example

        dataset = dataset.map(add_id_and_shuffle)

        if ids_to_load:
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
        # Flatten choices for easy formatting in the prompt template
        format_dict = {
            "question": question["Question"],
            "choice_A": question["shuffled_choices"]["A"],
            "choice_B": question["shuffled_choices"]["B"],
            "choice_C": question["shuffled_choices"]["C"],
            "choice_D": question["shuffled_choices"]["D"],
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
            formatted_system_text = system_content.format(**format_dict)
            messages.append({"role": "system", "content": formatted_system_text})

        formatted_user_text = user_template.format(**format_dict)
        messages.append({"role": "user", "content": formatted_user_text})

        return messages
