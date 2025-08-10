import os
from datasets import load_dataset, get_dataset_infos
from huggingface_hub.errors import HfHubHTTPError
from typing import Dict, Any, List, Type, Literal
from pydantic import BaseModel, Field
from .base import BaseBenchmark, MessageContentType
from dotenv import load_dotenv

load_dotenv()


class OfficialHLEEvaluation(BaseModel):
    """Pydantic model for the structured output of the official HLE evaluation prompt."""

    extracted_final_answer: str | None = None
    reasoning: str
    correct: Literal["yes", "no"] | None = None
    confidence: int = Field(ge=0, le=100, default=100)


class HLEBenchmark(BaseBenchmark):
    """Implementation for the Humanity's Last Exam (HLE) benchmark."""

    def __init__(self):
        self._name = "hle"
        self.dataset_name = "cais/hle"
        self.dataset_config = "default"
        self.dataset_split = "test"

    @property
    def name(self) -> str:
        return self._name

    @property
    def official_generation_prompt_name(self) -> str | None:
        return "official_generation_v1"

    @property
    def official_evaluation_prompt_name(self) -> str | None:
        return "official_evaluation_v1"

    @property
    def official_evaluation_model(self) -> Type[BaseModel] | None:
        return OfficialHLEEvaluation

    @property
    def is_multimodal(self) -> bool:
        return True

    def get_size(self) -> int:
        """Efficiently gets the total number of samples in the dataset split."""
        hf_token = os.getenv("HF_TOKEN")
        try:
            infos = get_dataset_infos(self.dataset_name, token=hf_token)
            return infos[self.dataset_config].splits[self.dataset_split].num_examples
        except (HfHubHTTPError, KeyError) as e:
            print("\n--- ⚠️  Warning: Slow Dataset Size Calculation ---")
            print(
                f"Could not quickly fetch dataset size due to a '{type(e).__name__}' error. "
                "This is expected if you are offline or the dataset requires authentication."
            )
            print(
                "Falling back to a full data load to determine the size. "
                "This will be slower and may consume significant memory and network bandwidth."
            )
            return len(self.load_data())

    def load_data(
        self,
        max_samples: int | None = None,
        ids_to_load: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not set. This may fail for gated datasets.")

        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.dataset_split,
            token=hf_token,
        )

        if ids_to_load:
            print(
                f"Optimized load: Loading {len(ids_to_load)} specific samples from the benchmark."
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
        messages: List[Dict[str, Any]] = []
        system_content = None
        user_template = prompt_template

        if "---[SYSTEM]---" in prompt_template:
            parts = prompt_template.split("---[SYSTEM]---", 1)[1].split(
                "---[USER]---", 1
            )
            system_content = parts[0].strip()
            if len(parts) > 1:
                user_template = parts[1].strip()
            else:
                user_template = ""
        elif "---[USER]---" in prompt_template:
            user_template = prompt_template.split("---[USER]---", 1)[1].strip()

        if system_content:
            messages.append({"role": "system", "content": system_content})

        formatted_user_text = user_template.format(question=question["question"])

        user_content_list: List[Dict[str, Any]] = [
            {"type": "text", "text": formatted_user_text}
        ]

        if question.get("image"):
            user_content_list.append(
                {"type": "image_url", "image_url": {"url": question["image"]}}
            )

        messages.append({"role": "user", "content": user_content_list})

        return messages