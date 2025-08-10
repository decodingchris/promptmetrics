import os
from datasets import load_dataset, get_dataset_infos
from typing import Dict, Any, List
from .base import BaseBenchmark, MessageContentType
from dotenv import load_dotenv

load_dotenv()


class HLEBenchmark(BaseBenchmark):
    """Implementation for the Humanity's Last Exam (HLE) benchmark."""

    def __init__(self):
        self.name = "hle"
        self.dataset_name = "cais/hle"
        self.dataset_config = "default"
        self.dataset_split = "test"

    @property
    def is_multimodal(self) -> bool:
        return True

    def get_size(self) -> int:
        """Efficiently gets the total number of samples in the dataset split."""
        hf_token = os.getenv("HF_TOKEN")
        try:
            infos = get_dataset_infos(self.dataset_name, token=hf_token)
            return infos[self.dataset_config].splits[self.dataset_split].num_examples
        except Exception:
            return len(self.load_data())

    def load_data(self, max_samples: int | None = None) -> List[Dict[str, Any]]:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not set. This may fail for gated datasets.")

        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.dataset_split,
            token=hf_token,
        )

        if max_samples:
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
