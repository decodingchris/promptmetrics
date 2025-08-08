import os
from datasets import load_dataset
from typing import Dict, Any, List
from .base import BaseBenchmark
from dotenv import load_dotenv

load_dotenv()

class HLEBenchmark(BaseBenchmark):
    """Implementation for the Humanity's Last Exam (HLE) benchmark."""

    def __init__(self):
        self.name = "hle"
        self.dataset_name = "cais/hle"
        self.dataset_config = "default"
        self.dataset_split = "test"

    def load_data(self, max_samples: int | None = None) -> List[Dict[str, Any]]:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not set. This may fail for gated datasets.")
        
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.dataset_split,
            token=hf_token
        )

        if max_samples:
            dataset = dataset.select(range(max_samples))
        
        return [sample for sample in dataset]

    def format_prompt(self, question: Dict[str, Any], prompt_template: str) -> str:
        return prompt_template.format(question=question['question'])