from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks.

    It defines a standard interface for loading data and formatting prompts,
    ensuring that the core evaluation scripts can work with any benchmark
    that implements this interface.
    """

    @abstractmethod
    def load_data(self, max_samples: int | None = None) -> List[Dict[str, Any]]:
        """
        Loads the dataset for the benchmark.

        Args:
            max_samples: The maximum number of samples to load. If None, load all.

        Returns:
            A list of question dictionaries.
        """
        pass

    @abstractmethod
    def format_prompt(self, question: Dict[str, Any], prompt_template: str) -> str:
        """
        Formats a single question into a prompt string using a template.

        Args:
            question: A dictionary representing a single question from the dataset.
            prompt_template: The string template to be filled.

        Returns:
            A formatted prompt string ready to be sent to an LLM.
        """
        pass