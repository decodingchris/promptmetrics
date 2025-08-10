from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union

MessageContentType = Union[str, List[Dict[str, Any]]]


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks.

    It defines a standard interface for loading data and formatting prompts,
    ensuring that the core evaluation scripts can work with any benchmark
    that implements this interface.
    """

    @property
    @abstractmethod
    def is_multimodal(self) -> bool:
        """Indicates if the benchmark contains multi-modal content (e.g., images)."""
        pass

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
    def format_prompt_messages(
        self, question: Dict[str, Any], prompt_template: str
    ) -> List[Dict[str, MessageContentType]]:
        """
        Formats a single question into a list of messages for an LLM.

        Args:
            question: A dictionary representing a single question from the dataset.
            prompt_template: The string template to be filled.

        Returns:
            A list of message dictionaries ready for an LLM API call.
        """
        pass
