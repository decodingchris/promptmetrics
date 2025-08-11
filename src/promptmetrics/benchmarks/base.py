from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type
from pydantic import BaseModel

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
    def name(self) -> str:
        """The canonical, lower-case name of the benchmark."""
        pass

    @property
    @abstractmethod
    def answer_key(self) -> str:
        """The key in the dataset dictionary that holds the correct answer."""
        pass

    @property
    @abstractmethod
    def official_generation_prompt_name(self) -> str | None:
        """The filename stem of the official generation prompt, if one exists."""
        pass

    @property
    @abstractmethod
    def official_evaluation_prompt_name(self) -> str | None:
        """The filename stem of the official evaluation prompt, if one exists."""
        pass

    @property
    @abstractmethod
    def official_evaluation_model(self) -> Type[BaseModel] | None:
        """The Pydantic model for parsing official evaluation verdicts."""
        pass

    @property
    @abstractmethod
    def is_multimodal(self) -> bool:
        """Indicates if the benchmark contains multi-modal content (e.g., images)."""
        pass

    @abstractmethod
    def load_data(
        self,
        max_samples: int | None = None,
        ids_to_load: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Loads the dataset for the benchmark.

        This method should prioritize `ids_to_load` over `max_samples` if both are provided.

        Args:
            max_samples: The maximum number of samples to load. If None, load all.
            ids_to_load: A specific list of sample IDs to load.

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
