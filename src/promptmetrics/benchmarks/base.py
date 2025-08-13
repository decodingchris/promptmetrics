"""Abstract interfaces and shared typing for PromptMetrics benchmarks."""

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
        """Canonical, lower-case name of the benchmark."""
        pass

    @property
    @abstractmethod
    def answer_key(self) -> str:
        """Dataset field that contains the correct answer for grading."""
        pass

    @property
    @abstractmethod
    def official_generation_prompt_name(self) -> str | None:
        """Filename stem for the benchmark's official generation prompt, if any."""
        pass

    @property
    @abstractmethod
    def official_evaluation_prompt_name(self) -> str | None:
        """Filename stem for the benchmark's official evaluation prompt, if any."""
        pass

    @property
    @abstractmethod
    def official_evaluation_model(self) -> Type[BaseModel] | None:
        """Pydantic model used to parse structured evaluation outputs, if supported."""
        pass

    @property
    @abstractmethod
    def is_multimodal(self) -> bool:
        """True if any sample can contain non-text content (e.g., images)."""
        pass

    @abstractmethod
    def load_data(
        self,
        max_samples: int | None = None,
        ids_to_load: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Load and adapt dataset samples for this benchmark.

        Implementations should prioritize ids_to_load over max_samples when both
        are provided.

        Args:
            max_samples: Optional upper bound on number of samples to return.
            ids_to_load: Optional list of exact sample IDs to return.

        Returns:
            List of adapted sample dictionaries (each must include a string 'id').
        """
        pass

    @abstractmethod
    def format_prompt_messages(
        self, question: Dict[str, Any], prompt_template: str
    ) -> List[Dict[str, MessageContentType]]:
        """
        Format a single adapted sample into chat messages for an LLM API.

        Args:
            question: Single adapted sample from load_data().
            prompt_template: Raw prompt template content (possibly with ---[SYSTEM]---
                and ---[USER]--- markers).

        Returns:
            Chat messages in OpenAI-compatible format. For multimodal benchmarks,
            the user 'content' may be a list containing both text and image parts.
        """
        pass
