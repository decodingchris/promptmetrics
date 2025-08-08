import pytest
from promptmetrics.benchmarks.hle import HLEBenchmark

@pytest.fixture
def hle_bench():
    """Provides a HLEBenchmark instance for tests."""
    return HLEBenchmark()

def test_hle_prompt_formatting(hle_bench):
    """
    Tests that the HLE benchmark correctly injects the question
    into the prompt template.
    """
    question_data = {
        "question": "This is a test question with choices A and B."
    }
    template = "System instruction.\n\n{question}\n\nAnswer:"

    expected_output = "System instruction.\n\nThis is a test question with choices A and B.\n\nAnswer:"
    
    formatted_prompt = hle_bench.format_prompt(question_data, template)

    assert formatted_prompt == expected_output