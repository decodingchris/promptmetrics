import pytest
from promptmetrics.benchmarks.hle import HLEBenchmark

@pytest.fixture
def hle_bench():
    """Provides a HLEBenchmark instance for tests."""
    return HLEBenchmark()

def test_hle_is_multimodal(hle_bench):
    """Tests that the HLE benchmark correctly identifies as multi-modal."""
    assert hle_bench.is_multimodal is True

def test_hle_system_and_user_prompt_parsing(hle_bench):
    """
    Tests that a template with both system and user separators is parsed correctly.
    """
    question_data = {"question": "Test Question"}
    template = """
    ---[SYSTEM]---
    System instruction here.
    ---[USER]---
    User instruction with {question}.
    """
    
    expected_output = [
        {"role": "system", "content": "System instruction here."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "User instruction with Test Question."}
            ]
        }
    ]
    
    formatted_messages = hle_bench.format_prompt_messages(question_data, template)
    assert formatted_messages == expected_output

def test_hle_user_only_prompt_parsing(hle_bench):
    """
    Tests that a template with only a user separator is parsed correctly.
    """
    question_data = {"question": "Another Test"}
    template = """
    ---[USER]---
    The question is: {question}
    """
    
    expected_output = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "The question is: Another Test"}
            ]
        }
    ]

    formatted_messages = hle_bench.format_prompt_messages(question_data, template)
    assert formatted_messages == expected_output

def test_hle_legacy_prompt_format(hle_bench):
    """
    Tests backward compatibility with templates that have no separators.
    """
    question_data = {"question": "Legacy Question"}
    template = "This is a simple prompt about a {question}."
    
    expected_output = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is a simple prompt about a Legacy Question."}
            ]
        }
    ]
    
    formatted_messages = hle_bench.format_prompt_messages(question_data, template)
    assert formatted_messages == expected_output