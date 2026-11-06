from unittest.mock import patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from holmes.core.llm import DefaultLLM


def make_llm(
    args: dict | None = None, model: str = "test-model", max_context_size=None
) -> DefaultLLM:
    """Skips __init__: check_llm wants a provider litellm knows plus its credentials,
    and update_custom_args would consume the custom_args the stream test checks."""
    llm = DefaultLLM.__new__(DefaultLLM)
    llm.model = model
    llm.api_key = None
    llm.api_base = None
    llm.api_version = None
    llm.args = dict(args or {})
    llm.tracer = None
    llm.name = None
    llm.is_robusta_model = False
    llm.max_context_size = max_context_size
    return llm


@pytest.fixture
def mock_completion():
    with patch("holmes.core.llm.litellm.completion") as mock:
        mock.return_value = ModelResponse(
            id="chatcmpl-test",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        yield mock
