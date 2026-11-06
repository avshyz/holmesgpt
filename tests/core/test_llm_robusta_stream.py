"""Robusta-hosted models are always called with stream=True and the chunks are
rebuilt into one ModelResponse (ROB-1752); other models are untouched."""

from unittest.mock import patch

import litellm
from litellm.types.utils import ModelResponse

from tests.core.test_llm_completion_max_tokens import _make_llm, _mock_model_response


def _chunks():
    for text, usage in (
        ("hel", None),
        ("lo", {"prompt_tokens": 5, "completion_tokens": 2}),
    ):
        yield litellm.ModelResponseStream(
            id="c",
            model="test-model",
            choices=[{"index": 0, "delta": {"role": "assistant", "content": text}}],
            usage=usage,
        )


def test_robusta_model_streams_and_rebuilds():
    llm = _make_llm({}, model="robusta/test-model")
    llm.is_robusta_model = True
    with patch("holmes.core.llm.litellm.completion", return_value=_chunks()) as mock:
        result = llm.completion(messages=[{"role": "user", "content": "hi"}])
    assert mock.call_args.kwargs["stream"] is True
    assert mock.call_args.kwargs["stream_options"] == {"include_usage": True}
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "hello"
    assert result.usage.completion_tokens == 2


def test_other_models_do_not_stream():
    llm = _make_llm({})
    with patch(
        "holmes.core.llm.litellm.completion", return_value=_mock_model_response()
    ) as mock:
        llm.completion(messages=[{"role": "user", "content": "hi"}])
    assert mock.call_args.kwargs["stream"] is None
    assert "stream_options" not in mock.call_args.kwargs
