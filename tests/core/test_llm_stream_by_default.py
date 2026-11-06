"""A model whose relay-served args say `stream` is always called with
stream=True and the chunks are rebuilt into one priced ModelResponse
(ROB-1752); other models are untouched."""

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


def test_relay_args_switch_streaming_on():
    llm = _make_llm({"custom_args": {"stream": True, "max_context_size": 7}})
    llm.update_custom_args()
    assert llm.stream_by_default is True
    assert llm.max_context_size == 7
    assert "custom_args" not in llm.args


def test_streams_and_rebuilds_a_priced_response():
    llm = _make_llm({})
    llm.stream_by_default = True
    litellm.register_model(
        {
            "test-model": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )
    with patch("holmes.core.llm.litellm.completion", return_value=_chunks()) as mock:
        result = llm.completion(messages=[{"role": "user", "content": "hi"}])
    assert mock.call_args.kwargs["stream"] is True
    assert mock.call_args.kwargs["stream_options"] == {"include_usage": True}
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "hello"
    assert result.usage.completion_tokens == 2
    assert result._hidden_params["response_cost"] == 5 * 1e-6 + 2 * 2e-6


def test_other_models_do_not_stream():
    llm = _make_llm({})
    with patch(
        "holmes.core.llm.litellm.completion", return_value=_mock_model_response()
    ) as mock:
        llm.completion(messages=[{"role": "user", "content": "hi"}])
    assert mock.call_args.kwargs["stream"] is None
    assert mock.call_args.kwargs["stream_options"] is None
