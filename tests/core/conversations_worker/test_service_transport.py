"""Unit tests for ConversationServiceTransport (contract section 6).

HTTP is mocked with the ``responses`` fixture from the root conftest.
"""
import json
import threading
from unittest.mock import MagicMock

import pytest
import requests
from responses import matchers
from tenacity import wait_none

from holmes.core.conversations_worker.models import ConversationReassignedError
from holmes.core.conversations_worker.service_transport import (
    CLAIM_READ_TIMEOUT_MARGIN_SECONDS,
    CLAIM_WAIT_SECONDS,
    CONNECT_TIMEOUT_SECONDS,
    ConversationServiceTransport,
)

BASE = "https://api.example.test/api/conversations/v1"
AGENT = f"{BASE}/agent"


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    # tenacity copies the policy at call time, so the decorated method's
    # ``retry`` object is the one place to turn the backoff off for tests.
    monkeypatch.setattr(
        ConversationServiceTransport._request_with_retry.retry, "wait", wait_none()
    )


def _transport():
    dal = MagicMock()
    dal.get_ai_credentials.return_value = ("acc-1", "tok-1")
    dal.token_cache = {"session_token": "tok-1"}
    dal.lock = threading.Lock()
    return ConversationServiceTransport(dal, base_url=BASE + "/"), dal


# ---- claim ----


def test_claim_sends_holmes_bearer_and_body(responses, conversation_row):
    responses.post(f"{AGENT}/claim", json={"conversations": [conversation_row]})
    transport, _ = _transport()

    rows = transport.claim("cluster-1", "h-1", 3)

    assert rows == [conversation_row]
    request = responses.calls[0].request
    assert request.headers["Authorization"] == "Bearer acc-1 tok-1"
    assert request.body is not None
    assert json.loads(request.body) == {
        "cluster_id": "cluster-1",
        "assignee": "h-1",
        "limit": 3,
        "wait_seconds": 25,
    }


def test_claim_read_timeout_outlives_the_long_poll():
    transport, _ = _transport()
    transport._session.request = MagicMock(
        return_value=MagicMock(status_code=200, json=lambda: {"conversations": []})
    )

    transport.claim("cluster-1", "h-1", 1)

    timeout = transport._session.request.call_args.kwargs["timeout"]
    assert timeout == (
        CONNECT_TIMEOUT_SECONDS,
        CLAIM_WAIT_SECONDS + CLAIM_READ_TIMEOUT_MARGIN_SECONDS,
    )
    assert timeout[1] > CLAIM_WAIT_SECONDS


# ---- events ----


def test_get_conversation_events_sends_query_and_returns_events(responses):
    events = [{"event": "user_message", "data": {"ask": "hi"}, "ts": "t"}]
    responses.get(
        f"{AGENT}/conversations/c1/events",
        json={"events": events},
        match=[
            matchers.query_param_matcher({"include_compacted": "true", "min_seq": "4"})
        ],
    )
    transport, _ = _transport()

    out = transport.get_conversation_events("c1", include_compacted=True, min_seq=4)

    assert out == events
    assert responses.calls[0].request.headers["Authorization"] == "Bearer acc-1 tok-1"


def test_post_conversation_events_returns_seq(responses):
    responses.post(f"{AGENT}/conversations/c1/events", json={"seq": 12})
    transport, _ = _transport()

    seq = transport.post_conversation_events(
        "c1", "h-1", 2, [{"event": "x", "data": {}, "ts": "t"}], compact=True
    )

    assert seq == 12
    assert json.loads(responses.calls[0].request.body) == {
        "assignee": "h-1",
        "request_sequence": 2,
        "events": [{"event": "x", "data": {}, "ts": "t"}],
        "compact": True,
    }


def test_post_conversation_events_409_reassigned_raises(responses):
    responses.post(
        f"{AGENT}/conversations/c1/events",
        status=409,
        json={"error": "reassigned", "message": "MISMATCH assignee"},
    )
    transport, _ = _transport()

    with pytest.raises(ConversationReassignedError, match="MISMATCH assignee"):
        transport.post_conversation_events("c1", "h-1", 1, [])

    assert len(responses.calls) == 1


def test_409_without_reassigned_is_an_http_error(responses):
    responses.post(f"{AGENT}/conversations/c1/status", status=409, json={"error": "busy"})
    transport, _ = _transport()

    with pytest.raises(requests.HTTPError) as info:
        transport.update_conversation_status("c1", 1, "h-1", "completed")

    assert info.value.response.status_code == 409
    assert len(responses.calls) == 1


def test_post_conversation_events_without_seq_is_a_value_error(responses):
    responses.post(f"{AGENT}/conversations/c1/events", json={})
    transport, _ = _transport()

    with pytest.raises(ValueError, match="no seq"):
        transport.post_conversation_events("c1", "h-1", 1, [])


# ---- status ----


def test_update_conversation_status_returns_ok(responses):
    responses.post(f"{AGENT}/conversations/c1/status", json={"ok": True})
    responses.post(f"{AGENT}/conversations/c1/status", json={"ok": False})
    transport, _ = _transport()

    assert transport.update_conversation_status("c1", 3, "h-1", "completed") is True
    assert transport.update_conversation_status("c1", 3, "h-1", "completed") is False
    assert json.loads(responses.calls[0].request.body) == {
        "assignee": "h-1",
        "request_sequence": 3,
        "status": "completed",
    }


# ---- retries ----


def test_transient_5xx_is_retried_then_succeeds(responses, conversation_row):
    responses.post(f"{AGENT}/claim", status=503, body="upstream down")
    responses.post(f"{AGENT}/claim", status=502, body="bad gateway")
    responses.post(f"{AGENT}/claim", json={"conversations": [conversation_row]})
    transport, _ = _transport()

    assert transport.claim("cluster-1", "h-1", 1) == [conversation_row]
    assert len(responses.calls) == 3


def test_5xx_raises_after_the_dal_attempt_budget(responses):
    for _ in range(3):
        responses.post(f"{AGENT}/claim", status=500, body="boom")
    transport, _ = _transport()

    with pytest.raises(requests.HTTPError) as info:
        transport.claim("cluster-1", "h-1", 1)

    assert info.value.response.status_code == 500
    assert len(responses.calls) == 3


def test_connection_error_is_retried(responses):
    responses.post(
        f"{AGENT}/conversations/c1/events",
        body=requests.exceptions.ConnectionError("reset by peer"),
    )
    responses.post(f"{AGENT}/conversations/c1/events", json={"seq": 1})
    transport, _ = _transport()

    assert transport.post_conversation_events("c1", "h-1", 1, []) == 1
    assert len(responses.calls) == 2


def test_4xx_is_not_retried(responses):
    responses.get(f"{AGENT}/conversations/c1/events", status=400, json={"error": "bad"})
    transport, _ = _transport()

    with pytest.raises(requests.HTTPError) as info:
        transport.get_conversation_events("c1")

    assert info.value.response.status_code == 400
    assert len(responses.calls) == 1


def test_401_drops_the_cached_token_without_retrying(responses):
    responses.post(f"{AGENT}/claim", status=401, json={"error": "unauthorized"})
    transport, dal = _transport()

    with pytest.raises(requests.HTTPError) as info:
        transport.claim("cluster-1", "h-1", 1)

    assert info.value.response.status_code == 401
    assert len(responses.calls) == 1
    assert "session_token" not in dal.token_cache
