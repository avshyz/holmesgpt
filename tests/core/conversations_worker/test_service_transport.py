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


@pytest.fixture
def transport():
    dal = MagicMock()
    dal.get_ai_credentials.return_value = ("acc-1", "tok-1")
    dal.token_cache = {"session_token": "tok-1"}
    dal.lock = threading.Lock()
    return ConversationServiceTransport(dal, base_url=BASE + "/")


def test_session_verify_follows_the_dal_ca_lookup(monkeypatch, tmp_path):
    bundle = tmp_path / "ca.pem"
    bundle.write_text("")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/elsewhere.pem")

    transport = ConversationServiceTransport(MagicMock(), base_url=BASE)

    assert transport._session.verify == str(bundle)


# ---- claim ----


def test_claim_sends_holmes_bearer_and_body(responses, conversation_row, transport):
    responses.post(f"{AGENT}/claim", json={"conversations": [conversation_row]})

    rows = transport.claim("cluster-1", "h-1", 3)

    assert rows == [conversation_row]
    request = responses.calls[0].request
    assert request.headers["Authorization"] == "Bearer acc-1 tok-1"
    assert json.loads(request.body) == {
        "cluster_id": "cluster-1",
        "assignee": "h-1",
        "limit": 3,
        "wait_seconds": 25,
    }
    # The read timeout outlives the service's long-poll.
    assert request.req_kwargs["timeout"] == (
        CONNECT_TIMEOUT_SECONDS,
        CLAIM_WAIT_SECONDS + CLAIM_READ_TIMEOUT_MARGIN_SECONDS,
    )


# ---- events ----


def test_get_conversation_events_sends_query_and_returns_events(responses, transport):
    events = [{"event": "user_message", "data": {"ask": "hi"}, "ts": "t"}]
    responses.get(
        f"{AGENT}/conversations/c1/events",
        json={"events": events},
        match=[
            matchers.query_param_matcher({"include_compacted": "true", "min_seq": "4"})
        ],
    )

    out = transport.get_conversation_events("c1", include_compacted=True, min_seq=4)

    assert out == events
    assert responses.calls[0].request.headers["Authorization"] == "Bearer acc-1 tok-1"


def test_post_conversation_events_returns_seq(responses, transport):
    responses.post(f"{AGENT}/conversations/c1/events", json={"seq": 12})

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


def test_post_conversation_events_409_reassigned_raises(responses, transport):
    responses.post(
        f"{AGENT}/conversations/c1/events",
        status=409,
        json={"error": "reassigned", "message": "MISMATCH assignee"},
    )

    with pytest.raises(ConversationReassignedError, match="MISMATCH assignee"):
        transport.post_conversation_events("c1", "h-1", 1, [])

    assert len(responses.calls) == 1


def test_409_without_reassigned_is_an_http_error(responses, transport):
    responses.post(
        f"{AGENT}/conversations/c1/status", status=409, json={"error": "busy"}
    )

    with pytest.raises(requests.HTTPError) as info:
        transport.update_conversation_status("c1", 1, "h-1", "completed")

    assert info.value.response.status_code == 409
    assert len(responses.calls) == 1


def test_post_conversation_events_without_seq_is_an_error(responses, transport):
    responses.post(f"{AGENT}/conversations/c1/events", json={})

    with pytest.raises(KeyError):
        transport.post_conversation_events("c1", "h-1", 1, [])


# ---- status ----


def test_update_conversation_status_returns_ok(responses, transport):
    responses.post(f"{AGENT}/conversations/c1/status", json={"ok": True})
    responses.post(f"{AGENT}/conversations/c1/status", json={"ok": False})

    assert transport.update_conversation_status("c1", 3, "h-1", "completed") is True
    assert transport.update_conversation_status("c1", 3, "h-1", "completed") is False
    assert json.loads(responses.calls[0].request.body) == {
        "assignee": "h-1",
        "request_sequence": 3,
        "status": "completed",
    }


# ---- retries ----


def test_transient_5xx_is_retried_then_succeeds(responses, conversation_row, transport):
    responses.post(f"{AGENT}/claim", status=503, body="upstream down")
    responses.post(f"{AGENT}/claim", status=502, body="bad gateway")
    responses.post(f"{AGENT}/claim", json={"conversations": [conversation_row]})

    assert transport.claim("cluster-1", "h-1", 1) == [conversation_row]
    assert len(responses.calls) == 3


def test_5xx_raises_after_the_dal_attempt_budget(responses, transport):
    for _ in range(3):
        responses.post(f"{AGENT}/claim", status=500, body="boom")

    with pytest.raises(requests.HTTPError) as info:
        transport.claim("cluster-1", "h-1", 1)

    assert info.value.response.status_code == 500
    assert len(responses.calls) == 3


@pytest.mark.parametrize("body", ["<html>login</html>", "", "null", "[]"])
def test_a_malformed_success_answer_is_an_error(responses, transport, body):
    responses.post(f"{AGENT}/claim", body=body)

    with pytest.raises(ValueError):
        transport.claim("cluster-1", "h-1", 1)

    assert len(responses.calls) == 1


def test_401_drops_the_cached_token_without_retrying(responses, transport):
    responses.post(f"{AGENT}/claim", status=401, json={"error": "unauthorized"})

    with pytest.raises(requests.HTTPError) as info:
        transport.claim("cluster-1", "h-1", 1)

    assert info.value.response.status_code == 401
    assert len(responses.calls) == 1
    assert "session_token" not in transport._dal.token_cache
