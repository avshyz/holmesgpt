"""The DAL's conversation methods route to the service transport on
CONVERSATION_TRANSPORT=service and leave everything else on Supabase."""
from unittest.mock import MagicMock

import pytest
import requests

from holmes.common.env_vars import ConversationTransport
from holmes.core.conversations_worker.models import ConversationReassignedError
from holmes.core.conversations_worker.service_transport import (
    ConversationServiceTransport,
)


@pytest.fixture
def service_dal(monkeypatch, build_dal):
    monkeypatch.setattr(
        "holmes.core.supabase_dal.CONVERSATION_TRANSPORT",
        ConversationTransport.SERVICE,
    )
    dal = build_dal(rpc_data=[])
    service = MagicMock(spec=ConversationServiceTransport)
    dal._service_transport = service
    return dal, service


def test_claim_routes_to_the_service(service_dal):
    dal, service = service_dal
    service.claim.return_value = [{"conversation_id": "c1"}]

    assert dal.claim_n_pending_conversations("h-1", 3) == [{"conversation_id": "c1"}]

    service.claim.assert_called_once_with("cluster-1", "h-1", 3)
    dal.client.rpc.assert_not_called()


def test_claim_on_the_service_still_skips_a_non_positive_limit(service_dal):
    dal, service = service_dal

    assert dal.claim_n_pending_conversations("h-1", 0) == []

    service.claim.assert_not_called()


def test_claim_on_the_service_propagates_failures(service_dal):
    """The service claim loop backs off on an exception, so unlike the
    Supabase path the DAL does not swallow one into an empty answer."""
    dal, service = service_dal
    service.claim.side_effect = requests.HTTPError("503 Service Unavailable")

    with pytest.raises(requests.HTTPError):
        dal.claim_n_pending_conversations("h-1", 1)


def test_get_conversation_events_routes_to_the_service(service_dal):
    dal, service = service_dal
    service.get_conversation_events.return_value = [{"event": "user_message"}]

    out = dal.get_conversation_events("c1", include_compacted=True, min_seq=5)

    assert out == [{"event": "user_message"}]
    service.get_conversation_events.assert_called_once_with(
        "c1", include_compacted=True, min_seq=5
    )
    dal.client.rpc.assert_not_called()


def test_get_conversation_events_on_the_service_returns_empty_on_failure(service_dal):
    dal, service = service_dal
    service.get_conversation_events.side_effect = requests.HTTPError("500")

    assert dal.get_conversation_events("c1") == []


def test_post_conversation_events_routes_to_the_service(service_dal):
    dal, service = service_dal
    service.post_conversation_events.return_value = 9
    events = [{"event": "x", "data": {}, "ts": "t"}]

    assert dal.post_conversation_events("c1", "h-1", 2, events, compact=True) == 9

    service.post_conversation_events.assert_called_once_with(
        conversation_id="c1",
        assignee="h-1",
        request_sequence=2,
        events=events,
        compact=True,
    )
    dal.client.rpc.assert_not_called()


def test_post_conversation_events_on_the_service_raises_reassignment(service_dal):
    dal, service = service_dal
    service.post_conversation_events.side_effect = ConversationReassignedError("gone")

    with pytest.raises(ConversationReassignedError):
        dal.post_conversation_events("c1", "h-1", 2, [])


def test_update_conversation_status_routes_to_the_service(service_dal):
    dal, service = service_dal
    service.update_conversation_status.return_value = True

    assert dal.update_conversation_status("c1", 2, "h-1", "completed") is True

    service.update_conversation_status.assert_called_once_with(
        conversation_id="c1",
        request_sequence=2,
        assignee="h-1",
        status="completed",
    )
    dal.client.rpc.assert_not_called()


def test_update_conversation_status_on_the_service_rejects_bad_status(service_dal):
    dal, service = service_dal

    assert dal.update_conversation_status("c1", 2, "h-1", "bogus") is False

    service.update_conversation_status.assert_not_called()


def test_update_conversation_status_on_the_service_returns_false_on_failure(
    service_dal,
):
    dal, service = service_dal
    service.update_conversation_status.side_effect = requests.HTTPError("500")

    assert dal.update_conversation_status("c1", 2, "h-1", "failed") is False


def test_update_conversation_status_on_the_service_raises_reassignment(service_dal):
    dal, service = service_dal
    service.update_conversation_status.side_effect = ConversationReassignedError("x")

    with pytest.raises(ConversationReassignedError):
        dal.update_conversation_status("c1", 2, "h-1", "timeout")


def test_disabled_dal_never_touches_the_service(service_dal):
    dal, service = service_dal
    dal.enabled = False

    assert dal.claim_n_pending_conversations("h-1", 1) == []
    assert dal.get_conversation_events("c1") == []
    assert dal.post_conversation_events("c1", "h-1", 1, []) is None
    assert dal.update_conversation_status("c1", 1, "h-1", "failed") is False

    assert not service.method_calls


def test_service_transport_is_built_once_on_first_use(service_dal):
    dal, _ = service_dal
    del dal._service_transport

    first = dal._service_transport
    second = dal._service_transport

    assert isinstance(first, ConversationServiceTransport)
    assert first is second


def test_supabase_transport_has_no_service(monkeypatch, build_dal):
    monkeypatch.setattr(
        "holmes.core.supabase_dal.CONVERSATION_TRANSPORT",
        ConversationTransport.SUPABASE,
    )
    assert build_dal()._service_transport is None
