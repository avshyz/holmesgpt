"""The DAL's conversation methods route to the service transport on
CONVERSATION_TRANSPORT=service and leave everything else on Supabase."""

from unittest.mock import MagicMock, call

import pytest
import requests

from holmes.common.env_vars import ConversationTransport
from holmes.core.conversations_worker.models import ConversationReassignedError
from holmes.core.conversations_worker.service_transport import (
    ConversationServiceTransport,
)

EVENTS = [{"event": "x", "data": {}, "ts": "t"}]


@pytest.fixture
def service_dal(monkeypatch, build_dal):
    monkeypatch.setattr(
        "holmes.core.supabase_dal.CONVERSATION_TRANSPORT",
        ConversationTransport.SERVICE,
    )
    dal = build_dal()
    service = MagicMock(spec=ConversationServiceTransport)
    dal._service_transport = service
    return dal, service


# fmt: off
# dal method, its args, its kwargs,
#   service method, the forwarded call, the answer
ROUTES = [
    ("claim_n_pending_conversations", ("h-1", 3), {},
     "claim", call("cluster-1", "h-1", 3), [{"conversation_id": "c1"}]),
    ("get_conversation_events", ("c1",), {"include_compacted": True, "min_seq": 5},
     "get_conversation_events", call("c1", True, 5), [{"event": "user_message"}]),
    ("post_conversation_events", ("c1", "h-1", 2, EVENTS), {"compact": True},
     "post_conversation_events", call("c1", "h-1", 2, EVENTS, True), 9),
    ("update_conversation_status", ("c1", 2, "h-1", "completed"), {},
     "update_conversation_status", call("c1", 2, "h-1", "completed"), True),
]
# dal method, its args,
#   service method, what the service raises, what the DAL raises or answers
FAILURES = [
    ("claim_n_pending_conversations", ("h-1", 1),
     "claim", requests.HTTPError("503 Service Unavailable"), requests.HTTPError),
    ("get_conversation_events", ("c1",),
     "get_conversation_events", requests.HTTPError("500"), []),
    ("post_conversation_events", ("c1", "h-1", 2, []), "post_conversation_events",
     ConversationReassignedError("gone"), ConversationReassignedError),
    ("update_conversation_status", ("c1", 2, "h-1", "failed"),
     "update_conversation_status", requests.HTTPError("500"), False),
    ("update_conversation_status", ("c1", 2, "h-1", "timeout"),
     "update_conversation_status",
     ConversationReassignedError("x"), ConversationReassignedError),
]
# fmt: on


@pytest.mark.parametrize(
    "dal_method, args, kwargs, service_method, expected, answer", ROUTES
)
def test_routes_to_the_service(
    service_dal, dal_method, args, kwargs, service_method, expected, answer
):
    dal, service = service_dal
    getattr(service, service_method).return_value = answer

    assert getattr(dal, dal_method)(*args, **kwargs) == answer

    assert getattr(service, service_method).call_args_list == [expected]
    dal.client.rpc.assert_not_called()


@pytest.mark.parametrize("dal_method, args, service_method, error, outcome", FAILURES)
def test_service_failures(
    service_dal, dal_method, args, service_method, error, outcome
):
    """A reassignment always propagates. So does a failed claim: the service
    claim loop backs off on an exception, so unlike the Supabase path the DAL
    does not swallow one into an empty answer."""
    dal, service = service_dal
    getattr(service, service_method).side_effect = error

    if isinstance(outcome, type):
        with pytest.raises(outcome):
            getattr(dal, dal_method)(*args)
    else:
        assert getattr(dal, dal_method)(*args) == outcome


def test_disabled_dal_never_touches_the_service(service_dal):
    dal, service = service_dal
    dal.enabled = False

    assert dal.claim_n_pending_conversations("h-1", 1) == []
    assert dal.get_conversation_events("c1") == []
    assert dal.post_conversation_events("c1", "h-1", 1, []) is None
    assert dal.update_conversation_status("c1", 1, "h-1", "failed") is False

    assert not service.method_calls


def test_service_transport_is_built_on_first_use(service_dal):
    dal, _ = service_dal
    del dal._service_transport

    assert isinstance(dal._service_transport, ConversationServiceTransport)


def test_supabase_transport_has_no_service(monkeypatch, build_dal):
    monkeypatch.setattr(
        "holmes.core.supabase_dal.CONVERSATION_TRANSPORT",
        ConversationTransport.SUPABASE,
    )
    assert build_dal()._service_transport is None
