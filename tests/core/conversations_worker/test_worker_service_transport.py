"""The ConversationWorker on CONVERSATION_TRANSPORT=service: no Realtime, no
verify loop, a long-poll claim loop that backs off on errors, and a
HolmesStatus row that names the transport."""
import json
import runpy
import threading
from unittest.mock import MagicMock, patch

import pytest

from holmes.common import env_vars
from holmes.common.env_vars import ConversationTransport
from holmes.core.conversations_worker.worker import (
    SERVICE_CLAIM_BACKOFF_MAX_SECONDS,
    SERVICE_CLAIM_BACKOFF_MIN_SECONDS,
    SERVICE_CLAIM_SATURATED_WAIT_SECONDS,
    ConversationWorker,
)
from holmes.utils.holmes_status import update_holmes_status_in_db
from tests.core.conversations_worker.conftest import _active

@pytest.fixture
def on_service(monkeypatch):
    monkeypatch.setattr(
        "holmes.core.conversations_worker.worker.CONVERSATION_TRANSPORT",
        ConversationTransport.SERVICE,
    )
    monkeypatch.setattr(
        "holmes.core.conversations_worker.worker.CONVERSATION_WORKER_MAX_CONCURRENT",
        2,
    )


def _record_waits(w):
    """Replace the loop's wait with one that records each timeout and
    returns at once, so a test sees the backoff schedule without sleeping."""
    waits = []

    def fake_wait(timeout=None):
        waits.append(timeout)
        return False

    w._notify_event.wait = fake_wait  # type: ignore[assignment]
    return waits


# ---- start / active workers ----


def test_start_on_service_starts_active_workers_without_a_verifier(
    on_service, worker_dal
):
    w = ConversationWorker(
        dal=worker_dal, config=MagicMock(), chat_function=MagicMock()
    )
    w._start_active_workers = MagicMock()

    w.start()

    w._start_active_workers.assert_called_once_with()
    assert w._running is True
    assert w._realtime_verify_thread is None
    worker_dal.is_realtime_enabled.assert_not_called()


def test_active_workers_on_service_create_no_realtime_worker(on_service, worker_dal):
    # Hold the claim open like the service does, so the loop is idle while
    # the test inspects the worker.
    release = threading.Event()

    def long_poll(_holmes_id, _limit):
        release.wait(timeout=5)
        return []

    worker_dal.claim_n_pending_conversations.side_effect = long_poll
    w = ConversationWorker(
        dal=worker_dal, config=MagicMock(), chat_function=MagicMock()
    )

    with patch("holmes.core.conversations_worker.worker.RealtimeWorker") as rt:
        try:
            w.start()
            assert w._executor is not None
            assert w._claim_thread is not None and w._claim_thread.is_alive()
            assert w._claim_thread._target == w._service_claim_loop  # type: ignore[attr-defined]
            assert w._realtime_manager is None
            rt.assert_not_called()
        finally:
            release.set()
            w.stop()

    worker_dal.is_realtime_enabled.assert_not_called()
    assert w._running is False


# ---- the long-poll claim loop ----


def test_service_claim_loop_dispatches_rows_and_claims_again_at_once(
    on_service, bare_worker, conversation_row
):
    w = bare_worker()
    waits = _record_waits(w)
    answers = iter([[conversation_row], []])

    def claim(_holmes_id, limit):
        rows = next(answers)
        if not rows:
            w._running = False
        return rows

    w.dal.claim_n_pending_conversations.side_effect = claim

    w._service_claim_loop()

    assert w.dal.claim_n_pending_conversations.call_count == 2
    # limit == free executor slots, as on the Supabase path.
    assert w.dal.claim_n_pending_conversations.call_args_list[0].args == ("h-test", 2)
    assert w._executor.submit.call_count == 1
    assert ("c1", 1) in w._active_conversation_ids
    # Nothing waited between the two calls: the long-poll is the wake.
    assert waits == []


def test_service_claim_loop_backs_off_on_errors_and_resets_on_success(
    on_service, bare_worker
):
    w = bare_worker()
    waits = _record_waits(w)
    answers = iter([
        ConnectionError("down"),
        ConnectionError("down"),
        ConnectionError("down"),
        [],
        ConnectionError("down"),
        [],
    ])

    def claim(_holmes_id, _limit):
        answer = next(answers, None)
        if answer is None:
            w._running = False
            return []
        if isinstance(answer, Exception):
            raise answer
        return answer

    w.dal.claim_n_pending_conversations.side_effect = claim

    w._service_claim_loop()

    assert waits == [1.0, 2.0, 4.0, 1.0]
    assert waits[0] == SERVICE_CLAIM_BACKOFF_MIN_SECONDS
    w._executor.submit.assert_not_called()


def test_service_claim_loop_backoff_is_capped(on_service, bare_worker):
    w = bare_worker()
    waits = _record_waits(w)
    failures = {"n": 0}

    def claim(_holmes_id, _limit):
        failures["n"] += 1
        if failures["n"] > 8:
            w._running = False
            return []
        raise ConnectionError("down")

    w.dal.claim_n_pending_conversations.side_effect = claim

    w._service_claim_loop()

    assert max(waits) == SERVICE_CLAIM_BACKOFF_MAX_SECONDS
    assert waits[-1] == SERVICE_CLAIM_BACKOFF_MAX_SECONDS


def test_service_claim_loop_waits_for_a_slot_at_capacity(
    on_service, monkeypatch, bare_worker
):
    monkeypatch.setattr(
        "holmes.core.conversations_worker.worker.CONVERSATION_WORKER_MAX_CONCURRENT",
        1,
    )
    w = bare_worker()
    _active(w, "busy")
    waits = []

    def wait_then_free_slot(timeout=None):
        waits.append(timeout)
        w._active_conversation_ids.clear()  # the conversation finished
        return True

    w._notify_event.wait = wait_then_free_slot  # type: ignore[assignment]

    def claim(_holmes_id, limit):
        w._running = False
        return []

    w.dal.claim_n_pending_conversations.side_effect = claim

    w._service_claim_loop()

    assert waits == [SERVICE_CLAIM_SATURATED_WAIT_SECONDS]
    w.dal.claim_n_pending_conversations.assert_called_once_with("h-test", 1)


def test_service_claim_loop_exits_on_stop_while_saturated(
    on_service, monkeypatch, bare_worker
):
    monkeypatch.setattr(
        "holmes.core.conversations_worker.worker.CONVERSATION_WORKER_MAX_CONCURRENT",
        1,
    )
    w = bare_worker()
    _active(w, "busy")

    t = threading.Thread(target=w._service_claim_loop)
    t.start()
    w._running = False
    w._notify_event.set()  # what stop() does
    t.join(timeout=3)

    assert not t.is_alive()
    w.dal.claim_n_pending_conversations.assert_not_called()


# ---- HolmesStatus ----


def _status_metadata(monkeypatch, transport, realtime_available=False):
    monkeypatch.setattr("holmes.utils.holmes_status.CONVERSATION_TRANSPORT", transport)
    monkeypatch.setattr("holmes.utils.holmes_status.ENABLE_CONVERSATION_WORKER", True)
    dal = MagicMock()
    config = MagicMock()
    config.cluster_name = "cl"
    config.should_try_robusta_ai = True
    config.get_models_list.return_value = ["m"]

    update_holmes_status_in_db(dal, config, realtime_available=realtime_available)

    row = dal.upsert_holmes_status.call_args.args[0]
    return json.loads(row["metadata"])


def test_holmes_status_on_service_advertises_the_transport(monkeypatch):
    metadata = _status_metadata(monkeypatch, ConversationTransport.SERVICE)

    assert metadata["conversation_transport"] == "service"
    # Written at startup, before any worker thread: the FE must already route
    # to Conversations, and there is no broadcast to send on the service.
    assert metadata["supports_realtime_conversations"] is True
    assert metadata["requires_realtime_broadcast"] is False


def test_holmes_status_on_supabase_is_unchanged(monkeypatch):
    before = _status_metadata(monkeypatch, ConversationTransport.SUPABASE)
    assert before["conversation_transport"] == "supabase"
    assert before["supports_realtime_conversations"] is False

    after = _status_metadata(
        monkeypatch, ConversationTransport.SUPABASE, realtime_available=True
    )
    assert after["conversation_transport"] == "supabase"
    assert after["supports_realtime_conversations"] is True


# ---- env ----


def test_conversation_transport_env_parsing(monkeypatch):
    # The value is read once at import, so the module runs afresh, in its
    # own namespace, under each environment.
    def load():
        return runpy.run_path(env_vars.__file__)["CONVERSATION_TRANSPORT"]

    monkeypatch.delenv("CONVERSATION_TRANSPORT", raising=False)
    assert load() == ConversationTransport.SUPABASE

    monkeypatch.setenv("CONVERSATION_TRANSPORT", " Service ")
    assert load() == ConversationTransport.SERVICE

    monkeypatch.setenv("CONVERSATION_TRANSPORT", "realtime")
    with pytest.raises(ValueError, match="ConversationTransport"):
        load()
