"""Fixtures shared by the conversation worker unit tests."""

import threading
import time
from typing import Any, Callable, Dict, Optional
from unittest.mock import MagicMock

import pytest

from holmes.core.conversations_worker.models import ConversationTask
from holmes.core.conversations_worker.worker import ConversationWorker, _ActiveTask
from holmes.core.supabase_dal import SupabaseDal


def _row(conversation_id: str, request_sequence: int = 1) -> Dict[str, Any]:
    """A claimed Conversations row, as the claim RPC and the service return it."""
    return {
        "conversation_id": conversation_id,
        "account_id": "acc-1",
        "cluster_id": "cluster-1",
        "origin": "chat",
        "request_sequence": request_sequence,
        "metadata": {},
    }


def _task(**kwargs: Any) -> ConversationTask:
    defaults: Dict[str, Any] = dict(
        conversation_id="c1",
        account_id="a1",
        cluster_id="cl1",
        origin="chat",
        request_sequence=1,
    )
    defaults.update(kwargs)
    return ConversationTask(**defaults)


@pytest.fixture
def conversation_row() -> Dict[str, Any]:
    return _row("c1")


@pytest.fixture
def build_dal() -> Callable[..., SupabaseDal]:
    """A DAL on a mocked Supabase client. ``rpc_data`` is what every
    ``rpc().execute()`` answers with; None leaves the client for the test to
    program."""

    def build(rpc_data: Any = None) -> SupabaseDal:
        dal = SupabaseDal.__new__(SupabaseDal)
        dal.enabled = True
        dal.account_id = "acc-1"
        dal.cluster = "cluster-1"
        dal.client = MagicMock()
        if rpc_data is not None:
            dal.client.rpc.return_value = MagicMock(
                execute=MagicMock(return_value=MagicMock(data=rpc_data))
            )
        return dal

    return build


@pytest.fixture
def worker_dal() -> MagicMock:
    """The DAL a start() test hands ConversationWorker(): enabled."""
    dal = MagicMock()
    dal.enabled = True
    return dal


@pytest.fixture
def worker(worker_dal: MagicMock) -> ConversationWorker:
    return ConversationWorker(
        dal=worker_dal, config=MagicMock(), chat_function=MagicMock()
    )


@pytest.fixture
def bare_worker() -> Callable[[], ConversationWorker]:
    """A ConversationWorker built without __init__: mocked DAL and executor,
    no threads, already running. Each call is a fresh worker, so a test can
    drive several side by side."""

    def build() -> ConversationWorker:
        w = ConversationWorker.__new__(ConversationWorker)
        w.dal = MagicMock()
        w.dal.enabled = True
        w.dal.update_conversation_status = MagicMock(return_value=True)
        w.config = MagicMock()
        w.chat_function = MagicMock()
        w.holmes_id = "h-test"
        w._running = True
        w._claim_thread = None
        w._notify_event = threading.Event()
        w._saturated_since = None
        w._saturation_logged = False
        w._last_stuck_warn = None
        w._executor = MagicMock()
        w._active_conversation_ids = {}
        w._active_lock = threading.Lock()
        w._dispatch_lock = threading.Lock()
        w._realtime_manager = None
        w._realtime_verify_thread = None
        w._realtime_verify_stop = threading.Event()
        return w

    return build


def _active(
    w: ConversationWorker,
    conversation_id: str = "c1",
    request_sequence: int = 1,
    started: Optional[float] = None,
) -> ConversationTask:
    """Register an in-flight conversation the way _dispatch does. ``started``
    backdates the slot for the stuck-slot tests."""
    task = _task(conversation_id=conversation_id, request_sequence=request_sequence)
    w._active_conversation_ids[task.active_key] = _ActiveTask(
        task, time.monotonic() if started is None else started
    )
    return task
