"""HTTP client for the platform-conversations agent endpoints.

The contract is relay/docs/design/2026-11-24_chat-service-poc.md, sections 6
and 7. Holmes only ever connects outbound, so this client carries the same CA
bundle and proxy settings as the Supabase client and authenticates with the
HOLMES session token the DAL already mints for Robusta AI.
"""

import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import quote

import requests
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from holmes.common.env_vars import CONVERSATIONS_SERVICE_URL
from holmes.core.conversations_worker.models import ConversationReassignedError

if TYPE_CHECKING:
    from holmes.core.supabase_dal import SupabaseDal

# The service holds a claim open for at most this long (contract, section 6);
# it answers at once when rows exist.
CLAIM_WAIT_SECONDS = 25
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 30
# The claim's read timeout is wait_seconds plus this, so the service ends the
# long-poll and the client never gives up on a healthy call.
CLAIM_READ_TIMEOUT_MARGIN_SECONDS = 15

# Same budget as the DAL's conversation RPCs.
_RETRY_ATTEMPTS = 3
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ConversationServiceError(Exception):
    """An answer the service will keep giving: a 4xx other than a 409
    reassignment, a malformed body, or a 5xx that outlived the retries."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ConversationServiceError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(
        exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    )


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logging.warning(
        "Conversation service request failed (attempt %d/%d): %s; retrying",
        retry_state.attempt_number,
        _RETRY_ATTEMPTS,
        exc,
    )


def _ca_bundle() -> "str | bool":
    """The DAL's lookup: SSL_CERT_FILE first, then REQUESTS_CA_BUNDLE.

    requests reads REQUESTS_CA_BUNDLE on its own but not SSL_CERT_FILE, so a
    cluster that sets only the latter would otherwise verify against certifi.
    Proxies need no mirroring: requests resolves HTTPS_PROXY and NO_PROXY per
    request, the way the DAL does by hand for httpx.
    """
    return (
        os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or True
    )


def _error_text(response: requests.Response) -> str:
    body = response.text or ""
    return f"{response.status_code} {response.reason}: {body[:500]}"


class ConversationServiceTransport:
    """The four agent calls of the contract, each raising
    ``ConversationReassignedError`` on ``409 {error: "reassigned"}`` and
    ``ConversationServiceError`` on any other failure that survived the retries.
    """

    def __init__(self, dal: "SupabaseDal", base_url: str = CONVERSATIONS_SERVICE_URL):
        self._dal = dal
        self._agent_url = base_url.rstrip("/") + "/agent"
        self._session = requests.Session()
        self._session.verify = _ca_bundle()

    @property
    def agent_url(self) -> str:
        return self._agent_url

    # ---- the contract ----

    def claim(
        self,
        cluster_id: str,
        assignee: str,
        limit: int,
        wait_seconds: int = CLAIM_WAIT_SECONDS,
    ) -> List[Dict[str, Any]]:
        body = self._request_with_retry(
            "POST",
            "/claim",
            json={
                "cluster_id": cluster_id,
                "assignee": assignee,
                "limit": limit,
                "wait_seconds": wait_seconds,
            },
            read_timeout=wait_seconds + CLAIM_READ_TIMEOUT_MARGIN_SECONDS,
        )
        return list(body.get("conversations") or [])

    def get_conversation_events(
        self,
        conversation_id: str,
        include_compacted: bool = False,
        min_seq: int = 1,
    ) -> List[Dict[str, Any]]:
        body = self._request_with_retry(
            "GET",
            f"/conversations/{quote(conversation_id, safe='')}/events",
            params={
                "include_compacted": "true" if include_compacted else "false",
                "min_seq": min_seq,
            },
        )
        return list(body.get("events") or [])

    def post_conversation_events(
        self,
        conversation_id: str,
        assignee: str,
        request_sequence: int,
        events: List[Dict[str, Any]],
        compact: bool = False,
    ) -> int:
        body = self._request_with_retry(
            "POST",
            f"/conversations/{quote(conversation_id, safe='')}/events",
            json={
                "assignee": assignee,
                "request_sequence": request_sequence,
                "events": events,
                "compact": compact,
            },
        )
        seq = body.get("seq")
        if seq is None:
            raise ConversationServiceError(
                f"post events answer carries no seq: {body!r}"
            )
        return int(seq)

    def update_conversation_status(
        self,
        conversation_id: str,
        request_sequence: int,
        assignee: str,
        status: str,
    ) -> bool:
        body = self._request_with_retry(
            "POST",
            f"/conversations/{quote(conversation_id, safe='')}/status",
            json={
                "assignee": assignee,
                "request_sequence": request_sequence,
                "status": status,
            },
        )
        return bool(body.get("ok"))

    # ---- transport ----

    @retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _request_with_retry(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        read_timeout: float = READ_TIMEOUT_SECONDS,
    ) -> Dict[str, Any]:
        account_id, token = self._dal.get_ai_credentials()
        response = self._session.request(
            method,
            self._agent_url + path,
            json=json,
            params=params,
            headers={"Authorization": f"Bearer {account_id} {token}"},
            timeout=(CONNECT_TIMEOUT_SECONDS, read_timeout),
        )
        if response.status_code == 409:
            body = _json_body(response)
            if body.get("error") == "reassigned":
                raise ConversationReassignedError(
                    body.get("message") or "conversation reassigned"
                )
        if response.status_code == 401:
            # The token was revoked or the AuthTokens row is gone: the cache
            # would otherwise serve it for up to 23 h. Not a retry (the
            # contract keeps 4xx final); the next call mints a fresh token.
            self._forget_credentials()
        if response.status_code >= 400:
            raise ConversationServiceError(
                _error_text(response), status_code=response.status_code
            )
        return _json_body(response)

    def _forget_credentials(self) -> None:
        with self._dal.lock:
            self._dal.token_cache.pop("session_token", None)


def _json_body(response: requests.Response) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = None
    return body if isinstance(body, dict) else {}
