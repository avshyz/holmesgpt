"""HTTP client for the platform-conversations agent endpoints.

The contract is relay/docs/design/2026-11-24_chat-service-poc.md, sections 6
and 7. Holmes only ever connects outbound: the session trusts the CA bundle
and the proxy environment the way the DAL does, and authenticates with the
HOLMES session token the DAL already mints for Robusta AI.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from holmes.clients.robusta_client import is_retryable_relay_error
from holmes.common.env_vars import CONVERSATIONS_SERVICE_URL
from holmes.core.conversations_worker.models import ConversationReassignedError

if TYPE_CHECKING:
    from holmes.core.supabase_dal import SupabaseDal

# The service holds a claim open for at most this long (contract, section 6);
# it answers at once when rows exist.
CLAIM_WAIT_SECONDS = 25
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 30
# The claim's read timeout is CLAIM_WAIT_SECONDS plus this, so the service ends
# the long-poll and the client never gives up on a healthy call.
CLAIM_READ_TIMEOUT_MARGIN_SECONDS = 15


class ConversationServiceTransport:
    """The four agent calls of the contract, each raising
    ``ConversationReassignedError`` on ``409 {error: "reassigned"}`` and
    ``requests.HTTPError`` on any other 4xx/5xx that survived the retries.
    """

    def __init__(self, dal: "SupabaseDal", base_url: str = CONVERSATIONS_SERVICE_URL):
        self._dal = dal
        self._agent_url = base_url.rstrip("/") + "/agent"
        self._session = requests.Session()
        # requests reads REQUESTS_CA_BUNDLE on its own but not SSL_CERT_FILE,
        # which the DAL and the Realtime socket honour.
        self._session.verify = os.environ.get("SSL_CERT_FILE") or True

    # ---- the contract ----

    def claim(self, cluster_id: str, assignee: str, limit: int) -> List[Dict[str, Any]]:
        body = self._request_with_retry(
            "POST",
            "/claim",
            json={
                "cluster_id": cluster_id,
                "assignee": assignee,
                "limit": limit,
                "wait_seconds": CLAIM_WAIT_SECONDS,
            },
            read_timeout=CLAIM_WAIT_SECONDS + CLAIM_READ_TIMEOUT_MARGIN_SECONDS,
        )
        return body.get("conversations") or []

    def get_conversation_events(
        self,
        conversation_id: str,
        include_compacted: bool = False,
        min_seq: int = 1,
    ) -> List[Dict[str, Any]]:
        body = self._request_with_retry(
            "GET",
            f"/conversations/{conversation_id}/events",
            params={
                "include_compacted": "true" if include_compacted else "false",
                "min_seq": min_seq,
            },
        )
        return body.get("events") or []

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
            f"/conversations/{conversation_id}/events",
            json={
                "assignee": assignee,
                "request_sequence": request_sequence,
                "events": events,
                "compact": compact,
            },
        )
        return int(body["seq"])

    def update_conversation_status(
        self,
        conversation_id: str,
        request_sequence: int,
        assignee: str,
        status: str,
    ) -> bool:
        body = self._request_with_retry(
            "POST",
            f"/conversations/{conversation_id}/status",
            json={
                "assignee": assignee,
                "request_sequence": request_sequence,
                "status": status,
            },
        )
        return bool(body.get("ok"))

    # ---- transport ----

    # At-least-once, as on the Supabase path: an answer lost after the service
    # committed re-sends the claim or the batch, and a duplicate is accepted
    # over a dropped turn.
    @retry(
        retry=retry_if_exception(is_retryable_relay_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
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
            with self._dal.lock:
                self._dal.token_cache.pop("session_token", None)
        if response.status_code >= 400:
            raise requests.HTTPError(
                f"{response.status_code} {response.reason}: {response.text[:500]}",
                response=response,
            )
        # A proxy answering 2xx with HTML, an empty body or JSON null must not
        # pass as an empty answer: the claim loop backs off only on an error.
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"unexpected answer body: {body!r}")
        return body


def _json_body(response: requests.Response) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = None
    return body if isinstance(body, dict) else {}
