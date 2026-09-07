"""API-key authentication without persistence or plaintext key storage."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from speech_intelligence_api.config import Settings
from speech_intelligence_api.domain.errors import AuthenticationError


@dataclass(frozen=True, slots=True)
class ApiPrincipal:
    """Non-sensitive identity derived from the matching key digest."""

    identifier: str


def api_key_digest(api_key: str, hmac_secret: str) -> str:
    """Return the configured HMAC-SHA256 representation of an API key."""

    return hmac.new(
        hmac_secret.encode("utf-8"),
        api_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


class ApiKeyAuthenticator:
    """Verify API keys against immutable HMAC digests."""

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.auth_enabled
        secret = settings.api_key_hmac_secret
        self._secret = secret.get_secret_value() if secret is not None else ""
        self._digests = settings.api_key_digests

    def authenticate(self, candidate: str | None) -> ApiPrincipal:
        """Return a safe principal or raise the public authentication error."""

        if not self._enabled:
            return ApiPrincipal(identifier="authentication-disabled")
        if candidate is None or not candidate.strip() or len(candidate) > 512:
            raise AuthenticationError

        candidate_digest = api_key_digest(candidate, self._secret)
        matching_identifier: str | None = None
        for configured_digest in self._digests:
            if hmac.compare_digest(candidate_digest, configured_digest):
                matching_identifier = configured_digest[:12]
        if matching_identifier is None:
            raise AuthenticationError
        return ApiPrincipal(identifier=matching_identifier)
