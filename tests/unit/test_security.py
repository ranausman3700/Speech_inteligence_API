"""API-key authentication unit tests."""

import pytest

from speech_intelligence_api.domain.errors import AuthenticationError
from speech_intelligence_api.entrypoints.http.security import ApiKeyAuthenticator, api_key_digest
from tests.factories import TEST_API_KEY, TEST_HMAC_SECRET, make_settings


def test_api_key_digest_is_deterministic_and_not_plaintext() -> None:
    first = api_key_digest(TEST_API_KEY, TEST_HMAC_SECRET)
    second = api_key_digest(TEST_API_KEY, TEST_HMAC_SECRET)

    assert first == second
    assert len(first) == 64
    assert TEST_API_KEY not in first


def test_authenticator_returns_only_digest_identifier() -> None:
    principal = ApiKeyAuthenticator(make_settings()).authenticate(TEST_API_KEY)

    assert len(principal.identifier) == 12
    assert TEST_API_KEY not in principal.identifier


@pytest.mark.parametrize("candidate", [None, "", "wrong-key", "x" * 513])
def test_authenticator_rejects_missing_or_invalid_keys(candidate: str | None) -> None:
    with pytest.raises(AuthenticationError):
        ApiKeyAuthenticator(make_settings()).authenticate(candidate)


def test_disabled_authenticator_allows_local_development() -> None:
    principal = ApiKeyAuthenticator(make_settings(auth_enabled=False)).authenticate(None)

    assert principal.identifier == "authentication-disabled"
