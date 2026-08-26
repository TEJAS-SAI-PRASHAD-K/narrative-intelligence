"""API key auth: hashing, verification, scopes and the failure modes.

The hashing tests need no database and always run. The routing tests need
Postgres and skip cleanly without it.
"""

from __future__ import annotations

import pytest

from tests.conftest_api import requires_postgres


# ---------------------------------------------------------------------------
# hashing / verification -- no database needed
# ---------------------------------------------------------------------------
@pytest.fixture
def peppered(monkeypatch):
    from app.config import get_api_settings

    monkeypatch.setenv("API_KEY_PEPPER", "a-test-pepper")
    get_api_settings.cache_clear()
    yield
    get_api_settings.cache_clear()


def test_minted_key_verifies_and_is_not_stored_in_the_clear(peppered):
    from app.security import KEY_PREFIX, mint, verify_key

    minted = mint()
    assert minted.plaintext.startswith(KEY_PREFIX)
    assert verify_key(minted.plaintext, minted.key_hash)
    # The whole point: the stored form must not contain the secret.
    assert minted.plaintext not in minted.key_hash
    assert minted.plaintext[len(minted.prefix) :] not in minted.key_hash


def test_wrong_key_does_not_verify(peppered):
    from app.security import mint, verify_key

    a, b = mint(), mint()
    assert not verify_key(a.plaintext, b.key_hash)
    assert not verify_key(a.plaintext + "x", a.key_hash)
    assert not verify_key(a.plaintext[:-1], a.key_hash)


def test_new_keys_use_the_fast_algorithm(peppered):
    """Argon2 cost 81ms per request for a property a 256-bit token does not need.

    See the reasoning in app/security.py. This pins the default so nobody
    reintroduces a memory-hard KDF on the request path by reflex.
    """
    from app.security import mint

    assert mint().key_hash.startswith("hmac-sha256$")


def test_a_legacy_argon2_hash_still_verifies(peppered):
    """The stored string names its algorithm, so old keys keep working."""
    import base64
    import secrets

    from app.security import mint, verify_key

    minted = mint()
    try:
        from argon2.low_level import Type, hash_secret_raw
    except ImportError:
        pytest.skip("argon2-cffi is not installed")

    salt = secrets.token_bytes(16)
    material = minted.plaintext.encode() + b"|" + b"a-test-pepper"
    digest = hash_secret_raw(
        secret=material,
        salt=salt,
        time_cost=2,
        memory_cost=65536,
        parallelism=1,
        hash_len=32,
        type=Type.ID,
    )

    def b64(raw):
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    legacy = "$".join(["argon2id", "t=2,m=65536,p=1", b64(salt), b64(digest)])
    assert verify_key(minted.plaintext, legacy)


def test_two_mints_of_the_same_plaintext_differ(peppered):
    """Salted, so an attacker cannot tell two identical keys apart."""
    from app.security import hash_key, verify_key

    plaintext = "nip_deterministic-input"
    first, second = hash_key(plaintext), hash_key(plaintext)
    assert first != second
    assert verify_key(plaintext, first)
    assert verify_key(plaintext, second)


def test_rotating_the_pepper_invalidates_every_key(peppered, monkeypatch):
    """The intended emergency lever, asserted rather than assumed."""
    from app.config import get_api_settings
    from app.security import mint, verify_key

    minted = mint()
    assert verify_key(minted.plaintext, minted.key_hash)

    monkeypatch.setenv("API_KEY_PEPPER", "a-different-pepper")
    get_api_settings.cache_clear()
    assert not verify_key(minted.plaintext, minted.key_hash)


def test_missing_pepper_refuses_to_authenticate(monkeypatch):
    """An empty pepper silently accepted would make the whole table attackable."""
    from app.config import get_api_settings
    from app.security import hash_key

    monkeypatch.setenv("API_KEY_PEPPER", "")
    get_api_settings.cache_clear()
    with pytest.raises(RuntimeError, match="API_KEY_PEPPER"):
        hash_key("nip_whatever")
    get_api_settings.cache_clear()


def test_malformed_stored_hash_is_a_non_match_not_a_crash(peppered):
    from app.security import verify_key

    for junk in ("", "garbage", "argon2id$broken", "unknown-algo$p$s$d"):
        assert verify_key("nip_anything", junk) is False


# ---------------------------------------------------------------------------
# routing -- needs Postgres
# ---------------------------------------------------------------------------
@requires_postgres
def test_missing_key_is_401(client):
    response = client.get("/api/v1/keys")
    assert response.status_code == 401
    body = response.json()["error"]
    assert body["code"] == "unauthorized"
    assert body["request_id"]


@requires_postgres
def test_invalid_key_is_401_and_does_not_leak_whether_the_prefix_exists(client, admin_key):
    unknown_prefix = client.get("/api/v1/keys", headers={"X-API-Key": "nip_zzzzzzzzzzzz"})
    right_prefix_wrong_secret = client.get(
        "/api/v1/keys", headers={"X-API-Key": admin_key[:8] + "wrongwrongwrong"}
    )
    assert unknown_prefix.status_code == right_prefix_wrong_secret.status_code == 401
    assert (
        unknown_prefix.json()["error"]["message"]
        == right_prefix_wrong_secret.json()["error"]["message"]
    )


@requires_postgres
def test_read_scope_gets_403_on_a_mutation(client, read_key):
    response = client.post(
        "/api/v1/keys",
        headers={"X-API-Key": read_key},
        json={"name": "escalation-attempt", "scopes": ["admin"]},
    )
    assert response.status_code == 403
    error = response.json()["error"]
    assert error["code"] == "forbidden"
    assert error["detail"]["required_scope"] == "admin"
    assert error["detail"]["granted_scopes"] == ["read"]


@requires_postgres
def test_admin_can_mint_and_the_key_is_shown_exactly_once(client, admin_key):
    created = client.post(
        "/api/v1/keys",
        headers={"X-API-Key": admin_key},
        json={"name": "phase5-dashboard", "scopes": ["read"]},
    )
    assert created.status_code == 201
    body = created.json()
    plaintext = body["key"]
    assert plaintext.startswith("nip_")
    assert body["prefix"] == plaintext[:8]
    assert body["scopes"] == ["read"]

    # The new key works...
    assert client.get("/api/v1/keys", headers={"X-API-Key": plaintext}).status_code == 403

    # ...and the listing never returns the secret again.
    listing = client.get("/api/v1/keys", headers={"X-API-Key": admin_key}).json()
    assert all("key" not in item for item in listing["items"])
    assert any(item["prefix"] == body["prefix"] for item in listing["items"])


@requires_postgres
def test_revoked_key_stops_working_but_the_row_survives(client, admin_key):
    created = client.post(
        "/api/v1/keys",
        headers={"X-API-Key": admin_key},
        json={"name": "short-lived", "scopes": ["read", "write", "admin"]},
    ).json()
    victim = created["key"]
    assert client.get("/api/v1/keys", headers={"X-API-Key": victim}).status_code == 200

    revoked = client.delete(f"/api/v1/keys/{created['id']}", headers={"X-API-Key": admin_key})
    assert revoked.status_code == 200
    assert revoked.json()["is_active"] is False

    assert client.get("/api/v1/keys", headers={"X-API-Key": victim}).status_code == 401

    # Tombstone, not a delete: the audit trail outlives the key.
    listing = client.get(
        "/api/v1/keys?include_revoked=true", headers={"X-API-Key": admin_key}
    ).json()
    assert any(item["id"] == created["id"] for item in listing["items"])


@requires_postgres
def test_a_key_with_no_scopes_is_rejected_at_validation(client, admin_key):
    response = client.post(
        "/api/v1/keys", headers={"X-API-Key": admin_key}, json={"name": "useless", "scopes": []}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@requires_postgres
def test_healthz_and_readyz_need_no_key(client):
    """An orchestrator probe must not require a credential."""
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code in (200, 503)
