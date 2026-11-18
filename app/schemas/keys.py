"""API key management schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from app.schemas.common import Camel

Scope = Literal["read", "write", "admin"]


class ApiKeyCreate(Camel):
    name: str = Field(
        min_length=1, max_length=128, description="Human label, e.g. 'phase5-dashboard'."
    )
    scopes: list[Scope] = Field(
        default_factory=lambda: ["read"],
        description="Reads need 'read', mutations 'write', key management 'admin'.",
    )

    @field_validator("scopes")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("a key with no scopes cannot do anything; grant at least 'read'")
        return sorted(set(v))


class ApiKeyOut(Camel):
    """A key as listed. Never contains the secret."""

    id: str
    name: str
    prefix: str = Field(description="First 8 characters, for identification only.")
    scopes: list[str]
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    is_active: bool


class ApiKeySelf(Camel):
    """The calling key as it sees itself. Never contains the secret or the hash."""

    id: str = Field(description="Key id, or 'demo' when DEMO_MODE answers without a key.")
    name: str
    prefix: str | None = Field(
        default=None,
        description="First 8 characters of the key. Null when the caller is not a stored key.",
    )
    scopes: list[str] = Field(description="Scopes granted to the key, sorted.")
    effective_scopes: list[str] = Field(
        description="Scopes the key can exercise: `admin` implies `read` and `write`."
    )
    can_read: bool
    can_write: bool
    can_admin: bool


class ApiKeyCreated(ApiKeyOut):
    """The mint response.

    ``key`` appears here and nowhere else, ever. It is not recoverable: only a
    peppered hash is stored, and a lost key is reissued rather than looked up.
    """

    key: str = Field(
        description="The plaintext key. Shown once. Store it now; it cannot be recovered."
    )
