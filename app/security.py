"""API key minting, hashing and verification.

Three properties this module has to hold, in order of how badly each one bites
if it is wrong:

1. **The plaintext key is never stored and never logged.** It is returned once,
   by the call that mints it, and after that only the hash exists. Every log
   line in this codebase records ``key_id``, never the key.
2. **Verification is constant-time and prefix-narrowed.** The presented key
   selects a single candidate row by its 8-character prefix, then the hash is
   compared with a constant-time primitive. Comparing hashes across the whole
   table per request would be both slow and a timing oracle on which prefixes
   exist.
3. **A server-side pepper is mixed in.** The pepper lives in the environment,
   not the database, so a dump of ``api_keys`` alone is not enough to mount an
   offline attack. Rotating ``API_KEY_PEPPER`` invalidates every issued key,
   which is the intended emergency lever.

**On the choice of hash.** New keys are stored as HMAC-SHA256 over the key and
the pepper, and that is deliberate rather than lazy. Argon2 and PBKDF2 exist to
make brute-forcing *low-entropy human passwords* expensive; an API key here is
32 bytes from ``secrets.token_urlsafe``, so brute force is infeasible no matter
how fast the hash is. What this actually needs is (a) never storing plaintext,
(b) a constant-time comparison, and (c) a pepper so a database dump alone is not
enough -- and HMAC-SHA256 gives all three in microseconds.

Argon2id was the first implementation and it cost **81ms of CPU on every single
request**, measured, which is a large fraction of the latency budget for an
endpoint that is supposed to answer in under 500ms. Paying that for a property
the threat model does not need is the wrong trade.

Both algorithms remain *verifiable*: the stored string names its own algorithm,
so keys minted under the old scheme keep working and can be rotated at leisure.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Keys are prefixed so a leaked one is greppable in a log aggregator and
#: identifiable on sight. ``nip`` = narrative intelligence platform.
KEY_PREFIX = "nip_"
#: 32 bytes of entropy, urlsafe-base64'd. Well beyond brute force, and short
#: enough to paste into a header by hand during a demo.
KEY_ENTROPY_BYTES = 32
#: How many leading characters are stored in the clear for identification. Eight
#: is enough to be unique in practice and far too few to be useful to an
#: attacker, since the remaining ~35 characters are the actual secret.
PREFIX_LEN = 8

_PBKDF2_ITERATIONS = 240_000

#: The algorithm new keys are minted with. See the module docstring.
_DEFAULT_ALGO = "hmac-sha256"


@dataclass(frozen=True)
class MintedKey:
    """The one and only time the plaintext exists."""

    plaintext: str
    prefix: str
    key_hash: str


def _pepper() -> bytes:
    from app.config import get_api_settings

    pepper = get_api_settings().api_key_pepper
    if not pepper:
        # Refusing to start would be worse: the API must boot to report what is
        # wrong. Refusing to *authenticate* is the correct failure -- an empty
        # pepper silently accepted would make every hash in the table
        # attackable offline, and nobody would find out.
        raise RuntimeError(
            "API_KEY_PEPPER is unset. Generate one with "
            '`python -c "import secrets; print(secrets.token_urlsafe(48))"` '
            "and put it in .env. Authentication is disabled until it is set."
        )
    return pepper.encode("utf-8")


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(KEY_ENTROPY_BYTES)


def key_prefix(plaintext: str) -> str:
    return plaintext[:PREFIX_LEN]


def hash_key(plaintext: str) -> str:
    """Hash a key for storage. Format: ``algo$params$salt$digest``.

    The salt is still per-key even though HMAC does not strictly need one for a
    high-entropy secret: it means two identical keys (which cannot happen, but
    still) do not produce identical rows, and it keeps the stored format
    uniform across all three algorithms.
    """
    salt = secrets.token_bytes(16)
    material = plaintext.encode("utf-8") + b"|" + _pepper()
    digest = hmac.new(salt + _pepper(), material, hashlib.sha256).digest()
    return "$".join([_DEFAULT_ALGO, "v=1", _b64(salt), _b64(digest)])


def verify_key(plaintext: str, stored: str) -> bool:
    """Constant-time verification against a stored hash. Never raises."""
    try:
        algo, params, salt_b64, digest_b64 = stored.split("$", 3)
        salt = _unb64(salt_b64)
        expected = _unb64(digest_b64)
        material = plaintext.encode("utf-8") + b"|" + _pepper()

        if algo == "hmac-sha256":
            actual = hmac.new(salt + _pepper(), material, hashlib.sha256).digest()
        elif algo == "argon2id":
            from argon2.low_level import Type, hash_secret_raw

            parsed = dict(part.split("=", 1) for part in params.split(","))
            actual = hash_secret_raw(
                secret=material,
                salt=salt,
                time_cost=int(parsed["t"]),
                memory_cost=int(parsed["m"]),
                parallelism=int(parsed["p"]),
                hash_len=len(expected),
                type=Type.ID,
            )
        elif algo == "pbkdf2-sha256":
            iterations = int(params.split("=", 1)[1])
            actual = hashlib.pbkdf2_hmac("sha256", material, salt, iterations, dklen=len(expected))
        else:
            log.error("unknown key hash algorithm %r; refusing to verify", algo)
            return False

        return hmac.compare_digest(actual, expected)
    except RuntimeError:
        # No pepper configured. Propagate: this is a misconfiguration the
        # operator has to see, not an invalid key.
        raise
    except Exception as exc:
        log.warning("malformed stored key hash (%s); treating as no match", type(exc).__name__)
        return False


def mint() -> MintedKey:
    plaintext = generate_key()
    return MintedKey(
        plaintext=plaintext,
        prefix=key_prefix(plaintext),
        key_hash=hash_key(plaintext),
    )


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
