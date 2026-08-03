"""Privacy-preserving pseudonymization service and prohibited-field scanner.

Pseudonymization
----------------
``PseudonymService`` generates stable, prefixed pseudonyms using HMAC-SHA-256
with domain separation.  The key must be at least 256 bits (32 bytes) and is
supplied exclusively via the ``PAD_PSEUDONYMIZATION_KEY`` environment variable
or an untracked ``.env`` file.

The key is stored only as a ``bytes`` instance variable and is never logged,
serialised, included in exception messages, or stored in class or module state.

Synthetic generation does not use ``PseudonymService``.  Synthetic identifiers
are produced deterministically from UUIDv5 without requiring a real secret.

Prohibited-field detection
--------------------------
``scan_prohibited_keys`` recursively scans a parsed JSON object for field names
that match the prohibited sensitive-field list after normalisation.  Normalisation
strips hyphens, underscores, and spaces, lower-cases the result, and collapses
camelCase boundaries so that variants such as ``passwordHash``, ``password_hash``,
and ``Password`` all match ``password``.

For CSV data, scan the header columns once.
For JSONL data, scan *every* record before processing any values.

A prohibited field in any record causes complete dataset rejection under both
the ``FAIL`` and ``QUARANTINE`` ingestion policies.  The scanner never reads,
copies, logs, or serialises prohibited values.

Privacy limitations
-------------------
Pseudonymization reduces re-identification risk but does not make an
authentication dataset anonymous.  Keyed HMAC pseudonyms are reversible for
anyone who holds the key.  Dataset operators are responsible for key management,
access control, and applicable privacy regulations.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import TYPE_CHECKING, Any, ClassVar

from password_attack_detector.exceptions import IngestionError, PseudonymizationError

if TYPE_CHECKING:
    from password_attack_detector.config import Settings

__all__ = [
    "MAX_NESTING_DEPTH",
    "PROHIBITED_NORMALIZED",
    "PseudonymService",
    "scan_prohibited_keys",
]

# Maximum JSON object nesting depth inspected by the prohibited-field scanner.
# Inputs exceeding this depth are rejected with an IngestionError.
MAX_NESTING_DEPTH: int = 5

# Normalized forms of prohibited sensitive field names.
# Normalization: lowercase, strip hyphens/underscores/spaces, collapse camelCase.
PROHIBITED_NORMALIZED: frozenset[str] = frozenset(
    {
        "password",
        "passwordhash",
        "secret",
        "token",
        "accesstoken",
        "refreshtoken",
        "authorization",
        "cookie",
        "privatekey",
    }
)

# Regex used to split camelCase at uppercase boundaries.
# "passwordHash" → "password Hash" → after lower+strip → "passwordhash"
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATOR_RE = re.compile(r"[-_\s]+")


def _normalize_field_name(name: str) -> str:
    """Return the normalized form of *name* for prohibited-field comparison.

    Normalization steps:
    1. Insert a space before each camelCase transition (e.g. ``Hash`` → `` Hash``)
    2. Lowercase the result
    3. Remove all hyphens, underscores, and spaces
    """
    spaced = _CAMEL_RE.sub(" ", name)
    lowered = spaced.lower()
    return _SEPARATOR_RE.sub("", lowered)


def scan_prohibited_keys(
    obj: dict[str, Any],
    depth: int = 0,
) -> list[str]:
    """Recursively scan *obj* for prohibited sensitive-field keys.

    Parameters
    ----------
    obj:
        Parsed JSON object (dict) to inspect.
    depth:
        Current recursion depth.  Callers should pass 0.

    Returns
    -------
    list[str]
        Original (un-normalized) names of any prohibited keys found.
        Empty when the object is clean.

    Raises
    ------
    IngestionError
        When the nesting depth exceeds ``MAX_NESTING_DEPTH``.

    Notes
    -----
    Field *values* are never read, copied, logged, or serialised.
    """
    if depth > MAX_NESTING_DEPTH:
        raise IngestionError(
            f"Input nesting depth exceeds maximum of {MAX_NESTING_DEPTH}; "
            "ingestion rejected"
        )
    found: list[str] = []
    for key, value in obj.items():
        if _normalize_field_name(key) in PROHIBITED_NORMALIZED:
            found.append(key)
        if isinstance(value, dict):
            found.extend(scan_prohibited_keys(value, depth + 1))
    return found


class PseudonymService:
    """Generate stable, prefixed HMAC-SHA-256 pseudonyms with domain separation.

    The pseudonymization key is accepted as a hex string, validated for minimum
    entropy, and stored only as ``bytes``.  It is never logged, serialised,
    included in exception messages, or exposed via ``repr``.

    Usage (real-data ingestion)::

        service = PseudonymService.from_settings(settings)
        user_pseudo = service.pseudonymize("user", original_user_id)

    Synthetic generation must not use this class.
    """

    # Output prefix per domain.
    DOMAIN_PREFIXES: ClassVar[dict[str, str]] = {
        "user": "u",
        "source": "s",
        "device": "d",
        "session": "sess",
    }

    # Minimum key length in bytes (256 bits).
    _MIN_KEY_BYTES: ClassVar[int] = 32

    def __init__(self, key_hex: str) -> None:
        """Initialise with a hex-encoded key of at least 256 bits.

        Parameters
        ----------
        key_hex:
            Hex-encoded secret key string.  Must decode to at least 32 bytes.

        Raises
        ------
        PseudonymizationError
            When *key_hex* is not valid hex or encodes fewer than 32 bytes.
        """
        try:
            raw: bytes = bytes.fromhex(key_hex)
        except ValueError as exc:
            raise PseudonymizationError(
                "Pseudonymization key must be a valid hexadecimal string"
            ) from exc
        if len(raw) < self._MIN_KEY_BYTES:
            raise PseudonymizationError(
                f"Pseudonymization key must be at least {self._MIN_KEY_BYTES * 8} bits "
                f"({self._MIN_KEY_BYTES} bytes); got {len(raw)} bytes"
            )
        # Store only as bytes.  Never retain the hex string.
        self._key = raw

    def __repr__(self) -> str:
        return "PseudonymService(<key redacted>)"

    def pseudonymize(self, domain: str, value: str) -> str:
        """Return a stable prefixed pseudonym for *value* in *domain*.

        Parameters
        ----------
        domain:
            One of ``"user"``, ``"source"``, ``"device"``, ``"session"``.
        value:
            Original identifier to pseudonymize.

        Returns
        -------
        str
            Prefixed pseudonym in the form ``"prefix:hex32"``
            (e.g. ``"u:a1b2c3d4..."``, 32 hex digits after the colon).

        Raises
        ------
        PseudonymizationError
            When *domain* is not a recognised domain name.
        """
        if domain not in self.DOMAIN_PREFIXES:
            raise PseudonymizationError(
                f"Unknown pseudonymization domain: {domain!r}. "
                f"Valid domains: {sorted(self.DOMAIN_PREFIXES)}"
            )
        prefix = self.DOMAIN_PREFIXES[domain]
        msg = f"{domain}:{value}".encode()
        digest = hmac.new(self._key, msg, hashlib.sha256).hexdigest()
        return f"{prefix}:{digest[:32]}"

    @classmethod
    def from_settings(cls, settings: Settings) -> PseudonymService:
        """Construct a ``PseudonymService`` from the active application settings.

        Parameters
        ----------
        settings:
            Loaded ``Settings`` instance.  ``pseudonymization_key`` must be set.

        Raises
        ------
        PseudonymizationError
            When ``settings.pseudonymization_key`` is ``None``.
        """
        if settings.pseudonymization_key is None:
            raise PseudonymizationError(
                "PAD_PSEUDONYMIZATION_KEY must be set via environment variable "
                "or untracked .env file for real-data ingestion"
            )
        return cls(settings.pseudonymization_key.get_secret_value())
