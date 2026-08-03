"""Base types shared by all ingestion adapters.

Privacy contract
----------------
- ``QuarantineEntry`` never contains source values, raw identifiers, or secrets.
  It records only: row number, stable error code, affected field name (canonical,
  not the source name), and a sanitized human-readable message.
- Ingestion adapters must never log, raise, or store raw field values in any
  form (strings, reprs, exception messages, or error context).
- ``PseudonymService`` keys must never appear in any output, log, or error.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from password_attack_detector.data.schemas import AuthEvent

__all__ = [
    "EmptyInputDiagnostic",
    "IngestionResult",
    "InvalidRowPolicy",
    "QuarantineEntry",
]


class InvalidRowPolicy(StrEnum):
    """Controls how the adapter handles rows that fail per-row validation.

    ``FAIL``
        Abort the entire ingestion as soon as any invalid row is encountered.
        No events are emitted and no partial result is returned.

    ``QUARANTINE``
        Skip the invalid row, record a privacy-safe ``QuarantineEntry``, and
        continue processing remaining rows.  If all rows are quarantined the
        result has ``accepted_count == 0`` and ``empty_diagnostic`` is set.
    """

    FAIL = "fail"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class QuarantineEntry:
    """Privacy-safe record of one row rejected during ingestion.

    Fields never contain raw values from the source data, original identifiers,
    file-system paths, or secret keys.
    """

    row_number: int
    """1-based row number in the source file (header counts as row 0 for CSV)."""

    error_code: str
    """Stable machine-readable code identifying the rejection reason."""

    field_name: str | None
    """Canonical field name involved in the rejection, or ``None`` if the
    rejection is not attributable to a single field."""

    sanitized_message: str
    """Human-readable description containing no raw source values."""


@dataclass(frozen=True)
class EmptyInputDiagnostic:
    """Explains why an ingestion run produced zero accepted events."""

    reason_code: str
    """Stable machine-readable code."""

    message: str
    """Human-readable explanation (no raw values, no paths, no secrets)."""


@dataclass(frozen=True)
class IngestionResult:
    """Complete outcome of one ingestion run.

    ``events`` contains the successfully parsed ``AuthEvent`` objects.
    When ``accepted_count == 0``, ``empty_diagnostic`` is set with the reason.
    When rows were quarantined, ``quarantine`` is non-empty.
    """

    accepted_count: int
    quarantine_count: int
    total_rows_read: int
    events: tuple[AuthEvent, ...]
    quarantine: tuple[QuarantineEntry, ...]
    empty_diagnostic: EmptyInputDiagnostic | None
