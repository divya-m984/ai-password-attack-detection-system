"""CSV ingestion adapter for canonical authentication-event data.

Usage::

    adapter = CSVIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE)
    result = adapter.ingest(Path("events.csv"))
    events = result.events

Header inspection
-----------------
The complete header row is inspected **before** any data row is read:

1. Any column whose normalized name is in ``PROHIBITED_NORMALIZED`` (sensitive
   credential fields such as ``password``, ``token``) causes the entire
   dataset to be rejected via ``IngestionError``.
2. Any column present in ``PROHIBITED_GT_COLUMNS`` (ground-truth leakage
   columns) causes the entire dataset to be rejected via ``IngestionError``.

Column normalization
--------------------
Header names are normalized before mapping to canonical field names:

- Leading/trailing whitespace stripped
- Lowercase
- Hyphens and internal spaces replaced with underscores

Extra columns (not in the canonical schema) are silently ignored.
Empty CSV string values (``""``, ``"none"``, ``"null"``, ``"na"``) are
converted to ``None`` before passing to the Pydantic model.

Privacy
-------
Raw source values never appear in ``QuarantineEntry.sanitized_message``,
exception messages, or log output.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from password_attack_detector.data.ingestion.base import (
    EmptyInputDiagnostic,
    IngestionResult,
    InvalidRowPolicy,
    QuarantineEntry,
)
from password_attack_detector.data.privacy import (
    PseudonymService,
    scan_prohibited_keys,
)
from password_attack_detector.data.schemas import (
    PROHIBITED_GT_COLUMNS,
    AuthEvent,
)
from password_attack_detector.exceptions import IngestionError

__all__ = ["CSVIngestionAdapter"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# String values (after stripping and lowercasing) treated as null.
_EMPTY_CSV_VALUES: frozenset[str] = frozenset({"", "none", "null", "na", "n/a"})

# Canonical field names from the AuthEvent schema.
_CANONICAL_FIELDS: frozenset[str] = frozenset(AuthEvent.model_fields.keys())

# Mapping from identifier field name to PseudonymService domain.
_PSEUDO_DOMAINS: dict[str, str] = {
    "user_id": "user",
    "source_id": "source",
    "device_id": "device",
    "session_id": "session",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_header(name: str) -> str:
    """Normalize a header name for canonical field matching.

    Strips whitespace, lowercases, and replaces hyphens or internal spaces
    with underscores.  Does **not** perform camelCase splitting.
    """
    cleaned = name.strip().lower()
    return re.sub(r"[-\s]+", "_", cleaned)


def _csv_val(raw: str | None) -> object:
    """Convert a CSV cell value to a Python object for ``AuthEvent``.

    Returns ``None`` for empty or null-like strings; otherwise returns the
    stripped string (Pydantic coerces string → int / float / UUID / datetime).
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if stripped.lower() in _EMPTY_CSV_VALUES:
        return None
    return stripped


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CSVIngestionAdapter:
    """Ingest a canonical CSV file into a sequence of ``AuthEvent`` objects.

    Parameters
    ----------
    policy:
        Controls per-row failure handling (FAIL aborts; QUARANTINE skips).
    pseudonym_service:
        When provided, the adapter pseudonymizes ``user_id``, ``source_id``,
        ``device_id``, and ``session_id`` fields before constructing events.
        The key is never included in any output or error message.
    """

    def __init__(
        self,
        policy: InvalidRowPolicy = InvalidRowPolicy.FAIL,
        pseudonym_service: PseudonymService | None = None,
        extra_field_map: dict[str, str] | None = None,
    ) -> None:
        self._policy = policy
        self._svc = pseudonym_service
        self._extra_field_map: dict[str, str] = extra_field_map or {}

    def ingest(self, path: Path) -> IngestionResult:
        """Read *path* and ingest all rows as ``AuthEvent`` objects.

        Parameters
        ----------
        path:
            Path to the CSV file (UTF-8 encoded, with header row).

        Returns
        -------
        IngestionResult
            Statistics and accepted events.

        Raises
        ------
        IngestionError
            When the file cannot be opened, the header contains prohibited
            columns, or a row fails validation under ``FAIL`` policy.
        """
        try:
            fh = path.open(encoding="utf-8", newline="")
        except OSError as exc:
            raise IngestionError(
                f"Cannot open CSV file ({type(exc).__name__})"
            ) from exc

        accepted: list[AuthEvent] = []
        quarantine: list[QuarantineEntry] = []
        total_rows = 0

        with fh:
            reader = csv.DictReader(fh)

            # Step 1: Inspect complete header before reading any rows.
            if reader.fieldnames is None:
                raise IngestionError("CSV file has no header row; ingestion rejected")

            raw_headers: list[str] = list(reader.fieldnames)
            normalized_map = self._check_and_map_headers(raw_headers)

            # Step 2: Process rows.
            for row in reader:
                total_rows += 1
                row_num = total_rows

                data = self._extract_row(row, normalized_map)
                event_or_none, entry = self._build_event(data, row_num)

                if event_or_none is not None:
                    accepted.append(event_or_none)
                elif entry is not None:
                    if self._policy == InvalidRowPolicy.FAIL:
                        raise IngestionError(
                            f"Row {row_num} failed ingestion "
                            f"(code: {entry.error_code}); aborted (policy=FAIL)"
                        )
                    quarantine.append(entry)

        # Build diagnostic for empty result.
        empty_diag: EmptyInputDiagnostic | None = None
        if total_rows == 0:
            empty_diag = EmptyInputDiagnostic(
                reason_code="E001",
                message="CSV file contains no data rows",
            )
        elif not accepted:
            empty_diag = EmptyInputDiagnostic(
                reason_code="E002",
                message="All rows were quarantined; no events accepted",
            )

        return IngestionResult(
            accepted_count=len(accepted),
            quarantine_count=len(quarantine),
            total_rows_read=total_rows,
            events=tuple(accepted),
            quarantine=tuple(quarantine),
            empty_diagnostic=empty_diag,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_and_map_headers(self, raw_headers: list[str]) -> dict[str, str]:
        """Inspect headers and return a mapping {raw_header: canonical_field}.

        Raises ``IngestionError`` if any header is prohibited.
        Returns only entries where the normalized name matches a canonical field.
        """
        # Check sensitive credential fields via scan_prohibited_keys.
        header_dict: dict[str, Any] = dict.fromkeys(raw_headers, "")
        prohibited = scan_prohibited_keys(header_dict)
        if prohibited:
            raise IngestionError(
                f"CSV header contains {len(prohibited)} prohibited sensitive "
                f"field name(s); ingestion rejected"
            )

        # Check for GT leakage columns (exact normalized match).
        normalized_to_raw: dict[str, str] = {
            _normalize_header(h): h for h in raw_headers
        }
        gt_present = PROHIBITED_GT_COLUMNS & set(normalized_to_raw.keys())
        if gt_present:
            raise IngestionError(
                f"CSV header contains {len(gt_present)} prohibited "
                f"ground-truth column(s); ingestion rejected"
            )

        # Build canonical field map; extra_field_map takes precedence.
        result: dict[str, str] = {}
        for norm, raw in normalized_to_raw.items():
            if raw in self._extra_field_map:
                canonical = self._extra_field_map[raw]
                if canonical in _CANONICAL_FIELDS:
                    result[raw] = canonical
            elif norm in _CANONICAL_FIELDS:
                result[raw] = norm
        return result

    def _extract_row(
        self,
        row: dict[str, str | None],
        normalized_map: dict[str, str],
    ) -> dict[str, Any]:
        """Extract canonical fields from one CSV row dict."""
        data: dict[str, Any] = {}
        for raw_key, canonical in normalized_map.items():
            data[canonical] = _csv_val(row.get(raw_key))
        return data

    def _build_event(
        self,
        data: dict[str, Any],
        row_num: int,
    ) -> tuple[AuthEvent | None, QuarantineEntry | None]:
        """Pseudonymize identifiers (if applicable) and construct ``AuthEvent``.

        Returns ``(event, None)`` on success or ``(None, entry)`` on failure.
        Raw source values never appear in the returned ``QuarantineEntry``.
        """
        # Pseudonymize identifier fields when service is provided.
        if self._svc is not None:
            for field, domain in _PSEUDO_DOMAINS.items():
                val = data.get(field)
                if val is not None:
                    try:
                        data[field] = self._svc.pseudonymize(domain, str(val))
                    except Exception:
                        return None, QuarantineEntry(
                            row_number=row_num,
                            error_code="I003",
                            field_name=field,
                            sanitized_message=(
                                f"Row {row_num}: pseudonymization failed for "
                                f"field {field!r}"
                            ),
                        )

        # Construct AuthEvent; Pydantic validates and coerces types.
        try:
            event = AuthEvent(**data)
        except ValidationError as exc:
            errs = exc.errors()
            first_loc = errs[0].get("loc", ()) if errs else ()
            field_name: str | None = str(first_loc[0]) if first_loc else None
            return None, QuarantineEntry(
                row_number=row_num,
                error_code="I002",
                field_name=field_name,
                sanitized_message=(
                    f"Row {row_num}: schema validation failed "
                    f"({len(errs)} field error(s))"
                ),
            )
        except Exception:
            return None, QuarantineEntry(
                row_number=row_num,
                error_code="I002",
                field_name=None,
                sanitized_message=f"Row {row_num}: unexpected error during row parsing",
            )

        return event, None
