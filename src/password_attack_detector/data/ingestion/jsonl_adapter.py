"""JSONL (newline-delimited JSON) ingestion adapter.

Usage::

    adapter = JSONLIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE)
    result = adapter.ingest(Path("events.jsonl"))
    events = result.events

Key inspection
--------------
For **every** JSONL record, all JSON keys are inspected recursively
*before* any field values are read or processed:

1. If any key (at any nesting depth) is in ``PROHIBITED_NORMALIZED`` (sensitive
   credential fields) the **entire dataset** is rejected via ``IngestionError``.
2. If any key is in ``PROHIBITED_GT_COLUMNS`` the entire dataset is rejected.
3. If nesting depth exceeds ``MAX_NESTING_DEPTH`` (5) the entire dataset is
   rejected via ``IngestionError``.

JSON parse errors are handled per the configured ``InvalidRowPolicy``:
- ``FAIL``: raises ``IngestionError`` on the first parse-error line.
- ``QUARANTINE``: records a ``QuarantineEntry`` and continues.

Privacy
-------
Raw source values never appear in ``QuarantineEntry.sanitized_message``,
exception messages, or log output.  The ``PseudonymService`` key is never
stored, logged, or included in any error.
"""

from __future__ import annotations

import json
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

__all__ = ["JSONLIngestionAdapter"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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


def _check_prohibited_keys(record: dict[str, Any], line_num: int) -> None:
    """Raise ``IngestionError`` if *record* contains prohibited keys.

    Checks both sensitive credential keys (``PROHIBITED_NORMALIZED``) and
    GT leakage keys (``PROHIBITED_GT_COLUMNS``).  Enforces ``MAX_NESTING_DEPTH``.

    Parameters
    ----------
    record:
        Parsed JSON object.
    line_num:
        1-based line number for diagnostic messages.

    Raises
    ------
    IngestionError
        If any prohibited key is found or nesting depth is exceeded.
    """
    # scan_prohibited_keys raises IngestionError on depth exceeded.
    sensitive = scan_prohibited_keys(record)
    if sensitive:
        raise IngestionError(
            f"Record at line {line_num} contains {len(sensitive)} prohibited "
            f"sensitive key(s); entire dataset rejected"
        )

    # Check GT leakage columns (exact key names).
    gt_present = PROHIBITED_GT_COLUMNS & set(record.keys())
    if gt_present:
        raise IngestionError(
            f"Record at line {line_num} contains {len(gt_present)} prohibited "
            f"ground-truth key(s); entire dataset rejected"
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class JSONLIngestionAdapter:
    """Ingest a JSONL (newline-delimited JSON) file into ``AuthEvent`` objects.

    Parameters
    ----------
    policy:
        Controls per-record failure handling (FAIL aborts; QUARANTINE skips).
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
        """Read *path* and ingest all records as ``AuthEvent`` objects.

        Parameters
        ----------
        path:
            Path to the JSONL file (UTF-8 encoded, one JSON object per line).

        Returns
        -------
        IngestionResult
            Statistics and accepted events.

        Raises
        ------
        IngestionError
            When the file cannot be opened, any record contains prohibited keys
            or exceeds the nesting depth limit, or a record fails validation
            under ``FAIL`` policy.
        """
        try:
            raw_lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise IngestionError(
                f"Cannot read JSONL file ({type(exc).__name__})"
            ) from exc

        accepted: list[AuthEvent] = []
        quarantine: list[QuarantineEntry] = []
        total_rows = 0

        for line_num, raw_line in enumerate(raw_lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                # Skip blank lines.
                continue

            total_rows += 1

            # Step 1: Parse JSON.
            try:
                record: dict[str, Any] = json.loads(stripped)
            except json.JSONDecodeError as exc:
                entry = QuarantineEntry(
                    row_number=line_num,
                    error_code="I005",
                    field_name=None,
                    sanitized_message=f"Line {line_num}: JSON parse error",
                )
                if self._policy == InvalidRowPolicy.FAIL:
                    raise IngestionError(
                        f"Line {line_num}: JSON parse error; aborted (policy=FAIL)"
                    ) from exc
                quarantine.append(entry)
                continue

            if not isinstance(record, dict):
                entry = QuarantineEntry(
                    row_number=line_num,
                    error_code="I005",
                    field_name=None,
                    sanitized_message=(
                        f"Line {line_num}: expected JSON object, got "
                        f"{type(record).__name__}"
                    ),
                )
                if self._policy == InvalidRowPolicy.FAIL:
                    raise IngestionError(
                        f"Line {line_num}: expected JSON object; aborted (policy=FAIL)"
                    )
                quarantine.append(entry)
                continue

            # Step 2: Inspect all keys recursively BEFORE reading any values.
            # _check_prohibited_keys raises IngestionError on any prohibited key
            # or nesting depth exceeded -- this rejects the entire dataset.
            _check_prohibited_keys(record, line_num)

            # Step 3: Extract canonical fields and build AuthEvent.
            event_or_none, q_entry = self._build_event(record, line_num)

            if event_or_none is not None:
                accepted.append(event_or_none)
            elif q_entry is not None:
                if self._policy == InvalidRowPolicy.FAIL:
                    raise IngestionError(
                        f"Line {line_num} failed ingestion "
                        f"(code: {q_entry.error_code}); aborted (policy=FAIL)"
                    )
                quarantine.append(q_entry)

        # Build diagnostic for empty result.
        empty_diag: EmptyInputDiagnostic | None = None
        if total_rows == 0:
            empty_diag = EmptyInputDiagnostic(
                reason_code="E001",
                message="JSONL file contains no records",
            )
        elif not accepted:
            empty_diag = EmptyInputDiagnostic(
                reason_code="E002",
                message="All records were quarantined; no events accepted",
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

    def _build_event(
        self,
        record: dict[str, Any],
        line_num: int,
    ) -> tuple[AuthEvent | None, QuarantineEntry | None]:
        """Pseudonymize identifiers (if applicable) and construct ``AuthEvent``.

        Extracts only canonical fields from *record*; extra keys are ignored.
        Returns ``(event, None)`` on success or ``(None, entry)`` on failure.
        Raw source values never appear in the returned ``QuarantineEntry``.
        """
        # Extract only canonical fields; extra_field_map renames source keys.
        data: dict[str, Any] = {}
        for k, v in record.items():
            if k in self._extra_field_map:
                canonical = self._extra_field_map[k]
                if canonical in _CANONICAL_FIELDS:
                    data[canonical] = v
            elif k in _CANONICAL_FIELDS:
                data[k] = v

        # Pseudonymize identifier fields when service is provided.
        if self._svc is not None:
            for field, domain in _PSEUDO_DOMAINS.items():
                val = data.get(field)
                if val is not None:
                    try:
                        data[field] = self._svc.pseudonymize(domain, str(val))
                    except Exception:
                        return None, QuarantineEntry(
                            row_number=line_num,
                            error_code="I003",
                            field_name=field,
                            sanitized_message=(
                                f"Line {line_num}: pseudonymization failed for "
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
                row_number=line_num,
                error_code="I002",
                field_name=field_name,
                sanitized_message=(
                    f"Line {line_num}: schema validation failed "
                    f"({len(errs)} field error(s))"
                ),
            )
        except Exception:
            return None, QuarantineEntry(
                row_number=line_num,
                error_code="I002",
                field_name=None,
                sanitized_message=(
                    f"Line {line_num}: unexpected error during record parsing"
                ),
            )

        return event, None
