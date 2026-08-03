"""CSV and JSONL ingestion adapters for real authentication-event data.

Public API
----------
- ``InvalidRowPolicy``  -- FAIL or QUARANTINE on bad rows
- ``QuarantineEntry``   -- privacy-safe record of one quarantined row
- ``EmptyInputDiagnostic`` -- explains why an input was rejected as empty
- ``IngestionResult``   -- outcome of a complete ingestion run
- ``CSVIngestionAdapter``  -- ingest canonical CSV files
- ``JSONLIngestionAdapter`` -- ingest newline-delimited JSON files
"""

from __future__ import annotations

from password_attack_detector.data.ingestion.base import (
    EmptyInputDiagnostic,
    IngestionResult,
    InvalidRowPolicy,
    QuarantineEntry,
)
from password_attack_detector.data.ingestion.csv_adapter import CSVIngestionAdapter
from password_attack_detector.data.ingestion.jsonl_adapter import JSONLIngestionAdapter

__all__ = [
    "CSVIngestionAdapter",
    "EmptyInputDiagnostic",
    "IngestionResult",
    "InvalidRowPolicy",
    "JSONLIngestionAdapter",
    "QuarantineEntry",
]
