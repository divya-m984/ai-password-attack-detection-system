"""Password Attack Detector — data foundation package.

Phase 2 modules:

- enums          Domain enumerations shared across data modules
- schemas        Canonical AuthEvent and GroundTruthLabel (Pydantic v2)
- privacy        PseudonymService and prohibited-field scanner
- validation     DatasetValidator and ValidationResult
- quality        QualityProfiler and QualityReport
- manifest       DatasetManifest and verify_manifest
- serialization  Parquet I/O and DatasetPublisher (staged rollback)
- cli            ``data`` command group for the Typer CLI
- ingestion      CSV and JSONL ingestion adapters
- synthetic      Deterministic synthetic event generation
"""

from __future__ import annotations

__all__: list[str] = []
