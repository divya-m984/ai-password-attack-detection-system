"""Project exception hierarchy for the Password Attack Detector."""

from __future__ import annotations

__all__ = [
    "ArtifactNotFoundError",
    "ConfigurationError",
    "DataValidationError",
    "IngestionError",
    "ManifestVerificationError",
    "ModelNotReadyError",
    "PasswordAttackDetectorError",
    "PseudonymizationError",
]


class PasswordAttackDetectorError(Exception):
    """Base exception for all project errors."""


class ConfigurationError(PasswordAttackDetectorError):
    """Raised when configuration is invalid, missing, or cannot be loaded."""


class DataValidationError(PasswordAttackDetectorError):
    """Raised when input data fails schema or type validation."""


class ArtifactNotFoundError(PasswordAttackDetectorError):
    """Raised when a required model artifact or file cannot be located."""


class ModelNotReadyError(PasswordAttackDetectorError):
    """Raised when a model is referenced before it has been trained or loaded."""


class PseudonymizationError(PasswordAttackDetectorError):
    """Raised when pseudonymization fails due to a missing or invalid key."""


class IngestionError(PasswordAttackDetectorError):
    """Raised when an ingestion adapter detects a fatal problem with the source data."""


class ManifestVerificationError(PasswordAttackDetectorError):
    """Raised when a dataset manifest fails integrity or path-safety verification."""
