"""Project exception hierarchy for the Password Attack Detector."""

from __future__ import annotations

__all__ = [
    "ArtifactNotFoundError",
    "ConfigurationError",
    "DataValidationError",
    "ModelNotReadyError",
    "PasswordAttackDetectorError",
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
