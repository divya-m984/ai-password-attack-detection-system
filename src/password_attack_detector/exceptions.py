"""Project exception hierarchy for the Password Attack Detector."""

from __future__ import annotations

__all__ = [
    "ArtifactNotFoundError",
    "BaselineFitError",
    "ConfigurationError",
    "DataValidationError",
    "DetectionConfigurationError",
    "FeatureComputationError",
    "IngestionError",
    "MLConfigurationError",
    "ManifestVerificationError",
    "ModelNotReadyError",
    "ModelSerializationError",
    "ModelTrainingError",
    "PasswordAttackDetectorError",
    "PseudonymizationError",
    "RuleEvaluationError",
    "SplitConfigurationError",
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


class FeatureComputationError(PasswordAttackDetectorError):
    """Raised when feature computation produces output inconsistent with the catalog."""


class BaselineFitError(PasswordAttackDetectorError):
    """Raised when a behavioral baseline is fitted from events it is not permitted to see."""


class SplitConfigurationError(ConfigurationError):
    """Raised when a chronological split cannot be produced from the given configuration."""


class DetectionConfigurationError(ConfigurationError):
    """Raised when a detection configuration or rule registration is invalid."""


class MLConfigurationError(ConfigurationError):
    """Raised when a machine-learning configuration or model registration is invalid.

    Mirrors :class:`DetectionConfigurationError` for the model layer: an
    unknown model family, a duplicate family declaration, an out-of-range
    hyperparameter, or a catalog entry whose implementation does not match its
    declared specification.
    """


class ModelTrainingError(PasswordAttackDetectorError):
    """Raised when a model is fitted outside its contract.

    Covers fitting on a split a model may not see, a design matrix whose
    columns do not match the declared feature order, and a task fitted with
    insufficient class support.
    """


class ModelSerializationError(PasswordAttackDetectorError):
    """Raised when a model cannot be canonically serialised or restored.

    Covers a non-finite parameter, an estimator attribute the serializer
    requires but the installed scikit-learn release does not provide, and a
    restored model whose scores do not match the original within tolerance.
    """


class RuleEvaluationError(PasswordAttackDetectorError):
    """Raised when a detection rule is prepared or evaluated outside its contract.

    Covers a rule declaring a feature the feature catalog does not provide, a
    rule reading a prohibited column, and a feature snapshot that does not
    supply a column the prepared rule requires.
    """
