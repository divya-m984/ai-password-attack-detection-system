"""Class weights from training counts. No resampling, ever.

Attacks are rare, and a model fitted on the raw prevalence can reach excellent
accuracy by predicting "benign" every time.  The answer here is **weighting**:
the rows stay exactly as the split contract fixed them, and the loss is told
that a minority row costs more to get wrong.

The answer is emphatically *not* resampling.  Oversampling duplicates rows a
split boundary already placed, undersampling discards evidence, and synthetic
minority generation invents authentication events that never happened and then
measures a detector against them.  :class:`ImbalanceConfig` admits one value for
``resampling`` so no configuration can turn any of that on.

**What this module may see.**  A sequence of class values and a declared class
order -- ordinary strings, handed over by whichever component already read the
labels.  It imports no label type, opens no file, and never learns which split
its counts came from except by being told; the caller is responsible for passing
training rows, and :attr:`ClassWeightState.computed_from` records the claim so a
later reader can check it against the run that produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.config import ImbalanceConfig
from password_attack_detector.ml.enums import MLTask

__all__ = [
    "BINARY_CLASS_ORDER",
    "IMBALANCE_SCHEMA_VERSION",
    "ClassSupport",
    "ClassWeightState",
    "compute_class_weights",
]

#: The class-weight state contract's own version.
IMBALANCE_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The binary task's class order, negative class first.
#:
#: Fixed here rather than derived from the data, because a class order that
#: depended on which classes a particular training split happened to contain
#: would reorder a weight vector when the data changed and leave every recorded
#: fingerprint pointing at the wrong arrangement.
BINARY_CLASS_ORDER: Final[tuple[str, ...]] = ("benign", "malicious")

#: Weights are quantized to this many decimals before being stored.
_FLOAT_PRECISION: Final[int] = 9


def _quantize(value: float) -> float:
    """Return *value* at the precision weights are stored and fingerprinted at."""
    if not math.isfinite(value):
        raise ValueError("a class weight must be finite")
    return float(f"{value:.{_FLOAT_PRECISION}f}")


class ClassSupport(BaseModel):
    """One class's training row count and the weight derived from it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_name: str
    train_row_count: int = Field(ge=0)
    weight: float

    @field_validator("weight")
    @classmethod
    def quantize_weight(cls, value: float) -> float:
        """Store the weight at serialized precision, so a round trip is exact."""
        return _quantize(value)

    @model_validator(mode="after")
    def check_weight_is_positive(self) -> Self:
        """A zero weight silences a class; a negative one inverts it."""
        if self.weight <= 0.0:
            raise ValueError(
                f"class {self.class_name!r} has weight {self.weight!r}; a weight "
                f"must be strictly positive"
            )
        return self


class ClassWeightState(BaseModel):
    """Frozen class weights, the counts behind them, and their identity.

    Immutable, JSON-round-trippable, and fingerprinted by content, so two runs
    over the same training labels under the same policy agree exactly and a run
    over different ones does not.  Carries counts as well as weights: a weight
    without its support is a number nobody can check.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    imbalance_schema_version: str = IMBALANCE_SCHEMA_VERSION
    task: MLTask
    policy: str
    computed_from: str
    class_order: tuple[str, ...]
    supports: tuple[ClassSupport, ...]
    imbalance_config_fingerprint: str

    @model_validator(mode="after")
    def check_alignment(self) -> Self:
        """The class order and the supports describe the same classes, in order."""
        if not self.class_order:
            raise ValueError("class_order must name at least one class")
        if len(set(self.class_order)) != len(self.class_order):
            raise ValueError("class_order repeats a class")
        if tuple(item.class_name for item in self.supports) != self.class_order:
            raise ValueError(
                "supports must be given in class_order, one entry per class"
            )
        return self

    @property
    def weights(self) -> tuple[float, ...]:
        """Return the weight per class, in :attr:`class_order`."""
        return tuple(item.weight for item in self.supports)

    @property
    def train_row_counts(self) -> tuple[int, ...]:
        """Return the training support per class, in :attr:`class_order`."""
        return tuple(item.train_row_count for item in self.supports)

    def weight_of(self, class_name: str) -> float:
        """Return the weight for *class_name*, or raise if it is not a class."""
        for item in self.supports:
            if item.class_name == class_name:
                return item.weight
        raise ModelTrainingError(
            f"{class_name!r} is not one of the {len(self.class_order)} declared "
            f"classes for task {str(self.task)!r}"
        )

    def as_mapping(self) -> dict[str, float]:
        """Return the weights keyed by class name, for an estimator that wants one."""
        return {item.class_name: item.weight for item in self.supports}

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready mapping this state serializes to."""
        payload = self.model_dump(mode="json")
        return dict(payload)

    @classmethod
    def from_dict(cls, payload: Any) -> ClassWeightState:
        """Return the state *payload* describes, or raise."""
        if not isinstance(payload, dict):
            raise ModelTrainingError(
                f"class-weight state must be a JSON object, got "
                f"{type(payload).__name__}"
            )
        version = payload.get("imbalance_schema_version")
        if version != IMBALANCE_SCHEMA_VERSION:
            raise ModelTrainingError(
                f"class-weight state declares schema version {version!r}; this "
                f"build implements {IMBALANCE_SCHEMA_VERSION!r}"
            )
        try:
            return cls.model_validate(payload)
        except Exception as exc:
            raise ModelTrainingError(
                f"class-weight state is not valid ({type(exc).__name__})"
            ) from None

    def to_json(self) -> str:
        """Return canonical JSON: sorted keys, ASCII, no incidental whitespace."""
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, text: str) -> ClassWeightState:
        """Return the state *text* encodes, or raise."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelTrainingError(
                f"class-weight state is not valid JSON ({type(exc).__name__})"
            ) from None
        return cls.from_dict(payload)

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic content that gives this state its identity."""
        return self.to_dict()

    def fingerprint(self) -> str:
        """Return the SHA-256 digest of this state's canonical rendering."""
        canonical = json.dumps(
            self.fingerprint_data(), sort_keys=True, ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def compute_class_weights(
    class_values: Sequence[str],
    *,
    task: MLTask,
    class_order: Sequence[str],
    config: ImbalanceConfig,
) -> ClassWeightState:
    """Return the class weights for *class_values* under *config*.

    Three policies, each fully determined by its inputs:

    ``none``
        Every class weighs ``1.0``.  Counted anyway, so the state records what
        the prevalence was even when nothing was done about it.

    ``balanced``
        ``n_samples / (n_classes * count(class))`` -- the standard formula,
        implemented here rather than delegated so the number in a manifest can
        be recomputed by hand from the counts beside it.  A class with no
        training rows makes this a division by zero, so it fails instead.

    ``fixed``
        The reviewed weights from the configuration, checked against the
        declared class order: an unknown class is a typo that would otherwise be
        ignored, and a missing one is a class that would otherwise be silently
        weighted differently from every other.

    Args:
        class_values: one class per training row, in canonical row order.  Order
            does not affect the result -- these are counted, not scanned -- and
            a test pins that.
        task: which head these weights are for.
        class_order: the deterministic class order the weights are reported in.
        config: the imbalance policy.

    Raises:
        ModelTrainingError: on an empty class order, a repeated class, a class
            value outside the declared order, a class with zero training support
            under ``balanced``, or a fixed mapping that does not describe exactly
            the declared classes.
    """
    order = tuple(class_order)
    if not order:
        raise ModelTrainingError(
            f"task {str(task)!r} was given no class order; weights have to be "
            f"reported in some deterministic arrangement"
        )
    if len(set(order)) != len(order):
        raise ModelTrainingError(f"the class order for {str(task)!r} repeats a class")

    counts = dict.fromkeys(order, 0)
    for value in class_values:
        if value not in counts:
            raise ModelTrainingError(
                f"a training row carries class {value!r}, which is not one of "
                f"the {len(order)} declared classes for task {str(task)!r}; a "
                f"class nobody declared cannot be weighted"
            )
        counts[value] += 1

    weights = _weights_for(counts, order=order, task=task, config=config)
    return ClassWeightState(
        task=task,
        policy=config.class_weight_policy,
        computed_from=config.computed_from,
        class_order=order,
        supports=tuple(
            ClassSupport(
                class_name=name, train_row_count=counts[name], weight=weights[name]
            )
            for name in order
        ),
        imbalance_config_fingerprint=_config_fingerprint(config),
    )


def _weights_for(
    counts: dict[str, int],
    *,
    order: tuple[str, ...],
    task: MLTask,
    config: ImbalanceConfig,
) -> dict[str, float]:
    """Return the weight per class under the configured policy."""
    if config.class_weight_policy == "none":
        return dict.fromkeys(order, 1.0)

    if config.class_weight_policy == "fixed":
        declared = dict(config.fixed_class_weights or ())
        unknown = sorted(set(declared) - set(order))
        if unknown:
            raise ModelTrainingError(
                f"fixed_class_weights names class(es) {unknown} that task "
                f"{str(task)!r} does not declare; a weight for a class that "
                f"does not exist is a typo, not a policy"
            )
        absent = sorted(set(order) - set(declared))
        if absent:
            raise ModelTrainingError(
                f"fixed_class_weights does not name class(es) {absent}; every "
                f"declared class needs an explicit weight, because a default "
                f"for the ones somebody forgot is exactly the silent behaviour "
                f"a fixed policy exists to avoid"
            )
        return {name: float(declared[name]) for name in order}

    total = sum(counts.values())
    if total == 0:
        raise ModelTrainingError(
            f"task {str(task)!r} was given no training rows; balanced weights "
            f"are a ratio of counts, and there is nothing to count"
        )
    empty = sorted(name for name in order if counts[name] == 0)
    if empty:
        raise ModelTrainingError(
            f"class(es) {empty} have no training rows, so a balanced weight "
            f"would divide by zero. A class the training split never saw is a "
            f"split or label problem to fix, not a weight to invent"
        )
    return {name: total / (len(order) * counts[name]) for name in order}


def _config_fingerprint(config: ImbalanceConfig) -> str:
    """Return the digest of the imbalance policy this state was computed under."""
    canonical = json.dumps(config.fingerprint_data(), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
