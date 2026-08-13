"""Deterministic builders for the machine-learning test suite.

Everything here is tiny and hand-specified.  No test in the ordinary suite
generates a synthetic dataset, builds the 201-column feature catalog over a long
horizon, or fits anything: a contract test that needs 720 hours of traffic to
express itself is testing the generator, not the contract.

Timestamps are literals.  Nothing calls ``datetime.now``, so a fixture built
today and the same fixture built next year produce identical fingerprints.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from password_attack_detector.features.catalog import FeatureCatalog, FeatureSpec
from password_attack_detector.features.config import AggregateKind, EntityKind
from password_attack_detector.ml.dataset import CampaignRow, LabelRow, SplitRow
from password_attack_detector.ml.features import FeatureAdmission, FeatureAllowlist

#: The instant every fixture counts from.  A literal, so no fixture's identity
#: depends on when the suite happened to run.
EPOCH: datetime = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)

#: A catalog fingerprint stand-in for tests that never build a real catalog.
PLACEHOLDER_FINGERPRINT: str = "0" * 64


def at(minutes: int) -> datetime:
    """Return an instant *minutes* after :data:`EPOCH`."""
    return EPOCH + timedelta(minutes=minutes)


def anchor_id(index: int) -> str:
    """Return a stable, sortable, obviously-synthetic anchor identifier."""
    return f"e{index:04d}"


def spec(
    name: str,
    *,
    group: str = "user_history",
    leakage_class: str = "prior_only",
    dtype: str = "float64",
    nullable: bool = True,
    requires_baseline: bool = False,
    deprecated: bool = False,
    intended_use: str = "supervised classification and anomaly detection",
    window: str | None = "5m",
    aggregate_kind: AggregateKind | None = AggregateKind.MEAN,
    privacy_class: str = "non_sensitive",
) -> FeatureSpec:
    """Return one feature specification with test-friendly defaults."""
    return FeatureSpec(
        name=name,
        group=group,  # type: ignore[arg-type]  # StrEnum coerces the value
        entity=EntityKind.USER,
        window=window,
        aggregate=aggregate_kind,
        dtype=dtype,  # type: ignore[arg-type]  # StrEnum coerces the value
        nullable=nullable,
        leakage_class=leakage_class,  # type: ignore[arg-type]  # StrEnum coerces
        null_semantics="undefined when no prior events fall in the window",
        description="A synthetic feature used only by the test suite.",
        intended_use=intended_use,
        requires_baseline=requires_baseline,
        deprecated=deprecated,
        privacy_class=privacy_class,  # type: ignore[arg-type]  # Literal coerces
    )


def catalog(specs: Sequence[FeatureSpec]) -> FeatureCatalog:
    """Return a catalog over *specs* with a fixed configuration fingerprint."""
    return FeatureCatalog(tuple(specs), config_fingerprint=PLACEHOLDER_FINGERPRINT)


def small_catalog() -> FeatureCatalog:
    """Return a four-feature catalog spanning every eligible leakage class."""
    return catalog(
        [
            spec("user_failure_rate"),
            spec("source_failure_rate", group="source_history"),
            spec(
                "current_authentication_outcome",
                group="current_context",
                leakage_class="current_event_context",
                dtype="string",
                window=None,
            ),
            spec(
                "login_hour_deviation",
                group="baseline",
                leakage_class="baseline_derived",
                requires_baseline=True,
                window=None,
            ),
        ]
    )


#: The six features every preprocessing fixture is built from, in catalog order.
#:
#: Chosen to span the encoding contract exactly once each: a nullable number, a
#: non-nullable number, a non-nullable category, a nullable category that rare
#: bucketing applies to, a nullable boolean, and a non-nullable boolean.  Six is
#: enough to exercise every branch and small enough that a failing assertion
#: names the column it means.
PREPROCESSING_FEATURES: tuple[str, ...] = (
    "user_failure_rate",
    "user_attempt_count",
    "current_authentication_outcome",
    "current_country_code",
    "is_new_device_for_user",
    "user_in_baseline",
)


def preprocessing_catalog() -> FeatureCatalog:
    """Return a catalog spanning every dtype and nullability preprocessing handles."""
    return catalog(
        [
            spec("user_failure_rate", dtype="float64", nullable=True),
            spec(
                "user_attempt_count",
                dtype="int64",
                nullable=False,
                aggregate_kind=AggregateKind.COUNT,
            ),
            spec(
                "current_authentication_outcome",
                group="current_context",
                leakage_class="current_event_context",
                dtype="string",
                nullable=False,
                window=None,
            ),
            spec(
                "current_country_code",
                group="current_context",
                leakage_class="current_event_context",
                dtype="string",
                nullable=True,
                window=None,
            ),
            spec(
                "is_new_device_for_user",
                group="baseline",
                leakage_class="baseline_derived",
                dtype="bool",
                nullable=True,
                requires_baseline=True,
                window=None,
            ),
            spec(
                "user_in_baseline",
                group="baseline",
                leakage_class="baseline_derived",
                dtype="bool",
                nullable=False,
                requires_baseline=True,
                window=None,
            ),
        ]
    )


def admission(
    name: str,
    *,
    decision_point: str = "post_event",
    leakage_class: str = "prior_only",
    feature_group: str = "user_history",
    rationale: str = "Admitted by the test suite for contract coverage only.",
    admitted_in: str = "test",
    experiments: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return one allowlist entry as a raw mapping, ready for YAML or the model."""
    return {
        "name": name,
        "decision_point": decision_point,
        "leakage_class": leakage_class,
        "feature_group": feature_group,
        "rationale": rationale,
        "admitted_in": admitted_in,
        "experiments": experiments,
    }


def admission_for(spec_: FeatureSpec, **overrides: Any) -> dict[str, Any]:
    """Return the allowlist entry that agrees with *spec_*, before *overrides*.

    Agreement is the default so a test that wants a *disagreement* has to state
    exactly which field it is perturbing.
    """
    entry = admission(
        spec_.name,
        decision_point=(
            "requires_fitted_baseline" if spec_.requires_baseline else "post_event"
        ),
        leakage_class=str(spec_.leakage_class),
        feature_group=str(spec_.group),
    )
    entry.update(overrides)
    return entry


def allowlist(
    entries: Sequence[dict[str, Any]],
    *,
    allowlist_id: str = "test_champion",
    allowlist_version: str = "1.0.0",
    catalog_fingerprints: Sequence[str] = (PLACEHOLDER_FINGERPRINT,),
    governed_leakage_classes: Sequence[str] | None = None,
    pending_review: Sequence[str] = (),
) -> FeatureAllowlist:
    """Return a validated allowlist over *entries*."""
    payload: dict[str, Any] = {
        "allowlist_id": allowlist_id,
        "allowlist_version": allowlist_version,
        "required_feature_schema_version": "1.0.0",
        "compatible_feature_catalog_fingerprints": tuple(catalog_fingerprints),
        "entries": tuple(FeatureAdmission(**entry) for entry in entries),
        "pending_review": tuple(pending_review),
    }
    if governed_leakage_classes is not None:
        payload["governed_leakage_classes"] = tuple(governed_leakage_classes)
    return FeatureAllowlist(**payload)


def allowlist_for(source: FeatureCatalog, **kwargs: Any) -> FeatureAllowlist:
    """Return an allowlist admitting every non-key feature of *source*."""
    return allowlist(
        [admission_for(item) for item in source.specs if str(item.group) != "key"],
        catalog_fingerprints=(source.fingerprint(),),
        **kwargs,
    )


def feature_row(
    index: int,
    *,
    names: Sequence[str],
    minutes: int | None = None,
    values: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return one feature snapshot row over *names*."""
    row: dict[str, Any] = {
        "anchor_event_id": anchor_id(index),
        "anchor_event_time": at(index if minutes is None else minutes),
    }
    if values is None:
        row.update({name: float(index) for name in names})
    else:
        row.update(dict(zip(names, values, strict=True)))
    return row


def label_row(
    index: int, *, malicious: bool = False, attack_class: str = "normal"
) -> LabelRow:
    """Return one label row, defaulting to a benign event."""
    return LabelRow(
        event_id=anchor_id(index),
        attack_class=attack_class,
        malicious=malicious,
        supervised_training_eligible=attack_class != "novel_anomaly_holdout",
    )


def split_row(index: int, split: str) -> SplitRow:
    """Return one split assignment."""
    return SplitRow(event_id=anchor_id(index), split=split)


def campaign_row(index: int, campaign: str, *, stage: str | None = None) -> CampaignRow:
    """Return one campaign membership record.

    *stage* is the metadata's declaration that a campaign exists.  It defaults to
    absent, which is what an ordinary campaign table looks like; pass one to
    declare a campaign whose identifier would otherwise read as the generator's
    benign placeholder.
    """
    return CampaignRow(
        event_id=anchor_id(index), campaign_id=campaign, campaign_stage=stage
    )
