"""The deterministic detection engine.

The engine turns a sequence of Phase 3 feature snapshots into fired detections
and aggregate diagnostics.  It decides nothing itself: every verdict comes from
a registered rule, and the engine's job is to run them in a fixed order, once
each per snapshot, and account for every outcome.

**Determinism is the contract.**  Rules are prepared once and iterated in
catalog order, which is sorted by rule identifier.  Snapshots are evaluated in
``(anchor_event_time, anchor_event_id)`` order, so shuffling the input changes
nothing about the output.  No wall clock, no random state, no process identity,
and no dictionary insertion order reaches a result.

**The boundary is structural, not conventional.**  :meth:`DetectionEngine.run`
accepts feature snapshots and nothing else.  There is no parameter for labels,
split assignments, campaign metadata, an entity scope, or the canonical event
table, and this module imports no reader for any of them -- so a caller trying
to pass one gets a type error rather than a leak.  Input validation rejects a
snapshot that carries such a column at all, before any rule sees a row.

**Cost is bounded.**  Validation is one pass over the rows and one pass over
the columns each rule actually reads; evaluation is exactly
``snapshot_count x enabled_rule_count`` rule calls.  Nothing rescans the
dataset per rule, and no rule reconstructs history: the snapshot it is handed
is the entire world it can see.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from password_attack_detector.detection.catalog import (
    RULE_CATALOG,
    RuleCatalog,
    resolve_rule_features,
)
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS, Rule
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.detection.schemas import (
    DETECTION_SCHEMA_VERSION,
    FiredDetection,
    RuleEvaluationResult,
)
from password_attack_detector.exceptions import (
    DataValidationError,
    DetectionConfigurationError,
)
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    PROHIBITED_FEATURE_COLUMNS,
    FeatureCatalog,
    build_catalog,
)
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
)

__all__ = [
    "PROHIBITED_SNAPSHOT_COLUMNS",
    "REQUIRED_FEATURE_SCHEMA_VERSION",
    "SCHEMA_VERSION_COLUMN",
    "DetectionEngine",
    "EngineResult",
    "EngineStats",
    "evaluate_snapshots",
]

#: The snapshot column declaring which feature schema a row conforms to.
SCHEMA_VERSION_COLUMN: str = "feature_schema_version"

#: Columns a detection input may never carry.  The Phase 3 prohibition set is
#: reused verbatim -- ground truth, campaign metadata, split assignment, and
#: model output are prohibited for the same reasons here as there -- extended
#: with the names an evaluation or model-scoring artifact would introduce if
#: one were ever joined onto a snapshot by mistake.
PROHIBITED_SNAPSHOT_COLUMNS: frozenset[str] = PROHIBITED_FEATURE_COLUMNS | frozenset(
    {
        "campaign",
        "ground_truth",
        "model_prediction",
        "model_score",
        "prediction",
        "predicted_label",
        "scenario_type",
        "y_pred",
        "y_true",
    }
)


@dataclass(frozen=True, slots=True)
class EngineStats:
    """Structural counters describing what a run actually did.

    These are operation counts, not timings.  A wall-clock measurement varies
    between machines and would make the invariants untestable; ``prepared_rule
    count == enabled rules`` and ``rule_evaluations == snapshots x rules`` are
    exact, so a regression that reprepares per row or rescans per rule fails a
    test rather than merely running slowly.
    """

    snapshot_count: int = 0
    enabled_rule_count: int = 0
    rule_preparations: int = 0
    rule_evaluations: int = 0
    validation_column_checks: int = 0


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Everything one detection run produced.

    Fired detections are the published artifact.  Everything else a run
    observed -- clean negatives, unavailable history, disabled rules -- stays
    here as aggregate counts, so a suppressed or silent outcome is still
    accounted for without publishing a row per non-event.
    """

    fired_detections: tuple[FiredDetection, ...] = ()
    status_counts: Mapping[str, int] = field(default_factory=dict)
    per_rule_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    evaluated_snapshot_count: int = 0
    total_rule_evaluation_count: int = 0
    insufficient_data_count: int = 0
    disabled_rule_count: int = 0
    configuration_fingerprint: str = ""
    rule_catalog_fingerprint: str = ""
    detection_schema_version: str = DETECTION_SCHEMA_VERSION
    stats: EngineStats = EngineStats()

    @property
    def fired_count(self) -> int:
        """Return how many detections fired."""
        return len(self.fired_detections)


class DetectionEngine:
    """Runs the enabled rules over feature snapshots, once each, in order.

    The constructor does all the work that must not happen per row: it resolves
    which rules are enabled, checks that every one has a registered
    implementation, prepares each exactly once, and confirms that what each
    implementation resolved against the feature catalog matches what its
    catalog entry declares.
    """

    __slots__ = (
        "_catalog",
        "_config",
        "_feature_catalog",
        "_preparation_count",
        "_prepared",
        "_required_columns",
    )

    def __init__(
        self,
        config: DetectionConfig,
        *,
        feature_catalog: FeatureCatalog | None = None,
        rules: Mapping[str, Rule] | None = None,
        catalog: RuleCatalog = RULE_CATALOG,
    ) -> None:
        """Prepare every enabled rule for a run.

        Note the parameters this does **not** take: there is no labels, splits,
        campaign, entity-scope, or events argument anywhere in the engine's
        interface.

        Raises:
            DetectionConfigurationError: if an enabled rule has no registered
                implementation, or an implementation's resolved feature set
                disagrees with what its catalog entry declares.
        """
        self._config = config
        self._catalog = catalog
        self._feature_catalog = (
            build_catalog(FeatureConfig())
            if feature_catalog is None
            else feature_catalog
        )
        implementations = RULE_IMPLEMENTATIONS if rules is None else rules

        prepared: list[tuple[str, PreparedRule]] = []
        required: set[str] = set()
        preparation_count = 0

        # Catalog order, not configuration order and not registry order: the
        # catalog is sorted by rule identifier, so evaluation order is a
        # property of the data rather than of how anything was written down.
        for spec in catalog.specs:
            if not config.is_enabled(spec.rule_id):
                continue
            rule = implementations.get(spec.rule_id)
            if rule is None:
                raise DetectionConfigurationError(
                    f"Rule {spec.rule_id!r} is enabled but has no registered "
                    f"implementation"
                )
            prepared_rule = rule.prepare(config, self._feature_catalog)
            preparation_count += 1

            self._check_agreement(spec.rule_id, prepared_rule, config)
            required.update(prepared_rule.preparation.features.values())
            prepared.append((spec.rule_id, prepared_rule))

        self._prepared = tuple(prepared)
        self._preparation_count = preparation_count
        self._required_columns = frozenset(required)

    def _check_agreement(
        self, rule_id: str, prepared: PreparedRule, config: DetectionConfig
    ) -> None:
        """Assert the implementation resolved exactly what the catalog declares.

        A rule's declared inputs and the columns its prepared form will read
        are two separate statements, and a mismatch would let an implementation
        read a column the catalog never vetted for prohibition or leakage class.

        Raises:
            DetectionConfigurationError: if the two disagree.
        """
        spec = self._catalog.get(rule_id)
        expected = resolve_rule_features(
            spec, config.parameters_for(rule_id), self._feature_catalog
        )
        if dict(prepared.preparation.features) != expected:
            raise DetectionConfigurationError(
                f"Rule {rule_id!r} resolved a feature set that does not match its "
                f"catalog declaration"
            )

    @property
    def enabled_rule_ids(self) -> tuple[str, ...]:
        """Return the enabled rule identifiers, in evaluation order."""
        return tuple(rule_id for rule_id, _ in self._prepared)

    @property
    def required_columns(self) -> frozenset[str]:
        """Return every snapshot column the enabled rules will read."""
        return self._required_columns

    # -- run ----------------------------------------------------------------

    def run(self, snapshots: Sequence[Mapping[str, Any]]) -> EngineResult:
        """Evaluate every enabled rule against every snapshot.

        *snapshots* is a sequence of feature-snapshot row mappings and is the
        engine's only input.  It is read, never mutated: no row is written to,
        and the ordering used for evaluation is built from a copy.

        An empty input is valid and produces a well-formed empty result -- a
        dataset with no rows is an ordinary state, not an error, and failing
        here would make a quiet interval indistinguishable from a broken one.

        Raises:
            DataValidationError: if the snapshots fail input validation.  The
                message names columns, codes, and counts only; no event
                identifier, row content, or value is ever included.
        """
        column_checks = self._validate(snapshots)
        ordered = self._ordered(snapshots)

        fired: list[FiredDetection] = []
        status_counts: dict[str, int] = {str(status): 0 for status in RuleStatus}
        per_rule: dict[str, dict[str, int]] = {
            rule_id: {str(status): 0 for status in RuleStatus}
            for rule_id in self.enabled_rule_ids
        }
        evaluations = 0

        for row in ordered:
            for rule_id, prepared in self._prepared:
                result = prepared.evaluate(row)
                evaluations += 1
                status = str(result.status)
                status_counts[status] += 1
                per_rule[rule_id][status] += 1
                if result.status is RuleStatus.FIRED:
                    fired.append(FiredDetection.from_result(result))

        # Disabled rules are never evaluated, so their count is structural:
        # how many registered rules this configuration switched off.  Reporting
        # it keeps the diagnostics honest about coverage rather than silently
        # presenting a partial run as a complete one.
        disabled_rule_count = len(self._catalog) - len(self._prepared)

        return EngineResult(
            fired_detections=tuple(fired),
            status_counts=dict(status_counts),
            per_rule_counts={
                rule_id: dict(counts) for rule_id, counts in per_rule.items()
            },
            evaluated_snapshot_count=len(ordered),
            total_rule_evaluation_count=evaluations,
            insufficient_data_count=status_counts[str(RuleStatus.INSUFFICIENT_DATA)],
            disabled_rule_count=disabled_rule_count,
            configuration_fingerprint=self._config.fingerprint(),
            rule_catalog_fingerprint=self._catalog.fingerprint(),
            detection_schema_version=DETECTION_SCHEMA_VERSION,
            stats=EngineStats(
                snapshot_count=len(ordered),
                enabled_rule_count=len(self._prepared),
                rule_preparations=self._preparation_count,
                rule_evaluations=evaluations,
                validation_column_checks=column_checks,
            ),
        )

    def run_diagnostic(
        self, snapshots: Sequence[Mapping[str, Any]]
    ) -> tuple[RuleEvaluationResult, ...]:
        """Return every rule result for *snapshots*, fired or not.

        The published artifact carries fired detections only; this exposes the
        full outcome set for quality reporting and tests without persisting a
        row per non-event.  Same validation, same ordering, same rule order as
        :meth:`run`.

        Raises:
            DataValidationError: if the snapshots fail input validation.
        """
        self._validate(snapshots)
        return tuple(
            prepared.evaluate(row)
            for row in self._ordered(snapshots)
            for _, prepared in self._prepared
        )

    # -- input validation ---------------------------------------------------

    def _validate(self, snapshots: Sequence[Mapping[str, Any]]) -> int:
        """Validate the input before any rule sees a row.

        Returns the number of column reads validation performed, which the
        structural tests use to confirm the cost stays linear.

        Raises:
            DataValidationError: on the first failing check.  Every message
                reports column names and counts only.
        """
        if not snapshots:
            return 0

        checks = 0
        seen_anchors: set[str] = set()
        duplicate_anchors = 0
        invalid_times = 0
        non_finite_columns: set[str] = set()

        for row in snapshots:
            self._check_columns(row)

            version = row.get(SCHEMA_VERSION_COLUMN)
            if version != self._config.required_feature_schema_version:
                raise DataValidationError(
                    f"Feature snapshot declares schema version {version!r}, which "
                    f"is incompatible with the configured requirement "
                    f"{self._config.required_feature_schema_version!r}"
                )

            anchor_id = row[ANCHOR_EVENT_ID]
            if not isinstance(anchor_id, str) or not anchor_id.strip():
                raise DataValidationError(
                    f"Column {ANCHOR_EVENT_ID!r} must hold a non-empty string"
                )
            if anchor_id in seen_anchors:
                duplicate_anchors += 1
            seen_anchors.add(anchor_id)

            anchor_time = row[ANCHOR_EVENT_TIME]
            if not isinstance(anchor_time, datetime) or anchor_time.tzinfo is None:
                invalid_times += 1

            # Only the columns the enabled rules will actually read are swept
            # for non-finite values.  A snapshot carries far more columns than
            # any run consumes, and validating what nothing reads would cost
            # real time to protect nothing.
            for column in self._required_columns:
                checks += 1
                value = row.get(column)
                if isinstance(value, float) and (
                    math.isnan(value) or math.isinf(value)
                ):
                    non_finite_columns.add(column)

        if duplicate_anchors:
            raise DataValidationError(
                f"Feature snapshots contain {duplicate_anchors} duplicate "
                f"{ANCHOR_EVENT_ID!r} value(s); each anchor must appear once"
            )
        if invalid_times:
            raise DataValidationError(
                f"{invalid_times} row(s) carry an absent, non-datetime, or "
                f"timezone-naive {ANCHOR_EVENT_TIME!r}"
            )
        if non_finite_columns:
            raise DataValidationError(
                f"Column(s) {sorted(non_finite_columns)} contain NaN or infinity, "
                f"which no rule input may hold"
            )
        return checks

    def _check_columns(self, row: Mapping[str, Any]) -> None:
        """Reject a row missing a required column or carrying a prohibited one.

        Raises:
            DataValidationError: naming the offending columns only.
        """
        present = row.keys()
        prohibited = sorted(PROHIBITED_SNAPSHOT_COLUMNS & set(present))
        if prohibited:
            raise DataValidationError(
                f"Feature snapshots carry prohibited column(s) {prohibited}; "
                f"ground truth, campaign metadata, split assignments, and model "
                f"output are not detection inputs"
            )
        missing = sorted(
            column
            for column in (
                SCHEMA_VERSION_COLUMN,
                ANCHOR_EVENT_ID,
                ANCHOR_EVENT_TIME,
                *self._required_columns,
            )
            if column not in present
        )
        if missing:
            raise DataValidationError(
                f"Feature snapshots are missing required column(s) {missing}"
            )

    @staticmethod
    def _ordered(
        snapshots: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        """Return the snapshots in deterministic evaluation order.

        Sorted by anchor time then anchor identifier.  The identifier breaks
        ties so that two events sharing a timestamp still evaluate in a fixed
        order, and the whole ordering is rebuilt from the data on every run --
        the caller's ordering never survives into a result.
        """
        return tuple(
            sorted(
                snapshots,
                key=lambda row: (row[ANCHOR_EVENT_TIME], row[ANCHOR_EVENT_ID]),
            )
        )


def evaluate_snapshots(
    config: DetectionConfig,
    snapshots: Sequence[Mapping[str, Any]],
    *,
    feature_catalog: FeatureCatalog | None = None,
) -> EngineResult:
    """Build an engine and run it once.

    A convenience for callers with a single batch; a caller evaluating several
    batches under one configuration should build a :class:`DetectionEngine` and
    reuse it, so rule preparation happens once rather than once per batch.
    """
    engine = DetectionEngine(config, feature_catalog=feature_catalog)
    return engine.run(snapshots)


#: The feature schema this engine is built against.  Re-exported so a caller
#: can check compatibility without importing the feature package.
REQUIRED_FEATURE_SCHEMA_VERSION: str = FEATURE_SCHEMA_VERSION
