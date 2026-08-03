"""The leakage auditor: twelve named checks over a built feature dataset.

Naming checks alone are not enough.  A column called ``user_failure_count__5m``
can still be computed wrongly, and no amount of inspecting its name would
reveal it.  So four of these checks are **behavioural**: they mutate the input
stream and assert that features which must not move, do not move.

The result is shaped deliberately like ``VerificationResult`` in
``data/manifest.py``, and the auditor is exposed as its own CLI command.  That
matters: burying these checks inside ``features build`` would mean they could
never be run against an artifact directory someone hands you.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from password_attack_detector.data.schemas import (
    PROHIBITED_GT_COLUMNS,
    AuthEvent,
    GroundTruthLabel,
)
from password_attack_detector.data.serialization import compute_events_fingerprint
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    PROHIBITED_FEATURE_COLUMNS,
    FeatureCatalog,
    LeakageClass,
)
from password_attack_detector.features.config import FeatureConfig
from password_attack_detector.features.engine import (
    BaselineTransform,
    FeatureEngine,
    FeatureFrame,
    sort_events_canonically,
)
from password_attack_detector.features.splitting import (
    SUPERVISED_SPLITS,
    SplitAssignment,
    SplitLabel,
)

__all__ = [
    "CHECK_NAMES",
    "LeakageAuditResult",
    "LeakageAuditor",
    "LeakageCheck",
    "audit_result_to_markdown",
]

#: Every check this auditor performs, in report order.
CHECK_NAMES: tuple[str, ...] = (
    "NO_GROUND_TRUTH_COLUMNS",
    "NO_CAMPAIGN_COLUMNS",
    "NO_SPLIT_COLUMNS",
    "NO_FUTURE_CONTRIBUTION",
    "NO_SAME_TIMESTAMP_CONTRIBUTION",
    "BASELINE_SOURCE_FINGERPRINT_MATCHES_TRAIN",
    "BASELINE_NOT_FIT_ON_EVALUATION_DATA",
    "PURGE_INTERVAL_RESPECTED",
    "CAMPAIGN_GROUPS_ISOLATED",
    "HOLDOUT_EXCLUDED_FROM_SUPERVISED",
    "CURRENT_VS_PRIOR_FIELDS_DISTINGUISHED",
    "JOIN_KEY_INTEGRITY",
)


def _group_by_time(
    events: Sequence[AuthEvent],
) -> Iterator[tuple[datetime, list[AuthEvent]]]:
    """Yield ``(event_time, events)`` for each run of simultaneous events."""
    block: list[AuthEvent] = []
    current: datetime | None = None
    for event in events:
        if current is not None and event.event_time != current:
            yield current, block
            block = []
        current = event.event_time
        block.append(event)
    if current is not None:
        yield current, block


@dataclass(frozen=True, slots=True)
class LeakageCheck:
    """The outcome of one named check."""

    name: str
    passed: bool
    severity: Literal["error", "warning"]
    message: str
    skipped: bool = False


@dataclass(frozen=True)
class LeakageAuditResult:
    """Structured outcome of a full leakage audit.

    Contains counts and column names only.  No event identifier, pseudonym, or
    raw row ever appears here, so the result is safe to embed in a report.
    """

    status: Literal["pass", "warning", "fail"]
    checks: tuple[LeakageCheck, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checked_feature_count: int
    checked_row_count: int
    baseline_interval: tuple[str, str] | None
    split_summary: dict[str, int]
    prohibited_column_findings: tuple[str, ...] = field(default=())

    @property
    def passed(self) -> bool:
        """Return whether the audit found no errors."""
        return self.status != "fail"

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-serialisable mapping."""
        return {
            "status": self.status,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "severity": check.severity,
                    "skipped": check.skipped,
                    "message": check.message,
                }
                for check in self.checks
            ],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checked_feature_count": self.checked_feature_count,
            "checked_row_count": self.checked_row_count,
            "baseline_interval": (
                None if self.baseline_interval is None else list(self.baseline_interval)
            ),
            "split_summary": self.split_summary,
            "prohibited_column_findings": list(self.prohibited_column_findings),
        }


class LeakageAuditor:
    """Runs the twelve leakage checks over a built feature dataset."""

    def __init__(self, config: FeatureConfig, catalog: FeatureCatalog) -> None:
        self._config = config
        self._catalog = catalog
        self._unmutated_cache: dict[str, dict[str, Any]] | None = None

    def audit(
        self,
        frame: FeatureFrame,
        *,
        events: Sequence[AuthEvent] | None = None,
        labels: Sequence[GroundTruthLabel] | None = None,
        assignments: Sequence[SplitAssignment] | None = None,
        baseline: BaselineTransform | None = None,
        baseline_artifact: Any | None = None,
    ) -> LeakageAuditResult:
        """Run every check that the supplied inputs make possible.

        Checks whose inputs are absent are recorded as *skipped* rather than
        quietly passing: an audit that silently degrades to a subset of its
        checks is worse than one that says what it could not verify.
        """
        self._unmutated_cache = None
        checks: list[LeakageCheck] = []
        columns = frame.catalog.column_order() if frame.rows else ()

        checks.append(self._check_no_ground_truth_columns(columns))
        checks.append(self._check_no_campaign_columns(columns))
        checks.append(self._check_no_split_columns(columns))
        checks.append(self._check_no_future_contribution(events, baseline))
        checks.append(self._check_no_same_timestamp_contribution(events, baseline))
        checks.append(
            self._check_baseline_fingerprint(baseline_artifact, events, assignments)
        )
        checks.append(
            self._check_baseline_not_fit_on_evaluation(baseline_artifact, assignments)
        )
        checks.append(self._check_purge_respected(events, assignments))
        checks.append(self._check_campaign_isolation(labels, assignments))
        checks.append(self._check_holdout_excluded(labels, assignments))
        checks.append(self._check_current_versus_prior())
        checks.append(self._check_join_integrity(frame, labels, assignments))

        errors = tuple(
            f"{c.name}: {c.message}"
            for c in checks
            if not c.passed and c.severity == "error"
        )
        warnings = tuple(
            f"{c.name}: {c.message}"
            for c in checks
            if (not c.passed and c.severity == "warning") or c.skipped
        )

        status: Literal["pass", "warning", "fail"]
        if errors:
            status = "fail"
        elif warnings:
            status = "warning"
        else:
            status = "pass"

        split_summary: dict[str, int] = {}
        if assignments is not None:
            for assignment in assignments:
                key = str(assignment.split)
                split_summary[key] = split_summary.get(key, 0) + 1

        interval: tuple[str, str] | None = None
        if baseline_artifact is not None:
            interval = (
                baseline_artifact.fitted_interval_start,
                baseline_artifact.fitted_interval_end,
            )

        return LeakageAuditResult(
            status=status,
            checks=tuple(checks),
            errors=errors,
            warnings=warnings,
            checked_feature_count=len(columns),
            checked_row_count=len(frame.rows),
            baseline_interval=interval,
            split_summary=split_summary,
            prohibited_column_findings=tuple(
                sorted(set(columns) & PROHIBITED_FEATURE_COLUMNS)
            ),
        )

    # -- 1-3: column-level checks -------------------------------------------

    @staticmethod
    def _check_no_ground_truth_columns(columns: Sequence[str]) -> LeakageCheck:
        found = sorted(set(columns) & PROHIBITED_GT_COLUMNS)
        return LeakageCheck(
            name="NO_GROUND_TRUTH_COLUMNS",
            passed=not found,
            severity="error",
            message=(
                "No ground-truth column appears in the feature snapshots"
                if not found
                else f"Ground-truth column(s) present in features: {found}"
            ),
        )

    @staticmethod
    def _check_no_campaign_columns(columns: Sequence[str]) -> LeakageCheck:
        campaign_fields = {"campaign_id", "scenario", "campaign_stage", "attack_class"}
        found = sorted(set(columns) & campaign_fields)
        return LeakageCheck(
            name="NO_CAMPAIGN_COLUMNS",
            passed=not found,
            severity="error",
            message=(
                "No campaign metadata appears in the feature snapshots"
                if not found
                else f"Campaign column(s) present in features: {found}"
            ),
        )

    @staticmethod
    def _check_no_split_columns(columns: Sequence[str]) -> LeakageCheck:
        found = sorted(set(columns) & {"split", "exclusion_reason"})
        return LeakageCheck(
            name="NO_SPLIT_COLUMNS",
            passed=not found,
            severity="error",
            message=(
                "No split assignment appears in the feature snapshots"
                if not found
                else f"Split column(s) present in features: {found}"
            ),
        )

    # -- 4-5: behavioural checks --------------------------------------------

    def _prior_only_columns(self) -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in self._catalog.specs_for_leakage_class(LeakageClass.PRIOR_ONLY)
        )

    def _recompute(
        self, events: Sequence[AuthEvent], baseline: BaselineTransform | None
    ) -> dict[str, dict[str, Any]]:
        engine = FeatureEngine(self._config, self._catalog, baseline=baseline)
        return {row[ANCHOR_EVENT_ID]: row for row in engine.run(events).rows}

    def _unmutated(
        self, events: Sequence[AuthEvent], baseline: BaselineTransform | None
    ) -> dict[str, dict[str, Any]]:
        """Return the baseline recomputation, shared across behavioural checks.

        Both mutation checks compare against the same unmutated frame, so
        computing it once rather than twice removes a full engine pass from
        every audit.
        """
        if self._unmutated_cache is None:
            self._unmutated_cache = self._recompute(events, baseline)
        return self._unmutated_cache

    def _check_no_future_contribution(
        self,
        events: Sequence[AuthEvent] | None,
        baseline: BaselineTransform | None,
    ) -> LeakageCheck:
        """Append an event beyond the stream and assert nothing earlier moves."""
        name = "NO_FUTURE_CONTRIBUTION"
        if not events:
            return LeakageCheck(
                name=name,
                passed=True,
                severity="error",
                skipped=True,
                message="Skipped: the source events were not supplied",
            )

        ordered = sort_events_canonically(events)
        probe = ordered[-1].model_copy(
            update={
                "event_time": ordered[-1].event_time + timedelta(days=1),
                "event_id": UUID(int=(ordered[-1].event_id.int + 1) % (1 << 128)),
            }
        )

        before = self._unmutated(ordered, baseline)
        after = self._recompute([*ordered, probe], baseline)

        moved = [
            column
            for column in self._prior_only_columns()
            for key, row in before.items()
            if after[key][column] != row[column]
        ]
        found = sorted(set(moved))
        return LeakageCheck(
            name=name,
            passed=not found,
            severity="error",
            message=(
                "Appending a later event left every earlier prior-history "
                "feature unchanged"
                if not found
                else f"{len(found)} prior-history column(s) changed when a later "
                f"event was appended, starting with {found[0]!r}"
            ),
        )

    def _check_no_same_timestamp_contribution(
        self,
        events: Sequence[AuthEvent] | None,
        baseline: BaselineTransform | None,
    ) -> LeakageCheck:
        """Duplicate a timestamp and assert its peers' features do not move."""
        name = "NO_SAME_TIMESTAMP_CONTRIBUTION"
        if not events:
            return LeakageCheck(
                name=name,
                passed=True,
                severity="error",
                skipped=True,
                message="Skipped: the source events were not supplied",
            )

        ordered = sort_events_canonically(events)
        pivot = ordered[len(ordered) // 2]
        peer = pivot.model_copy(
            update={"event_id": UUID(int=(pivot.event_id.int + 7) % (1 << 128))}
        )

        before = self._unmutated(ordered, baseline)
        after = self._recompute([*ordered, peer], baseline)

        # Only rows at or before the duplicated timestamp are expected to be
        # invariant.  Later events legitimately see one more event in their
        # history -- that is the peer contributing to the *future*, which is
        # allowed.  What must not happen is the peer contributing to its own
        # simultaneous cohort or to anything earlier.
        cutoff = pivot.event_time
        unchanged_keys = {
            str(event.event_id) for event in ordered if event.event_time <= cutoff
        }

        moved = [
            column
            for column in self._prior_only_columns()
            for key in unchanged_keys
            if after[key][column] != before[key][column]
        ]
        found = sorted(set(moved))
        return LeakageCheck(
            name=name,
            passed=not found,
            severity="error",
            message=(
                f"Adding a simultaneous peer left all {len(unchanged_keys)} row(s) "
                f"at or before that instant unchanged"
                if not found
                else f"{len(found)} prior-history column(s) changed at or before "
                f"the duplicated timestamp, starting with {found[0]!r}"
            ),
        )

    # -- 6-7: baseline provenance -------------------------------------------

    @staticmethod
    def _train_ids(
        assignments: Sequence[SplitAssignment] | None,
    ) -> frozenset[str] | None:
        if assignments is None:
            return None
        return frozenset(a.event_id for a in assignments if a.split is SplitLabel.TRAIN)

    def _check_baseline_fingerprint(
        self,
        artifact: Any | None,
        events: Sequence[AuthEvent] | None,
        assignments: Sequence[SplitAssignment] | None,
    ) -> LeakageCheck:
        """Recompute the training fingerprint independently and compare.

        This is what turns "the baseline was fitted on training data only" from
        a claim in a code review into something a machine can verify: the
        auditor derives the expected fingerprint from the split table without
        consulting the baseline at all.
        """
        name = "BASELINE_SOURCE_FINGERPRINT_MATCHES_TRAIN"
        train_ids = self._train_ids(assignments)
        if artifact is None or events is None or train_ids is None:
            return LeakageCheck(
                name=name,
                passed=True,
                severity="error",
                skipped=True,
                message="Skipped: no baseline artifact, events, or split table",
            )

        train_events = [e for e in events if str(e.event_id) in train_ids]
        expected = compute_events_fingerprint(train_events)
        actual = artifact.fitted_source_fingerprint
        return LeakageCheck(
            name=name,
            passed=expected == actual,
            severity="error",
            message=(
                "The baseline was fitted on exactly the training events"
                if expected == actual
                else "The baseline's recorded source fingerprint does not match "
                "the training split recomputed from the split table"
            ),
        )

    def _check_baseline_not_fit_on_evaluation(
        self,
        artifact: Any | None,
        assignments: Sequence[SplitAssignment] | None,
    ) -> LeakageCheck:
        name = "BASELINE_NOT_FIT_ON_EVALUATION_DATA"
        train_ids = self._train_ids(assignments)
        if artifact is None or train_ids is None:
            return LeakageCheck(
                name=name,
                passed=True,
                severity="error",
                skipped=True,
                message="Skipped: no baseline artifact or split table",
            )

        evaluation = sum(
            1 for a in assignments or () if a.split is not SplitLabel.TRAIN
        )
        consistent = artifact.total_fit_events <= len(train_ids)
        return LeakageCheck(
            name=name,
            passed=consistent,
            severity="error",
            message=(
                f"The baseline consumed {artifact.total_fit_events} event(s), "
                f"within the {len(train_ids)}-event training split "
                f"({evaluation} evaluation event(s) were withheld)"
                if consistent
                else f"The baseline consumed {artifact.total_fit_events} event(s), "
                f"more than the {len(train_ids)} in the training split"
            ),
        )

    # -- 8-10: split integrity ----------------------------------------------

    def _check_purge_respected(
        self,
        events: Sequence[AuthEvent] | None,
        assignments: Sequence[SplitAssignment] | None,
    ) -> LeakageCheck:
        """Verify no supervised event's lookback reaches into another split.

        Checked behaviourally against the actual assignments rather than by
        re-deriving the boundaries, so a purge that was configured but not
        applied is still caught.
        """
        name = "PURGE_INTERVAL_RESPECTED"
        if not events or assignments is None:
            return LeakageCheck(
                name=name,
                passed=True,
                severity="error",
                skipped=True,
                message="Skipped: no events or split table supplied",
            )

        by_id = {a.event_id: a.split for a in assignments}
        ordered = sort_events_canonically(events)
        window = self._config.max_window

        # Track only the most recent event time per supervised split.  If any
        # event of split T lies inside this anchor's lookback, the *latest*
        # such event does too, because it is the closest one below the anchor.
        # That turns the check from "scan every event in the window" -- which
        # is quadratic on a dense stream -- into constant work per event.
        last_seen: dict[SplitLabel, datetime] = {}
        violations = 0

        for moment, block in _group_by_time(ordered):
            lower = moment - window
            for event in block:
                split = by_id.get(str(event.event_id))
                if split not in SUPERVISED_SPLITS:
                    continue
                if any(
                    other is not split and seen >= lower
                    for other, seen in last_seen.items()
                ):
                    violations += 1

            # Updated only after the whole block is checked, so simultaneous
            # events never count against one another.
            for event in block:
                split = by_id.get(str(event.event_id))
                if split in SUPERVISED_SPLITS:
                    assert split is not None
                    last_seen[split] = moment

        return LeakageCheck(
            name=name,
            passed=violations == 0,
            severity="error",
            message=(
                "No supervised event's lookback window reaches into a different "
                "supervised split"
                if violations == 0
                else f"{violations} supervised event(s) have a lookback window "
                f"reaching into a different supervised split; the purge "
                f"interval is too short or was not applied"
            ),
        )

    @staticmethod
    def _check_campaign_isolation(
        labels: Sequence[GroundTruthLabel] | None,
        assignments: Sequence[SplitAssignment] | None,
    ) -> LeakageCheck:
        from password_attack_detector.data.enums import ScenarioType

        name = "CAMPAIGN_GROUPS_ISOLATED"
        if labels is None or assignments is None:
            return LeakageCheck(
                name=name,
                passed=True,
                severity="error",
                skipped=True,
                message="Skipped: no labels or split table supplied",
            )

        by_id = {a.event_id: a.split for a in assignments}
        per_campaign: dict[str, set[SplitLabel]] = {}
        for label in labels:
            if label.scenario is ScenarioType.NORMAL:
                continue
            split = by_id.get(str(label.event_id))
            if split not in SUPERVISED_SPLITS:
                continue
            assert split is not None
            per_campaign.setdefault(label.campaign_id, set()).add(split)

        straddling = sum(1 for splits in per_campaign.values() if len(splits) > 1)
        return LeakageCheck(
            name=name,
            passed=straddling == 0,
            severity="error",
            message=(
                f"All {len(per_campaign)} attack campaign(s) sit within a single "
                f"supervised split"
                if straddling == 0
                else f"{straddling} campaign(s) span more than one supervised split"
            ),
        )

    @staticmethod
    def _check_holdout_excluded(
        labels: Sequence[GroundTruthLabel] | None,
        assignments: Sequence[SplitAssignment] | None,
    ) -> LeakageCheck:
        name = "HOLDOUT_EXCLUDED_FROM_SUPERVISED"
        if labels is None or assignments is None:
            return LeakageCheck(
                name=name,
                passed=True,
                severity="error",
                skipped=True,
                message="Skipped: no labels or split table supplied",
            )

        by_id = {a.event_id: a.split for a in assignments}
        leaked = sum(
            1
            for label in labels
            if not label.supervised_training_eligible
            and by_id.get(str(label.event_id)) in SUPERVISED_SPLITS
        )
        return LeakageCheck(
            name=name,
            passed=leaked == 0,
            severity="error",
            message=(
                "No holdout event appears in a supervised split"
                if leaked == 0
                else f"{leaked} novel-anomaly holdout event(s) leaked into a "
                f"supervised split"
            ),
        )

    # -- 11-12: schema-level checks -----------------------------------------

    def _check_current_versus_prior(self) -> LeakageCheck:
        """Verify every column's declared timing class matches its name."""
        name = "CURRENT_VS_PRIOR_FIELDS_DISTINGUISHED"
        problems: list[str] = []
        for spec in self._catalog.specs:
            if spec.name.startswith("current_") and (
                spec.leakage_class is not LeakageClass.CURRENT_EVENT_CONTEXT
            ):
                problems.append(spec.name)
            if (
                spec.name.startswith(("prior_", "previous_")) or spec.window is not None
            ) and spec.leakage_class is not LeakageClass.PRIOR_ONLY:
                problems.append(spec.name)
            if spec.uses_current_event and spec.window is not None:
                problems.append(spec.name)

        found = sorted(set(problems))
        return LeakageCheck(
            name=name,
            passed=not found,
            severity="error",
            message=(
                "Current-event context and prior-history features are cleanly "
                "distinguished by declared leakage class"
                if not found
                else f"{len(found)} column(s) have a leakage class inconsistent "
                f"with their role, starting with {found[0]!r}"
            ),
        )

    @staticmethod
    def _check_join_integrity(
        frame: FeatureFrame,
        labels: Sequence[GroundTruthLabel] | None,
        assignments: Sequence[SplitAssignment] | None,
    ) -> LeakageCheck:
        name = "JOIN_KEY_INTEGRITY"
        if labels is None or assignments is None:
            return LeakageCheck(
                name=name,
                passed=True,
                severity="error",
                skipped=True,
                message="Skipped: no labels or split table supplied",
            )

        feature_ids = {row[ANCHOR_EVENT_ID] for row in frame.rows}
        label_ids = {str(label.event_id) for label in labels}
        split_ids = {a.event_id for a in assignments}

        problems: list[str] = []
        if len(feature_ids) != len(frame.rows):
            problems.append("duplicate anchor identifiers in the feature table")
        if feature_ids != label_ids:
            problems.append(
                f"{len(feature_ids ^ label_ids)} identifier(s) differ between "
                f"features and labels"
            )
        if feature_ids != split_ids:
            problems.append(
                f"{len(feature_ids ^ split_ids)} identifier(s) differ between "
                f"features and splits"
            )

        return LeakageCheck(
            name=name,
            passed=not problems,
            severity="error",
            message=(
                "Features, labels, and splits join one-to-one on event_id"
                if not problems
                else "; ".join(problems)
            ),
        )


def audit_result_to_markdown(result: LeakageAuditResult) -> str:
    """Render a leakage audit as Markdown, using aggregates only."""
    lines = [
        "# Leakage Audit",
        "",
        f"- Status: **{result.status}**",
        f"- Features checked: {result.checked_feature_count}",
        f"- Rows checked: {result.checked_row_count}",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|--------|-------|-------|",
    ]
    for check in result.checks:
        if check.skipped:
            verdict = "skipped"
        elif check.passed:
            verdict = "pass"
        else:
            verdict = "FAIL"
        lines.append(f"| `{check.name}` | {verdict} | {check.message} |")

    if result.split_summary:
        lines.extend(
            ["", "## Split sizes", "", "| Split | Events |", "|--------|-------|"]
        )
        for split, count in sorted(result.split_summary.items()):
            lines.append(f"| `{split}` | {count} |")

    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- The behavioural checks probe the configured feature set on the "
            "supplied events. They demonstrate the timing contract holds for "
            "that data; they are not a proof for all possible inputs.",
            "- Baseline provenance is verified by fingerprint comparison, which "
            "detects a mismatched fit but not a baseline fitted from data that "
            "never entered the split table.",
            "- Passing this audit says nothing about detection effectiveness.",
            "",
        ]
    )
    return "\n".join(lines)
