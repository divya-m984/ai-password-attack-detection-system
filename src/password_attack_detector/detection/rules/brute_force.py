"""Concentrated credential-guessing rules: PAD-BF-001 and PAD-BF-002.

The two rules describe the same behaviour from opposite ends.  PAD-BF-001 sees
a guessing run in progress: repeated failures concentrated on one account from
one source.  PAD-BF-002 sees the moment a run stops -- a success immediately
after a sustained burst of failures.

They share the ``credential_guessing_single_target`` correlation group with
PAD-DBF-001, so a later scoring stage collapses them rather than counting one
failure burst twice.  On any single anchor they are mutually exclusive: the
current outcome cannot be both a failure feeding PAD-BF-001's rate condition
and the success PAD-BF-002 requires.
"""

from __future__ import annotations

from datetime import datetime

from password_attack_detector.data.enums import AuthOutcome
from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.rules.base import (
    BasePreparedRule,
    BaseRule,
    PreparedRule,
    RulePreparation,
    SignalComponent,
    SnapshotView,
    insufficient_history_reason_code,
    saturate,
    saturate_inverse,
)
from password_attack_detector.detection.schemas import (
    EvidenceItem,
    RuleEvaluationResult,
)
from password_attack_detector.exceptions import RuleEvaluationError
from password_attack_detector.features.catalog import FeatureCatalog, LeakageClass

__all__ = [
    "ConcentratedBruteForceRule",
    "SuccessAfterFailureBurstRule",
]

#: The one current-event field PAD-BF-002 is permitted to read.  Everything
#: else it requires must be prior-only history, which :func:`_assert_prior_only`
#: enforces at preparation time.
_CURRENT_OUTCOME: str = "current_authentication_outcome"

#: Outcomes that count as a preceding non-success for PAD-BF-002.  A challenge
#: is deliberately excluded: it is a control engaging, not a rejected guess.
_PRECEDING_FAILURE_OUTCOMES: frozenset[str] = frozenset(
    {str(AuthOutcome.FAILURE), str(AuthOutcome.BLOCKED)}
)


# ---------------------------------------------------------------------------
# PAD-BF-001 -- concentrated brute force
# ---------------------------------------------------------------------------


class _ConcentratedBruteForcePreparedRule(BasePreparedRule):
    """Evaluate PAD-BF-001 against one snapshot.

    Five conditions must hold together.  Volume alone is not enough -- an
    account that simply gets used heavily clears the count thresholds -- so the
    failure *share*, the unbroken failure run, and the source's account
    concentration all have to agree.  The concentration ceiling is what keeps a
    password-spraying source out of this rule: a source touching many accounts
    is fan-out, and PAD-PS-001's business.
    """

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        preparation = self.preparation
        pair_failures = view.count("pair_failure_count__{window}")
        user_failures = view.count("user_failure_count__{window}")
        consecutive = view.count("prior_consecutive_user_failures")
        source_users = view.count("source_unique_user_count__{cardinality_window}")

        # The minimum-history gate has already established this is not null.
        failure_rate = view.number("pair_failure_rate__{window}")
        if failure_rate is None:  # pragma: no cover - defended by the gate above
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=(
                    insufficient_history_reason_code(
                        preparation.feature("pair_failure_rate__{window}")
                    ),
                ),
            )

        min_pair_failures = preparation.param_int("min_pair_failures")
        min_user_failures = preparation.param_int("min_user_failures")
        min_failure_rate = preparation.param_float("min_pair_failure_rate")
        min_consecutive = preparation.param_int("min_consecutive_failures")
        max_source_users = preparation.param_int("max_source_unique_users")

        # Every condition is evaluated with "at or above" / "at or below", so
        # exact threshold equality is a match rather than a near miss.
        unmet: list[str] = []
        if pair_failures < min_pair_failures:
            unmet.append("BELOW_PAIR_FAILURE_COUNT")
        if user_failures < min_user_failures:
            unmet.append("BELOW_USER_FAILURE_COUNT")
        if failure_rate < min_failure_rate:
            unmet.append("BELOW_PAIR_FAILURE_RATE")
        if consecutive < min_consecutive:
            unmet.append("BELOW_CONSECUTIVE_FAILURES")
        if source_users > max_source_users:
            unmet.append("SOURCE_TARGETS_TOO_MANY_ACCOUNTS")
        if unmet:
            return self.not_fired(anchor_id, anchor_time, reason_codes=tuple(unmet))

        multiple = preparation.signal.saturation_multiple
        evidence: list[EvidenceItem] = [
            self.evidence_for(
                "BF_PAIR_FAILURE_COUNT",
                pair_failures,
                threshold_value=min_pair_failures,
            ),
            self.evidence_for(
                "BF_USER_FAILURE_COUNT",
                user_failures,
                threshold_value=min_user_failures,
            ),
            self.evidence_for(
                "BF_PAIR_FAILURE_RATE", failure_rate, threshold_value=min_failure_rate
            ),
            self.evidence_for(
                "BF_CONSECUTIVE_USER_FAILURES",
                consecutive,
                threshold_value=min_consecutive,
            ),
            self.evidence_for(
                "BF_SOURCE_TARGET_CONCENTRATION",
                source_users,
                threshold_value=max_source_users,
            ),
        ]
        components = [
            SignalComponent(
                name="pair_failures",
                weight=1.0,
                value=saturate(pair_failures, min_pair_failures, multiple),
            ),
            SignalComponent(
                name="user_failures",
                weight=0.8,
                value=saturate(user_failures, min_user_failures, multiple),
            ),
            # A rate is already normalised to [0, 1] and is used directly, so a
            # near-total failure share reads as strong on its own terms rather
            # than being re-scaled against its own threshold.
            SignalComponent(name="failure_rate", weight=0.6, value=failure_rate),
            SignalComponent(
                name="consecutive_failures",
                weight=0.8,
                value=saturate(consecutive, min_consecutive, multiple),
            ),
        ]

        self._add_cadence_support(view, evidence, components)
        self._add_blocked_support(view, evidence, components)

        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength(components),
            evidence=evidence,
            reason_codes=tuple(item.evidence_code for item in evidence),
        )

    def _add_cadence_support(
        self,
        view: SnapshotView,
        evidence: list[EvidenceItem],
        components: list[SignalComponent],
    ) -> None:
        """Add the optional cadence contribution when the gap is short enough.

        Supporting only: a null mean interarrival, or one above the ceiling,
        removes the contribution rather than the detection.
        """
        observed = view.number("pair_mean_interarrival_seconds__{window}")
        ceiling = self.preparation.param_float("max_mean_interarrival_seconds")
        if observed is None or observed > ceiling:
            return
        evidence.append(
            self.evidence_for("BF_PAIR_INTERARRIVAL", observed, threshold_value=ceiling)
        )
        components.append(
            SignalComponent(
                name="pair_interarrival",
                weight=0.4,
                value=saturate_inverse(
                    observed, ceiling, self.preparation.signal.saturation_multiple
                ),
            )
        )

    def _add_blocked_support(
        self,
        view: SnapshotView,
        evidence: list[EvidenceItem],
        components: list[SignalComponent],
    ) -> None:
        """Add the optional blocked-account contribution.

        This is the project's only use of blocked-account activity.  It is
        supporting evidence inside a brute-force rule and never a condition: no
        standalone blocked-account-targeting rule exists, because lockouts are
        a control working as designed at least as often as they are an attack.
        """
        blocked = view.count("user_blocked_count__{window}")
        threshold = self.preparation.param_int("blocked_support_threshold")
        if blocked < threshold:
            return
        evidence.append(
            self.evidence_for("BF_BLOCKED_ACTIVITY", blocked, threshold_value=threshold)
        )
        components.append(
            SignalComponent(
                name="blocked_activity",
                weight=0.3,
                value=saturate(
                    blocked, threshold, self.preparation.signal.saturation_multiple
                ),
            )
        )


class ConcentratedBruteForceRule(BaseRule):
    """PAD-BF-001 -- repeated failures concentrated on one account and source."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-BF-001"))

    def _build(self, preparation: RulePreparation) -> PreparedRule:
        return _ConcentratedBruteForcePreparedRule(preparation)


# ---------------------------------------------------------------------------
# PAD-BF-002 -- success after a failure burst
# ---------------------------------------------------------------------------


class _SuccessAfterFailureBurstPreparedRule(BasePreparedRule):
    """Evaluate PAD-BF-002 against one snapshot.

    The anchor's success is the only current-event fact consulted.  Everything
    that makes the success interesting -- the size of the preceding burst, the
    preceding outcome, how recently the last failure occurred -- comes from
    prior-only sequence features, so the rule cannot learn anything about the
    anchor beyond the outcome the event already recorded.
    """

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        preparation = self.preparation
        if view.text(_CURRENT_OUTCOME) != str(AuthOutcome.SUCCESS):
            return self.not_fired(
                anchor_id, anchor_time, reason_codes=("ANCHOR_DID_NOT_SUCCEED",)
            )

        pair_burst = view.count("prior_failures_since_pair_success")
        user_burst = view.count("prior_failures_since_user_success")
        min_pair_burst = preparation.param_int("min_pair_failures_since_success")
        min_user_burst = preparation.param_int("min_user_failures_since_success")
        pair_met = pair_burst >= min_pair_burst
        user_met = user_burst >= min_user_burst

        # The minimum-history gate established both of these are non-null.
        previous_outcome = view.text("previous_user_outcome")
        seconds_since_failure = view.number("seconds_since_user_previous_failure")
        if (
            previous_outcome is None or seconds_since_failure is None
        ):  # pragma: no cover - defended by the gate above
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("MISSING_PRIOR_SEQUENCE_HISTORY",)
            )
        max_seconds = preparation.param_float("max_seconds_since_previous_failure")

        unmet: list[str] = []
        if not (pair_met or user_met):
            unmet.append("BELOW_FAILURE_BURST_THRESHOLD")
        if previous_outcome not in _PRECEDING_FAILURE_OUTCOMES:
            unmet.append("PRECEDING_EVENT_WAS_NOT_A_FAILURE")
        if seconds_since_failure > max_seconds:
            unmet.append("PRECEDING_FAILURE_TOO_DISTANT")
        if unmet:
            return self.not_fired(anchor_id, anchor_time, reason_codes=tuple(unmet))

        multiple = preparation.signal.saturation_multiple
        evidence: list[EvidenceItem] = [
            self.evidence_for(
                "BF2_CURRENT_SUCCESS", str(AuthOutcome.SUCCESS), threshold_value=None
            )
        ]
        components: list[SignalComponent] = []

        # Both bursts are reported when both cleared, so the evidence records
        # the full shape of the run rather than only the first branch that met.
        if pair_met:
            evidence.append(
                self.evidence_for(
                    "BF2_PAIR_FAILURE_BURST",
                    pair_burst,
                    threshold_value=min_pair_burst,
                )
            )
            components.append(
                SignalComponent(
                    name="pair_burst",
                    weight=1.0,
                    value=saturate(pair_burst, min_pair_burst, multiple),
                )
            )
        if user_met:
            evidence.append(
                self.evidence_for(
                    "BF2_USER_FAILURE_BURST",
                    user_burst,
                    threshold_value=min_user_burst,
                )
            )
            components.append(
                SignalComponent(
                    name="user_burst",
                    weight=1.0,
                    value=saturate(user_burst, min_user_burst, multiple),
                )
            )

        evidence.append(self.evidence_for("BF2_PREVIOUS_OUTCOME", previous_outcome))
        evidence.append(
            self.evidence_for(
                "BF2_FAILURE_RECENCY",
                seconds_since_failure,
                threshold_value=max_seconds,
            )
        )
        components.append(
            SignalComponent(
                name="failure_recency",
                weight=0.5,
                value=saturate_inverse(seconds_since_failure, max_seconds, multiple),
            )
        )

        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength(components),
            evidence=evidence,
            reason_codes=tuple(item.evidence_code for item in evidence),
        )


class SuccessAfterFailureBurstRule(BaseRule):
    """PAD-BF-002 -- a success closing out a sustained run of failures."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-BF-002"))

    def prepare(
        self, config: DetectionConfig, feature_catalog: FeatureCatalog
    ) -> PreparedRule:
        """Prepare the rule, pinning its input surface first.

        The extra check exists because this rule's whole justification is that
        it reads exactly one current-event field.  Asserting that at preparation
        time means a later edit that quietly adds a second one fails the run
        rather than passing review.
        """
        _assert_prior_only_inputs(self.spec.required_features, feature_catalog)
        return super().prepare(config, feature_catalog)

    def _build(self, preparation: RulePreparation) -> PreparedRule:
        return _SuccessAfterFailureBurstPreparedRule(preparation)


def _assert_prior_only_inputs(
    required_features: tuple[str, ...], feature_catalog: FeatureCatalog
) -> None:
    """Assert every required feature but the current outcome is prior-only.

    Raises:
        RuleEvaluationError: if a required feature other than
            ``current_authentication_outcome`` carries any leakage class other
            than :data:`LeakageClass.PRIOR_ONLY`.
    """
    for template in required_features:
        if template == _CURRENT_OUTCOME:
            continue
        # PAD-BF-002 declares no windowed feature, so a template is already a
        # concrete column name and needs no parameter substitution.
        feature = feature_catalog.get(template)
        if feature.leakage_class is not LeakageClass.PRIOR_ONLY:
            raise RuleEvaluationError(
                f"PAD-BF-002 may read only prior-only history beyond the current "
                f"outcome; {template!r} is classified {feature.leakage_class}"
            )
