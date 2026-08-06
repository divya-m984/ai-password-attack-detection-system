"""MFA sequence anomaly rule: PAD-MFA-001.

The rule reports a *sequence anomaly*, not a defeated control.  A run of
multi-factor challenges and failures ending in another abnormal outcome is the
shape of an attacker pushing prompts at someone; it is also the shape of a
failing authenticator app or a clock-drifted hardware token.

Two gates keep an ordinary challenge from firing it.  A minimum-history gate
runs first: below the configured attempt count, or below the configured number
of combined challenge and multi-factor-failure observations, the rule returns
``insufficient_data`` rather than a clean negative -- it has not seen enough
multi-factor history to call any sequence anomalous.  Only then are the two
required conditions evaluated, and both must hold: elevated prior activity
*and* an abnormal outcome on the anchor itself.

Shares the ``session_anomaly`` correlation group with PAD-ATO-001 so one
compromise narrative is not counted twice.
"""

from __future__ import annotations

from datetime import datetime

from password_attack_detector.data.enums import MFAOutcome
from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.rules.base import (
    BasePreparedRule,
    BaseRule,
    PreparedRule,
    RulePreparation,
    SignalComponent,
    SnapshotView,
    saturate,
)
from password_attack_detector.detection.schemas import (
    EvidenceItem,
    RuleEvaluationResult,
)

__all__ = ["MFASequenceAnomalyRule"]

#: Anchor multi-factor outcomes treated as abnormal.  ``not_required`` and
#: ``not_enrolled`` describe a policy rather than an anomaly, and ``passed`` is
#: the control working.
_ABNORMAL_MFA_OUTCOMES: frozenset[str] = frozenset(
    {str(MFAOutcome.FAILED), str(MFAOutcome.BYPASSED)}
)


class _MFASequenceAnomalyPreparedRule(BasePreparedRule):
    """Evaluate PAD-MFA-001 against one snapshot."""

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        preparation = self.preparation
        attempts = view.count("user_attempt_count__{window}")
        mfa_failures = view.count("user_mfa_failure_count__{window}")
        challenges = view.count("user_challenge_count__{window}")

        min_history_events = preparation.param_int("min_mfa_history_events")
        min_observations = preparation.param_int("min_mfa_observations")

        # The history gate is checked before any threshold, so a quiet account
        # is reported as unobserved rather than as clean.  This rule's
        # min_history specification is empty on purpose: the requirement is a
        # count comparison, not a null check, and the base class only gates on
        # nulls.
        if attempts < min_history_events:
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=("MFA_INSUFFICIENT_ATTEMPT_HISTORY",),
            )
        if challenges + mfa_failures < min_observations:
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("MFA_INSUFFICIENT_OBSERVATIONS",)
            )

        current_outcome = view.text("current_mfa_outcome")
        if current_outcome is None:
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("MFA_OUTCOME_NOT_RECORDED",)
            )

        min_mfa_failures = preparation.param_int("min_mfa_failures")
        min_challenges = preparation.param_int("min_challenges")
        failures_met = mfa_failures >= min_mfa_failures
        challenges_met = challenges >= min_challenges

        unmet: list[str] = []
        if not (failures_met or challenges_met):
            unmet.append("PRIOR_MFA_ACTIVITY_NOT_ELEVATED")
        if current_outcome not in _ABNORMAL_MFA_OUTCOMES:
            unmet.append("ANCHOR_MFA_OUTCOME_NOT_ABNORMAL")
        if unmet:
            return self.not_fired(anchor_id, anchor_time, reason_codes=tuple(unmet))

        multiple = preparation.signal.saturation_multiple
        evidence: list[EvidenceItem] = [
            self.evidence_for(
                "MFA_HISTORY_SUFFICIENT", attempts, threshold_value=min_history_events
            )
        ]
        components: list[SignalComponent] = []

        if failures_met:
            evidence.append(
                self.evidence_for(
                    "MFA_PRIOR_FAILURES",
                    mfa_failures,
                    threshold_value=min_mfa_failures,
                )
            )
            components.append(
                SignalComponent(
                    name="prior_mfa_failures",
                    weight=1.0,
                    value=saturate(mfa_failures, min_mfa_failures, multiple),
                )
            )
        if challenges_met:
            evidence.append(
                self.evidence_for(
                    "MFA_PRIOR_CHALLENGES", challenges, threshold_value=min_challenges
                )
            )
            components.append(
                SignalComponent(
                    name="prior_challenges",
                    weight=0.8,
                    value=saturate(challenges, min_challenges, multiple),
                )
            )

        evidence.append(self.evidence_for("MFA_CURRENT_OUTCOME", current_outcome))

        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength(components),
            evidence=evidence,
            reason_codes=tuple(item.evidence_code for item in evidence),
        )


class MFASequenceAnomalyRule(BaseRule):
    """PAD-MFA-001 -- elevated multi-factor activity plus an abnormal outcome."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-MFA-001"))

    def _build(self, preparation: RulePreparation) -> PreparedRule:
        return _MFASequenceAnomalyPreparedRule(preparation)
