"""Password-spraying rule: PAD-PS-001.

Spraying is the inverse shape of concentrated brute force.  Instead of many
guesses against one account, it is a few guesses against many accounts, chosen
so that no single account accumulates enough failures to trip a lockout.  The
attempts-per-account ceiling is therefore not a refinement -- it is the
discriminator.  Without it, any high-volume failing source would match, and a
single-account brute force would be reported twice.
"""

from __future__ import annotations

from datetime import datetime

from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.rules.base import (
    BasePreparedRule,
    BaseRule,
    PreparedRule,
    RulePreparation,
    SignalComponent,
    SnapshotView,
    insufficient_history_reason_code,
    safe_ratio,
    saturate,
    saturate_inverse,
)
from password_attack_detector.detection.schemas import (
    EvidenceItem,
    RuleEvaluationResult,
)

__all__ = ["PasswordSprayingRule"]


class _PasswordSprayingPreparedRule(BasePreparedRule):
    """Evaluate PAD-PS-001 against one snapshot."""

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        preparation = self.preparation
        unique_users = view.count("source_unique_user_count__{cardinality_window}")
        failures = view.count("source_failure_count__{window}")
        attempts = view.count("source_attempt_count__{window}")

        # The minimum-history gate has already established this is not null.
        failure_rate = view.number("source_failure_rate__{window}")
        if failure_rate is None:  # pragma: no cover - defended by the gate above
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=(
                    insufficient_history_reason_code(
                        preparation.feature("source_failure_rate__{window}")
                    ),
                ),
            )

        min_unique_users = preparation.param_int("min_unique_users")
        min_failures = preparation.param_int("min_source_failures")
        min_failure_rate = preparation.param_float("min_source_failure_rate")
        max_attempts_per_user = preparation.param_float("max_attempts_per_user")

        # Computed rather than read: no feature expresses attempts per targeted
        # account, and deriving it here keeps the discriminator explicit.  The
        # ratio is undefined for a source with no observed accounts, which is
        # absent history rather than a clean negative.
        attempts_per_user = safe_ratio(attempts, unique_users)
        if attempts_per_user is None:
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("NO_TARGETED_ACCOUNTS_OBSERVED",)
            )

        unmet: list[str] = []
        if unique_users < min_unique_users:
            unmet.append("BELOW_ACCOUNT_FANOUT_THRESHOLD")
        if failures < min_failures:
            unmet.append("BELOW_SOURCE_FAILURE_COUNT")
        if failure_rate < min_failure_rate:
            unmet.append("BELOW_SOURCE_FAILURE_RATE")
        if attempts_per_user > max_attempts_per_user:
            unmet.append("ATTEMPTS_PER_ACCOUNT_TOO_HIGH")
        if unmet:
            return self.not_fired(anchor_id, anchor_time, reason_codes=tuple(unmet))

        multiple = preparation.signal.saturation_multiple
        evidence: list[EvidenceItem] = [
            self.evidence_for(
                "PS_SOURCE_USER_FANOUT", unique_users, threshold_value=min_unique_users
            ),
            self.evidence_for(
                "PS_SOURCE_FAILURE_COUNT", failures, threshold_value=min_failures
            ),
            self.evidence_for(
                "PS_SOURCE_FAILURE_RATE",
                failure_rate,
                threshold_value=min_failure_rate,
            ),
            self.evidence_for(
                "PS_ATTEMPTS_PER_USER",
                attempts_per_user,
                threshold_value=max_attempts_per_user,
            ),
        ]
        components = [
            SignalComponent(
                name="account_fanout",
                weight=1.0,
                value=saturate(unique_users, min_unique_users, multiple),
            ),
            SignalComponent(
                name="source_failures",
                weight=0.8,
                value=saturate(failures, min_failures, multiple),
            ),
            SignalComponent(name="failure_rate", weight=0.6, value=failure_rate),
            SignalComponent(
                name="attempts_per_user",
                weight=0.6,
                value=saturate_inverse(
                    attempts_per_user, max_attempts_per_user, multiple
                ),
            ),
        ]

        self._add_cadence_support(view, evidence, components)

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
        """Add the optional deliberate-cadence contribution.

        Disabled by default.  A floor of zero would match every source, so the
        component is skipped entirely rather than contributing a constant --
        an always-true signal carries no information and would only dilute the
        components that do.
        """
        floor = self.preparation.param_float("min_mean_interarrival_seconds")
        if floor <= 0.0:
            return
        observed = view.number("source_mean_interarrival_seconds__{window}")
        if observed is None or observed < floor:
            return
        evidence.append(
            self.evidence_for("PS_SOURCE_CADENCE", observed, threshold_value=floor)
        )
        components.append(
            SignalComponent(
                name="deliberate_cadence",
                weight=0.4,
                value=saturate(
                    observed, floor, self.preparation.signal.saturation_multiple
                ),
            )
        )


class PasswordSprayingRule(BaseRule):
    """PAD-PS-001 -- few attempts each against many accounts, mostly failing."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-PS-001"))

    def _build(self, preparation: RulePreparation) -> PreparedRule:
        return _PasswordSprayingPreparedRule(preparation)
