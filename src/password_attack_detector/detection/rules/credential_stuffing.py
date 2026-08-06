"""Credential-stuffing rule: PAD-CS-001.

Stuffing replays credentials obtained elsewhere, so most attempts fail and a
few succeed.  That mixed outcome is the signature separating it from spraying,
where a run of pure failures is the norm.

The rule reads behaviour only.  It has no visibility into whether any password
was reused from another service, holds no credential list, and compares no
secret -- the "reuse" in the name describes the attacker's method, not this
rule's inputs.  Novelty of device or country for the targeted account is the
closest observable proxy, and it is required rather than optional: broad
fan-out from a corporate gateway is ordinary, and without an unfamiliar context
this rule would report every busy proxy.
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
    safe_ratio,
    saturate,
    saturate_inverse,
)
from password_attack_detector.detection.schemas import (
    EvidenceItem,
    RuleEvaluationResult,
)

__all__ = ["CredentialStuffingRule"]


class _CredentialStuffingPreparedRule(BasePreparedRule):
    """Evaluate PAD-CS-001 against one snapshot."""

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        preparation = self.preparation

        # An account absent from the fitted baseline has no "unfamiliar" to
        # measure against.  Reporting that as a clean negative would conflate
        # "this device is known for this account" with "this account has never
        # been seen", which are different facts.
        if not view.flag("user_in_baseline"):
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("ACCOUNT_ABSENT_FROM_BASELINE",)
            )

        unique_users = view.count("source_unique_user_count__{cardinality_window}")
        successes = view.count("source_success_count__{window}")
        failures = view.count("source_failure_count__{window}")
        attempts = view.count("source_attempt_count__{window}")
        devices = view.count("source_unique_device_count__{cardinality_window}")
        user_agents = view.count("source_unique_user_agent_count__{cardinality_window}")

        min_unique_users = preparation.param_int("min_unique_users")
        min_successes = preparation.param_int("min_successes")
        min_failures = preparation.param_int("min_failures")
        min_devices = preparation.param_int("min_unique_devices")
        min_user_agents = preparation.param_int("min_unique_user_agents")
        max_attempts_per_user = preparation.param_float("max_attempts_per_user")

        attempts_per_user = safe_ratio(attempts, unique_users)
        if attempts_per_user is None:
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("NO_TARGETED_ACCOUNTS_OBSERVED",)
            )

        # Either kind of client diversity satisfies the condition; the anchor
        # need not present both a new device and a new user agent.
        devices_met = devices >= min_devices
        user_agents_met = user_agents >= min_user_agents

        # A null novelty flag is treated as "not new" rather than as unknown.
        # The baseline gate above has already established the account is known,
        # so a null here means the anchor carried no such attribute at all.
        new_device = view.flag("is_new_device_for_user") is True
        new_country = view.flag("is_new_country_for_user") is True

        unmet: list[str] = []
        if unique_users < min_unique_users:
            unmet.append("BELOW_ACCOUNT_FANOUT_THRESHOLD")
        if successes < min_successes or failures < min_failures:
            unmet.append("OUTCOME_MIX_NOT_PRESENT")
        if not (devices_met or user_agents_met):
            unmet.append("BELOW_CLIENT_DIVERSITY_THRESHOLD")
        if attempts_per_user > max_attempts_per_user:
            unmet.append("ATTEMPTS_PER_ACCOUNT_TOO_HIGH")
        if not (new_device or new_country):
            unmet.append("CONTEXT_ALREADY_FAMILIAR_FOR_ACCOUNT")
        if unmet:
            return self.not_fired(anchor_id, anchor_time, reason_codes=tuple(unmet))

        multiple = preparation.signal.saturation_multiple

        # The catalog binds this evidence item to the device-count column, and
        # either count can satisfy the condition.  The device count is reported
        # when it met its own threshold, otherwise the user-agent count that
        # did -- so the comparator in the evidence is always true of the value
        # it carries.
        if devices_met:
            diversity_observed, diversity_threshold = devices, min_devices
        else:
            diversity_observed, diversity_threshold = user_agents, min_user_agents

        evidence: list[EvidenceItem] = [
            self.evidence_for(
                "CS_SOURCE_USER_FANOUT", unique_users, threshold_value=min_unique_users
            ),
            self.evidence_for(
                "CS_MIXED_OUTCOMES", successes, threshold_value=min_successes
            ),
            self.evidence_for(
                "CS_CLIENT_DIVERSITY",
                diversity_observed,
                threshold_value=diversity_threshold,
            ),
            self.evidence_for(
                "CS_ATTEMPTS_PER_USER",
                attempts_per_user,
                threshold_value=max_attempts_per_user,
            ),
            self.evidence_for("CS_UNFAMILIAR_CONTEXT", True),
        ]
        components = [
            SignalComponent(
                name="account_fanout",
                weight=1.0,
                value=saturate(unique_users, min_unique_users, multiple),
            ),
            SignalComponent(
                name="failure_volume",
                weight=0.8,
                value=saturate(failures, min_failures, multiple),
            ),
            SignalComponent(
                name="client_diversity",
                weight=0.6,
                value=saturate(diversity_observed, diversity_threshold, multiple),
            ),
            SignalComponent(
                name="attempts_per_user",
                weight=0.6,
                value=saturate_inverse(
                    attempts_per_user, max_attempts_per_user, multiple
                ),
            ),
            # Two unfamiliar contexts are more than one, and a boolean cannot
            # saturate, so novelty contributes a graded share rather than a
            # constant.
            SignalComponent(
                name="unfamiliar_context",
                weight=0.7,
                value=0.5 * (int(new_device) + int(new_country)),
            ),
        ]

        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength(components),
            evidence=evidence,
            reason_codes=tuple(item.evidence_code for item in evidence),
        )


class CredentialStuffingRule(BaseRule):
    """PAD-CS-001 -- broad fan-out with mixed outcomes and unfamiliar context."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-CS-001"))

    def _build(self, preparation: RulePreparation) -> PreparedRule:
        return _CredentialStuffingPreparedRule(preparation)
