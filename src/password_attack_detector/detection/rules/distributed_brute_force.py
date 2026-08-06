"""Distributed brute-force rule: PAD-DBF-001.

One account, many sources, few attempts from each.  The shape exists to defeat
per-source rate limiting, and it is exactly what PAD-BF-001's concentration
conditions are built to miss: no single user-source pair accumulates enough
failures to look like a guessing run.

Two ceilings do the discriminating work.  The per-source attempt ceiling keeps
one high-volume source from being reported as distributed, and the source
fan-out ceiling keeps a spraying source -- which also touches many pairs -- out
of a rule about a single targeted account.
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
    saturate,
    saturate_inverse,
)
from password_attack_detector.detection.schemas import (
    EvidenceItem,
    RuleEvaluationResult,
)

__all__ = ["DistributedBruteForceRule"]


class _DistributedBruteForcePreparedRule(BasePreparedRule):
    """Evaluate PAD-DBF-001 against one snapshot."""

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        preparation = self.preparation
        unique_sources = view.count("user_unique_source_count__{cardinality_window}")
        user_failures = view.count("user_failure_count__{window}")
        pair_attempts = view.count("pair_attempt_count__{window}")
        source_users = view.count("source_unique_user_count__{cardinality_window}")

        # The minimum-history gate has already established this is not null.
        failure_rate = view.number("user_failure_rate__{window}")
        if failure_rate is None:  # pragma: no cover - defended by the gate above
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=(
                    insufficient_history_reason_code(
                        preparation.feature("user_failure_rate__{window}")
                    ),
                ),
            )

        min_unique_sources = preparation.param_int("min_unique_sources")
        min_user_failures = preparation.param_int("min_user_failures")
        min_failure_rate = preparation.param_float("min_user_failure_rate")
        max_pair_attempts = preparation.param_int("max_pair_attempts")
        max_source_users = preparation.param_int("max_source_unique_users")

        unmet: list[str] = []
        if unique_sources < min_unique_sources:
            unmet.append("BELOW_SOURCE_FANOUT_THRESHOLD")
        if user_failures < min_user_failures:
            unmet.append("BELOW_USER_FAILURE_COUNT")
        if failure_rate < min_failure_rate:
            unmet.append("BELOW_USER_FAILURE_RATE")
        if pair_attempts > max_pair_attempts:
            unmet.append("PER_SOURCE_VOLUME_TOO_HIGH")
        if source_users > max_source_users:
            unmet.append("SOURCE_TARGETS_TOO_MANY_ACCOUNTS")
        if unmet:
            return self.not_fired(anchor_id, anchor_time, reason_codes=tuple(unmet))

        multiple = preparation.signal.saturation_multiple
        evidence: list[EvidenceItem] = [
            self.evidence_for(
                "DBF_USER_SOURCE_FANOUT",
                unique_sources,
                threshold_value=min_unique_sources,
            ),
            self.evidence_for(
                "DBF_USER_FAILURE_COUNT",
                user_failures,
                threshold_value=min_user_failures,
            ),
            self.evidence_for(
                "DBF_USER_FAILURE_RATE",
                failure_rate,
                threshold_value=min_failure_rate,
            ),
            self.evidence_for(
                "DBF_LOW_PER_SOURCE_VOLUME",
                pair_attempts,
                threshold_value=max_pair_attempts,
            ),
            self.evidence_for(
                "DBF_SOURCE_NOT_FANNED_OUT",
                source_users,
                threshold_value=max_source_users,
            ),
        ]
        components = [
            SignalComponent(
                name="source_fanout",
                weight=1.0,
                value=saturate(unique_sources, min_unique_sources, multiple),
            ),
            SignalComponent(
                name="user_failures",
                weight=0.9,
                value=saturate(user_failures, min_user_failures, multiple),
            ),
            SignalComponent(name="failure_rate", weight=0.6, value=failure_rate),
            SignalComponent(
                name="per_source_volume",
                weight=0.6,
                value=saturate_inverse(pair_attempts, max_pair_attempts, multiple),
            ),
        ]

        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength(components),
            evidence=evidence,
            reason_codes=tuple(item.evidence_code for item in evidence),
        )


class DistributedBruteForceRule(BaseRule):
    """PAD-DBF-001 -- one account failing across many low-volume sources."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-DBF-001"))

    def _build(self, preparation: RulePreparation) -> PreparedRule:
        return _DistributedBruteForcePreparedRule(preparation)
