"""Bot-like authentication rule: PAD-BOT-001.

Regular timing describes a client, not an intent.  Monitoring probes, service
accounts, and scheduled integrations authenticate on exactly the rhythm this
rule looks for, which is why it carries the lowest default severity of the nine
and why its evidence says the timing *is consistent with* machine generation
rather than that a bot was found.

Three conditions together are what make the finding worth anything.  A volume
floor means no short sequence can fire it -- two fast attempts have a
coefficient of variation, but not a meaningful one.  The dispersion ceiling is
the actual discriminator: irregular human-like timing fails it however high the
volume goes.  Client uniformity separates one automated client from many
different clients that happen to share an address.
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

__all__ = ["BotActivityRule"]


class _BotActivityPreparedRule(BasePreparedRule):
    """Evaluate PAD-BOT-001 against one snapshot."""

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        preparation = self.preparation
        attempts = view.count("source_attempt_count__{dispersion_window}")
        user_agents = view.count("source_unique_user_agent_count__{cardinality_window}")

        # The minimum-history gate has already established this is not null.
        cov = view.number(
            "source_interarrival_coefficient_of_variation__{dispersion_window}"
        )
        if cov is None:  # pragma: no cover - defended by the gate above
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=(
                    insufficient_history_reason_code(
                        preparation.feature(
                            "source_interarrival_coefficient_of_variation__"
                            "{dispersion_window}"
                        )
                    ),
                ),
            )

        # The mean is not part of the history gate, and a window can carry a
        # coefficient of variation without one only in a degenerate snapshot;
        # treating that as a clean negative would be a claim the data does not
        # support.
        mean_interarrival = view.number(
            "source_mean_interarrival_seconds__{dispersion_window}"
        )
        if mean_interarrival is None:
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=(
                    insufficient_history_reason_code(
                        preparation.feature(
                            "source_mean_interarrival_seconds__{dispersion_window}"
                        )
                    ),
                ),
            )

        min_attempts = preparation.param_int("min_attempts")
        max_cov = preparation.param_float("max_interarrival_cov")
        max_mean = preparation.param_float("max_mean_interarrival_seconds")
        max_user_agents = preparation.param_int("max_unique_user_agents")

        unmet: list[str] = []
        if attempts < min_attempts:
            unmet.append("BELOW_ATTEMPT_VOLUME_FLOOR")
        if cov > max_cov:
            unmet.append("TIMING_TOO_IRREGULAR")
        if mean_interarrival > max_mean:
            unmet.append("MEAN_INTERVAL_TOO_LONG")
        if user_agents > max_user_agents:
            unmet.append("CLIENT_CHARACTERISTICS_TOO_VARIED")
        if unmet:
            return self.not_fired(anchor_id, anchor_time, reason_codes=tuple(unmet))

        multiple = preparation.signal.saturation_multiple
        evidence: list[EvidenceItem] = [
            self.evidence_for(
                "BOT_ATTEMPT_VOLUME", attempts, threshold_value=min_attempts
            ),
            self.evidence_for("BOT_TIMING_REGULARITY", cov, threshold_value=max_cov),
            self.evidence_for(
                "BOT_MEAN_INTERARRIVAL", mean_interarrival, threshold_value=max_mean
            ),
            self.evidence_for(
                "BOT_CLIENT_UNIFORMITY", user_agents, threshold_value=max_user_agents
            ),
        ]
        components = [
            SignalComponent(
                name="attempt_volume",
                weight=0.8,
                value=saturate(attempts, min_attempts, multiple),
            ),
            # The dominant component: dispersion is what distinguishes a
            # machine from a person, and volume alone does not.
            SignalComponent(
                name="timing_regularity",
                weight=1.0,
                value=saturate_inverse(cov, max_cov, multiple),
            ),
            SignalComponent(
                name="mean_interarrival",
                weight=0.5,
                value=saturate_inverse(mean_interarrival, max_mean, multiple),
            ),
            SignalComponent(
                name="client_uniformity",
                weight=0.5,
                value=saturate_inverse(user_agents, max_user_agents, multiple),
            ),
        ]

        self._add_fanout_support(view, evidence, components)

        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength(components),
            evidence=evidence,
            reason_codes=tuple(item.evidence_code for item in evidence),
        )

    def _add_fanout_support(
        self,
        view: SnapshotView,
        evidence: list[EvidenceItem],
        components: list[SignalComponent],
    ) -> None:
        """Add the optional account fan-out contribution.

        Disabled by default.  A floor of zero is satisfied by every source, so
        the component is skipped rather than contributing a constant that would
        only dilute the timing evidence.
        """
        floor = self.preparation.param_int("min_unique_users")
        if floor <= 0:
            return
        observed = view.count("source_unique_user_count__{cardinality_window}")
        if observed < floor:
            return
        evidence.append(
            self.evidence_for("BOT_SOURCE_FANOUT", observed, threshold_value=floor)
        )
        components.append(
            SignalComponent(
                name="account_fanout",
                weight=0.4,
                value=saturate(
                    observed, floor, self.preparation.signal.saturation_multiple
                ),
            )
        )


class BotActivityRule(BaseRule):
    """PAD-BOT-001 -- sustained, regular, uniform-client authentication."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-BOT-001"))

    def _build(self, preparation: RulePreparation) -> PreparedRule:
        return _BotActivityPreparedRule(preparation)
