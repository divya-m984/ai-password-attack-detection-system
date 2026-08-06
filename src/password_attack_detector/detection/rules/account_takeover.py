"""Account-takeover indicator rule: PAD-ATO-001.

This rule reports an *indicator*.  A successful authentication from unfamiliar
contexts is what a takeover looks like from the outside, and it is also what a
new laptop on a business trip looks like.  Nothing here distinguishes the two,
and the naming, the evidence wording, and the catalog description all say so.

Novelty alone therefore never fires it.  Two independent things must hold: the
anchor must present at least the configured number of new contexts, *and* at
least one behavioural deviation must support them.  A first login from a new
country with the account's usual rhythm, hour, and multi-factor result stays
silent, because that is a description of travel.
"""

from __future__ import annotations

from datetime import datetime

from password_attack_detector.data.enums import AuthOutcome, MFAOutcome
from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.rules.base import (
    BasePreparedRule,
    BaseRule,
    PreparedRule,
    RulePreparation,
    SignalComponent,
    SnapshotView,
    clamp,
    saturate,
)
from password_attack_detector.detection.schemas import (
    EvidenceItem,
    RuleEvaluationResult,
)

__all__ = ["AccountTakeoverRule"]

#: The five novelty indicators, in a fixed order so the count and the evidence
#: are identical across runs.
_NOVELTY_FEATURES: tuple[str, ...] = (
    "is_new_device_for_user",
    "is_new_source_for_user",
    "is_new_country_for_user",
    "is_new_application_for_user",
    "is_new_auth_method_for_user",
)

#: Multi-factor outcomes that count as a supporting deviation.  ``not_required``
#: and ``not_enrolled`` are excluded: they describe a policy, not an anomaly.
_DEVIANT_MFA_OUTCOMES: frozenset[str] = frozenset(
    {str(MFAOutcome.FAILED), str(MFAOutcome.BYPASSED)}
)


class _AccountTakeoverPreparedRule(BasePreparedRule):
    """Evaluate PAD-ATO-001 against one snapshot."""

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        preparation = self.preparation
        if view.text("current_authentication_outcome") != str(AuthOutcome.SUCCESS):
            return self.not_fired(
                anchor_id, anchor_time, reason_codes=("ANCHOR_DID_NOT_SUCCEED",)
            )

        # Novelty is meaningless for an account the baseline has never seen: a
        # cold account's first device is not a *new* device.
        if not view.flag("user_in_baseline"):
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("ACCOUNT_ABSENT_FROM_BASELINE",)
            )

        novel_count = sum(
            1 for template in _NOVELTY_FEATURES if view.flag(template) is True
        )
        min_novel = preparation.param_int("min_novel_context_count")
        min_supporting = preparation.param_int("min_supporting_signals")

        supporting = self._supporting_signals(view)
        unmet: list[str] = []
        if novel_count < min_novel:
            unmet.append("BELOW_NOVEL_CONTEXT_THRESHOLD")
        if len(supporting) < min_supporting:
            unmet.append("NO_SUPPORTING_BEHAVIOURAL_DEVIATION")
        if unmet:
            return self.not_fired(anchor_id, anchor_time, reason_codes=tuple(unmet))

        multiple = preparation.signal.saturation_multiple
        evidence: list[EvidenceItem] = [
            self.evidence_for(
                "ATO_CURRENT_SUCCESS", str(AuthOutcome.SUCCESS), threshold_value=None
            ),
            self.evidence_for(
                "ATO_NOVEL_CONTEXT_COUNT", novel_count, threshold_value=min_novel
            ),
        ]
        components: list[SignalComponent] = [
            SignalComponent(
                name="novel_contexts",
                weight=1.0,
                value=saturate(novel_count, min_novel, multiple),
            ),
            # Breadth of corroboration, normalised against the six declared
            # supporting signals so the component stays bounded no matter how
            # many are configured as required.
            SignalComponent(
                name="supporting_breadth",
                weight=0.8,
                value=clamp(len(supporting) / len(_SUPPORTING_CODES)),
            ),
        ]
        for code, observed, threshold in supporting:
            evidence.append(
                self.evidence_for(code, observed, threshold_value=threshold)
            )

        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength(components),
            evidence=evidence,
            reason_codes=tuple(item.evidence_code for item in evidence),
        )

    def _supporting_signals(
        self, view: SnapshotView
    ) -> list[tuple[str, bool | int | float | str, bool | int | float | str | None]]:
        """Return every supporting deviation present, in a fixed order.

        Each entry is the evidence code, the observed value, and the threshold
        it cleared.  A null optional feature contributes nothing: the account
        may simply have no fitted baseline for that dimension, which is not
        evidence of anything.
        """
        preparation = self.preparation
        found: list[
            tuple[str, bool | int | float | str, bool | int | float | str | None]
        ] = []

        prior_failures = view.count("prior_failures_since_user_success")
        min_prior_failures = preparation.param_int("min_prior_failures")
        if prior_failures >= min_prior_failures:
            found.append(("ATO_PRIOR_FAILURES", prior_failures, min_prior_failures))

        hour_deviation = view.number("login_hour_deviation")
        min_hour_deviation = preparation.param_float("min_login_hour_deviation")
        if hour_deviation is not None and hour_deviation >= min_hour_deviation:
            found.append(
                ("ATO_LOGIN_HOUR_DEVIATION", hour_deviation, min_hour_deviation)
            )

        # A response time far *below* the account's baseline is as much a
        # deviation as one far above, so the magnitude is compared rather than
        # the signed score.
        zscore = view.number("response_time_zscore")
        min_zscore = preparation.param_float("min_abs_response_time_zscore")
        if zscore is not None and abs(zscore) >= min_zscore:
            found.append(("ATO_RESPONSE_TIME_DEVIATION", abs(zscore), min_zscore))

        rate_ratio = view.number("user_event_rate_ratio")
        min_rate_ratio = preparation.param_float("min_event_rate_ratio")
        if rate_ratio is not None and rate_ratio >= min_rate_ratio:
            found.append(("ATO_EVENT_RATE_DEVIATION", rate_ratio, min_rate_ratio))

        mfa_outcome = view.text("current_mfa_outcome")
        if mfa_outcome is not None and mfa_outcome in _DEVIANT_MFA_OUTCOMES:
            found.append(("ATO_MFA_DEVIATION", mfa_outcome, None))

        distance = view.number("distance_from_user_baseline_centroid_km")
        min_distance = preparation.param_float("min_baseline_distance_km")
        if distance is not None and distance >= min_distance:
            found.append(("ATO_BASELINE_DISTANCE", distance, min_distance))

        return found


#: The supporting evidence codes, used to normalise the breadth component.
_SUPPORTING_CODES: tuple[str, ...] = (
    "ATO_PRIOR_FAILURES",
    "ATO_LOGIN_HOUR_DEVIATION",
    "ATO_RESPONSE_TIME_DEVIATION",
    "ATO_EVENT_RATE_DEVIATION",
    "ATO_MFA_DEVIATION",
    "ATO_BASELINE_DISTANCE",
)


class AccountTakeoverRule(BaseRule):
    """PAD-ATO-001 -- a success from new contexts with corroborating deviation."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-ATO-001"))

    def _build(self, preparation: RulePreparation) -> PreparedRule:
        return _AccountTakeoverPreparedRule(preparation)
