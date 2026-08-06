"""Impossible-travel indicator rule: PAD-GEO-001.

The rule consumes derived distance, velocity, and status columns.  No
coordinate exists in a feature snapshot and none can reach an artifact: the
Phase 3 catalog exposes a great-circle distance, a capped implied velocity, and
two categorical status columns explaining why either is unavailable.  Evidence
rounds the distance to a configurable multiple, so even the derived figure
cannot be used to narrow a location.

"Impossible" describes the arithmetic, not the world.  Coarse location is an
approximation, and a VPN, a corporate egress point, or a carrier gateway
relocates an apparent origin by thousands of kilometres with nobody moving.
The rule is named, described, and worded as an indicator throughout.

The zero-elapsed case is handled explicitly rather than folded into the
velocity comparison.  Two located successes sharing a timestamp make implied
velocity undefined, not merely large, and treating that as ordinary travel
would silently discard the most anomalous shape the data can hold.
"""

from __future__ import annotations

from datetime import datetime

from password_attack_detector.data.enums import AuthOutcome
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

__all__ = ["ImpossibleTravelRule"]

#: The previous-success geo status that means a comparison could be made.
_GEO_STATUS_OK: str = "ok"

#: Implied-velocity statuses carrying a usable magnitude.  ``capped`` is
#: included: the value is a floor on the true velocity, which is sufficient for
#: an "at or above" comparison.
_VELOCITY_STATUS_USABLE: frozenset[str] = frozenset({"ok", "capped"})

#: Implied-velocity status meaning the two events share a timestamp.
_VELOCITY_STATUS_ZERO_ELAPSED: str = "zero_elapsed"


def _reason_for_status(prefix: str, status: str) -> str:
    """Return a machine-readable reason code naming a status value.

    The status vocabulary is a fixed schema constant, so surfacing it leaks
    nothing: it says which comparison could not be made, never where anyone is.
    """
    normalized = "".join(
        character if character.isalnum() else "_" for character in status.upper()
    )
    return f"{prefix}_{normalized}"


class _ImpossibleTravelPreparedRule(BasePreparedRule):
    """Evaluate PAD-GEO-001 against one snapshot."""

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        preparation = self.preparation
        if view.text("current_authentication_outcome") != str(AuthOutcome.SUCCESS):
            return self.not_fired(
                anchor_id, anchor_time, reason_codes=("ANCHOR_DID_NOT_SUCCEED",)
            )

        geo_status = view.text("user_previous_success_geo__status")
        if geo_status is None:  # pragma: no cover - the column is non-nullable
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("GEO_STATUS_UNAVAILABLE",)
            )
        if geo_status != _GEO_STATUS_OK:
            # No prior located success, or a missing coordinate on either side.
            # Each is unseen history, and the status names which.
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=(_reason_for_status("GEO_STATUS", geo_status),),
            )

        distance = view.number("distance_km_from_user_previous_success")
        if distance is None:
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("GEO_DISTANCE_UNAVAILABLE",)
            )

        min_distance = preparation.param_float("min_distance_km")
        if distance < min_distance:
            # Below this, coarse-location error dominates the measurement and
            # an implied velocity computed from it means nothing.
            return self.not_fired(
                anchor_id, anchor_time, reason_codes=("BELOW_MINIMUM_DISTANCE",)
            )

        # A null country flag fails the requirement rather than satisfying it:
        # "the country is unknown" is not "the country changed".
        if (
            preparation.param_bool("require_country_change")
            and view.flag("country_changed_since_previous_success") is not True
        ):
            return self.not_fired(
                anchor_id, anchor_time, reason_codes=("NO_COUNTRY_CHANGE",)
            )

        velocity_status = view.text("implied_velocity__status")
        if velocity_status is None:  # pragma: no cover - the column is non-nullable
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=("GEO_VELOCITY_STATUS_UNAVAILABLE",),
            )

        if velocity_status == _VELOCITY_STATUS_ZERO_ELAPSED:
            return self._evaluate_zero_elapsed(view, anchor_id, anchor_time, distance)
        if velocity_status not in _VELOCITY_STATUS_USABLE:
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=(_reason_for_status("GEO_VELOCITY", velocity_status),),
            )

        velocity = view.number("implied_velocity_kmh_from_previous_success")
        if velocity is None:
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("GEO_VELOCITY_UNAVAILABLE",)
            )

        min_velocity = preparation.param_float("min_velocity_kmh")
        if velocity < min_velocity:
            return self.not_fired(
                anchor_id, anchor_time, reason_codes=("BELOW_MINIMUM_VELOCITY",)
            )

        multiple = preparation.signal.saturation_multiple
        evidence = self._base_evidence(view, distance, min_distance)
        evidence.append(
            self.evidence_for(
                "GEO_IMPLIED_VELOCITY", velocity, threshold_value=min_velocity
            )
        )
        components = [
            SignalComponent(
                name="implied_velocity",
                weight=1.0,
                value=saturate(velocity, min_velocity, multiple),
            ),
            SignalComponent(
                name="distance",
                weight=0.5,
                value=saturate(distance, min_distance, multiple),
            ),
        ]
        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength(components),
            evidence=evidence,
            reason_codes=tuple(item.evidence_code for item in evidence),
        )

    def _evaluate_zero_elapsed(
        self,
        view: SnapshotView,
        anchor_id: str,
        anchor_time: datetime,
        distance: float,
    ) -> RuleEvaluationResult:
        """Handle two located successes sharing a timestamp.

        Velocity is undefined here rather than merely high, so the configured
        policy decides.  Firing is the default and carries maximum strength:
        an unbounded implied velocity is the strongest form of the shape this
        rule looks for, not a weaker one.
        """
        if self.preparation.param_str("zero_elapsed_policy") != "fire":
            return self.insufficient_data(
                anchor_id, anchor_time, reason_codes=("GEO_VELOCITY_ZERO_ELAPSED",)
            )

        min_distance = self.preparation.param_float("min_distance_km")
        evidence = self._base_evidence(view, distance, min_distance)
        evidence.append(
            self.evidence_for(
                "GEO_ZERO_ELAPSED_INTERVAL", _VELOCITY_STATUS_ZERO_ELAPSED
            )
        )
        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength(
                [SignalComponent(name="zero_elapsed_interval", weight=1.0, value=1.0)]
            ),
            evidence=evidence,
            reason_codes=tuple(item.evidence_code for item in evidence),
        )

    def _base_evidence(
        self, view: SnapshotView, distance: float, min_distance: float
    ) -> list[EvidenceItem]:
        """Build the evidence every firing path shares."""
        evidence: list[EvidenceItem] = [
            self.evidence_for(
                "GEO_CURRENT_SUCCESS", str(AuthOutcome.SUCCESS), threshold_value=None
            ),
            self.evidence_for(
                "GEO_DISTANCE",
                self._round_distance(distance),
                threshold_value=min_distance,
            ),
        ]
        # Reported whenever it is known to be true, whether or not the
        # configuration required it, because a country change is the part of
        # this finding an analyst can act on without any coordinate.
        if view.flag("country_changed_since_previous_success") is True:
            evidence.append(self.evidence_for("GEO_COUNTRY_CHANGE", True))
        return evidence

    def _round_distance(self, distance: float) -> float:
        """Round *distance* to the configured multiple.

        Coarsening the published figure is deliberate: an exact great-circle
        distance from an unstated origin still narrows a location, and nothing
        downstream needs more precision than "roughly this far".
        """
        step = self.preparation.param_float("distance_rounding_km")
        if step <= 0.0:  # pragma: no cover - the parameter declares minimum 1.0
            return distance
        return round(distance / step) * step


class ImpossibleTravelRule(BaseRule):
    """PAD-GEO-001 -- movement implying an implausible travel speed."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-GEO-001"))

    def _build(self, preparation: RulePreparation) -> PreparedRule:
        return _ImpossibleTravelPreparedRule(preparation)
