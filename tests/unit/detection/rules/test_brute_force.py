"""Thresholds, discriminators, and boundary behaviour for PAD-BF-001/002."""

from __future__ import annotations

import math
from typing import Any

import pytest

from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import DetectionConfig, RuleSettings
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.detection.rules.brute_force import (
    SuccessAfterFailureBurstRule,
)
from password_attack_detector.exceptions import RuleEvaluationError
from password_attack_detector.features.catalog import (
    FeatureCatalog,
    FeatureSpec,
    LeakageClass,
)
from tests.unit.detection import factories
from tests.unit.detection.factories import brute_force_row, success_after_burst_row


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


@pytest.fixture()
def bf1(catalog: FeatureCatalog) -> PreparedRule:
    return RULE_IMPLEMENTATIONS["PAD-BF-001"].prepare(DetectionConfig(), catalog)


@pytest.fixture()
def bf2(catalog: FeatureCatalog) -> PreparedRule:
    return RULE_IMPLEMENTATIONS["PAD-BF-002"].prepare(DetectionConfig(), catalog)


# ---------------------------------------------------------------------------
# PAD-BF-001 -- thresholds
# ---------------------------------------------------------------------------


def test_fires_at_exact_threshold_equality(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    """Every condition is "at or above" / "at or below": equality matches."""
    row = brute_force_row(
        catalog,
        pair_failure_count__5m=8,
        user_failure_count__5m=8,
        pair_failure_rate__5m=0.80,
        prior_consecutive_user_failures=5,
        source_unique_user_count__5m=3,
    )
    result = bf1.evaluate(row)
    assert result.status is RuleStatus.FIRED
    # Every saturating component contributes zero at exact equality, so the
    # strength here comes only from the failure-rate component and the two
    # optional supports.  It is strictly positive -- a firing is never scored
    # as though nothing fired -- and strictly below a comfortably-clear row.
    assert result.signal_strength > 0.0
    clear = bf1.evaluate(brute_force_row(catalog))
    assert result.signal_strength < clear.signal_strength


def test_signal_strength_is_the_configured_floor_when_every_component_ties(
    bf2: PreparedRule, catalog: FeatureCatalog
) -> None:
    """PAD-BF-002 at exact equality on both saturating conditions.

    With no rate component and no optional support to lift it, the strength is
    exactly ``min_signal_strength`` -- the guarantee that makes "a fired rule
    always outscores nothing fired" true rather than approximately true.
    """
    row = success_after_burst_row(
        catalog,
        prior_failures_since_pair_success=6,
        prior_failures_since_user_success=0,
        seconds_since_user_previous_failure=300.0,
    )
    result = bf2.evaluate(row)
    assert result.status is RuleStatus.FIRED
    assert result.signal_strength == pytest.approx(
        DetectionConfig().signal.min_signal_strength, abs=1e-12
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("pair_failure_count__5m", 7),
        ("user_failure_count__5m", 7),
        ("prior_consecutive_user_failures", 4),
    ],
)
def test_one_step_below_an_integer_threshold_does_not_fire(
    bf1: PreparedRule, catalog: FeatureCatalog, column: str, value: int
) -> None:
    overrides: dict[str, Any] = {
        "pair_failure_count__5m": 8,
        "user_failure_count__5m": 8,
        "prior_consecutive_user_failures": 5,
        column: value,
    }
    assert bf1.evaluate(brute_force_row(catalog, **overrides)).status is (
        RuleStatus.NOT_FIRED
    )


def test_one_representable_value_below_the_rate_threshold_does_not_fire(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = brute_force_row(catalog, pair_failure_rate__5m=math.nextafter(0.80, 0.0))
    assert bf1.evaluate(row).status is RuleStatus.NOT_FIRED


def test_stronger_evidence_never_lowers_signal_strength(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    weaker = bf1.evaluate(
        brute_force_row(
            catalog,
            pair_failure_count__5m=8,
            user_failure_count__5m=8,
            prior_consecutive_user_failures=5,
        )
    )
    stronger = bf1.evaluate(
        brute_force_row(
            catalog,
            pair_failure_count__5m=40,
            user_failure_count__5m=40,
            prior_consecutive_user_failures=30,
        )
    )
    assert stronger.signal_strength >= weaker.signal_strength
    assert stronger.signal_strength > weaker.signal_strength


def test_configured_thresholds_move_the_decision(catalog: FeatureCatalog) -> None:
    """Thresholds are configuration, not literals baked into the rule."""
    row = brute_force_row(catalog, pair_failure_count__5m=5)
    strict = RULE_IMPLEMENTATIONS["PAD-BF-001"].prepare(DetectionConfig(), catalog)
    assert strict.evaluate(row).status is RuleStatus.NOT_FIRED

    relaxed_config = DetectionConfig(
        rules={"PAD-BF-001": RuleSettings(parameters={"min_pair_failures": 5})}
    )
    relaxed = RULE_IMPLEMENTATIONS["PAD-BF-001"].prepare(relaxed_config, catalog)
    assert relaxed.evaluate(row).status is RuleStatus.FIRED


# ---------------------------------------------------------------------------
# PAD-BF-001 -- history and false-positive controls
# ---------------------------------------------------------------------------


def test_a_null_pair_failure_rate_is_insufficient_data_not_a_negative(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    """An unobserved window is not the same as an observed quiet one."""
    result = bf1.evaluate(brute_force_row(catalog, pair_failure_rate__5m=None))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("INSUFFICIENT_HISTORY_PAIR_FAILURE_RATE__5M",)
    assert result.evidence == ()


def test_broad_account_fanout_does_not_fire_concentrated_brute_force(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    """The concentration ceiling is what keeps a spraying source out."""
    result = bf1.evaluate(brute_force_row(catalog, source_unique_user_count__5m=40))
    assert result.status is RuleStatus.NOT_FIRED
    assert "SOURCE_TARGETS_TOO_MANY_ACCOUNTS" in result.reason_codes


def test_the_concentration_ceiling_is_inclusive(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    assert (
        bf1.evaluate(brute_force_row(catalog, source_unique_user_count__5m=3)).status
        is RuleStatus.FIRED
    )
    assert (
        bf1.evaluate(brute_force_row(catalog, source_unique_user_count__5m=4)).status
        is RuleStatus.NOT_FIRED
    )


def test_isolated_ordinary_failures_do_not_fire(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = factories.quiet_row(
        catalog,
        pair_failure_count__5m=2,
        user_failure_count__5m=2,
        pair_failure_rate__5m=0.2,
        prior_consecutive_user_failures=1,
    )
    assert bf1.evaluate(row).status is RuleStatus.NOT_FIRED


# ---------------------------------------------------------------------------
# PAD-BF-001 -- blocked-account activity is supporting evidence only
# ---------------------------------------------------------------------------


def test_blocked_activity_is_supporting_and_never_required(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    without = bf1.evaluate(brute_force_row(catalog, user_blocked_count__5m=0))
    assert without.status is RuleStatus.FIRED
    assert "BF_BLOCKED_ACTIVITY" not in without.reason_codes

    with_blocked = bf1.evaluate(brute_force_row(catalog, user_blocked_count__5m=8))
    assert "BF_BLOCKED_ACTIVITY" in with_blocked.reason_codes


def test_blocked_activity_alone_does_not_fire(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = factories.quiet_row(catalog, user_blocked_count__5m=50)
    assert bf1.evaluate(row).status is RuleStatus.NOT_FIRED


def test_no_standalone_blocked_account_rule_is_registered() -> None:
    for spec in RULE_CATALOG.specs:
        assert "blocked" not in str(spec.attack_category)
        assert "blocked" not in str(spec.family)


def test_the_cadence_component_is_optional(
    bf1: PreparedRule, catalog: FeatureCatalog
) -> None:
    """A null or slow interarrival removes the contribution, not the finding."""
    for value in (None, 600.0):
        result = bf1.evaluate(
            brute_force_row(catalog, pair_mean_interarrival_seconds__5m=value)
        )
        assert result.status is RuleStatus.FIRED
        assert "BF_PAIR_INTERARRIVAL" not in result.reason_codes


# ---------------------------------------------------------------------------
# PAD-BF-002 -- the success discriminator
# ---------------------------------------------------------------------------


def test_a_failing_anchor_never_fires_however_large_the_burst(
    bf2: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = bf2.evaluate(
        success_after_burst_row(
            catalog,
            current_authentication_outcome="failure",
            prior_failures_since_pair_success=500,
            prior_failures_since_user_success=500,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert result.reason_codes == ("ANCHOR_DID_NOT_SUCCEED",)


def test_an_ordinary_successful_login_does_not_fire(
    bf2: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = factories.quiet_row(
        catalog,
        current_authentication_outcome="success",
        previous_user_outcome="success",
        seconds_since_user_previous_failure=90000.0,
    )
    assert bf2.evaluate(row).status is RuleStatus.NOT_FIRED


def test_fires_at_exact_burst_threshold_equality(
    bf2: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = success_after_burst_row(
        catalog,
        prior_failures_since_pair_success=6,
        prior_failures_since_user_success=0,
        seconds_since_user_previous_failure=300.0,
    )
    result = bf2.evaluate(row)
    assert result.status is RuleStatus.FIRED
    assert "BF2_PAIR_FAILURE_BURST" in result.reason_codes
    assert "BF2_USER_FAILURE_BURST" not in result.reason_codes


def test_one_below_both_burst_thresholds_does_not_fire(
    bf2: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = success_after_burst_row(
        catalog,
        prior_failures_since_pair_success=5,
        prior_failures_since_user_success=7,
    )
    result = bf2.evaluate(row)
    assert result.status is RuleStatus.NOT_FIRED
    assert "BELOW_FAILURE_BURST_THRESHOLD" in result.reason_codes


def test_either_burst_alone_satisfies_the_condition(
    bf2: PreparedRule, catalog: FeatureCatalog
) -> None:
    pair_only = bf2.evaluate(
        success_after_burst_row(
            catalog,
            prior_failures_since_pair_success=9,
            prior_failures_since_user_success=0,
        )
    )
    user_only = bf2.evaluate(
        success_after_burst_row(
            catalog,
            prior_failures_since_pair_success=0,
            prior_failures_since_user_success=9,
        )
    )
    assert pair_only.status is RuleStatus.FIRED
    assert user_only.status is RuleStatus.FIRED


def test_a_recency_boundary_is_inclusive(
    bf2: PreparedRule, catalog: FeatureCatalog
) -> None:
    at = success_after_burst_row(catalog, seconds_since_user_previous_failure=300.0)
    above = success_after_burst_row(
        catalog,
        seconds_since_user_previous_failure=math.nextafter(300.0, math.inf),
    )
    assert bf2.evaluate(at).status is RuleStatus.FIRED
    assert bf2.evaluate(above).status is RuleStatus.NOT_FIRED


@pytest.mark.parametrize("outcome", ["success", "challenged"])
def test_a_preceding_non_failure_does_not_fire(
    bf2: PreparedRule, catalog: FeatureCatalog, outcome: str
) -> None:
    """A challenge is a control engaging, not a rejected guess."""
    result = bf2.evaluate(
        success_after_burst_row(catalog, previous_user_outcome=outcome)
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "PRECEDING_EVENT_WAS_NOT_A_FAILURE" in result.reason_codes


@pytest.mark.parametrize("outcome", ["failure", "blocked"])
def test_a_preceding_failure_or_block_satisfies_the_condition(
    bf2: PreparedRule, catalog: FeatureCatalog, outcome: str
) -> None:
    row = success_after_burst_row(catalog, previous_user_outcome=outcome)
    assert bf2.evaluate(row).status is RuleStatus.FIRED


@pytest.mark.parametrize(
    "column", ["previous_user_outcome", "seconds_since_user_previous_failure"]
)
def test_absent_sequence_history_is_insufficient_data(
    bf2: PreparedRule, catalog: FeatureCatalog, column: str
) -> None:
    result = bf2.evaluate(success_after_burst_row(catalog, **{column: None}))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.evidence == ()


def test_stronger_bursts_never_lower_signal_strength(
    bf2: PreparedRule, catalog: FeatureCatalog
) -> None:
    weaker = bf2.evaluate(
        success_after_burst_row(
            catalog,
            prior_failures_since_pair_success=6,
            prior_failures_since_user_success=8,
            seconds_since_user_previous_failure=300.0,
        )
    )
    stronger = bf2.evaluate(
        success_after_burst_row(
            catalog,
            prior_failures_since_pair_success=60,
            prior_failures_since_user_success=80,
            seconds_since_user_previous_failure=1.0,
        )
    )
    assert stronger.signal_strength > weaker.signal_strength


# ---------------------------------------------------------------------------
# PAD-BF-002 -- the pinned input surface
# ---------------------------------------------------------------------------


def test_the_only_current_event_input_is_the_outcome() -> None:
    spec = RULE_CATALOG.get("PAD-BF-002")
    catalog = factories.feature_catalog()
    for template in spec.required_features:
        if template == "current_authentication_outcome":
            continue
        assert catalog.get(template).leakage_class is LeakageClass.PRIOR_ONLY
    assert spec.optional_features == ()


def test_preparation_refuses_a_widened_input_surface(
    catalog: FeatureCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later edit adding a second current-event input must fail the run.

    The rule's justification is that it reads exactly one current-event field.
    Reclassifying one of its prior-only inputs simulates that edit.
    """
    original: FeatureSpec = catalog.get("prior_failures_since_pair_success")
    widened = original.model_copy(
        update={"leakage_class": LeakageClass.CURRENT_EVENT_CONTEXT}
    )

    def fake_get(name: str) -> Any:
        return widened if name == original.name else catalog.get(name)

    class _Shim:
        get = staticmethod(fake_get)

    with pytest.raises(RuleEvaluationError, match="only prior-only history"):
        SuccessAfterFailureBurstRule().prepare(DetectionConfig(), _Shim())  # type: ignore[arg-type]
