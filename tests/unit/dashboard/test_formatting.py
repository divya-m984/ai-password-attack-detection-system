"""Tests for the console's presentation vocabulary.

Most of this file is not about aesthetics.  Three of the rules being pinned here
are the same rules the API's own schemas enforce, restated at the point where a
number becomes a string a human reads -- which is the last place they can be
broken and the easiest place to break them:

* a **risk score** is an ordinal 0-100 magnitude and is never a percentage;
* a **decision score** is never relabelled a probability;
* nothing user-entered is ever rendered as markup.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from password_attack_detector.dashboard.formatting import (
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    STATE_COLORS,
    escape_text,
    format_component_state,
    format_detail,
    format_fusion_strategy,
    format_model_score,
    format_reason_code,
    format_risk_score,
    format_timestamp,
    format_verdict,
    severity_rank,
    truncate_identifier,
)

# ---------------------------------------------------------------------------
# A risk score is not a percentage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "0.0 / 100"), (72.38, "72.4 / 100"), (100.0, "100.0 / 100")],
)
def test_a_risk_score_states_its_scale(value: float, expected: str) -> None:
    """The denominator is shown because the number is an ordinal, not a fraction."""
    assert format_risk_score(value) == expected


def test_a_risk_score_is_never_rendered_as_a_percentage() -> None:
    """A percent sign would invite the comparison the whole system refuses."""
    for value in (0.0, 1.0, 50.0, 99.9, 100.0):
        assert "%" not in format_risk_score(value)


def test_an_absent_risk_score_is_a_dash_not_a_zero() -> None:
    """Zero is a real reading; absence is not."""
    assert format_risk_score(None) == "—"


# ---------------------------------------------------------------------------
# A decision score is not a probability
# ---------------------------------------------------------------------------


def test_a_calibrated_probability_is_labelled_as_one() -> None:
    """The label comes from the declared score kind, not from what is populated."""
    label, value = format_model_score(
        score_kind="calibrated_probability", decision_score=0.9, probability=0.731
    )
    assert label == "Calibrated probability"
    assert value == "0.731000"


def test_an_uncalibrated_score_is_never_called_a_probability() -> None:
    """A lineage with no calibrator has no probability, and none is invented."""
    label, value = format_model_score(
        score_kind="decision_score", decision_score=2.5, probability=None
    )
    assert label == "Decision score (uncalibrated)"
    assert "probability" not in label.lower().replace("uncalibrated", "")
    assert value == "2.500000"


def test_a_probability_present_without_a_calibrated_score_kind_is_not_promoted() -> (
    None
):
    """The score kind is the authority; a populated field is not."""
    label, _ = format_model_score(
        score_kind="decision_score", decision_score=1.0, probability=0.5
    )
    assert label == "Decision score (uncalibrated)"


def test_no_model_score_renders_as_an_absence() -> None:
    """An unavailable layer has no number, and no number is supplied."""
    label, value = format_model_score(
        score_kind=None, decision_score=None, probability=None
    )
    assert label == "Model score"
    assert value == "—"


def test_no_model_score_is_ever_rendered_as_a_percentage() -> None:
    """Neither quantity is a percentage, so neither gets a percent sign."""
    for kind, score, probability in (
        ("calibrated_probability", 0.1, 0.1),
        ("decision_score", 3.0, None),
        (None, None, None),
    ):
        _, value = format_model_score(
            score_kind=kind, decision_score=score, probability=probability
        )
        assert "%" not in value


# ---------------------------------------------------------------------------
# Verdicts, states, reasons
# ---------------------------------------------------------------------------


def test_an_absent_verdict_is_distinct_from_a_negative() -> None:
    """'Did not flag' and 'produced no verdict' are different facts."""
    assert format_verdict(True) == "FLAGGED"
    assert format_verdict(False) == "not flagged"
    assert format_verdict(None) == "—"


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("or_gate", "OR gate"),
        ("and_gate", "AND gate"),
        ("stacked", "Stacked (fitted meta-learner)"),
        (None, "—"),
    ],
)
def test_a_fusion_strategy_reads_in_the_projects_own_terms(
    strategy: str | None, expected: str
) -> None:
    """The vocabulary matches the documents, so a reader does not translate."""
    assert format_fusion_strategy(strategy) == expected


def test_an_unknown_strategy_renders_verbatim() -> None:
    """A value from a later release renders as itself rather than as a blank."""
    assert format_fusion_strategy("something_new") == "something_new"


def test_a_reason_code_keeps_the_code_beside_the_prose() -> None:
    """The code is what an operator greps a log for; it is never replaced."""
    rendered = format_reason_code("serving_bundle_lineage_mismatch")
    assert "serving_bundle_lineage_mismatch" in rendered
    assert "serving bundle lineage mismatch" in rendered


def test_an_absent_reason_is_a_dash() -> None:
    """A ready component names no reason, and none is manufactured."""
    assert format_reason_code(None) == "—"
    assert format_reason_code("") == "—"


def test_an_unknown_component_state_renders_verbatim() -> None:
    """A state from a later release is shown, not swallowed."""
    assert format_component_state("degraded") == "degraded"
    assert format_component_state("ready") == "Ready"


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_the_severity_order_is_the_phase_four_ordinal_scale() -> None:
    """Charts and sorts read left to right as increasing seriousness."""
    assert SEVERITY_ORDER == ("none", "low", "medium", "high", "critical")
    ranks = [severity_rank(name) for name in SEVERITY_ORDER]
    assert ranks == sorted(ranks)


def test_an_unknown_severity_sorts_first_rather_than_last() -> None:
    """An unrecognised value must not outrank ``critical``."""
    assert severity_rank("not-a-severity") < severity_rank("none")


def test_nothing_below_high_is_rendered_red() -> None:
    """A red cell means the same thing on every page, or it means nothing."""
    red = {"#f85149", "#f0883e"}
    assert SEVERITY_COLORS["high"] in red
    assert SEVERITY_COLORS["critical"] in red
    for name in ("none", "low", "medium"):
        assert SEVERITY_COLORS[name] not in red


def test_a_healthy_state_is_green_and_a_failed_one_is_not() -> None:
    """The colour carries meaning, so it has to be consistent."""
    assert STATE_COLORS["ready"] == "#3fb950"
    assert STATE_COLORS["unavailable"] != STATE_COLORS["ready"]
    assert STATE_COLORS["disabled"] != STATE_COLORS["unavailable"]


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "<script>alert(1)</script>",
        '"><img src=x onerror=alert(1)>',
        "<div style='position:fixed'>overlay</div>",
        "javascript:alert(1)",
        "' onmouseover='alert(1)",
    ],
)
def test_hostile_text_is_escaped_rather_than_rendered(hostile: str) -> None:
    """Nothing a user typed reaches a styled block as markup."""
    escaped = escape_text(hostile)
    assert "<script" not in escaped
    assert "<img" not in escaped
    assert "<div" not in escaped
    for character in ("<", ">"):
        assert character not in escaped


def test_quotes_are_escaped_because_values_land_in_attributes() -> None:
    """The accent colours are interpolated into ``style``; a quote would break out."""
    assert '"' not in escape_text('a "quoted" value')
    assert "'" not in escape_text("an 'apostrophed' value")


def test_a_non_string_is_escaped_too() -> None:
    """Callers pass counts and floats; the function must not assume a string."""
    assert escape_text(42) == "42"
    assert escape_text(None) == "None"


# ---------------------------------------------------------------------------
# Odds and ends
# ---------------------------------------------------------------------------


def test_a_short_identifier_is_not_truncated() -> None:
    """Shortening shortens a display, never a value."""
    assert truncate_identifier("abc") == "abc"


def test_a_long_identifier_is_marked_as_shortened() -> None:
    """An ellipsis so nobody copies a truncated digest thinking it is whole."""
    rendered = truncate_identifier("f" * 64, keep=12)
    assert rendered == "f" * 12 + "…"


def test_a_timestamp_renders_at_seconds_resolution() -> None:
    """Sub-second precision is noise on a console; the zone is not."""
    moment = datetime(2026, 3, 4, 12, 30, 45, 123456, tzinfo=UTC)
    assert format_timestamp(moment) == "2026-03-04 12:30:45 UTC"


def test_an_absent_timestamp_is_a_dash() -> None:
    """An empty cell would read as midnight."""
    assert format_timestamp(None) == "—"


def test_aggregate_detail_renders_as_one_sorted_clause() -> None:
    """Sorted, so the same failure reads the same way twice."""
    assert (
        format_detail({"max_batch_events": 5, "event_count": 9})
        == "event_count: 9, max_batch_events: 5"
    )
    assert format_detail(None) == ""
    assert format_detail({}) == ""
