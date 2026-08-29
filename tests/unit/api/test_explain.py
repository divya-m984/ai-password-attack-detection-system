"""Tests for the attribution response contract and the service's refusals.

The decomposition itself needs a real frozen champion and is exercised in
``tests/integration/test_api_explain.py``, where the residual is checked against
a model that actually scored the row.  What is worth testing cheaply is
everything around it: the invariants the response schema refuses to publish
without, the refusals the service makes before any artifact is touched, and the
structural guarantee that gives this endpoint its licence to exist -- that it
attributes a live row without naming an experimental population.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from password_attack_detector.api.config import APISettings
from password_attack_detector.api.errors import APIError, ErrorCode
from password_attack_detector.api.schemas import (
    MAX_REPORTED_CONTRIBUTIONS,
    DetectionWindowRequest,
    ExplanationContribution,
    ExplanationResponse,
)
from password_attack_detector.api.services import (
    RuntimeState,
    build_runtime,
    explain_document,
)
from password_attack_detector.ml.enums import ExplanationMethod, ScoreKind
from tests.api.factories import brute_force_window

ANCHOR = "8f0b2f4a-2f0e-5a5c-9c8d-1c1e5b6a7d20"
WHEN = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)


@pytest.fixture()
def rule_only() -> RuntimeState:
    """A runtime serving the rule layer alone: no artifacts, no model."""
    return build_runtime(APISettings(require_ml_champion=False))


def _available(**overrides: object) -> dict[str, object]:
    """Return the fields of a minimal available explanation."""
    body: dict[str, object] = {
        "anchor_event_id": ANCHOR,
        "anchor_event_time": WHEN,
        "available": True,
        "method": ExplanationMethod.LINEAR_LOGIT_CONTRIBUTION,
        "model_family": "logistic_regression",
        "score_kind": ScoreKind.CALIBRATED_PROBABILITY,
        "decision_value": 1.5,
        "baseline_value": 0.5,
        "contributions": (
            ExplanationContribution(transformed_feature="a", contribution=0.6),
            ExplanationContribution(transformed_feature="b", contribution=0.4),
        ),
        "transformed_feature_count": 2,
        "omitted_contribution_count": 0,
        "reconstruction_residual": 0.0,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# The response contract
# ---------------------------------------------------------------------------


def test_an_available_explanation_reports_what_it_decomposed() -> None:
    """The ordinary shape, stated once so the negatives below are readable."""
    document = ExplanationResponse.model_validate(_available())
    assert document.available is True
    assert document.unavailable_reason is None
    assert document.transformed_feature_count == 2


def test_an_available_explanation_must_carry_its_residual() -> None:
    """The residual is the check; an explanation without one checked nothing."""
    with pytest.raises(ValidationError, match="residual"):
        ExplanationResponse.model_validate(_available(reconstruction_residual=None))


def test_an_available_explanation_must_name_its_method() -> None:
    """Which decomposition produced the numbers is not an optional detail."""
    with pytest.raises(ValidationError, match="method"):
        ExplanationResponse.model_validate(_available(method=None))


def test_an_available_explanation_names_no_unavailable_reason() -> None:
    """Reporting both would leave a reader to guess which one is true."""
    with pytest.raises(ValidationError, match="no unavailable reason"):
        ExplanationResponse.model_validate(_available(unavailable_reason="whatever"))


def test_an_unavailable_explanation_must_name_why() -> None:
    """An absence with no reason code is indistinguishable from a bug."""
    with pytest.raises(ValidationError, match="must name why"):
        ExplanationResponse.model_validate(
            {
                "anchor_event_id": ANCHOR,
                "anchor_event_time": WHEN,
                "available": False,
            }
        )


def test_an_unavailable_explanation_decomposes_nothing() -> None:
    """Contributions beside an unavailable status would be the worst of both."""
    with pytest.raises(ValidationError, match="decomposes nothing"):
        ExplanationResponse.model_validate(
            _available(
                available=False,
                unavailable_reason="explanation_method_unavailable",
            )
        )


def test_the_counts_must_account_for_every_column() -> None:
    """A truncated table that under-reports what it dropped reads as complete."""
    with pytest.raises(ValidationError, match="account for every"):
        ExplanationResponse.model_validate(_available(transformed_feature_count=40))


def test_a_repeated_column_is_refused() -> None:
    """A column credited twice double-counts under any reading of the table."""
    with pytest.raises(ValidationError, match="at most once"):
        ExplanationResponse.model_validate(
            _available(
                contributions=(
                    ExplanationContribution(transformed_feature="a", contribution=0.6),
                    ExplanationContribution(transformed_feature="a", contribution=0.4),
                )
            )
        )


def test_the_reported_contributions_are_bounded() -> None:
    """The ceiling is enforced by the schema, not only by the service."""
    count = MAX_REPORTED_CONTRIBUTIONS + 1
    with pytest.raises(ValidationError, match="at most"):
        ExplanationResponse.model_validate(
            _available(
                contributions=tuple(
                    ExplanationContribution(
                        transformed_feature=f"f{index}", contribution=1.0
                    )
                    for index in range(count)
                ),
                transformed_feature_count=count,
            )
        )


def test_a_contribution_has_no_field_for_a_feature_value() -> None:
    """Phase 5 gates value disclosure; the wire surface has no gate to turn on."""
    assert set(ExplanationContribution.model_fields) == {
        "transformed_feature",
        "contribution",
    }
    with pytest.raises(ValidationError):
        ExplanationContribution.model_validate(
            {"transformed_feature": "a", "contribution": 1.0, "transformed_value": 2.0}
        )


def test_the_response_declares_no_threshold_or_verdict_field() -> None:
    """The operating point belongs to the decision, not the decomposition."""
    declared = set(ExplanationResponse.model_fields)
    assert not declared & {
        "decision_threshold",
        "threshold",
        "flagged",
        "probability",
        "calibrated_probability",
        "split",
        "scope",
    }


# ---------------------------------------------------------------------------
# Service refusals
# ---------------------------------------------------------------------------


def test_a_multi_anchor_selection_is_refused(rule_only: RuntimeState) -> None:
    """One explanation answers one row; two would silently drop one."""
    events = brute_force_window(3)
    request = DetectionWindowRequest.model_validate(
        {
            "events": events,
            "anchor_selection": "explicit",
            "anchor_event_ids": [events[0]["event_id"], events[1]["event_id"]],
        }
    )
    with pytest.raises(APIError) as caught:
        explain_document(rule_only, request)
    assert caught.value.code is ErrorCode.ANCHOR_SELECTION_ERROR


def test_a_deployment_with_no_champion_cannot_explain_one(
    rule_only: RuntimeState,
) -> None:
    """There is no model to decompose, so the refusal names the model layer.

    Reported as a refusal rather than an unavailable explanation because the
    caller asked about a champion this deployment does not have -- which is a
    different thing from a champion whose family has no exact method.
    """
    request = DetectionWindowRequest.model_validate({"events": brute_force_window(3)})
    with pytest.raises(APIError) as caught:
        explain_document(rule_only, request)
    assert caught.value.code is ErrorCode.ML_CHAMPION_UNAVAILABLE


# ---------------------------------------------------------------------------
# The structural guarantee
# ---------------------------------------------------------------------------


def test_the_attribution_path_takes_no_scope_argument() -> None:
    """Explaining a live row requires no claim about which split it came from.

    This is the whole reason the endpoint could be built without touching Phase
    5's semantics. ``local_contributions`` decomposes a verified model over a
    transformed matrix and asks nothing else; ``explain_predictions``, which does
    take a ``scope: MLSplit`` and refuses TEST, is a population-level entry point
    and is not reachable from here.
    """
    from password_attack_detector.ml.explain import local_contributions

    parameters = set(inspect.signature(local_contributions).parameters)
    assert parameters == {"model", "matrix"}
    assert not parameters & {"scope", "split", "test", "labels"}


def test_the_serving_module_never_imports_the_population_level_explainer() -> None:
    """The import-time guard, asserted so it cannot be quietly deleted."""
    from password_attack_detector.api import services

    namespace = vars(services)
    assert "explain_predictions" not in namespace
    assert "MLSplit" not in namespace
    assert "local_contributions" in namespace


def test_explaining_takes_no_argument_that_could_name_a_population() -> None:
    """The service signature, stated rather than promised."""
    parameters = set(inspect.signature(explain_document).parameters)
    assert parameters == {"runtime", "request"}
