"""Tests for the wire contracts.

Grouped by what each group protects:

* **What a request may carry** -- field names, formats, bounds, and the one
  thing that is refused outright under every spelling: credential material.
* **What a window must be** -- ordered, non-repeating, and honest about which
  events it is asking a verdict for.
* **What a response may say** -- three layers kept apart, an unavailable layer
  that names why, and no field anywhere that blends an ordinal magnitude with a
  probability.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from password_attack_detector.api.errors import (
    ErrorCode,
    rejection_code,
    rejection_marker,
)
from password_attack_detector.api.schemas import (
    API_SCHEMA_VERSION,
    MAX_WINDOW_EVENTS,
    AnchorSelection,
    AuthEventRequest,
    BatchDetectionResponse,
    ComponentReport,
    ComponentState,
    DetectionBatchRequest,
    DetectionResponse,
    DetectionWindowRequest,
    HybridLayerResult,
    MLLayerResult,
    ModelInfoResponse,
    ReadinessResponse,
    ReadinessState,
    RuleLayerResult,
    SystemStatusResponse,
    VersionResponse,
    WindowSummary,
)
from password_attack_detector.data.enums import AuthOutcome
from password_attack_detector.ml.enums import FusionStrategy, ScoreKind
from tests.api.factories import brute_force_window, event


def _codes(exc: ValidationError) -> ErrorCode | None:
    """Return the error code the failing validator claimed, if any."""
    return rejection_code(str(item["msg"]) for item in exc.errors())


# ---------------------------------------------------------------------------
# One event
# ---------------------------------------------------------------------------


def test_a_well_formed_event_is_accepted() -> None:
    """The ordinary case: canonical terminology, pseudonymous identifiers."""
    request = AuthEventRequest.model_validate(event("a"))
    assert request.authentication_outcome is AuthOutcome.FAILURE
    assert request.source_id is not None
    assert request.source_ip is None


def test_the_canonical_event_it_converts_to_carries_no_address() -> None:
    """Conversion hands the canonical schema a pseudonym, never an address."""
    request = AuthEventRequest.model_validate(
        {k: v for k, v in event("a").items() if k != "source_id"}
        | {"source_ip": "203.0.113.9"}
    )
    canonical = request.to_canonical_event(source_id="s:" + "a" * 32)
    assert canonical.source_id == "s:" + "a" * 32
    assert "203.0.113.9" not in canonical.model_dump_json()


def test_a_naive_timestamp_is_refused() -> None:
    """A timestamp with no offset cannot be placed on the timeline."""
    with pytest.raises(ValidationError):
        AuthEventRequest.model_validate(event("a", event_time="2026-03-04T12:00:00"))


def test_an_unparseable_timestamp_is_refused() -> None:
    """Malformed prose is not a time."""
    with pytest.raises(ValidationError):
        AuthEventRequest.model_validate(event("a", event_time="last tuesday"))


@pytest.mark.parametrize(
    "address", ["999.1.1.1", "not-an-address", "203.0.113.9/24", "", "1.2.3"]
)
def test_a_malformed_source_address_is_refused(address: str) -> None:
    """A syntactically invalid address never reaches the pseudonym service."""
    body = {k: v for k, v in event("a").items() if k != "source_id"}
    with pytest.raises(ValidationError):
        AuthEventRequest.model_validate(body | {"source_ip": address})


@pytest.mark.parametrize("address", ["203.0.113.9", "2001:db8::1"])
def test_a_valid_address_of_either_family_is_accepted(address: str) -> None:
    """IPv4 and IPv6 are both ordinary inputs."""
    body = {k: v for k, v in event("a").items() if k != "source_id"}
    assert AuthEventRequest.model_validate(body | {"source_ip": address}).source_ip


def test_exactly_one_source_identity_is_required() -> None:
    """Neither is ambiguous and both is contradictory."""
    body = {k: v for k, v in event("a").items() if k != "source_id"}
    with pytest.raises(ValidationError, match="exactly one"):
        AuthEventRequest.model_validate(body)
    with pytest.raises(ValidationError, match="exactly one"):
        AuthEventRequest.model_validate(
            event("a") | {"source_ip": "203.0.113.9"},
        )


def test_an_extra_field_is_refused() -> None:
    """A key nobody declared is an error, never a silently ignored one."""
    with pytest.raises(ValidationError):
        AuthEventRequest.model_validate(event("a") | {"confidence": 0.9})


@pytest.mark.parametrize(
    "field",
    ["password", "passwordHash", "password_hash", "secret", "token", "Authorization"],
)
def test_credential_material_is_refused_under_every_spelling(field: str) -> None:
    """The refusal is by normalised field *name*; the value is never read."""
    with pytest.raises(ValidationError) as caught:
        AuthEventRequest.model_validate(event("a") | {field: "anything"})
    assert _codes(caught.value) is ErrorCode.CREDENTIAL_FIELD_REJECTED


def test_a_credential_refusal_outranks_any_other_claimed_code() -> None:
    """If two markers ever met, the secret is the part that matters."""
    assert (
        rejection_code(
            [
                f"{rejection_marker(ErrorCode.EVENT_ORDERING_ERROR)} misordered",
                f"{rejection_marker(ErrorCode.CREDENTIAL_FIELD_REJECTED)} refused",
            ]
        )
        is ErrorCode.CREDENTIAL_FIELD_REJECTED
    )


def test_an_unknown_marker_is_ignored_rather_than_raised_on() -> None:
    """This runs inside an exception handler; it must not become the exception."""
    assert rejection_code(["[API777] from a future build"]) is None
    assert rejection_code(["no marker at all"]) is None


def test_the_lowest_code_wins_when_neither_is_a_credential() -> None:
    """A written-down tie-break, so the answer never depends on ordering."""
    assert (
        rejection_code(
            [
                f"{rejection_marker(ErrorCode.ANCHOR_SELECTION_ERROR)} anchors",
                f"{rejection_marker(ErrorCode.DUPLICATE_EVENT_IDENTITY)} duplicate",
            ]
        )
        is ErrorCode.DUPLICATE_EVENT_IDENTITY
    )


def test_the_credential_refusal_does_not_echo_the_value() -> None:
    """A message quoting the rejected secret would be the leak it prevents."""
    with pytest.raises(ValidationError) as caught:
        AuthEventRequest.model_validate(event("a") | {"password": "hunter2-SECRET"})
    assert "hunter2-SECRET" not in str(caught.value.errors()[0]["msg"])


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("user_id", "s:" + "0" * 32),
        ("source_id", "u:" + "0" * 32),
        ("device_id", "sess:" + "0" * 32),
        ("session_id", "d:" + "0" * 32),
    ],
)
def test_a_pseudonym_from_the_wrong_domain_is_refused(field: str, wrong: str) -> None:
    """Domain separation is part of what a pseudonym means."""
    with pytest.raises(ValidationError, match="domain pseudonym"):
        AuthEventRequest.model_validate(event("a") | {field: wrong})


def test_an_unshaped_identifier_is_refused() -> None:
    """A plaintext username is not a pseudonym."""
    with pytest.raises(ValidationError):
        AuthEventRequest.model_validate(event("a") | {"user_id": "alice@example.com"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 91.0, -91.0])
def test_a_non_finite_or_out_of_range_latitude_is_refused(value: float) -> None:
    """NaN compares false against every bound, so it is checked explicitly."""
    with pytest.raises(ValidationError):
        AuthEventRequest.model_validate(event("a") | {"coarse_latitude": value})


def test_an_over_long_application_identifier_is_refused() -> None:
    """Strings are bounded so one request cannot carry an unbounded payload."""
    with pytest.raises(ValidationError):
        AuthEventRequest.model_validate(event("a", application="x" * 200))


def test_an_impossible_outcome_pairing_is_refused_by_the_canonical_schema() -> None:
    """The API restates no cross-field rule; the canonical schema owns them."""
    request = AuthEventRequest.model_validate(
        event("a", outcome="success", failure_reason="invalid_credentials")
    )
    with pytest.raises(ValidationError):
        request.to_canonical_event(source_id=request.source_id or "")


def test_a_bad_country_code_is_refused() -> None:
    """ISO 3166-1 alpha-2, or nothing."""
    with pytest.raises(ValidationError):
        AuthEventRequest.model_validate(event("a", country="United States"))


# ---------------------------------------------------------------------------
# A window
# ---------------------------------------------------------------------------


def test_a_window_defaults_to_its_latest_event() -> None:
    """The streaming case: score what just happened, given what came before."""
    request = DetectionWindowRequest.model_validate({"events": brute_force_window(5)})
    assert request.anchor_selection is AnchorSelection.LAST
    assert request.resolved_anchor_ids() == (str(request.events[-1].event_id),)


def test_a_batch_window_defaults_to_every_event() -> None:
    """The batch case answers more anchors, not more input."""
    request = DetectionBatchRequest.model_validate({"events": brute_force_window(4)})
    assert request.anchor_selection is AnchorSelection.ALL
    assert len(request.resolved_anchor_ids()) == 4


def test_explicit_anchors_are_returned_exactly() -> None:
    """An explicit selection is honoured, not widened."""
    events = brute_force_window(4)
    chosen = [events[1]["event_id"], events[2]["event_id"]]
    request = DetectionBatchRequest.model_validate(
        {"events": events, "anchor_selection": "explicit", "anchor_event_ids": chosen}
    )
    assert set(request.resolved_anchor_ids()) == set(chosen)


def test_an_empty_window_is_refused() -> None:
    """There is nothing to score and no history to score it against."""
    with pytest.raises(ValidationError):
        DetectionWindowRequest.model_validate({"events": []})


def test_a_misordered_window_is_refused_with_its_own_code() -> None:
    """Ordering is a contract, and the client is told which contract it broke."""
    events = brute_force_window(3)
    with pytest.raises(ValidationError) as caught:
        DetectionWindowRequest.model_validate({"events": list(reversed(events))})
    assert _codes(caught.value) is ErrorCode.EVENT_ORDERING_ERROR


def test_events_sharing_a_timestamp_are_accepted() -> None:
    """Simultaneous events are ordinary; the contract is non-decreasing, not strict."""
    same = [event("x", offset_seconds=0.0), event("y", offset_seconds=0.0)]
    assert DetectionBatchRequest.model_validate({"events": same}).events


def test_a_repeated_event_identity_is_refused_with_its_own_code() -> None:
    """One identity, one event: a window is not a place to restate one."""
    body = event("a")
    with pytest.raises(ValidationError) as caught:
        DetectionWindowRequest.model_validate({"events": [body, body]})
    assert _codes(caught.value) is ErrorCode.DUPLICATE_EVENT_IDENTITY


def test_an_anchor_outside_the_window_is_refused() -> None:
    """A verdict for an event nobody supplied would be a verdict about nothing."""
    with pytest.raises(ValidationError) as caught:
        DetectionWindowRequest.model_validate(
            {
                "events": brute_force_window(2),
                "anchor_selection": "explicit",
                "anchor_event_ids": [str(uuid.uuid4())],
            }
        )
    assert _codes(caught.value) is ErrorCode.ANCHOR_SELECTION_ERROR


def test_anchor_ids_without_explicit_selection_are_refused() -> None:
    """Silently ignoring them would answer a different question than was asked."""
    events = brute_force_window(2)
    with pytest.raises(ValidationError) as caught:
        DetectionWindowRequest.model_validate(
            {"events": events, "anchor_event_ids": [events[0]["event_id"]]}
        )
    assert _codes(caught.value) is ErrorCode.ANCHOR_SELECTION_ERROR


def test_explicit_selection_without_anchors_is_refused() -> None:
    """'Exactly these' has to name at least one."""
    with pytest.raises(ValidationError) as caught:
        DetectionWindowRequest.model_validate(
            {"events": brute_force_window(2), "anchor_selection": "explicit"}
        )
    assert _codes(caught.value) is ErrorCode.ANCHOR_SELECTION_ERROR


def test_a_window_beyond_the_absolute_ceiling_is_refused() -> None:
    """The parser's own ceiling, below whatever a deployment configures."""
    oversized = [
        event(f"n-{index}", offset_seconds=float(index))
        for index in range(MAX_WINDOW_EVENTS + 1)
    ]
    with pytest.raises(ValidationError) as caught:
        DetectionBatchRequest.model_validate({"events": oversized})
    assert _codes(caught.value) is ErrorCode.BATCH_LIMIT_EXCEEDED


def test_the_envelope_forbids_an_extra_field() -> None:
    """A request cannot smuggle a knob past the schema by inventing a key."""
    for extra in ("model_id", "decision_threshold", "fusion_strategy", "artifact_root"):
        with pytest.raises(ValidationError):
            DetectionWindowRequest.model_validate(
                {"events": brute_force_window(2), extra: "anything"}
            )


def test_the_envelope_refuses_credential_material() -> None:
    """The refusal applies at the window level as well as per event."""
    with pytest.raises(ValidationError) as caught:
        DetectionWindowRequest.model_validate(
            {"events": brute_force_window(2), "token": "abc"}
        )
    assert _codes(caught.value) is ErrorCode.CREDENTIAL_FIELD_REJECTED


def test_the_wire_version_is_pinned() -> None:
    """A request declaring another contract version is a request for another API."""
    assert API_SCHEMA_VERSION == "1.0.0"
    with pytest.raises(ValidationError):
        DetectionWindowRequest.model_validate(
            {"events": brute_force_window(2), "api_schema_version": "2.0.0"}
        )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def _rule_layer(**overrides: Any) -> RuleLayerResult:
    """Return a minimal clean-negative rule layer."""
    payload: dict[str, Any] = {
        "flagged": False,
        "risk_score": 0.0,
        "severity": "low",
        "scoring_version": "1.0.0",
    }
    payload.update(overrides)
    return RuleLayerResult.model_validate(payload)


def test_an_available_model_layer_reports_a_decision() -> None:
    """Available means there is a verdict to report."""
    layer = MLLayerResult(
        available=True,
        flagged=True,
        score_kind=ScoreKind.CALIBRATED_PROBABILITY,
        decision_score=0.9,
        probability=0.8,
        decision_threshold=0.5,
    )
    assert layer.flagged is True


def test_an_unavailable_model_layer_must_name_why() -> None:
    """A silent absence is indistinguishable from a negative verdict."""
    with pytest.raises(ValidationError):
        MLLayerResult(available=False)
    with pytest.raises(ValidationError):
        MLLayerResult(available=True, unavailable_reason="whatever")
    with pytest.raises(ValidationError):
        MLLayerResult(available=False, unavailable_reason="gone", flagged=True)


def test_an_unavailable_hybrid_reports_no_verdict() -> None:
    """No fallback strategy, and nowhere to put a verdict from one."""
    layer = HybridLayerResult(available=False, unavailable_reason="no_fusion_selection")
    assert layer.strategy is None
    with pytest.raises(ValidationError):
        HybridLayerResult(
            available=False, unavailable_reason="none", strategy=FusionStrategy.OR_GATE
        )
    with pytest.raises(ValidationError):
        HybridLayerResult(available=True, flagged=True)


def test_no_response_model_carries_a_blended_score() -> None:
    """An ordinal magnitude and a probability do not add, so nothing sums them.

    The same guard the fusion layer applies to its own schemas, applied to the
    wire contracts: a future field named for a combined quantity fails here.
    """
    forbidden = {
        "combined_score",
        "blended_score",
        "fused_score",
        "weighted_score",
        "overall_score",
        "total_score",
        "rule_probability",
        "risk_probability",
        "confidence",
    }
    models: tuple[type[BaseModel], ...] = (
        RuleLayerResult,
        MLLayerResult,
        HybridLayerResult,
        DetectionResponse,
        BatchDetectionResponse,
        SystemStatusResponse,
        ModelInfoResponse,
    )
    for model in models:
        assert set(model.model_fields) & forbidden == set(), model.__name__


def test_the_rule_layer_score_stays_on_its_declared_scale() -> None:
    """0-100, because that is the scale Phase 4 defined."""
    with pytest.raises(ValidationError):
        _rule_layer(risk_score=101.0)
    with pytest.raises(ValidationError):
        _rule_layer(risk_score=-1.0)


def _status(**overrides: Any) -> SystemStatusResponse:
    """Return a system-status document, varying only the hybrid arm."""
    fields: dict[str, Any] = {
        "status": ReadinessState.READY,
        "package_version": "0.0.0",
        "rule_detection_enabled": True,
        "ml_detection_enabled": True,
        "hybrid_detection_enabled": False,
        "hybrid_required": False,
        "enabled_rule_count": 1,
        "registered_rule_count": 9,
        "max_batch_events": 500,
        "fusion_unavailable_reason": "no_fusion_selection",
    }
    return SystemStatusResponse(**(fields | overrides))


def test_a_status_document_reports_the_scientific_outcome_when_none_qualified() -> None:
    """No hybrid frozen: not required, not running, and the reason says so."""
    document = _status()
    assert document.hybrid_required is False
    assert document.frozen_fusion_strategy is None
    assert document.fusion_strategy is None
    assert document.stacked_state_fingerprint is None


def test_a_running_hybrid_must_be_the_frozen_one() -> None:
    """There is no fallback strategy, and no way to report one."""
    with pytest.raises(ValidationError, match="not the frozen one"):
        _status(
            hybrid_detection_enabled=True,
            hybrid_required=True,
            fusion_strategy=FusionStrategy.OR_GATE,
            frozen_fusion_strategy=FusionStrategy.STACKED,
            fusion_unavailable_reason=None,
        )


def test_a_disabled_hybrid_executes_no_strategy() -> None:
    """ "Unavailable" and "running" are not both true of one deployment."""
    with pytest.raises(ValidationError, match="executes no strategy"):
        _status(
            hybrid_detection_enabled=False,
            hybrid_required=True,
            fusion_strategy=FusionStrategy.OR_GATE,
            frozen_fusion_strategy=FusionStrategy.OR_GATE,
        )


def test_a_frozen_hybrid_makes_the_component_required() -> None:
    """A selected hybrid that is not required would be an optional extra."""
    with pytest.raises(ValidationError, match="required exactly when one was frozen"):
        _status(
            hybrid_required=False,
            frozen_fusion_strategy=FusionStrategy.STACKED,
            fusion_unavailable_reason="serving_bundle_not_published",
        )


def test_an_ambiguous_lineage_may_require_a_hybrid_it_cannot_name() -> None:
    """Two receipts naming different strategies: required, and unattributable."""
    document = _status(
        hybrid_required=True,
        frozen_fusion_strategy=None,
        fusion_unavailable_reason="ambiguous_fusion_selection",
        status=ReadinessState.NOT_READY,
    )
    assert document.hybrid_required is True
    assert document.fusion_strategy is None


def test_only_a_running_stacked_hybrid_names_a_fitted_state() -> None:
    """A gate has no meta-learner, so it has no digest to report."""
    with pytest.raises(ValidationError, match="fitted state to name"):
        _status(
            hybrid_detection_enabled=True,
            hybrid_required=True,
            fusion_strategy=FusionStrategy.OR_GATE,
            frozen_fusion_strategy=FusionStrategy.OR_GATE,
            fusion_unavailable_reason=None,
            stacked_state_fingerprint="a" * 64,
        )


def test_a_component_that_is_not_ready_must_name_a_reason() -> None:
    """A readiness document with an unexplained absence explains nothing."""
    with pytest.raises(ValidationError):
        ComponentReport(component="ml_champion", state=ComponentState.UNAVAILABLE)
    with pytest.raises(ValidationError):
        ComponentReport(
            component="rule_engine", state=ComponentState.READY, reason="fine"
        )


def test_readiness_is_not_partial_credit() -> None:
    """One unavailable required component makes the runtime not ready."""
    broken = ComponentReport(
        component="ml_champion",
        state=ComponentState.UNAVAILABLE,
        reason="champion_verification_failed",
    )
    with pytest.raises(ValidationError):
        ReadinessResponse(
            status=ReadinessState.READY, version="0.5.0", components=(broken,)
        )
    document = ReadinessResponse(
        status=ReadinessState.NOT_READY, version="0.5.0", components=(broken,)
    )
    assert document.status is ReadinessState.NOT_READY


def test_an_optional_component_does_not_block_readiness() -> None:
    """An unselected hybrid is a finding, not a fault."""
    optional = ComponentReport(
        component="fusion",
        state=ComponentState.DISABLED,
        reason="no_fusion_selection",
        required=False,
    )
    document = ReadinessResponse(
        status=ReadinessState.READY, version="0.5.0", components=(optional,)
    )
    assert document.status is ReadinessState.READY


def test_an_unavailable_model_document_reports_no_identity() -> None:
    """There is no model, so there is nothing to attribute a threshold to."""
    with pytest.raises(ValidationError):
        ModelInfoResponse(available=False, unavailable_reason="none", model_id="x")
    with pytest.raises(ValidationError):
        ModelInfoResponse(available=False)


def test_the_version_document_names_no_host() -> None:
    """Every field is a property of the build, not of the machine."""
    document = VersionResponse(
        package_version="0.5.0",
        feature_schema_version="1.0.0",
        detection_schema_version="1.0.0",
        scoring_version="1.0.0",
        ml_schema_version="1.0.0",
        fusion_schema_version="1.0.0",
    )
    for name in ("hostname", "host", "pid", "started_at", "path", "root"):
        assert name not in type(document).model_fields


def test_the_window_summary_is_counts_and_versions_only() -> None:
    """Nothing per-row, nothing private, nothing located."""
    summary = WindowSummary(
        event_count=3,
        anchor_count=1,
        feature_schema_version="1.0.0",
        detection_schema_version="1.0.0",
        enabled_rule_count=9,
        evaluated_snapshot_count=3,
    )
    assert summary.model_dump().keys() == {
        "event_count",
        "anchor_count",
        "feature_schema_version",
        "detection_schema_version",
        "enabled_rule_count",
        "evaluated_snapshot_count",
    }
