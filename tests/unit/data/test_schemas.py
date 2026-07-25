"""Unit tests for password_attack_detector.data.schemas."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from password_attack_detector.data.enums import (
    AuthMethod,
    AuthOutcome,
    CampaignStage,
    ClientType,
    FailureReason,
    MFAOutcome,
    ScenarioType,
)
from password_attack_detector.data.schemas import (
    PROHIBITED_GT_COLUMNS,
    SCHEMA_VERSION,
    AuthEvent,
    GroundTruthLabel,
)

# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

_EVENT_ID = UUID("12345678-1234-5678-1234-567812345678")
_NOW_UTC = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

# Valid prefixed pseudonyms (prefix + 32 lowercase hex chars)
_USER_ID = "u:" + "a" * 32
_SOURCE_ID = "s:" + "b" * 32
_DEVICE_ID = "d:" + "c" * 32
_SESSION_ID = "sess:" + "d" * 32


def _make_event(**kwargs: Any) -> AuthEvent:
    """Return a valid AuthEvent, overriding defaults with *kwargs*."""
    defaults: dict[str, Any] = {
        "event_id": _EVENT_ID,
        "event_time": _NOW_UTC,
        "user_id": _USER_ID,
        "source_id": _SOURCE_ID,
        "device_id": _DEVICE_ID,
        "session_id": _SESSION_ID,
        "application_id": "app-001",
        "authentication_method": AuthMethod.PASSWORD,
        "authentication_outcome": AuthOutcome.SUCCESS,
        "failure_reason": None,
    }
    defaults.update(kwargs)
    return AuthEvent(**defaults)


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_schema_version_constant(self) -> None:
        assert SCHEMA_VERSION == "1.0.0"

    def test_default_schema_version_on_event(self) -> None:
        event = _make_event()
        assert event.schema_version == "1.0.0"

    def test_schema_version_explicit_valid(self) -> None:
        event = _make_event(schema_version="1.0.0")
        assert event.schema_version == "1.0.0"

    def test_schema_version_wrong_literal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(schema_version="2.0.0")

    def test_schema_version_not_in_prohibited_columns(self) -> None:
        # schema_version is part of AuthEvent, not ground truth
        assert "schema_version" not in PROHIBITED_GT_COLUMNS


# ---------------------------------------------------------------------------
# Ground-truth leakage guard
# ---------------------------------------------------------------------------


class TestLeakageGuard:
    def test_prohibited_gt_columns_disjoint_from_auth_event_fields(self) -> None:
        """No ground-truth column may appear in AuthEvent.model_fields."""
        event_fields = set(AuthEvent.model_fields)
        leaked = PROHIBITED_GT_COLUMNS & event_fields
        assert not leaked, f"Ground-truth columns leaked into AuthEvent: {leaked}"

    def test_prohibited_columns_non_empty(self) -> None:
        assert len(PROHIBITED_GT_COLUMNS) >= 8


# ---------------------------------------------------------------------------
# Valid event construction
# ---------------------------------------------------------------------------


class TestValidAuthEvent:
    def test_minimal_success_event(self) -> None:
        event = _make_event()
        assert event.authentication_outcome == AuthOutcome.SUCCESS
        assert event.failure_reason is None

    def test_failure_event_with_reason(self) -> None:
        event = _make_event(
            authentication_outcome=AuthOutcome.FAILURE,
            failure_reason=FailureReason.INVALID_CREDENTIALS,
        )
        assert event.failure_reason == FailureReason.INVALID_CREDENTIALS

    def test_failure_event_with_unknown_reason(self) -> None:
        event = _make_event(
            authentication_outcome=AuthOutcome.FAILURE,
            failure_reason=FailureReason.UNKNOWN,
        )
        assert event.failure_reason == FailureReason.UNKNOWN

    def test_blocked_event_with_valid_reason(self) -> None:
        event = _make_event(
            authentication_outcome=AuthOutcome.BLOCKED,
            failure_reason=FailureReason.IP_BLOCKED,
        )
        assert event.authentication_outcome == AuthOutcome.BLOCKED

    def test_challenged_event_no_reason(self) -> None:
        event = _make_event(
            authentication_outcome=AuthOutcome.CHALLENGED,
            failure_reason=None,
        )
        assert event.authentication_outcome == AuthOutcome.CHALLENGED

    def test_all_optional_fields_none(self) -> None:
        event = _make_event()
        assert event.mfa_outcome is None
        assert event.country_code is None
        assert event.coarse_latitude is None
        assert event.coarse_longitude is None
        assert event.response_time_ms is None
        assert event.client_type is None

    def test_all_optional_fields_set(self) -> None:
        event = _make_event(
            mfa_outcome=MFAOutcome.PASSED,
            country_code="US",
            region_code="CA",
            coarse_latitude=37.8,
            coarse_longitude=-122.4,
            user_agent_family="Chrome",
            operating_system_family="Linux",
            client_type=ClientType.WEB_BROWSER,
            response_time_ms=250,
        )
        assert event.country_code == "US"
        assert event.coarse_latitude == pytest.approx(37.8)

    def test_event_is_frozen(self) -> None:
        event = _make_event()
        with pytest.raises((TypeError, ValidationError)):
            event.application_id = "changed"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(attack_label="brute_force")


# ---------------------------------------------------------------------------
# event_time validation
# ---------------------------------------------------------------------------


class TestEventTime:
    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(event_time=datetime(2024, 1, 15, 10, 30, 0))

    def test_utc_datetime_accepted(self) -> None:
        event = _make_event(event_time=_NOW_UTC)
        assert event.event_time.tzinfo is not None

    def test_non_utc_aware_normalised_to_utc(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        dt_eastern = datetime(2024, 1, 15, 5, 30, 0, tzinfo=eastern)
        event = _make_event(event_time=dt_eastern)
        # Normalised: 5:30 AM EST = 10:30 AM UTC
        assert event.event_time == _NOW_UTC
        assert str(event.event_time.tzinfo) == "UTC"

    def test_utc_plus_offset_normalised(self) -> None:
        plus2 = timezone(timedelta(hours=2))
        dt_plus2 = datetime(2024, 1, 15, 12, 30, 0, tzinfo=plus2)
        event = _make_event(event_time=dt_plus2)
        assert event.event_time == _NOW_UTC


# ---------------------------------------------------------------------------
# Identifier format validation
# ---------------------------------------------------------------------------


class TestIdentifierFormat:
    @pytest.mark.parametrize(
        "field,valid_value",
        [
            ("user_id", "u:" + "a" * 32),
            ("source_id", "s:" + "b" * 32),
            ("device_id", "d:" + "c" * 32),
            ("session_id", "sess:" + "e" * 32),
        ],
    )
    def test_valid_pseudonym_format(self, field: str, valid_value: str) -> None:
        event = _make_event(**{field: valid_value})
        assert getattr(event, field) == valid_value

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("user_id", "plaintext_username"),
            ("source_id", "192.168.1.1"),
            ("device_id", "device-fingerprint"),
            ("session_id", "session-token-abc"),
            ("user_id", "u:ABCDEF"),  # uppercase hex rejected
            ("user_id", "u:" + "a" * 31),  # too short
            ("user_id", "u:" + "a" * 33),  # too long
            ("user_id", "x:" + "a" * 32),  # unknown prefix
            ("user_id", "u:" + "g" * 32),  # non-hex char
        ],
    )
    def test_invalid_pseudonym_format_rejected(
        self, field: str, bad_value: str
    ) -> None:
        with pytest.raises(ValidationError):
            _make_event(**{field: bad_value})


# ---------------------------------------------------------------------------
# Outcome / failure_reason consistency
# ---------------------------------------------------------------------------


class TestOutcomeFailureReasonConsistency:
    def test_success_with_failure_reason_rejected(self) -> None:
        with pytest.raises(ValidationError, match="failure_reason must be None"):
            _make_event(
                authentication_outcome=AuthOutcome.SUCCESS,
                failure_reason=FailureReason.INVALID_CREDENTIALS,
            )

    def test_success_with_unknown_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(
                authentication_outcome=AuthOutcome.SUCCESS,
                failure_reason=FailureReason.UNKNOWN,
            )

    def test_failure_without_reason_rejected(self) -> None:
        with pytest.raises(ValidationError, match="failure_reason must be set"):
            _make_event(
                authentication_outcome=AuthOutcome.FAILURE,
                failure_reason=None,
            )

    def test_failure_with_unknown_accepted(self) -> None:
        event = _make_event(
            authentication_outcome=AuthOutcome.FAILURE,
            failure_reason=FailureReason.UNKNOWN,
        )
        assert event.failure_reason == FailureReason.UNKNOWN

    def test_blocked_without_reason_rejected(self) -> None:
        with pytest.raises(ValidationError, match="failure_reason must be set"):
            _make_event(
                authentication_outcome=AuthOutcome.BLOCKED,
                failure_reason=None,
            )

    def test_blocked_with_invalid_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(
                authentication_outcome=AuthOutcome.BLOCKED,
                failure_reason=FailureReason.INVALID_CREDENTIALS,
            )

    def test_blocked_with_ip_blocked_accepted(self) -> None:
        event = _make_event(
            authentication_outcome=AuthOutcome.BLOCKED,
            failure_reason=FailureReason.IP_BLOCKED,
        )
        assert event.authentication_outcome == AuthOutcome.BLOCKED

    def test_blocked_with_unknown_accepted(self) -> None:
        event = _make_event(
            authentication_outcome=AuthOutcome.BLOCKED,
            failure_reason=FailureReason.UNKNOWN,
        )
        assert event.failure_reason == FailureReason.UNKNOWN

    def test_challenged_with_reason_rejected(self) -> None:
        with pytest.raises(ValidationError, match="failure_reason must be None"):
            _make_event(
                authentication_outcome=AuthOutcome.CHALLENGED,
                failure_reason=FailureReason.MFA_FAILED,
            )

    def test_challenged_without_reason_accepted(self) -> None:
        event = _make_event(
            authentication_outcome=AuthOutcome.CHALLENGED,
            failure_reason=None,
        )
        assert event.authentication_outcome == AuthOutcome.CHALLENGED


# ---------------------------------------------------------------------------
# MFA consistency
# ---------------------------------------------------------------------------


class TestMFAConsistency:
    def test_mfa_bypassed_with_success_accepted(self) -> None:
        event = _make_event(mfa_outcome=MFAOutcome.BYPASSED)
        assert event.mfa_outcome == MFAOutcome.BYPASSED

    def test_mfa_bypassed_with_failure_accepted(self) -> None:
        event = _make_event(
            authentication_outcome=AuthOutcome.FAILURE,
            failure_reason=FailureReason.INVALID_CREDENTIALS,
            mfa_outcome=MFAOutcome.BYPASSED,
        )
        assert event.mfa_outcome == MFAOutcome.BYPASSED

    def test_mfa_bypassed_with_blocked_rejected(self) -> None:
        with pytest.raises(ValidationError, match="BYPASSED"):
            _make_event(
                authentication_outcome=AuthOutcome.BLOCKED,
                failure_reason=FailureReason.IP_BLOCKED,
                mfa_outcome=MFAOutcome.BYPASSED,
            )

    def test_mfa_bypassed_with_challenged_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(
                authentication_outcome=AuthOutcome.CHALLENGED,
                failure_reason=None,
                mfa_outcome=MFAOutcome.BYPASSED,
            )

    def test_mfa_not_required_with_any_outcome(self) -> None:
        for outcome, reason in [
            (AuthOutcome.SUCCESS, None),
            (AuthOutcome.FAILURE, FailureReason.UNKNOWN),
            (AuthOutcome.CHALLENGED, None),
        ]:
            event = _make_event(
                authentication_outcome=outcome,
                failure_reason=reason,
                mfa_outcome=MFAOutcome.NOT_REQUIRED,
            )
            assert event.mfa_outcome == MFAOutcome.NOT_REQUIRED


# ---------------------------------------------------------------------------
# Coordinate and response-time validation
# ---------------------------------------------------------------------------


class TestCoordinateValidation:
    @pytest.mark.parametrize("lat", [-90.0, 0.0, 90.0, 37.8])
    def test_valid_latitude(self, lat: float) -> None:
        event = _make_event(coarse_latitude=lat)
        assert event.coarse_latitude == pytest.approx(lat)

    @pytest.mark.parametrize("lat", [-90.001, 90.001, 180.0, -180.0])
    def test_invalid_latitude_rejected(self, lat: float) -> None:
        with pytest.raises(ValidationError):
            _make_event(coarse_latitude=lat)

    @pytest.mark.parametrize("lon", [-180.0, 0.0, 180.0, -122.4])
    def test_valid_longitude(self, lon: float) -> None:
        event = _make_event(coarse_longitude=lon)
        assert event.coarse_longitude == pytest.approx(lon)

    @pytest.mark.parametrize("lon", [-180.001, 180.001, 270.0])
    def test_invalid_longitude_rejected(self, lon: float) -> None:
        with pytest.raises(ValidationError):
            _make_event(coarse_longitude=lon)

    def test_both_coordinates_none(self) -> None:
        event = _make_event()
        assert event.coarse_latitude is None
        assert event.coarse_longitude is None


class TestResponseTimeValidation:
    @pytest.mark.parametrize("ms", [0, 1, 100, 30_000])
    def test_valid_response_time(self, ms: int) -> None:
        event = _make_event(response_time_ms=ms)
        assert event.response_time_ms == ms

    @pytest.mark.parametrize("ms", [-1, 30_001, 100_000])
    def test_invalid_response_time_rejected(self, ms: int) -> None:
        with pytest.raises(ValidationError):
            _make_event(response_time_ms=ms)

    def test_none_response_time_accepted(self) -> None:
        event = _make_event(response_time_ms=None)
        assert event.response_time_ms is None


# ---------------------------------------------------------------------------
# Country code validation
# ---------------------------------------------------------------------------


class TestCountryCodeValidation:
    @pytest.mark.parametrize("code", ["US", "GB", "DE", "JP", "ZZ"])
    def test_valid_country_codes(self, code: str) -> None:
        event = _make_event(country_code=code)
        assert event.country_code == code

    @pytest.mark.parametrize("code", ["us", "usa", "U", "U1", "123", ""])
    def test_invalid_country_codes_rejected(self, code: str) -> None:
        with pytest.raises(ValidationError):
            _make_event(country_code=code)

    def test_none_country_code_accepted(self) -> None:
        event = _make_event(country_code=None)
        assert event.country_code is None


# ---------------------------------------------------------------------------
# GroundTruthLabel
# ---------------------------------------------------------------------------


class TestGroundTruthLabel:
    def test_valid_label(self) -> None:
        label = GroundTruthLabel(
            event_id=_EVENT_ID,
            campaign_id="campaign:brute_force:42:0",
            scenario=ScenarioType.BRUTE_FORCE,
            malicious=True,
            supervised_training_eligible=True,
            generator_version="0.2.0",
        )
        assert label.malicious is True
        assert label.supervised_training_eligible is True

    def test_novel_anomaly_not_training_eligible(self) -> None:
        label = GroundTruthLabel(
            event_id=_EVENT_ID,
            campaign_id="campaign:novel:42:0",
            scenario=ScenarioType.NOVEL_ANOMALY_HOLDOUT,
            malicious=False,
            supervised_training_eligible=False,
            generator_version="0.2.0",
        )
        assert label.supervised_training_eligible is False

    def test_optional_fields_none(self) -> None:
        label = GroundTruthLabel(
            event_id=_EVENT_ID,
            campaign_id="c:normal:42:0",
            scenario=ScenarioType.NORMAL,
            malicious=False,
            supervised_training_eligible=True,
            generator_version="0.2.0",
        )
        assert label.scenario_variant is None
        assert label.campaign_stage is None

    def test_typed_metadata_fields(self) -> None:
        label = GroundTruthLabel(
            event_id=_EVENT_ID,
            campaign_id="c:ato:42:0",
            scenario=ScenarioType.ACCOUNT_TAKEOVER_INDICATOR,
            malicious=True,
            supervised_training_eligible=True,
            generator_version="0.2.0",
            scenario_variant="with_precursor_failures",
            campaign_stage=CampaignStage.ACTIVE,
        )
        assert label.scenario_variant == "with_precursor_failures"
        assert label.campaign_stage == CampaignStage.ACTIVE

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruthLabel(
                event_id=_EVENT_ID,
                campaign_id="c:0",
                scenario=ScenarioType.NORMAL,
                malicious=False,
                supervised_training_eligible=True,
                generator_version="0.2.0",
                notes="free text",  # type: ignore[call-arg]
            )

    def test_label_is_frozen(self) -> None:
        label = GroundTruthLabel(
            event_id=_EVENT_ID,
            campaign_id="c:0",
            scenario=ScenarioType.NORMAL,
            malicious=False,
            supervised_training_eligible=True,
            generator_version="0.2.0",
        )
        with pytest.raises((TypeError, ValidationError)):
            label.malicious = True

    def test_ground_truth_does_not_share_fields_with_auth_event(self) -> None:
        """No AuthEvent field may appear in PROHIBITED_GT_COLUMNS."""
        event_fields = set(AuthEvent.model_fields)
        assert PROHIBITED_GT_COLUMNS.isdisjoint(event_fields)
