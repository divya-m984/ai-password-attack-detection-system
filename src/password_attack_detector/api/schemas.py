"""The API's own request and response contracts.

These are **not** the canonical schemas re-exported.  They are a deliberately
narrower wire surface that converts *into* the canonical ones, and the narrowing
is the point:

* Every model forbids extra fields, so a key nobody declared is a validation
  error rather than a silently ignored one.
* Every request model rejects credential-shaped field names **before** any other
  validation, using the same prohibited-key scanner the Phase 2 ingestion
  adapters use.  There is no field on any request through which a password, a
  hash, a token, or a secret could arrive, and a request that offers one under
  any spelling is refused rather than partially processed.
* Pseudonymous identifiers are checked against their own domain prefix, which is
  stricter than the canonical schema's generic pattern.  The canonical schema
  has to accept whatever a historical dataset carries; a live request does not.
* A raw ``source_ip`` may be supplied *instead of* a pseudonymous ``source_id``.
  It is validated syntactically, pseudonymized through the project's keyed HMAC
  service, and never echoed back in any response.

The response models keep the three detection layers in separate typed objects.
There is no field anywhere below that blends the rule layer's ordinal 0-100
severity magnitude with the model's probability: they are quantities on
different scales, and a single number combining them would have units nobody
defined.
"""

from __future__ import annotations

import ipaddress
import math
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from password_attack_detector.api.config import MAX_MAX_BATCH_EVENTS
from password_attack_detector.api.errors import ErrorCode, rejection_marker
from password_attack_detector.data.enums import (
    AuthMethod,
    AuthOutcome,
    ClientType,
    FailureReason,
    MFAOutcome,
)
from password_attack_detector.data.privacy import scan_prohibited_keys
from password_attack_detector.data.schemas import SCHEMA_VERSION, AuthEvent
from password_attack_detector.detection.enums import (
    AttackCategory,
    RuleFamily,
    Severity,
)
from password_attack_detector.detection.schemas import EvidenceItem
from password_attack_detector.ml.enums import (
    ExplanationMethod,
    FusionStrategy,
    ScoreKind,
)

__all__ = [
    "API_SCHEMA_VERSION",
    "MAX_REPORTED_CONTRIBUTIONS",
    "MAX_WINDOW_EVENTS",
    "AnchorDetection",
    "AnchorSelection",
    "AuthEventRequest",
    "BatchDetectionResponse",
    "ComponentReport",
    "ComponentState",
    "DetectionBatchRequest",
    "DetectionResponse",
    "DetectionWindowRequest",
    "ExplanationContribution",
    "ExplanationResponse",
    "HealthResponse",
    "HybridLayerResult",
    "MLLayerResult",
    "ModelInfoResponse",
    "ReadinessResponse",
    "ReadinessState",
    "RuleCatalogResponse",
    "RuleLayerResult",
    "RuleSummary",
    "SystemStatusResponse",
    "VersionResponse",
    "WindowRequestBase",
    "WindowSummary",
]

#: The wire contract's own version.  Independent of the package version, the
#: canonical event schema version, the detection schema version, and the ML
#: schema version: what a request or response carries can change without any of
#: those changing, and the reverse.
#:
#: Declared ``Final`` without an explicit annotation so its type narrows to
#: ``Literal["1.0.0"]``, letting the request envelopes pin their
#: ``api_schema_version`` to this constant instead of repeating the literal.
API_SCHEMA_VERSION: Final = "1.0.0"

#: The service name every document reports itself under.
SERVICE_NAME: Final[str] = "password-attack-detector"

#: The absolute ceiling on how many events one window may carry, whatever a
#: deployment configures.  The per-deployment limit is enforced by the service
#: and is never larger than this; this one exists so a body that slipped past the
#: byte ceiling still cannot make the parser build an unbounded list.
MAX_WINDOW_EVENTS: Final[int] = MAX_MAX_BATCH_EVENTS

#: The hard ceiling on how many per-column contributions one explanation
#: reports, whatever the reviewed configuration asks for.
#:
#: An explanation is a *summary of what moved this decision*, and a response
#: listing every transformed column would be a row-by-row export of the fitted
#: function's shape.  The bound is applied after ranking by magnitude, so what
#: is dropped is always the part that moved the decision least, and the response
#: says how many were dropped rather than presenting the remainder as the whole.
MAX_REPORTED_CONTRIBUTIONS: Final[int] = 25

#: Domain prefix required of each pseudonymous identifier field.
_PSEUDONYM_PREFIX: Final[dict[str, str]] = {
    "user_id": "u",
    "source_id": "s",
    "device_id": "d",
    "session_id": "sess",
}

BoundedName = Annotated[
    str, StringConstraints(min_length=1, max_length=128, strip_whitespace=True)
]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
RegionCode = Annotated[
    str, StringConstraints(min_length=1, max_length=16, strip_whitespace=True)
]
Pseudonym = Annotated[str, StringConstraints(pattern=r"^(u|s|d|sess):[0-9a-f]{32}$")]
ReasonCode = Annotated[str, StringConstraints(min_length=1, max_length=64)]


def _reject_credential_fields(data: Any) -> Any:
    """Refuse a payload that offers credential material under any spelling.

    Runs as a ``mode="before"`` validator so it sees the raw keys, ahead of the
    ``extra="forbid"`` rejection that would otherwise report a password field as
    an ordinary unexpected key.  The scanner reads names only: a prohibited
    field's *value* is never read, copied, or included in the message.
    """
    if isinstance(data, Mapping):
        offending = scan_prohibited_keys(dict(data))
        if offending:
            raise ValueError(
                f"{rejection_marker(ErrorCode.CREDENTIAL_FIELD_REJECTED)} this "
                f"service accepts no credential material; {len(offending)} "
                f"prohibited field name(s) were offered"
            )
    return data


class AnchorSelection(StrEnum):
    """Which events in a window the caller wants a verdict for.

    A window exists because point-in-time features need history.  The anchors
    are the subset of it the caller is actually asking about; the rest is
    context, and context still has to be supplied honestly rather than invented.
    """

    #: The single latest event in the window, in canonical order.
    LAST = "last"
    #: Every event in the window.
    ALL = "all"
    #: Exactly the events named in ``anchor_event_ids``.
    EXPLICIT = "explicit"


class AuthEventRequest(BaseModel):
    """One authentication event, in the terminology of the canonical schema.

    Field names, enumerations, and semantics are the canonical ones: an
    integrator who has read ``docs/data-contract.md`` already knows this model.
    What differs is the identity contract -- ``source_ip`` is admitted as an
    alternative to ``source_id`` and is pseudonymized on the way in -- and the
    stricter per-domain prefix check on the pseudonymous fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "description": (
                "An authentication event. Never send passwords, hashes, tokens, "
                "or any other credential material: such fields are refused."
            )
        },
    )

    event_id: UUID = Field(description="Caller-assigned unique identity for the event.")
    event_time: AwareDatetime = Field(
        description="When the attempt completed. Must be timezone-aware."
    )
    user_id: Pseudonym = Field(
        description="Pseudonymous user identifier, 'u:<32 hex>'."
    )
    source_id: Pseudonym | None = Field(
        default=None,
        description=(
            "Pseudonymous source identifier, 's:<32 hex>'. Supply this or "
            "source_ip, not both."
        ),
    )
    source_ip: str | None = Field(
        default=None,
        description=(
            "Source IP address. Pseudonymized on arrival under the project's "
            "keyed-HMAC privacy contract and never returned in any response."
        ),
    )
    device_id: Pseudonym = Field(
        description="Pseudonymous device identifier, 'd:<32 hex>'."
    )
    session_id: Pseudonym = Field(
        description="Pseudonymous session identifier, 'sess:<32 hex>'."
    )
    application_id: BoundedName = Field(
        description="Identifier of the application that was authenticated to."
    )
    authentication_method: AuthMethod
    authentication_outcome: AuthOutcome
    failure_reason: FailureReason | None = None
    mfa_outcome: MFAOutcome | None = None
    country_code: CountryCode | None = Field(
        default=None, description="ISO 3166-1 alpha-2 country code."
    )
    region_code: RegionCode | None = None
    coarse_latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    coarse_longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    user_agent_family: BoundedName | None = None
    operating_system_family: BoundedName | None = None
    client_type: ClientType | None = None
    response_time_ms: int | None = Field(default=None, ge=0, le=30_000)

    @model_validator(mode="before")
    @classmethod
    def refuse_credentials(cls, data: Any) -> Any:
        """Refuse credential material before anything else is looked at."""
        return _reject_credential_fields(data)

    @field_validator("coarse_latitude", "coarse_longitude")
    @classmethod
    def check_finite(cls, value: float | None) -> float | None:
        """Reject a non-finite coordinate.

        ``Field(ge=..., le=...)`` already rejects an infinity, but ``NaN``
        compares false against every bound and would slip through.
        """
        if value is not None and not math.isfinite(value):
            raise ValueError("a coordinate must be a finite number")
        return value

    @field_validator("source_ip")
    @classmethod
    def check_source_ip(cls, value: str | None) -> str | None:
        """Require a syntactically valid IPv4 or IPv6 address."""
        if value is None:
            return None
        try:
            return str(ipaddress.ip_address(value.strip()))
        except ValueError:
            raise ValueError("source_ip must be a valid IPv4 or IPv6 address") from None

    @model_validator(mode="after")
    def check_identity(self) -> Self:
        """Require exactly one source identity, in the right pseudonym domain."""
        if (self.source_id is None) == (self.source_ip is None):
            raise ValueError(
                "supply exactly one of source_id (pseudonymous) or source_ip (raw)"
            )
        for field_name, prefix in _PSEUDONYM_PREFIX.items():
            value = getattr(self, field_name)
            if value is not None and not value.startswith(f"{prefix}:"):
                raise ValueError(f"{field_name} must be a {prefix!r}-domain pseudonym")
        return self

    def to_canonical_event(self, *, source_id: str) -> AuthEvent:
        """Return the canonical :class:`AuthEvent` this request describes.

        *source_id* is supplied by the caller rather than read off this model,
        because a request carrying ``source_ip`` has no pseudonym until the
        service has produced one.  The canonical schema then applies its own
        cross-field rules -- outcome against failure reason, MFA consistency --
        so this method never has to restate them.
        """
        return AuthEvent(
            event_id=self.event_id,
            event_time=self.event_time,
            user_id=self.user_id,
            source_id=source_id,
            device_id=self.device_id,
            session_id=self.session_id,
            application_id=self.application_id,
            authentication_method=self.authentication_method,
            authentication_outcome=self.authentication_outcome,
            failure_reason=self.failure_reason,
            mfa_outcome=self.mfa_outcome,
            country_code=self.country_code,
            region_code=self.region_code,
            coarse_latitude=self.coarse_latitude,
            coarse_longitude=self.coarse_longitude,
            user_agent_family=self.user_agent_family,
            operating_system_family=self.operating_system_family,
            client_type=self.client_type,
            response_time_ms=self.response_time_ms,
        )


class WindowRequestBase(BaseModel):
    """The shared shape of a detection window: ordered events plus anchors.

    A single stateless event cannot honestly be scored.  Almost every feature the
    rules and the model read is a windowed or sequence quantity over the anchor's
    own strictly-prior history, and a request carrying one event would produce a
    row whose history is empty -- not "unknown", but *wrong*.  So the unit of
    work is a window, the caller supplies the history, and nothing here
    fabricates one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_schema_version: Literal["1.0.0"] = API_SCHEMA_VERSION
    events: tuple[AuthEventRequest, ...] = Field(
        min_length=1,
        description=(
            "The window, in non-decreasing event_time order. Includes the "
            "anchors and the prior context their point-in-time features need."
        ),
    )
    anchor_event_ids: tuple[UUID, ...] = Field(
        default=(),
        description=(
            "Anchors to report on. Required when anchor_selection is 'explicit', "
            "and must be empty otherwise."
        ),
    )
    anchor_selection: AnchorSelection

    @model_validator(mode="before")
    @classmethod
    def refuse_credentials(cls, data: Any) -> Any:
        """Refuse credential material at the envelope level too."""
        return _reject_credential_fields(data)

    @model_validator(mode="after")
    def check_window(self) -> Self:
        """Enforce the size, ordering, uniqueness, and anchor-membership contracts.

        Each refusal states the error code it means, so the client sees
        ``API003`` for a repeated event identity and ``API004`` for a misordered
        window rather than one undifferentiated "malformed request".
        """
        if len(self.events) > MAX_WINDOW_EVENTS:
            raise ValueError(
                f"{rejection_marker(ErrorCode.BATCH_LIMIT_EXCEEDED)} a window "
                f"carries at most {MAX_WINDOW_EVENTS} events"
            )

        seen: set[UUID] = set()
        previous: datetime | None = None
        for event in self.events:
            if event.event_id in seen:
                raise ValueError(
                    f"{rejection_marker(ErrorCode.DUPLICATE_EVENT_IDENTITY)} each "
                    f"event_id may appear once in a window"
                )
            seen.add(event.event_id)
            if previous is not None and event.event_time < previous:
                raise ValueError(
                    f"{rejection_marker(ErrorCode.EVENT_ORDERING_ERROR)} events "
                    f"must be supplied in non-decreasing event_time order"
                )
            previous = event.event_time

        marker = rejection_marker(ErrorCode.ANCHOR_SELECTION_ERROR)
        explicit = self.anchor_selection is AnchorSelection.EXPLICIT
        if explicit and not self.anchor_event_ids:
            raise ValueError(
                f"{marker} anchor_selection 'explicit' requires at least one "
                f"anchor_event_id"
            )
        if not explicit and self.anchor_event_ids:
            raise ValueError(
                f"{marker} anchor_event_ids may only be supplied with "
                f"anchor_selection 'explicit'"
            )
        if len(set(self.anchor_event_ids)) != len(self.anchor_event_ids):
            raise ValueError(f"{marker} anchor_event_ids must not repeat an identifier")
        unknown = [item for item in self.anchor_event_ids if item not in seen]
        if unknown:
            raise ValueError(
                f"{marker} {len(unknown)} requested anchor(s) are not present in "
                f"the window"
            )
        return self

    def resolved_anchor_ids(self) -> tuple[str, ...]:
        """Return the anchors this request selects, as canonical id strings."""
        if self.anchor_selection is AnchorSelection.EXPLICIT:
            return tuple(str(item) for item in self.anchor_event_ids)
        if self.anchor_selection is AnchorSelection.ALL:
            return tuple(str(event.event_id) for event in self.events)
        latest = max(self.events, key=lambda event: (event.event_time, event.event_id))
        return (str(latest.event_id),)


class DetectionWindowRequest(WindowRequestBase):
    """A window scored for one anchor.

    Defaults to the window's latest event, which is the ordinary streaming case:
    "here is what just happened, and here is the recent history it has to be
    read against".
    """

    anchor_selection: AnchorSelection = AnchorSelection.LAST


class DetectionBatchRequest(WindowRequestBase):
    """A window scored for many anchors at once.

    Defaults to every event in the window.  The window is still one ordered
    batch and still bounded: this endpoint answers more anchors, not more input.
    """

    anchor_selection: AnchorSelection = AnchorSelection.ALL


# ---------------------------------------------------------------------------
# Detection responses
# ---------------------------------------------------------------------------


class RuleLayerResult(BaseModel):
    """What the frozen Phase 4 rule layer said about one anchor.

    ``risk_score`` is a bounded **ordinal severity magnitude on a 0-100 scale**.
    It is not a probability, it is not comparable with the model's probability,
    and nothing in this API combines the two arithmetically.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    flagged: bool = Field(description="Whether at least one rule fired.")
    risk_score: float = Field(
        ge=0.0, le=100.0, description="Ordinal severity magnitude. Not a probability."
    )
    severity: Severity
    primary_attack_category: AttackCategory | None = None
    contributing_categories: tuple[AttackCategory, ...] = ()
    fired_rule_ids: tuple[str, ...] = ()
    fired_rule_count: int = Field(default=0, ge=0)
    insufficient_data_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Rules that could not see the history they need. Distinct from a "
            "clean negative."
        ),
    )
    scoring_version: str
    evidence: tuple[EvidenceItem, ...] = Field(
        default=(),
        description=(
            "The strongest sanitized behavioral evidence behind the score. "
            "Carries counts, rates and durations; never an identifier."
        ),
    )


class MLLayerResult(BaseModel):
    """What the frozen ML champion said about one anchor.

    ``probability`` is present exactly when the frozen operating point was
    selected against a calibrated probability.  An uncalibrated decision score is
    never relabelled a probability, and its absence is a null rather than a copy
    of the decision score.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    unavailable_reason: ReasonCode | None = Field(
        default=None,
        description="Stable reason code when the model layer produced no verdict.",
    )
    flagged: bool | None = None
    score_kind: ScoreKind | None = None
    decision_score: float | None = None
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    decision_threshold: float | None = Field(
        default=None, description="The frozen operating point. Never client-settable."
    )

    @model_validator(mode="after")
    def check_layer(self) -> Self:
        """An available layer carries a verdict; an unavailable one names why."""
        if self.available:
            if self.unavailable_reason is not None:
                raise ValueError("an available model layer names no unavailable reason")
            if self.flagged is None or self.decision_score is None:
                raise ValueError("an available model layer reports a decision")
        else:
            if self.unavailable_reason is None:
                raise ValueError("an unavailable model layer must name why")
            if self.flagged is not None:
                raise ValueError("an unavailable model layer reports no decision")
        return self


class HybridLayerResult(BaseModel):
    """The fused verdict, under the strategy validation selected before TEST.

    There is no combined score field and there will not be one.  The strategy is
    frozen: the API cannot choose it, and a deployment with no selected strategy
    reports the hybrid as unavailable rather than defaulting to one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    unavailable_reason: ReasonCode | None = None
    flagged: bool | None = None
    strategy: FusionStrategy | None = None

    @model_validator(mode="after")
    def check_layer(self) -> Self:
        """An available hybrid names its strategy; an unavailable one names why."""
        if self.available:
            if self.unavailable_reason is not None:
                raise ValueError("an available hybrid names no unavailable reason")
            if self.flagged is None or self.strategy is None:
                raise ValueError("an available hybrid reports a strategy and a verdict")
        else:
            if self.unavailable_reason is None:
                raise ValueError("an unavailable hybrid must name why")
            if self.flagged is not None or self.strategy is not None:
                raise ValueError("an unavailable hybrid reports no verdict")
        return self


class AnchorDetection(BaseModel):
    """Everything the three layers said about one anchor, kept separate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_event_id: str = Field(
        description="The caller's own event_id for this anchor."
    )
    anchor_event_time: datetime
    rule: RuleLayerResult
    ml: MLLayerResult
    hybrid: HybridLayerResult
    severity: Severity = Field(
        description=(
            "The Phase 4 ordinal severity for this anchor. Derived from the rule "
            "layer alone; a probability is not a severity."
        )
    )


class WindowSummary(BaseModel):
    """Aggregate facts about the window that was evaluated.

    Counts and versions only.  No path, no fingerprint of anything private, no
    per-row content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_count: int = Field(ge=1)
    anchor_count: int = Field(ge=1)
    feature_schema_version: str
    detection_schema_version: str
    enabled_rule_count: int = Field(ge=0)
    evaluated_snapshot_count: int = Field(ge=0)


class DetectionResponse(BaseModel):
    """The response for a single-anchor detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_schema_version: str = API_SCHEMA_VERSION
    window: WindowSummary
    anchor: AnchorDetection


class BatchDetectionResponse(BaseModel):
    """The response for a many-anchor detection, in canonical anchor order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_schema_version: str = API_SCHEMA_VERSION
    window: WindowSummary
    anchors: tuple[AnchorDetection, ...]


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


class ExplanationContribution(BaseModel):
    """One transformed column's signed contribution to one anchor's decision.

    ``transformed_feature`` is an engineered column name the reviewed allowlist
    admitted.  There is no ``transformed_value`` field and there will not be one:
    Phase 5 gates value disclosure behind a reviewed configuration flag because a
    transformed value can be a country code, and a live wire surface is not the
    place that flag gets turned on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transformed_feature: str
    contribution: float = Field(
        description=(
            "Signed contribution to decision_value, on the decision function's "
            "own scale. Not a probability and not a percentage."
        )
    )


class ExplanationResponse(BaseModel):
    """Which transformed columns moved the frozen model's decision for one anchor.

    The same decomposition Phase 5 publishes, computed by the same function, over
    a live row.  Three things it deliberately is **not**:

    * **Not a decomposition of the probability.**  :attr:`decision_value` is the
      decision function's own quantity -- the logit for a linear head, the mean
      leaf score for the forest, the step for the threshold baseline.  The
      calibrated probability and the frozen operating point are reported by
      ``/api/v1/detect``, and no contribution here sums toward either.
    * **Not causal.**  A contribution says how the fitted function decomposes
      over the columns it was handed.  It does not say the behaviour caused the
      outcome, and the vocabulary stays flat: ``contribution``, never
      ``importance``, ``driver``, or ``because``.
    * **Not complete.**  The contributions are ranked by magnitude and bounded;
      :attr:`omitted_contribution_count` says how many were left out, so a reader
      never mistakes the reported set for the whole decomposition.

    :attr:`reconstruction_residual` is the check, reported rather than asserted
    away: the full decomposition -- not the truncated one below -- reconstructs
    the model's own decision value to within the declared tolerance, or this
    response reports the explanation unavailable instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_schema_version: str = API_SCHEMA_VERSION
    anchor_event_id: str
    anchor_event_time: datetime
    available: bool
    unavailable_reason: ReasonCode | None = Field(
        default=None,
        description="Stable reason code when no exact decomposition was produced.",
    )
    method: ExplanationMethod | None = Field(
        default=None,
        description=(
            "The exact decomposition used. A family without one reports "
            "unavailable rather than an approximation."
        ),
    )
    model_family: str | None = None
    score_kind: ScoreKind | None = Field(
        default=None,
        description=(
            "What the frozen operating point was applied to. Context only; "
            "never summed with a contribution."
        ),
    )
    decision_value: float | None = Field(
        default=None,
        description="The decision-function quantity the contributions sum to.",
    )
    baseline_value: float | None = Field(
        default=None,
        description=(
            "The additive constant contributions are measured against: a linear "
            "intercept, the forest's ensemble-mean root value, or zero."
        ),
    )
    contributions: tuple[ExplanationContribution, ...] = ()
    transformed_feature_count: int = Field(
        default=0, ge=0, description="Columns the full decomposition covered."
    )
    omitted_contribution_count: int = Field(
        default=0, ge=0, description="Columns ranked below the reported bound."
    )
    reconstruction_residual: float | None = Field(
        default=None,
        description=(
            "decision_value - (baseline_value + sum of the FULL decomposition). "
            "Checked against the declared tolerance before this response is built."
        ),
    )

    @model_validator(mode="after")
    def check_availability(self) -> Self:
        """An available explanation decomposes something; an absent one names why."""
        if self.available:
            if self.unavailable_reason is not None:
                raise ValueError("an available explanation names no unavailable reason")
            if (
                self.method is None
                or self.decision_value is None
                or self.baseline_value is None
                or self.reconstruction_residual is None
            ):
                raise ValueError(
                    "an available explanation reports a method, a decision value, "
                    "a baseline, and the residual that checked them"
                )
        else:
            if self.unavailable_reason is None:
                raise ValueError("an unavailable explanation must name why")
            if self.method is not None or self.contributions:
                raise ValueError("an unavailable explanation decomposes nothing")
        if len(self.contributions) > MAX_REPORTED_CONTRIBUTIONS:
            raise ValueError(
                f"an explanation reports at most {MAX_REPORTED_CONTRIBUTIONS} "
                f"contributions"
            )
        names = [item.transformed_feature for item in self.contributions]
        if len(set(names)) != len(names):
            raise ValueError("an explanation credits each column at most once")
        if len(self.contributions) + self.omitted_contribution_count != (
            self.transformed_feature_count
        ):
            raise ValueError(
                "the reported and omitted contributions must account for every "
                "column the decomposition covered"
            )
        return self


# ---------------------------------------------------------------------------
# Operational documents
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Process liveness.  Deliberately cheap and deliberately uninformative.

    Answers "is this process running and able to serve a request", and nothing
    else.  It performs no artifact check, touches no model, and reads no file, so
    a liveness probe cannot become a load source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok"] = "ok"
    service: str = SERVICE_NAME
    version: str


class ComponentState(StrEnum):
    """The readiness of one runtime component."""

    READY = "ready"
    UNAVAILABLE = "unavailable"
    #: Deliberately switched off by configuration, not broken.
    DISABLED = "disabled"


class ComponentReport(BaseModel):
    """One component's readiness, with a stable reason when it is not ready."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: str
    state: ComponentState
    reason: ReasonCode | None = Field(
        default=None,
        description=(
            "Stable sanitized reason code. Never a path, a message, or a trace."
        ),
    )
    #: Whether overall readiness depends on this component.
    required: bool = True

    @model_validator(mode="after")
    def check_component(self) -> Self:
        """A component that is not ready must name a stable reason."""
        if self.state is ComponentState.READY and self.reason is not None:
            raise ValueError("a ready component names no reason")
        if self.state is not ComponentState.READY and self.reason is None:
            raise ValueError("a component that is not ready must name a reason code")
        return self


class ReadinessState(StrEnum):
    """Whether this runtime can actually serve detection."""

    READY = "ready"
    NOT_READY = "not_ready"


class ReadinessResponse(BaseModel):
    """Aggregate readiness across every runtime component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ReadinessState
    service: str = SERVICE_NAME
    version: str
    components: tuple[ComponentReport, ...]

    @model_validator(mode="after")
    def check_aggregate(self) -> Self:
        """Ready means every required component is ready.  No partial credit."""
        blocking = [
            item
            for item in self.components
            if item.required and item.state is not ComponentState.READY
        ]
        if blocking and self.status is ReadinessState.READY:
            raise ValueError(
                "a runtime with an unavailable required component is not ready"
            )
        if not blocking and self.status is not ReadinessState.READY:
            raise ValueError(
                "a runtime with every required component ready is not not-ready"
            )
        return self


class VersionResponse(BaseModel):
    """Deterministic contract versions.

    Every value here is a property of the build and the frozen artifacts.
    Nothing host-specific appears: no hostname, no path, no process identity, no
    interpreter build, no start time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    service: str = SERVICE_NAME
    package_version: str
    api_schema_version: str = API_SCHEMA_VERSION
    event_schema_version: str = SCHEMA_VERSION
    feature_schema_version: str
    detection_schema_version: str
    scoring_version: str
    ml_schema_version: str
    fusion_schema_version: str


class SystemStatusResponse(BaseModel):
    """Which detection layers this deployment is actually running.

    The hybrid arm reports three separate things, because collapsing them would
    hide the distinction that matters most operationally:

    * :attr:`frozen_fusion_strategy` -- what validation **selected** before TEST.
      ``None`` means no hybrid qualified, which is a scientific outcome rather
      than a deployment fault.
    * :attr:`hybrid_required` -- whether readiness depends on it.  True exactly
      when a strategy was frozen: a deployment that cannot run the hybrid it
      selected is not a working deployment.
    * :attr:`fusion_strategy` -- what is actually **executing**.  Equal to the
      frozen strategy or ``None``; it is never a substitute, because a fallback
      would publish a hybrid nobody selected and would look exactly like one that
      had been.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    service: str = SERVICE_NAME
    status: ReadinessState
    api_schema_version: str = API_SCHEMA_VERSION
    package_version: str
    rule_detection_enabled: bool
    ml_detection_enabled: bool
    hybrid_detection_enabled: bool
    fusion_strategy: FusionStrategy | None = Field(
        default=None,
        description="The frozen strategy this deployment is executing, if any.",
    )
    frozen_fusion_strategy: FusionStrategy | None = Field(
        default=None,
        description=(
            "The strategy validation selected before TEST. Null means no hybrid "
            "qualified, which is a measured outcome and not a missing artifact."
        ),
    )
    hybrid_required: bool = Field(
        default=False,
        description=(
            "Whether readiness depends on the hybrid. True exactly when a "
            "strategy was frozen."
        ),
    )
    stacked_state_fingerprint: str | None = Field(
        default=None,
        description=(
            "Digest of the fitted meta-learner this deployment loaded, for a "
            "verified stacked hybrid. Identity only: no coefficient is reachable."
        ),
    )
    fusion_unavailable_reason: ReasonCode | None = None
    champion_model_family: str | None = None
    enabled_rule_count: int = Field(ge=0)
    registered_rule_count: int = Field(ge=0)
    max_batch_events: int = Field(ge=1)

    # -- demonstration replay ----------------------------------------------
    #: Reported here rather than on ``/health`` deliberately. Liveness stays
    #: cheap and uninformative; whether an *optional* subsystem initialised is
    #: exactly the sort of thing a status document is for.
    replay_enabled: bool = Field(
        default=False,
        description="Whether this deployment serves the replay endpoints.",
    )
    replay_available: bool = Field(
        default=False,
        description="Whether a replay run can actually be started right now.",
    )
    replay_required: bool = Field(
        default=False,
        description=(
            "Whether overall readiness depends on replay. False on an ordinary "
            "deployment: a detection service does not become unavailable because "
            "an optional demonstration facility did not initialise."
        ),
    )
    replay_unavailable_reason: ReasonCode | None = None
    replay_scenario_count: int = Field(
        default=0, ge=0, description="Scenarios in the built-in replay catalog."
    )
    max_active_replay_runs: int = Field(
        default=0, ge=0, description="Concurrent replay runs this deployment admits."
    )

    @model_validator(mode="after")
    def check_replay(self) -> Self:
        """An available subsystem names no reason; an enabled-but-broken one does.

        A *disabled* subsystem is exempt from the reason requirement: "switched
        off" is already the whole explanation, and demanding a code for it would
        make every deployment that does not want replay carry a field that reads
        like a fault.
        """
        if self.replay_available:
            if not self.replay_enabled:
                raise ValueError("a disabled replay subsystem is not available")
            if self.replay_unavailable_reason is not None:
                raise ValueError(
                    "an available replay subsystem names no unavailable reason"
                )
        elif self.replay_enabled and self.replay_unavailable_reason is None:
            raise ValueError(
                "a replay subsystem that is enabled and not available must name "
                "a reason code"
            )
        return self

    @model_validator(mode="after")
    def check_hybrid(self) -> Self:
        """An executing hybrid is the frozen one, and names no reason."""
        if self.hybrid_detection_enabled:
            if self.fusion_strategy is None:
                raise ValueError("an enabled hybrid names the strategy it runs")
            if self.fusion_strategy is not self.frozen_fusion_strategy:
                raise ValueError(
                    "the executing strategy is not the frozen one; there is no "
                    "fallback hybrid and nothing may report one"
                )
            if self.fusion_unavailable_reason is not None:
                raise ValueError("an enabled hybrid names no unavailable reason")
        elif self.fusion_strategy is not None:
            raise ValueError("a disabled hybrid executes no strategy")
        # Required exactly when a hybrid was frozen, with one exception:
        # ambiguous lineage requires a hybrid without a nameable strategy. That
        # case is reported by the readiness document's component reason rather
        # than by inventing a strategy here.
        if self.hybrid_required != (self.frozen_fusion_strategy is not None) and not (
            self.hybrid_required and self.frozen_fusion_strategy is None
        ):
            raise ValueError("the hybrid is required exactly when one was frozen")
        if self.stacked_state_fingerprint is not None and (
            self.fusion_strategy is not FusionStrategy.STACKED
        ):
            raise ValueError("only a running stacked hybrid has a fitted state to name")
        return self


class ModelInfoResponse(BaseModel):
    """Public-safe identity and lineage of the frozen champion.

    Carries what the model *is* and how it decides, never how it computes:
    no coefficient, no tree array, no feature value, no artifact path, and no
    entity pseudonym appears here or can be reached from here.
    """

    # ``protected_namespaces`` is cleared because this model's subject *is* a
    # model: ``model_family`` and ``model_id`` are the project's own published
    # terminology, and renaming them to dodge a Pydantic namespace warning would
    # make the API disagree with every document that describes it.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    available: bool
    unavailable_reason: ReasonCode | None = None
    model_family: str | None = None
    catalog_model_id: str | None = None
    model_id: str | None = Field(
        default=None,
        description="The content-derived model identifier the freeze recorded.",
    )
    task: str | None = None
    score_kind: ScoreKind | None = None
    calibrated: bool | None = None
    decision_threshold: float | None = None
    champion_scope_key: str | None = None
    freeze_record_id: str | None = None
    training_run_id: str | None = None
    validation_selection_id: str | None = None
    ml_schema_version: str | None = None
    required_feature_schema_version: str | None = None
    category_head_available: bool = False

    @model_validator(mode="after")
    def check_availability(self) -> Self:
        """An unavailable model names why and reports nothing else."""
        if self.available and self.unavailable_reason is not None:
            raise ValueError("an available model names no unavailable reason")
        if not self.available:
            if self.unavailable_reason is None:
                raise ValueError("an unavailable model must name why")
            if self.model_id is not None or self.decision_threshold is not None:
                raise ValueError("an unavailable model reports no identity")
        return self


class RuleSummary(BaseModel):
    """One rule, as the public catalog describes it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    rule_version: str
    name: str
    description: str
    family: RuleFamily
    attack_category: AttackCategory
    default_severity: Severity
    enabled: bool
    deprecated: bool = False


class RuleCatalogResponse(BaseModel):
    """The public rule catalog, ordered by rule identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_schema_version: str
    rule_count: int = Field(ge=0)
    enabled_rule_count: int = Field(ge=0)
    rules: tuple[RuleSummary, ...]
