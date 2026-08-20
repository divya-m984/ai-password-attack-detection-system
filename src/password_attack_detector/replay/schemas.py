"""The replay layer's wire contract.

Two rules shape everything below, and both are inherited from the serving layer
rather than re-decided here.

**A replay record reports a detection, it does not restate one.**  Every timeline
entry embeds the serving layer's own
:class:`~password_attack_detector.api.schemas.AnchorDetection` verbatim.  The
three layers stay in their three separate objects, the rule layer's ordinal
magnitude stays next to the model's own score without either being combined with
the other, and a field that is public-safe in a detection response is public-safe
here for exactly the same reason.  Re-declaring those fields would create a second
opinion about which of them may be published.

**The only thing a caller may say is which scenario and how fast.**
:class:`CreateReplayRunRequest` declares two fields, both closed vocabularies,
and forbids everything else.  There is no field on it for a model, a threshold, a
strategy, an artifact root, a filesystem path, an external target, an event
definition, or a credential -- and ``extra="forbid"`` means offering one is a
refusal rather than a silently ignored key.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.api.errors import ErrorCode, rejection_marker
from password_attack_detector.api.schemas import AnchorDetection
from password_attack_detector.data.privacy import scan_prohibited_keys
from password_attack_detector.replay.enums import ReplayPace, ReplayState, ScenarioId

__all__ = [
    "REPLAY_SCHEMA_VERSION",
    "CreateReplayRunRequest",
    "ReplayRunListResponse",
    "ReplayRunResponse",
    "ReplayRunSummary",
    "ReplayTimelineRecord",
    "ReplayTimelineResponse",
    "ScenarioCatalogResponse",
    "ScenarioSummary",
]

#: The replay wire contract's own version.  Independent of the API schema
#: version: the replay surface can grow a field without the detection contract
#: changing, and the reverse.
REPLAY_SCHEMA_VERSION: Final[str] = "1.0.0"


def _reject_credential_fields(data: Any) -> Any:
    """Refuse a payload offering credential material under any spelling.

    The same ``mode="before"`` scan the detection request schemas run, applied to
    a body that has no credential-shaped field to begin with.  It is here anyway
    because "this model declares no such field" and "this model refuses such a
    field" are different guarantees, and only the second survives somebody adding
    a field later.
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


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class ScenarioSummary(BaseModel):
    """One built-in scenario, as the catalog publishes it.

    Everything here is a property of the *scenario*, not of any run: the events
    it will emit, how long its simulated timeline is, and what replaying it has
    been shown to demonstrate.  The rule expectations are the ones an integration
    test asserts against the frozen system; a rule the scenario would like to
    demonstrate and does not is described in :attr:`limitations` instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: ScenarioId
    name: str
    description: str
    purpose: str = Field(description="What an analyst is meant to learn from it.")
    scenario_schema_version: str
    revision: int = Field(
        ge=1, description="Bumped whenever the event content changes."
    )
    scenario_fingerprint: str = Field(
        description=(
            "Digest over the schema version, identifier, revision and every "
            "event. Two runs agreeing on this replayed identical timelines."
        )
    )
    event_count: int = Field(ge=1)
    duration_seconds: float = Field(
        ge=0.0,
        description=(
            "Span of the scenario's own simulated timeline. Unaffected by the "
            "pace it is replayed at."
        ),
    )
    expected_rule_ids: tuple[str, ...] = Field(
        default=(),
        description=(
            "Rules an integration test proved fire during this scenario under "
            "the repository's demo rule configuration."
        ),
    )
    expected_rule_families: tuple[str, ...] = ()
    expected_severity_at_least: str | None = Field(
        default=None,
        description=(
            "The weakest severity the run's worst step is guaranteed to reach. "
            "Null where the scenario makes no severity claim."
        ),
    )
    demonstrates_ml: bool = Field(
        description=(
            "Whether the frozen model layer produces a verdict. A verdict, not a "
            "positive one: the catalog does not predict what the model decides."
        )
    )
    demonstrates_hybrid: bool
    limitations: str = Field(
        default="", description="What this scenario cannot demonstrate, and why."
    )


class ScenarioCatalogResponse(BaseModel):
    """``GET /api/v1/demo/scenarios``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    replay_schema_version: str = REPLAY_SCHEMA_VERSION
    scenario_schema_version: str
    scenario_count: int = Field(ge=0)
    scenarios: tuple[ScenarioSummary, ...]


# ---------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------


class ReplayTimelineRecord(BaseModel):
    """One replayed step: what was emitted, and what the frozen system said.

    :attr:`source_event_time` and :attr:`emitted_at` are deliberately both
    present and deliberately different things.  The first is the scenario's own
    fixed timestamp -- the instant the fabricated event claims to have happened,
    and the one every point-in-time feature is computed against.  The second is
    when this process presented the step, which moves with the replay pace and
    is presentation metadata only.  Nothing scientific reads it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(
        ge=1,
        description=(
            "Position in this run's timeline, from 1, strictly increasing. The "
            "cursor clients poll with."
        ),
    )
    run_id: str
    scenario_id: ScenarioId
    replay_state: ReplayState = Field(
        description="The run's state at the moment this record was appended."
    )
    event_index: int = Field(ge=0, description="Index of the emitted scenario event.")
    window_event_count: int = Field(
        ge=1,
        description=(
            "How many events the detection window carried: every event emitted "
            "so far, which is the anchor's own strictly-prior history plus the "
            "anchor. No history is fabricated."
        ),
    )
    source_event_time: datetime = Field(
        description="The scenario event's own fixed timestamp."
    )
    emitted_at: datetime = Field(
        description=(
            "When this process presented the step. Presentation metadata: it "
            "moves with the replay pace and no feature, rule, or model reads it."
        )
    )
    authentication_outcome: str = Field(
        description="The synthetic event's own outcome, for the analyst timeline."
    )
    detection: AnchorDetection = Field(
        description=(
            "The serving layer's own three-layer verdict for this anchor, "
            "embedded verbatim."
        )
    )

    @property
    def severity(self) -> str:
        """Return the anchor's Phase 4 ordinal severity."""
        return str(self.detection.severity)


class ReplayRunSummary(BaseModel):
    """Aggregate counts over one run's timeline records.

    Every field is derived from the records the run actually produced.  There is
    no global total here and there is no persistent history behind it: a summary
    describes one demonstration run in one process, and says so.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_count: int = Field(ge=0, description="Timeline records produced.")
    rule_flagged_count: int = Field(ge=0)
    ml_flagged_count: int = Field(ge=0)
    ml_unavailable_count: int = Field(ge=0)
    hybrid_flagged_count: int = Field(ge=0)
    hybrid_unavailable_count: int = Field(ge=0)
    highest_severity: str | None = Field(
        default=None, description="Worst severity observed, on the Phase 4 scale."
    )
    severity_counts: dict[str, int] = Field(
        default_factory=dict, description="How many records carried each severity."
    )
    triggered_rule_counts: dict[str, int] = Field(
        default_factory=dict, description="How often each public rule identifier fired."
    )
    fusion_strategies: tuple[str, ...] = Field(
        default=(),
        description=(
            "The frozen strategies observed executing. More than one entry would "
            "mean the deployment changed underneath the run."
        ),
    )


class ReplayRunResponse(BaseModel):
    """One run's identity, state, progress, and summary.

    Two identities, and the distinction matters:

    * :attr:`scenario_fingerprint` is **content identity**.  It is the same on
      every machine for the same catalog entry, and it is what makes "the same
      scenario produced the same verdicts" checkable.
    * :attr:`run_id` is **instance identity**.  It distinguishes two simultaneous
      executions of the same scenario in the same process and means nothing
      outside it.  It is opaque: no host path, process identifier, port, user, or
      secret contributes to it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    replay_schema_version: str = REPLAY_SCHEMA_VERSION
    run_id: str
    scenario_id: ScenarioId
    scenario_name: str
    scenario_revision: int = Field(ge=1)
    scenario_fingerprint: str
    pace: ReplayPace
    state: ReplayState
    event_count: int = Field(
        ge=1, description="Events the scenario will emit in total."
    )
    emitted_count: int = Field(ge=0, description="Events emitted and scored so far.")
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    next_sequence: int = Field(
        ge=0,
        description=(
            "The cursor to poll the timeline from to receive only new records."
        ),
    )
    more_expected: bool = Field(
        description=(
            "Whether further records may still arrive. False once the run "
            "reached a terminal state; a client stops polling on it."
        )
    )
    failure_reason: str | None = Field(
        default=None,
        description=(
            "Stable reason code when the run failed. Never an exception message, "
            "a path, or a stack frame."
        ),
    )
    summary: ReplayRunSummary

    @model_validator(mode="after")
    def check_progress(self) -> Self:
        """A run reports no more progress than its scenario has, and names failure."""
        if self.emitted_count > self.event_count:
            raise ValueError("a run cannot emit more events than its scenario carries")
        if (self.state is ReplayState.FAILED) != (self.failure_reason is not None):
            raise ValueError(
                "a failed run names a stable reason code, and only a failed one does"
            )
        if self.more_expected and self.state in {
            ReplayState.COMPLETED,
            ReplayState.STOPPED,
            ReplayState.FAILED,
        }:
            raise ValueError("a run in a terminal state expects no further records")
        return self


class ReplayRunListResponse(BaseModel):
    """``GET /api/v1/demo/runs`` -- the runs this *process* is retaining.

    Named for what it is.  This is not a history: it is a bounded window over
    one process's memory, newest first, cleared by a restart. The retention
    bound is published beside it so a caller can see that an absent run may have
    been evicted rather than never have existed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    replay_schema_version: str = REPLAY_SCHEMA_VERSION
    run_count: int = Field(ge=0)
    active_run_count: int = Field(ge=0)
    max_active_runs: int = Field(ge=1)
    max_retained_runs: int = Field(ge=1)
    runs: tuple[ReplayRunResponse, ...]


class ReplayTimelineResponse(BaseModel):
    """``GET /api/v1/demo/runs/{run_id}/timeline``.

    Incremental by construction.  A client polls with the ``next_sequence`` it
    was last given and receives only what it has not seen; the page is bounded,
    and :attr:`more_expected` distinguishes "there is already more waiting" from
    "the run is still producing" from "this is everything there will ever be".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    replay_schema_version: str = REPLAY_SCHEMA_VERSION
    run_id: str
    scenario_id: ScenarioId
    state: ReplayState
    after_sequence: int = Field(ge=0, description="The cursor this page was read from.")
    next_sequence: int = Field(
        ge=0, description="The cursor to poll with next. Never moves backwards."
    )
    record_count: int = Field(ge=0)
    records: tuple[ReplayTimelineRecord, ...]
    more_expected: bool = Field(
        description=(
            "True while more records may arrive: either this page was truncated, "
            "or the run has not reached a terminal state."
        )
    )
    emitted_count: int = Field(ge=0)
    event_count: int = Field(ge=1)

    @model_validator(mode="after")
    def check_page(self) -> Self:
        """The page is contiguous, ordered, and consistent with its own cursors."""
        if self.record_count != len(self.records):
            raise ValueError("record_count must match the records returned")
        previous = self.after_sequence
        for item in self.records:
            if item.sequence <= previous:
                raise ValueError("timeline records must strictly increase in sequence")
            previous = item.sequence
        if self.next_sequence < self.after_sequence:
            raise ValueError("a timeline cursor never moves backwards")
        if self.records and self.next_sequence != self.records[-1].sequence:
            raise ValueError("next_sequence must be the last record's sequence")
        return self


# ---------------------------------------------------------------------------
# The one request a caller may make
# ---------------------------------------------------------------------------


class CreateReplayRunRequest(BaseModel):
    """``POST /api/v1/demo/runs`` -- start one built-in scenario.

    The entire admissible input surface: a name from the reviewed catalog, and a
    word from the pace vocabulary.  Both are enumerations, so an unknown value is
    a refusal rather than something to interpret.

    There is no field here for an event list, a source address, an external
    target, a filesystem path, a model identifier, a threshold, or a fusion
    strategy -- and ``extra="forbid"`` means offering one is refused rather than
    ignored. That is deliberate to the point of being the design: a replay
    endpoint that accepted a caller-supplied event stream would be an
    unauthenticated way to make this process do work of the caller's choosing,
    and one that accepted a scientific parameter would let a demonstration
    publish a system nobody deployed.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "description": (
                "Start a built-in synthetic scenario. Scenarios cannot be "
                "uploaded, parameterised, or pointed at an external target."
            )
        },
    )

    scenario_id: ScenarioId = Field(description="A scenario from the built-in catalog.")
    pace: ReplayPace = Field(
        default=ReplayPace.NORMAL,
        description=(
            "How fast the presentation advances. Affects wall-clock spacing "
            "only; the scenario's event times and every verdict are identical at "
            "every pace."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def refuse_credentials(cls, data: Any) -> Any:
        """Refuse credential material before anything else is looked at."""
        return _reject_credential_fields(data)
