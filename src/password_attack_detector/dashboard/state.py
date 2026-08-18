"""What one browser session remembers, and what it deliberately does not.

The whole of this module's contents live in a single Streamlit session and end
with it.  That is a limitation, and it is stated everywhere it is visible rather
than papered over: there is no alert database yet, no event store, and no
server-side history, so a page that showed "1,284 alerts today" would be showing
a number nobody measured.  Persistent history is a later milestone's work.

**The record is built from what the service returned.**  A
:class:`DetectionRecord` is assembled from a response document; the dashboard
never re-derives a verdict, re-thresholds a score, or combines the layers to
produce a flag of its own.  The one number this module computes is a sequence
counter, which is a fact about this session and about nothing else.

**No request body is kept.**  What a submitted window contained -- the
identifiers, the addresses, the application names -- stays in the draft the
analyst is editing and is not copied into the history.  A resend re-reads the
draft the analyst can see, so nothing can be re-submitted that they cannot
inspect first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from password_attack_detector.dashboard.contracts import (
    AnchorDetection,
    BatchDetectionDocument,
    DetectionDocument,
    ExplanationDocument,
)
from password_attack_detector.dashboard.scenarios import prohibited_field_names

__all__ = [
    "MAX_HISTORY",
    "DashboardSession",
    "DetectionRecord",
    "severity_counts",
    "triggered_rule_counts",
]

#: How many detection results one session keeps.  Bounded because a session that
#: never forgets is a memory leak with a chart on top; the oldest are dropped and
#: the counter keeps counting, so the sequence numbers stay honest.
MAX_HISTORY: Final[int] = 200


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """One detection this session performed, as the service reported it.

    Dashboard-safe metadata only.  No pseudonym, no address, no feature value,
    no artifact identifier, and no request body: everything here either came out
    of a response the service already deemed publishable, or is a counter.
    """

    #: Position in this session, starting at 1.  Local to the browser tab and
    #: explicitly not a server-side alert identifier.
    sequence: int
    #: When the dashboard received the response.  A display timestamp, not an
    #: event time -- the event's own time is :attr:`anchor_event_time`.
    observed_at: datetime
    scenario: str
    event_count: int
    anchor_event_id: str
    anchor_event_time: datetime
    severity: str
    rule_flagged: bool
    rule_risk_score: float
    fired_rule_ids: tuple[str, ...]
    ml_available: bool
    ml_flagged: bool | None
    ml_score_kind: str | None
    ml_decision_score: float | None
    ml_probability: float | None
    hybrid_available: bool
    hybrid_flagged: bool | None
    hybrid_strategy: str | None

    @classmethod
    def from_document(
        cls,
        document: DetectionDocument,
        *,
        sequence: int,
        observed_at: datetime,
        scenario: str,
    ) -> DetectionRecord:
        """Build a record from a single-anchor detection response.

        *observed_at* is passed in rather than read from a clock here, so the
        whole module stays free of ambient time and a test can state exactly
        what a session's history looks like.
        """
        return cls.from_anchor(
            document.anchor,
            event_count=document.window.event_count,
            sequence=sequence,
            observed_at=observed_at,
            scenario=scenario,
        )

    @classmethod
    def from_anchor(
        cls,
        anchor: AnchorDetection,
        *,
        event_count: int,
        sequence: int,
        observed_at: datetime,
        scenario: str,
    ) -> DetectionRecord:
        """Build a record from one anchor's verdict.

        Factored out of :meth:`from_document` so a batch response -- which
        carries many anchors over one window -- produces one record per anchor
        through the same code. A batch whose results were rendered but not
        recorded would show a verdict on screen and then claim, on the alerts
        page, that the session had no activity.
        """
        return cls(
            sequence=sequence,
            observed_at=observed_at,
            scenario=scenario,
            event_count=event_count,
            anchor_event_id=anchor.anchor_event_id,
            anchor_event_time=anchor.anchor_event_time,
            severity=anchor.severity,
            rule_flagged=anchor.rule.flagged,
            rule_risk_score=anchor.rule.risk_score,
            fired_rule_ids=anchor.rule.fired_rule_ids,
            ml_available=anchor.ml.available,
            ml_flagged=anchor.ml.flagged,
            ml_score_kind=anchor.ml.score_kind,
            ml_decision_score=anchor.ml.decision_score,
            ml_probability=anchor.ml.probability,
            hybrid_available=anchor.hybrid.available,
            hybrid_flagged=anchor.hybrid.flagged,
            hybrid_strategy=anchor.hybrid.strategy,
        )

    @property
    def any_layer_flagged(self) -> bool:
        """Return whether any available layer raised a flag.

        A disjunction for *display grouping only*. It is not a verdict, it is
        not the hybrid, and nothing downstream treats it as one -- the fused
        decision is :attr:`hybrid_flagged` and is produced by the frozen
        strategy on the server.
        """
        return bool(self.rule_flagged or self.ml_flagged or self.hybrid_flagged)


@dataclass
class DashboardSession:
    """Everything one browser session holds, in one mutable object.

    Kept as a plain object rather than a scatter of ``st.session_state`` keys so
    the logic below is testable without Streamlit running, and so there is one
    place to look for "what does this session remember".
    """

    #: The window being composed. A list of request-shaped event mappings,
    #: exactly as the API's schema defines them.
    draft_events: list[dict[str, Any]] = field(default_factory=list)
    #: Results, oldest first, bounded by :data:`MAX_HISTORY`.
    history: list[DetectionRecord] = field(default_factory=list)
    #: The most recent full detection document, for the result panels that
    #: render evidence and per-layer detail. Only the latest is kept: a session
    #: holding every full response would grow without bound, and the bounded
    #: :class:`DetectionRecord` is what the history pages actually read.
    last_result: DetectionDocument | None = None
    #: The most recent attribution, when one was requested.
    last_explanation: ExplanationDocument | None = None
    #: The template last loaded into the draft, for display.
    selected_scenario: str = "custom"
    #: How many detections this session has performed, including any dropped
    #: from :attr:`history`. The source of the sequence numbers.
    detection_count: int = 0

    # -- the draft window ---------------------------------------------------

    def add_event(self, event: Mapping[str, Any]) -> None:
        """Append one request-shaped event to the draft.

        Refuses a credential-shaped key rather than storing it.  Nothing in the
        console can produce one -- the event builder writes a fixed set of keys
        -- so this is defence in depth, and it is placed *here* deliberately:
        this is the only way into :attr:`draft_events`, so the draft is provably
        free of credential material, and therefore so is everything that renders
        it and everything that posts it.

        The service refuses such a request regardless.  What this adds is that a
        secret is never written into browser-session state and never echoed back
        onto a page, both of which would happen before the service ever saw it.

        Raises:
            ValueError: naming the offending field names and nothing else.  The
                value is not read, not copied, and not included in the message.
        """
        offending = prohibited_field_names(event)
        if offending:
            raise ValueError(
                f"this console holds no credential material; "
                f"{len(offending)} prohibited field name(s) were offered"
            )
        self.draft_events.append(dict(event))

    def extend_events(self, events: Iterable[Mapping[str, Any]]) -> None:
        """Append several events to the draft, in the order given."""
        for item in events:
            self.add_event(item)

    def remove_event(self, index: int) -> None:
        """Drop the event at *index*, ignoring an index that is not there."""
        if 0 <= index < len(self.draft_events):
            del self.draft_events[index]

    def clear_window(self) -> None:
        """Empty the draft and forget which template produced it."""
        self.draft_events.clear()
        self.selected_scenario = "custom"

    def load_scenario(self, name: str, events: Sequence[Mapping[str, Any]]) -> None:
        """Replace the draft with a template's events.

        Replaces rather than appends: a template is a complete scenario, and
        merging one into whatever was already there would produce a window
        describing neither.

        The draft is cleared *before* the new events are admitted, so a template
        that failed the credential check cannot leave a half-loaded window behind
        it.

        Raises:
            ValueError: when any event carries a credential-shaped field name.
        """
        self.draft_events = []
        self.extend_events(events)
        self.selected_scenario = name

    # -- results ------------------------------------------------------------

    def record(
        self, document: DetectionDocument, *, observed_at: datetime
    ) -> DetectionRecord:
        """Append one detection result and return the record that was stored."""
        self.detection_count += 1
        entry = DetectionRecord.from_document(
            document,
            sequence=self.detection_count,
            observed_at=observed_at,
            scenario=self.selected_scenario,
        )
        self.last_result = document
        self._append(entry)
        return entry

    def record_batch(
        self, document: BatchDetectionDocument, *, observed_at: datetime
    ) -> tuple[DetectionRecord, ...]:
        """Append one record per anchor in a batch response, in its own order.

        One record per anchor, because one anchor is one verdict: a batch stored
        as a single entry would under-count the session's own activity, and a
        batch stored as nothing at all would render results on screen while the
        alerts page reported an empty session.

        :attr:`last_result` is left alone. It holds the document the detailed
        result panels render, and those panels take a single anchor; a batch has
        no single anchor to be the latest one, and picking one would be arbitrary.
        """
        stored: list[DetectionRecord] = []
        for anchor in document.anchors:
            self.detection_count += 1
            entry = DetectionRecord.from_anchor(
                anchor,
                event_count=document.window.event_count,
                sequence=self.detection_count,
                observed_at=observed_at,
                scenario=self.selected_scenario,
            )
            self._append(entry)
            stored.append(entry)
        return tuple(stored)

    def _append(self, entry: DetectionRecord) -> None:
        """Store one record, dropping the oldest once the bound is reached."""
        self.history.append(entry)
        if len(self.history) > MAX_HISTORY:
            del self.history[: len(self.history) - MAX_HISTORY]

    def clear_history(self) -> None:
        """Forget every result, and reset the sequence numbering with them.

        The counter is reset too: leaving it running would number the next
        detection ``#57`` in a list whose first entry is ``#57``, which reads as
        a display bug rather than as the deliberate clearing it was.
        """
        self.history.clear()
        self.last_result = None
        self.last_explanation = None
        self.detection_count = 0

    @property
    def latest(self) -> DetectionRecord | None:
        """Return the most recent result, or ``None`` when there is none."""
        return self.history[-1] if self.history else None

    @property
    def flagged_history(self) -> tuple[DetectionRecord, ...]:
        """Return the results where at least one layer raised a flag."""
        return tuple(item for item in self.history if item.any_layer_flagged)


def severity_counts(history: Sequence[DetectionRecord]) -> dict[str, int]:
    """Return how many results carried each severity, ordered by severity.

    Ordered by the Phase 4 ordinal scale rather than alphabetically or by count,
    so a chart of it reads left to right as increasing seriousness.
    """
    order = ("none", "low", "medium", "high", "critical")
    counts = dict.fromkeys(order, 0)
    for item in history:
        counts[item.severity] = counts.get(item.severity, 0) + 1
    return {name: counts[name] for name in order if name in counts}


def triggered_rule_counts(history: Sequence[DetectionRecord]) -> dict[str, int]:
    """Return how often each rule fired, most frequent first.

    Ties are broken by rule identifier, so the same session always renders the
    same chart rather than one that depends on dictionary order.
    """
    counts: dict[str, int] = {}
    for item in history:
        for rule_id in item.fired_rule_ids:
            counts[rule_id] = counts.get(rule_id, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))
