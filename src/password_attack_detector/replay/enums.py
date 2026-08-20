"""The replay layer's closed vocabularies.

Three of them, and each is closed for a reason a demonstration system makes
sharper than an ordinary one.

**A scenario is a name from a reviewed list, never a payload.**  A demo that
accepted an uploaded scenario would be a detection service that runs whatever
event stream a caller composed, on the caller's schedule, under a label the
caller chose.  :class:`ScenarioId` is the whole admissible input surface.

**A pace is a word, never a duration.**  Accepting a number of seconds would let
one request occupy a run slot for as long as it liked, and accepting an
expression would be worse.  Four words map to four bounded intervals, and the
mapping is checked at import.

**A state machine is a graph, not a convention.**  :data:`ALLOWED_TRANSITIONS`
is the graph; :func:`can_transition` is the only thing that reads it.  A run that
has completed, stopped, or failed has no outgoing edge, so "a terminal run cannot
restart" is a property of the data rather than a rule somebody remembered to
write in every branch.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = [
    "ALLOWED_TRANSITIONS",
    "MAX_STEP_INTERVAL_SECONDS",
    "PACE_INTERVAL_SECONDS",
    "TERMINAL_STATES",
    "ReplayPace",
    "ReplayState",
    "ScenarioId",
    "can_transition",
]


class ScenarioId(StrEnum):
    """The built-in synthetic scenarios, and the only ones that can be run.

    Every member names a *shape of authentication behaviour* that the detection
    system was built to reason about.  None of them is an attack: they are
    fabricated event streams describing activity that did not happen, replayed
    into a detector the operator is running themselves.
    """

    #: Ordinary, well-spaced successful sign-ins from a stable identity.
    NORMAL_ACTIVITY = "normal_activity"
    #: Repeated failures against one account from one source.
    BRUTE_FORCE = "brute_force"
    #: One source trying one credential attempt against many accounts.
    PASSWORD_SPRAYING = "password_spraying"
    #: Broad account fan-out from one source with mixed outcomes and varied
    #: clients.  No credential value, list, or reuse data is involved anywhere.
    CREDENTIAL_STUFFING = "credential_stuffing"
    #: A failure burst against one account followed by a success from a context
    #: the account has not used before.
    ACCOUNT_TAKEOVER = "account_takeover"
    #: Sustained machine-cadence authentication from one source.
    BOT_ACTIVITY = "bot_activity"
    #: Several of the above in one timeline, as an analyst would meet them.
    MIXED_ATTACK = "mixed_attack"


class ReplayState(StrEnum):
    """Where one replay run is in its lifecycle."""

    #: Accepted and registered.  Nothing has been emitted.
    CREATED = "created"
    #: Emitting events and scoring them.
    RUNNING = "running"
    #: Every event was emitted and scored.
    COMPLETED = "completed"
    #: Stopped by an explicit request before the scenario ran out.
    STOPPED = "stopped"
    #: Abandoned because a step could not be scored.
    FAILED = "failed"


#: The states a run never leaves.  A run in one of these emits no further
#: record, and its timeline is final.
TERMINAL_STATES: Final[frozenset[ReplayState]] = frozenset(
    {ReplayState.COMPLETED, ReplayState.STOPPED, ReplayState.FAILED}
)

#: Every transition the lifecycle permits, as an adjacency map.  A state absent
#: from a value set is a transition that cannot happen, including every edge out
#: of a terminal state: a completed run does not restart, and a new execution of
#: the same scenario is a *new run* with its own identity.
ALLOWED_TRANSITIONS: Final[dict[ReplayState, frozenset[ReplayState]]] = {
    ReplayState.CREATED: frozenset(
        {ReplayState.RUNNING, ReplayState.STOPPED, ReplayState.FAILED}
    ),
    ReplayState.RUNNING: frozenset(
        {ReplayState.COMPLETED, ReplayState.STOPPED, ReplayState.FAILED}
    ),
    ReplayState.COMPLETED: frozenset(),
    ReplayState.STOPPED: frozenset(),
    ReplayState.FAILED: frozenset(),
}


def can_transition(current: ReplayState, target: ReplayState) -> bool:
    """Return whether *current* may become *target*.

    A self-transition is not permitted.  It reads as harmless -- "stopping a
    stopped run" -- but the idempotence a caller wants there is *the request
    succeeding without changing anything*, which is a decision for the layer
    handling the request rather than an edge in the graph.
    """
    return target in ALLOWED_TRANSITIONS[current]


class ReplayPace(StrEnum):
    """How fast the presentation advances.  Never what the events *are*.

    A pace changes the wall-clock spacing between steps and nothing else.  The
    scenario's ``event_time`` values are fixed in the catalog, the window each
    step is scored against is fixed by the scenario's own order, and the frozen
    detection system is the same one either way -- so the same scenario at
    ``instant`` and at ``slow`` produces byte-identical verdicts.
    """

    #: No delay at all.  What the tests run at, and what an operator picks to
    #: see a whole scenario's outcome immediately.
    INSTANT = "instant"
    FAST = "fast"
    NORMAL = "normal"
    SLOW = "slow"


#: The hard ceiling on any pace's per-step delay.  A pace vocabulary whose
#: slowest member could be minutes would let one run hold a slot indefinitely.
MAX_STEP_INTERVAL_SECONDS: Final[float] = 2.0

#: The wall-clock delay between steps, per pace.  Chosen for a demonstration
#: someone is watching: ``fast`` is quick enough to feel live and slow enough to
#: read, ``slow`` is deliberate enough to narrate over.
PACE_INTERVAL_SECONDS: Final[dict[ReplayPace, float]] = {
    ReplayPace.INSTANT: 0.0,
    ReplayPace.FAST: 0.25,
    ReplayPace.NORMAL: 0.75,
    ReplayPace.SLOW: 1.75,
}


def _assert_the_pace_vocabulary_is_bounded_and_total() -> None:
    """Fail at import if a pace is unmapped, negative, or over the ceiling.

    The interval table is the only place a replay can acquire a delay, so a
    missing entry would be a ``KeyError`` at run time and an over-large one would
    be an unbounded run.  Both are cheaper to refuse here.
    """
    missing = sorted(
        str(pace) for pace in ReplayPace if pace not in PACE_INTERVAL_SECONDS
    )
    if missing:
        raise ValueError(f"pace(s) {missing} have no bounded interval")
    for pace, seconds in PACE_INTERVAL_SECONDS.items():
        if not 0.0 <= seconds <= MAX_STEP_INTERVAL_SECONDS:
            raise ValueError(
                f"pace {pace!s} maps to {seconds}s, outside "
                f"[0, {MAX_STEP_INTERVAL_SECONDS}]; a replay pace is a "
                f"presentation control and must stay bounded"
            )


_assert_the_pace_vocabulary_is_bounded_and_total()


def _assert_every_terminal_state_is_absorbing() -> None:
    """Fail at import if a terminal state ever acquires an outgoing edge.

    The property the whole store depends on: once a run is finished, no further
    record can be appended to it and no second execution can adopt its identity.
    """
    for state in TERMINAL_STATES:
        if ALLOWED_TRANSITIONS[state]:
            raise ValueError(
                f"terminal state {state!s} has outgoing transitions; a finished "
                f"run must not be able to resume under its own identity"
            )
    unreachable = sorted(
        str(state) for state in ReplayState if state not in ALLOWED_TRANSITIONS
    )
    if unreachable:
        raise ValueError(f"state(s) {unreachable} are missing from the transition map")


_assert_every_terminal_state_is_absorbing()
