# Temporal Semantics

## Overview

Every feature in this project is computed at a fixed point in time called the
**anchor**: the canonical authentication event the feature snapshot describes.
This document states exactly what an anchor is allowed to see. It is the most
important contract in the feature layer, because a violation of it is silent —
the resulting model looks better, not worse.

## Detection happens after the event

The system scores an authentication event **after** it has completed and been
recorded. It does not intercept an in-flight login. That timing is what makes
the anchor's own recorded fields legitimate inputs: by the time detection runs,
the outcome, method, MFA result, client type, response time, and country are
already facts in the log.

Those fields are published as `current_*` columns and are classified
`current_event_context` in the catalog.

## The half-open history window

For an anchor at time `t` and a configured window `w`, every historical
aggregate is computed from events in:

    [t - w, t)

- The left edge is **inclusive**: an event at exactly `t - w` is in the window.
- The right edge is **exclusive**: the anchor itself is never in its own
  history.

## Simultaneous events are mutually invisible

Events sharing the anchor's exact `event_time` never enter the anchor's
history, and the anchor never enters theirs. A group of simultaneous events —
a *timestamp block* — all see byte-identical history.

This avoids an ambiguity rather than solving one. Two events recorded at the
same microsecond have no true ordering; any rule that let one contribute to the
other would be inventing one.

## Canonical ordering

Events are processed in:

    event_time ascending, then event_id ascending

The `event_id` tie-break governs **output row order only**. It never affects
state content, so it cannot let simultaneous events observe one another.

## How the guarantee is enforced

The engine does not compare timestamps to decide what to include. It groups
events into timestamp blocks and runs two strictly separated phases per block:

1. **Emit** — every event in the block produces its row by *reading* state.
2. **Ingest** — only after every row in the block exists does any event in the
   block *write* to state.

Because blocks are processed in ascending time order, the invariant "state
contains only events strictly earlier than this block" holds at the top of
every iteration. Anchor exclusion and same-timestamp exclusion both follow from
that structure; neither is a special case anyone has to remember.

## The invariance property

The contract implies a property that can be tested directly:

> Adding, modifying, or removing any event at a time later than `t` must never
> change any feature for an anchor at or before `t`.

`tests/unit/features/test_engine.py` asserts this by building features twice —
once on a stream, once on a mutated stream — and comparing rows for exact
equality. The leakage auditor performs the same check as
`NO_FUTURE_CONTRIBUTION` and `NO_SAME_TIMESTAMP_CONTRIBUTION`.

## Prior-only sequence features

Sequence features (`prior_*`, `previous_*`, `seconds_since_*`) are maintained
in per-entity state updated **only during the ingest phase**. A field named
`prior_consecutive_user_failures` therefore cannot observe the anchor's own
outcome, even though that outcome is available as current-event context in the
same row.

`consecutive_failures` counts an unbroken run of `failure` outcomes; any other
outcome, including `blocked`, ends the run. `failures_since_success` resets
only on `success`.

## Determinism

Feature values are computed with exact integer arithmetic: response times are
integers, and event times are integer microseconds, so sums and sums-of-squares
accumulate in Python's arbitrary-precision `int` and convert to float only at
read time.

This is not primarily about numerical stability. Floating-point addition is not
associative, so an incremental accumulator and a recompute-from-scratch would
disagree in the last bits, and the invariance tests above could not assert
exact equality. Integer accumulators make the engine bit-for-bit reproducible.

Consequences that are tested:

- Shuffling the input rows does not change any output value.
- Running the same engine twice produces identical rows.
- The engine agrees exactly with a naive `O(n^2)` reference implementation.

## Known limitations

- Within a single timestamp block, events belonging to the *same* entity are
  folded into sequence state in `event_id` order. That order is deterministic
  and stable, but arbitrary: simultaneous events for one entity have no true
  ordering. It is observable only when one entity records two or more events at
  the identical microsecond.
- The invariance property holds for the engine given a fixed baseline and fixed
  split boundaries. Under fraction-derived boundaries, appending events moves
  the boundary, which moves the training set and therefore the fitted baseline;
  baseline-derived columns for early rows can then legitimately change. Pin the
  boundaries and the baseline when the property is being tested.
- All calculations use UTC. A user's local timezone is never inferred from
  their country: a country can span several offsets, and no trusted per-user
  timezone source exists in the canonical schema.
