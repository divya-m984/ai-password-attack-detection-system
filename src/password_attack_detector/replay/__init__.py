"""Safe synthetic live/replay demonstration for the detection service.

This package makes a detection system *watchable*.  It takes a reviewed,
deterministic scenario from its own catalog, emits it one event at a time into
the existing serving orchestration, and records what the frozen system said about
each step -- so an analyst can see rules fire, the model decide, and the frozen
hybrid fuse the two, on a timeline, in order, as it happens.

**Nothing here attacks anything.**  There is no credential, no credential list,
no password cracking, no external request, no network scan, and no login
endpoint.  The scenarios are fabricated event streams describing activity that
did not happen, involving entities that do not exist, replayed into a service the
operator is running themselves.  ``tests/integration/test_replay_security.py``
asserts each of those properties from outside the package, and this package's own
modules assert several of them at import.

**Nothing here scores anything.**  The layers are deliberately arranged so the
replay code *cannot*::

    scenarios.py  the reviewed catalog -- deterministic events, and nothing else
    enums.py      the closed vocabularies: scenario, state, pace
    schemas.py    the wire contract, embedding the serving layer's own verdicts
    store.py      bounded, process-local, non-persistent run storage
    engine.py     the state machine and the pace; the detector is *injected*
    service.py    the operations the API namespace is a shell over

The detector the engine calls is bound in
:mod:`password_attack_detector.api.services`, beside the ``detect_single`` it
wraps, so every replayed step goes through exactly the orchestration that
``POST /api/v1/detect`` goes through -- the same canonical events, the same
point-in-time feature engine, the same rules, the same frozen champion, the same
frozen fusion.

**Nothing here persists.**  Runs and timelines live in one process's memory,
bounded, and are gone when it restarts.  That limitation is stated in the API
documents, in ``docs/live-replay.md``, and on the dashboard page itself.
"""

from __future__ import annotations

__all__: list[str] = []
