"""Top-level synthetic dataset generation entry point.

``generate_dataset`` is the single public function.  It:

1. Seeds ``np.random.default_rng(config.seed)`` — no global RNG state.
2. Builds the entity population.
3. Dispatches each enabled scenario generator in a fixed order,
   passing a shared mutable counter that ensures globally unique event IDs.
4. Returns an immutable ``GenerationResult``.

Determinism guarantee
---------------------
The same ``SyntheticConfig`` (including ``seed``) always produces identical
events, labels, and ``config_fingerprint`` regardless of platform, Python
version, or wall-clock time.  The generator never reads ``datetime.now()``,
``time.time()``, ``os.urandom()``, or any other non-deterministic source.

Different seeds are expected to produce materially different output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.data.synthetic.campaigns import (
    generate_account_takeover_indicator,
    generate_bot_activity,
    generate_brute_force,
    generate_credential_stuffing,
    generate_distributed_brute_force,
    generate_impossible_travel,
    generate_normal,
    generate_novel_anomaly_holdout,
    generate_password_spraying,
)
from password_attack_detector.data.synthetic.config import SyntheticConfig
from password_attack_detector.data.synthetic.entities import build_entity_population

__all__ = [
    "GenerationResult",
    "generate_dataset",
]


@dataclass(frozen=True)
class GenerationResult:
    """Immutable container for a completed synthetic generation run.

    Attributes
    ----------
    events:
        Canonical authentication events; no ground-truth columns.
    labels:
        Ground-truth labels; one per event, joined by ``event_id``.
    config:
        The ``SyntheticConfig`` used for this run.
    config_fingerprint:
        SHA-256 hex fingerprint of ``config`` (see ``SyntheticConfig.fingerprint``).
    """

    events: tuple[AuthEvent, ...]
    labels: tuple[GroundTruthLabel, ...]
    config: SyntheticConfig
    config_fingerprint: str


def generate_dataset(config: SyntheticConfig) -> GenerationResult:
    """Generate a complete synthetic authentication-event dataset.

    Parameters
    ----------
    config:
        Fully validated generation configuration.

    Returns
    -------
    GenerationResult
        Immutable result containing events, labels, config, and fingerprint.

    Raises
    ------
    DataValidationError
        (Propagated from Pydantic validators inside scenario generators if
        an event fails schema constraints.)
    """
    rng = np.random.default_rng(config.seed)
    population = build_entity_population(config, rng)

    all_events: list[AuthEvent] = []
    all_labels: list[GroundTruthLabel] = []
    # Shared counter guarantees unique (seed, counter) pairs across all generators.
    counter: list[int] = [0]

    es = config.enabled_scenarios

    if es.normal:
        pairs = generate_normal(rng, config, population, counter)
        _extend(all_events, all_labels, pairs)

    if es.brute_force:
        pairs = generate_brute_force(rng, config, population, counter)
        _extend(all_events, all_labels, pairs)

    if es.password_spraying:
        pairs = generate_password_spraying(rng, config, population, counter)
        _extend(all_events, all_labels, pairs)

    if es.credential_stuffing:
        pairs = generate_credential_stuffing(rng, config, population, counter)
        _extend(all_events, all_labels, pairs)

    if es.distributed_brute_force:
        pairs = generate_distributed_brute_force(rng, config, population, counter)
        _extend(all_events, all_labels, pairs)

    if es.account_takeover_indicator:
        pairs = generate_account_takeover_indicator(rng, config, population, counter)
        _extend(all_events, all_labels, pairs)

    if es.impossible_travel:
        pairs = generate_impossible_travel(rng, config, population, counter)
        _extend(all_events, all_labels, pairs)

    if es.bot_activity:
        pairs = generate_bot_activity(rng, config, population, counter)
        _extend(all_events, all_labels, pairs)

    if es.novel_anomaly_holdout:
        pairs = generate_novel_anomaly_holdout(rng, config, population, counter)
        _extend(all_events, all_labels, pairs)

    return GenerationResult(
        events=tuple(all_events),
        labels=tuple(all_labels),
        config=config,
        config_fingerprint=config.fingerprint(),
    )


def _extend(
    events: list[AuthEvent],
    labels: list[GroundTruthLabel],
    pairs: list[tuple[AuthEvent, GroundTruthLabel]],
) -> None:
    for event, label in pairs:
        events.append(event)
        labels.append(label)
