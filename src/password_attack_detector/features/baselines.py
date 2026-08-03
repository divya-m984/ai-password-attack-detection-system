"""Behavioral baselines with explicit fit and transform separation.

A baseline summarises what "usual" looks like for a user or a source.  It is
fitted **only** from an approved reference interval -- in practice, the events
assigned to the training split -- and is then applied unchanged to every event.

Two properties make that guarantee mechanical rather than aspirational:

* :meth:`BehavioralBaselineModel.fit` takes an explicit ``permitted_event_ids``
  set and **raises** on any event outside it.  It never silently skips.  It
  also records a fingerprint of exactly what it consumed, which the leakage
  auditor recomputes independently from the split table.
* The fitted state is frozen, so :meth:`transform_one` structurally cannot
  mutate it.  Validation and test events cannot move a training baseline.

**Privacy.**  Fitted state is keyed by pseudonymous identifiers and holds
per-entity known-value sets, so it is sensitive operational metadata.  It is
written only to git-ignored artifact paths, with the pseudonym-bearing tables
in mode 0600 and a separate metadata-only ``baseline.json`` that reports and
CLI output read instead.  Real-data baselines require protected storage.

**A documented caveat.**  Transforming a training event uses a baseline that
saw that same event, so training rows carry mild in-sample optimism.  This
phase accepts that and records ``baseline_in_sample_for_train`` in the
artifact; leave-one-out baselines are a later modelling concern.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Self
from uuid import UUID

from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.data.serialization import compute_events_fingerprint
from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    BaselineFitError,
)
from password_attack_detector.features.config import BaselineConfig, FeatureConfig
from password_attack_detector.features.geospatial import haversine_km, valid_coordinates
from password_attack_detector.features.temporal import (
    MICROSECONDS_PER_SECOND,
    mean_std,
    to_microseconds,
)

__all__ = [
    "BASELINE_COLUMNS",
    "BASELINE_JSON",
    "BaselineArtifact",
    "BehavioralBaselineModel",
    "SourceBaseline",
    "UserBaseline",
]

BASELINE_SCHEMA_VERSION: str = "1.0.0"

BASELINE_JSON: str = "baseline.json"
_USER_TABLE: str = "user_baselines.parquet"
_SOURCE_TABLE: str = "source_baselines.parquet"

#: Restrictive permissions for the pseudonym-bearing tables.  Set explicitly
#: rather than relying on the process umask.
_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIR_MODE = 0o700

#: Every column this module contributes to a feature row.
BASELINE_COLUMNS: tuple[str, ...] = (
    "is_new_device_for_user",
    "is_new_source_for_user",
    "is_new_country_for_user",
    "is_new_application_for_user",
    "is_new_auth_method_for_user",
    "login_hour_deviation",
    "user_success_rate_deviation",
    "user_event_rate_ratio",
    "response_time_zscore",
    "source_user_fanout_ratio",
    "source_event_rate_ratio",
    "distance_from_user_baseline_centroid_km",
    "user_in_baseline",
    "source_in_baseline",
)

_HOURS = 24


@dataclass(frozen=True, slots=True)
class UserBaseline:
    """What usual behaviour looks like for one user.

    Frozen so ``transform`` cannot mutate it: purity is enforced by the type,
    not by convention.
    """

    event_count: int
    known_device_ids: frozenset[str]
    known_source_ids: frozenset[str]
    known_country_codes: frozenset[str]
    known_application_ids: frozenset[str]
    known_auth_methods: frozenset[str]
    hour_histogram: tuple[float, ...]
    success_rate: float | None
    event_rate_per_hour: float | None
    response_time_mean_ms: float | None
    response_time_std_ms: float | None
    interarrival_median_s: float | None
    interarrival_p90_s: float | None
    centroid_latitude: float | None
    centroid_longitude: float | None
    located_event_count: int
    truncated_sets: frozenset[str]


@dataclass(frozen=True, slots=True)
class SourceBaseline:
    """What usual behaviour looks like for one source."""

    event_count: int
    targeted_user_count: int
    success_rate: float | None
    event_rate_per_hour: float | None
    distinct_client_type_count: int
    distinct_user_agent_count: int
    client_type_entropy: float | None
    user_agent_entropy: float | None
    response_time_mean_ms: float | None
    response_time_std_ms: float | None


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    """Metadata describing a fitted baseline.

    This is the **only** part written to a world-readable file and the only
    part reports and CLI output are allowed to read: it contains no
    pseudonymous identifiers at all.
    """

    baseline_schema_version: str
    fitted_interval_start: str
    fitted_interval_end: str
    fitted_source_fingerprint: str
    config_fingerprint: str
    content_fingerprint: str
    user_count: int
    source_count: int
    total_fit_events: int
    truncated_set_count: int
    baseline_in_sample_for_train: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the artifact as a JSON-serialisable mapping."""
        return {
            "baseline_schema_version": self.baseline_schema_version,
            "fitted_interval_start": self.fitted_interval_start,
            "fitted_interval_end": self.fitted_interval_end,
            "fitted_source_fingerprint": self.fitted_source_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "user_count": self.user_count,
            "source_count": self.source_count,
            "total_fit_events": self.total_fit_events,
            "truncated_set_count": self.truncated_set_count,
            "baseline_in_sample_for_train": self.baseline_in_sample_for_train,
            "created_at": self.created_at,
        }


# --- accumulation helpers --------------------------------------------------


@dataclass(slots=True)
class _UserAccumulator:
    """Mutable tallies used while fitting one user's baseline."""

    events: int = 0
    successes: int = 0
    devices: Counter[str] = field(default_factory=Counter)
    sources: Counter[str] = field(default_factory=Counter)
    countries: Counter[str] = field(default_factory=Counter)
    applications: Counter[str] = field(default_factory=Counter)
    methods: Counter[str] = field(default_factory=Counter)
    hours: list[int] = field(default_factory=lambda: [0] * _HOURS)
    rt_values: list[int] = field(default_factory=list)
    stamps: list[int] = field(default_factory=list)
    latitudes: list[float] = field(default_factory=list)
    longitudes: list[float] = field(default_factory=list)


@dataclass(slots=True)
class _SourceAccumulator:
    """Mutable tallies used while fitting one source's baseline."""

    events: int = 0
    successes: int = 0
    users: set[str] = field(default_factory=set)
    client_types: Counter[str] = field(default_factory=Counter)
    user_agents: Counter[str] = field(default_factory=Counter)
    rt_values: list[int] = field(default_factory=list)


def _shannon_entropy(counts: Mapping[str, int]) -> float | None:
    """Return the Shannon entropy of a distribution, in bits.

    ``None`` for an empty distribution; ``0.0`` for a single observed value.
    """
    total = sum(counts.values())
    if total <= 0:
        return None
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _capped_set(
    counts: Mapping[str, int], *, max_size: int, min_occurrences: int
) -> tuple[frozenset[str], bool]:
    """Return the most frequent values, and whether the set was truncated.

    Capping matters: without it one high-volume entity could produce a
    ragged Parquet cell with tens of thousands of entries.  Ties are broken by
    value so the result is deterministic.
    """
    eligible = [
        (value, count) for value, count in counts.items() if count >= min_occurrences
    ]
    if len(eligible) <= max_size:
        return frozenset(value for value, _ in eligible), False
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return frozenset(value for value, _ in eligible[:max_size]), True


def _percentile(sorted_values: Sequence[float], fraction: float) -> float | None:
    """Return a percentile using nearest-rank, which needs no interpolation."""
    if not sorted_values:
        return None
    index = min(
        len(sorted_values) - 1,
        max(0, math.ceil(fraction * len(sorted_values)) - 1),
    )
    return sorted_values[index]


def _round(value: float | None) -> str | None:
    """Render a float as a fixed-precision string for fingerprinting.

    Fixed precision removes the last-bit differences that would otherwise make
    a fingerprint depend on accumulation order.
    """
    return None if value is None else f"{round(value, 9):.9f}"


class BehavioralBaselineModel:
    """Fits and applies user and source behavioral baselines."""

    def __init__(self, config: BaselineConfig) -> None:
        self._config = config
        self._users: dict[str, UserBaseline] = {}
        self._sources: dict[str, SourceBaseline] = {}
        self._artifact: BaselineArtifact | None = None

    # -- properties ---------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Return whether :meth:`fit` or :meth:`load` has populated this model."""
        return self._artifact is not None

    @property
    def config(self) -> BaselineConfig:
        """Return the configuration this baseline was built with."""
        return self._config

    @property
    def artifact(self) -> BaselineArtifact:
        """Return the fitted artifact metadata.

        Raises:
            BaselineFitError: if the model has not been fitted or loaded.
        """
        if self._artifact is None:
            raise BaselineFitError("Baseline has not been fitted")
        return self._artifact

    @property
    def user_count(self) -> int:
        """Return the number of users in the fitted baseline."""
        return len(self._users)

    @property
    def source_count(self) -> int:
        """Return the number of sources in the fitted baseline."""
        return len(self._sources)

    def has_user(self, user_id: str) -> bool:
        """Return whether *user_id* is present in the fitted baseline."""
        return user_id in self._users

    def has_source(self, source_id: str) -> bool:
        """Return whether *source_id* is present in the fitted baseline."""
        return source_id in self._sources

    # -- fitting ------------------------------------------------------------

    def fit(
        self,
        events: Sequence[AuthEvent],
        *,
        permitted_event_ids: frozenset[UUID],
        interval: tuple[datetime, datetime],
    ) -> Self:
        """Rebuild all baseline state from *events*.

        Every event must lie inside *interval* and appear in
        *permitted_event_ids*.  Violations raise rather than being skipped:
        silently dropping a disallowed event would hide exactly the mistake
        this parameter exists to catch.

        Raises:
            BaselineFitError: if any event is outside the interval or not
                permitted, or if the interval is empty or inverted.
        """
        start, end = interval
        if end <= start:
            raise BaselineFitError("Baseline interval must be non-empty and increasing")

        outside = 0
        not_permitted = 0
        for event in events:
            if not (start <= event.event_time <= end):
                outside += 1
            elif event.event_id not in permitted_event_ids:
                not_permitted += 1
        if outside or not_permitted:
            raise BaselineFitError(
                f"Baseline fit rejected: {outside} event(s) outside the fitted "
                f"interval and {not_permitted} event(s) not in the permitted "
                f"set. Fit only on approved reference data."
            )

        interval_hours = (end - start).total_seconds() / 3600.0
        users, sources = self._accumulate(events)
        self._users = {
            key: self._build_user(accumulator, interval_hours)
            for key, accumulator in sorted(users.items())
        }
        self._sources = {
            key: self._build_source(accumulator, interval_hours)
            for key, accumulator in sorted(sources.items())
        }

        truncated = sum(len(u.truncated_sets) for u in self._users.values())
        content = self._content_fingerprint(start, end, events)
        self._artifact = BaselineArtifact(
            baseline_schema_version=BASELINE_SCHEMA_VERSION,
            fitted_interval_start=start.astimezone(UTC).isoformat(),
            fitted_interval_end=end.astimezone(UTC).isoformat(),
            fitted_source_fingerprint=compute_events_fingerprint(events),
            config_fingerprint=self._config.fingerprint(),
            content_fingerprint=content,
            user_count=len(self._users),
            source_count=len(self._sources),
            total_fit_events=len(events),
            truncated_set_count=truncated,
            baseline_in_sample_for_train=True,
            created_at=datetime.now(UTC).isoformat(),
        )
        return self

    @staticmethod
    def _accumulate(
        events: Sequence[AuthEvent],
    ) -> tuple[dict[str, _UserAccumulator], dict[str, _SourceAccumulator]]:
        """Tally per-entity observations in a single pass."""
        from password_attack_detector.data.enums import AuthOutcome

        users: dict[str, _UserAccumulator] = {}
        sources: dict[str, _SourceAccumulator] = {}

        for event in events:
            succeeded = event.authentication_outcome is AuthOutcome.SUCCESS

            user = users.setdefault(event.user_id, _UserAccumulator())
            user.events += 1
            user.successes += int(succeeded)
            user.devices[event.device_id] += 1
            user.sources[event.source_id] += 1
            user.applications[event.application_id] += 1
            user.methods[str(event.authentication_method)] += 1
            if event.country_code is not None:
                user.countries[event.country_code] += 1
            user.hours[event.event_time.astimezone(UTC).hour] += 1
            if event.response_time_ms is not None:
                user.rt_values.append(event.response_time_ms)
            user.stamps.append(to_microseconds(event.event_time))
            if valid_coordinates(event.coarse_latitude, event.coarse_longitude):
                assert event.coarse_latitude is not None
                assert event.coarse_longitude is not None
                user.latitudes.append(event.coarse_latitude)
                user.longitudes.append(event.coarse_longitude)

            source = sources.setdefault(event.source_id, _SourceAccumulator())
            source.events += 1
            source.successes += int(succeeded)
            source.users.add(event.user_id)
            if event.client_type is not None:
                source.client_types[str(event.client_type)] += 1
            if event.user_agent_family is not None:
                source.user_agents[event.user_agent_family] += 1
            if event.response_time_ms is not None:
                source.rt_values.append(event.response_time_ms)

        return users, sources

    def _build_user(
        self, accumulator: _UserAccumulator, interval_hours: float
    ) -> UserBaseline:
        """Summarise one user's tallies into a frozen baseline."""
        config = self._config
        truncated: set[str] = set()

        def known(counts: Counter[str], label: str) -> frozenset[str]:
            values, was_truncated = _capped_set(
                counts,
                max_size=config.known_set_max_size,
                min_occurrences=config.known_set_min_occurrences,
            )
            if was_truncated:
                truncated.add(label)
            return values

        devices = known(accumulator.devices, "known_device_ids")
        source_ids = known(accumulator.sources, "known_source_ids")
        countries = known(accumulator.countries, "known_country_codes")
        applications = known(accumulator.applications, "known_application_ids")
        methods = known(accumulator.methods, "known_auth_methods")

        alpha = config.hour_histogram_alpha
        total = accumulator.events + alpha * _HOURS
        histogram = tuple(
            (count + alpha) / total if total > 0 else 1.0 / _HOURS
            for count in accumulator.hours
        )

        enough = accumulator.events >= config.min_events_per_user
        success_rate = (
            accumulator.successes / accumulator.events
            if enough and accumulator.events
            else None
        )
        event_rate = (
            accumulator.events / interval_hours
            if enough and interval_hours > 0
            else None
        )

        rt_mean, rt_std = (
            mean_std(
                len(accumulator.rt_values),
                sum(accumulator.rt_values),
                sum(v * v for v in accumulator.rt_values),
            )
            if len(accumulator.rt_values) >= config.response_time_min_events
            else (None, None)
        )

        stamps = sorted(accumulator.stamps)
        gaps = sorted((b - a) / MICROSECONDS_PER_SECOND for a, b in pairwise(stamps))

        located = len(accumulator.latitudes)
        centroid_lat = sum(accumulator.latitudes) / located if located else None
        centroid_lon = sum(accumulator.longitudes) / located if located else None

        return UserBaseline(
            event_count=accumulator.events,
            known_device_ids=devices,
            known_source_ids=source_ids,
            known_country_codes=countries,
            known_application_ids=applications,
            known_auth_methods=methods,
            hour_histogram=histogram,
            success_rate=success_rate,
            event_rate_per_hour=event_rate,
            response_time_mean_ms=rt_mean,
            response_time_std_ms=rt_std,
            interarrival_median_s=_percentile(gaps, 0.5),
            interarrival_p90_s=_percentile(gaps, 0.9),
            centroid_latitude=centroid_lat,
            centroid_longitude=centroid_lon,
            located_event_count=located,
            truncated_sets=frozenset(truncated),
        )

    def _build_source(
        self, accumulator: _SourceAccumulator, interval_hours: float
    ) -> SourceBaseline:
        """Summarise one source's tallies into a frozen baseline."""
        config = self._config
        enough = accumulator.events >= config.min_events_per_source

        rt_mean, rt_std = (
            mean_std(
                len(accumulator.rt_values),
                sum(accumulator.rt_values),
                sum(v * v for v in accumulator.rt_values),
            )
            if len(accumulator.rt_values) >= config.response_time_min_events
            else (None, None)
        )

        return SourceBaseline(
            event_count=accumulator.events,
            targeted_user_count=len(accumulator.users),
            success_rate=(
                accumulator.successes / accumulator.events
                if enough and accumulator.events
                else None
            ),
            event_rate_per_hour=(
                accumulator.events / interval_hours
                if enough and interval_hours > 0
                else None
            ),
            distinct_client_type_count=len(accumulator.client_types),
            distinct_user_agent_count=len(accumulator.user_agents),
            client_type_entropy=_shannon_entropy(accumulator.client_types),
            user_agent_entropy=_shannon_entropy(accumulator.user_agents),
            response_time_mean_ms=rt_mean,
            response_time_std_ms=rt_std,
        )

    # -- fingerprinting -----------------------------------------------------

    def _content_fingerprint(
        self, start: datetime, end: datetime, events: Sequence[AuthEvent]
    ) -> str:
        """Return a deterministic digest of the fitted state.

        Fingerprints the *logical* state rather than any serialised bytes, so
        it does not vary with pyarrow version or compression settings.  Sets
        are sorted, floats are rendered at fixed precision, and the creation
        timestamp is deliberately excluded -- the same events fitted twice
        must fingerprint identically.
        """
        payload = {
            "baseline_schema_version": BASELINE_SCHEMA_VERSION,
            "config_fingerprint": self._config.fingerprint(),
            "fitted_interval_start": start.astimezone(UTC).isoformat(),
            "fitted_interval_end": end.astimezone(UTC).isoformat(),
            "fitted_source_fingerprint": compute_events_fingerprint(events),
            "users": [
                self._canonical_user(key, self._users[key])
                for key in sorted(self._users)
            ],
            "sources": [
                self._canonical_source(key, self._sources[key])
                for key in sorted(self._sources)
            ],
        }
        canonical = json.dumps(
            payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _canonical_user(key: str, baseline: UserBaseline) -> dict[str, Any]:
        return {
            "user_id": key,
            "event_count": baseline.event_count,
            "known_device_ids": sorted(baseline.known_device_ids),
            "known_source_ids": sorted(baseline.known_source_ids),
            "known_country_codes": sorted(baseline.known_country_codes),
            "known_application_ids": sorted(baseline.known_application_ids),
            "known_auth_methods": sorted(baseline.known_auth_methods),
            "hour_histogram": [_round(v) for v in baseline.hour_histogram],
            "success_rate": _round(baseline.success_rate),
            "event_rate_per_hour": _round(baseline.event_rate_per_hour),
            "response_time_mean_ms": _round(baseline.response_time_mean_ms),
            "response_time_std_ms": _round(baseline.response_time_std_ms),
            "interarrival_median_s": _round(baseline.interarrival_median_s),
            "interarrival_p90_s": _round(baseline.interarrival_p90_s),
            "centroid_latitude": _round(baseline.centroid_latitude),
            "centroid_longitude": _round(baseline.centroid_longitude),
            "located_event_count": baseline.located_event_count,
            "truncated_sets": sorted(baseline.truncated_sets),
        }

    @staticmethod
    def _canonical_source(key: str, baseline: SourceBaseline) -> dict[str, Any]:
        return {
            "source_id": key,
            "event_count": baseline.event_count,
            "targeted_user_count": baseline.targeted_user_count,
            "success_rate": _round(baseline.success_rate),
            "event_rate_per_hour": _round(baseline.event_rate_per_hour),
            "distinct_client_type_count": baseline.distinct_client_type_count,
            "distinct_user_agent_count": baseline.distinct_user_agent_count,
            "client_type_entropy": _round(baseline.client_type_entropy),
            "user_agent_entropy": _round(baseline.user_agent_entropy),
            "response_time_mean_ms": _round(baseline.response_time_mean_ms),
            "response_time_std_ms": _round(baseline.response_time_std_ms),
        }

    # -- transformation -----------------------------------------------------

    def transform_one(
        self, event: AuthEvent, features: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return the baseline-derived columns for one event.

        Pure: reads fitted state and the already-computed point-in-time
        features, and mutates nothing.

        Entities absent from the fitted baseline receive **null** for every
        derived column, never a novelty verdict.  Reporting an unknown user's
        device as "new" would conflate "never seen this user" with "know this
        user and this device is new", which is precisely the encoding a model
        could exploit to recover the split boundary.
        """
        from password_attack_detector.data.enums import AuthOutcome

        window = self._config.rate_reference_window
        row: dict[str, Any] = dict.fromkeys(BASELINE_COLUMNS)

        user = self._users.get(event.user_id)
        source = self._sources.get(event.source_id)
        row["user_in_baseline"] = user is not None
        row["source_in_baseline"] = source is not None

        if user is not None:
            row["is_new_device_for_user"] = event.device_id not in user.known_device_ids
            row["is_new_source_for_user"] = event.source_id not in user.known_source_ids
            row["is_new_country_for_user"] = (
                None
                if event.country_code is None
                else event.country_code not in user.known_country_codes
            )
            row["is_new_application_for_user"] = (
                event.application_id not in user.known_application_ids
            )
            row["is_new_auth_method_for_user"] = (
                str(event.authentication_method) not in user.known_auth_methods
            )

            hour = event.event_time.astimezone(UTC).hour
            row["login_hour_deviation"] = 1.0 - user.hour_histogram[hour]

            if user.success_rate is not None:
                observed = float(event.authentication_outcome is AuthOutcome.SUCCESS)
                row["user_success_rate_deviation"] = observed - user.success_rate

            row["user_event_rate_ratio"] = self._rate_ratio(
                features, f"user_attempt_count__{window}", user.event_rate_per_hour
            )

            if (
                event.response_time_ms is not None
                and user.response_time_mean_ms is not None
                and user.response_time_std_ms
            ):
                raw = (
                    event.response_time_ms - user.response_time_mean_ms
                ) / user.response_time_std_ms
                limit = self._config.max_response_time_zscore
                row["response_time_zscore"] = max(-limit, min(limit, raw))

            if user.centroid_latitude is not None and valid_coordinates(
                event.coarse_latitude, event.coarse_longitude
            ):
                assert event.coarse_latitude is not None
                assert event.coarse_longitude is not None
                assert user.centroid_longitude is not None
                row["distance_from_user_baseline_centroid_km"] = haversine_km(
                    user.centroid_latitude,
                    user.centroid_longitude,
                    event.coarse_latitude,
                    event.coarse_longitude,
                )

        if source is not None:
            if source.targeted_user_count > 0 and features is not None:
                observed_users = features.get(f"source_unique_user_count__{window}")
                if observed_users is not None:
                    row["source_user_fanout_ratio"] = (
                        observed_users / source.targeted_user_count
                    )
            row["source_event_rate_ratio"] = self._rate_ratio(
                features,
                f"source_attempt_count__{window}",
                source.event_rate_per_hour,
            )

        return row

    def _rate_ratio(
        self,
        features: Mapping[str, Any] | None,
        column: str,
        baseline_rate: float | None,
    ) -> float | None:
        """Compare an observed windowed count against a baseline hourly rate."""
        if features is None or baseline_rate is None or baseline_rate <= 0.0:
            return None
        count = features.get(column)
        if count is None:
            return None
        hours = self._reference_window_hours()
        if hours <= 0.0:
            return None
        return float(count) / hours / baseline_rate

    def _reference_window_hours(self) -> float:
        from password_attack_detector.features.config import parse_duration

        return (
            parse_duration(self._config.rate_reference_window).total_seconds() / 3600.0
        )

    def transform(self, events: Sequence[AuthEvent]) -> list[dict[str, Any]]:
        """Apply :meth:`transform_one` to a sequence of events."""
        return [self.transform_one(event) for event in events]

    # -- persistence --------------------------------------------------------

    def save(self, directory: Path, *, overwrite: bool = False) -> Path:
        """Write the fitted baseline to *directory*.

        The pseudonym-bearing tables are written with restrictive permissions
        and are separated from the metadata-only ``baseline.json`` that
        reports and CLI output read.

        Raises:
            BaselineFitError: if the model is not fitted, or if artifacts
                already exist and *overwrite* is false.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        artifact = self.artifact
        directory = Path(directory)

        targets = [
            directory / name for name in (BASELINE_JSON, _USER_TABLE, _SOURCE_TABLE)
        ]
        if not overwrite and any(path.exists() for path in targets):
            raise BaselineFitError(
                "Baseline artifacts already exist; pass overwrite=True to replace"
            )

        directory.mkdir(parents=True, exist_ok=True)
        # Restrictive permissions are best-effort: some filesystems (notably
        # mounted Windows shares) do not support them, and failing the whole
        # fit over that would be worse than the weaker protection.
        with contextlib.suppress(OSError):  # pragma: no cover - platform dependent
            Path(directory).chmod(_PRIVATE_DIR_MODE)

        (directory / BASELINE_JSON).write_text(
            json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        pq.write_table(
            pa.table(self._user_table_data(), schema=_user_schema()),
            directory / _USER_TABLE,
        )
        pq.write_table(
            pa.table(self._source_table_data(), schema=_source_schema()),
            directory / _SOURCE_TABLE,
        )

        for name in (_USER_TABLE, _SOURCE_TABLE):
            with contextlib.suppress(OSError):  # pragma: no cover - platform dependent
                (directory / name).chmod(_PRIVATE_FILE_MODE)

        return directory / BASELINE_JSON

    def _user_table_data(self) -> dict[str, list[Any]]:
        keys = sorted(self._users)
        rows = [self._users[key] for key in keys]
        return {
            "user_id": keys,
            "event_count": [r.event_count for r in rows],
            "known_device_ids": [sorted(r.known_device_ids) for r in rows],
            "known_source_ids": [sorted(r.known_source_ids) for r in rows],
            "known_country_codes": [sorted(r.known_country_codes) for r in rows],
            "known_application_ids": [sorted(r.known_application_ids) for r in rows],
            "known_auth_methods": [sorted(r.known_auth_methods) for r in rows],
            "hour_histogram": [list(r.hour_histogram) for r in rows],
            "success_rate": [r.success_rate for r in rows],
            "event_rate_per_hour": [r.event_rate_per_hour for r in rows],
            "response_time_mean_ms": [r.response_time_mean_ms for r in rows],
            "response_time_std_ms": [r.response_time_std_ms for r in rows],
            "interarrival_median_s": [r.interarrival_median_s for r in rows],
            "interarrival_p90_s": [r.interarrival_p90_s for r in rows],
            "centroid_latitude": [r.centroid_latitude for r in rows],
            "centroid_longitude": [r.centroid_longitude for r in rows],
            "located_event_count": [r.located_event_count for r in rows],
            "truncated_sets": [sorted(r.truncated_sets) for r in rows],
        }

    def _source_table_data(self) -> dict[str, list[Any]]:
        keys = sorted(self._sources)
        rows = [self._sources[key] for key in keys]
        return {
            "source_id": keys,
            "event_count": [r.event_count for r in rows],
            "targeted_user_count": [r.targeted_user_count for r in rows],
            "success_rate": [r.success_rate for r in rows],
            "event_rate_per_hour": [r.event_rate_per_hour for r in rows],
            "distinct_client_type_count": [r.distinct_client_type_count for r in rows],
            "distinct_user_agent_count": [r.distinct_user_agent_count for r in rows],
            "client_type_entropy": [r.client_type_entropy for r in rows],
            "user_agent_entropy": [r.user_agent_entropy for r in rows],
            "response_time_mean_ms": [r.response_time_mean_ms for r in rows],
            "response_time_std_ms": [r.response_time_std_ms for r in rows],
        }

    @classmethod
    def load(cls, directory: Path, config: BaselineConfig | None = None) -> Self:
        """Load a baseline previously written by :meth:`save`.

        Raises:
            ArtifactNotFoundError: if the directory does not hold a baseline.
        """
        import pyarrow.parquet as pq

        directory = Path(directory)
        metadata_path = directory / BASELINE_JSON
        if not metadata_path.exists():
            raise ArtifactNotFoundError(
                "No baseline metadata found in the requested directory"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = cls(config if config is not None else BaselineConfig())

        user_table = pq.read_table(directory / _USER_TABLE).to_pylist()
        for record in user_table:
            model._users[record["user_id"]] = UserBaseline(
                event_count=record["event_count"],
                known_device_ids=frozenset(record["known_device_ids"]),
                known_source_ids=frozenset(record["known_source_ids"]),
                known_country_codes=frozenset(record["known_country_codes"]),
                known_application_ids=frozenset(record["known_application_ids"]),
                known_auth_methods=frozenset(record["known_auth_methods"]),
                hour_histogram=tuple(record["hour_histogram"]),
                success_rate=record["success_rate"],
                event_rate_per_hour=record["event_rate_per_hour"],
                response_time_mean_ms=record["response_time_mean_ms"],
                response_time_std_ms=record["response_time_std_ms"],
                interarrival_median_s=record["interarrival_median_s"],
                interarrival_p90_s=record["interarrival_p90_s"],
                centroid_latitude=record["centroid_latitude"],
                centroid_longitude=record["centroid_longitude"],
                located_event_count=record["located_event_count"],
                truncated_sets=frozenset(record["truncated_sets"]),
            )

        source_table = pq.read_table(directory / _SOURCE_TABLE).to_pylist()
        for record in source_table:
            model._sources[record["source_id"]] = SourceBaseline(
                event_count=record["event_count"],
                targeted_user_count=record["targeted_user_count"],
                success_rate=record["success_rate"],
                event_rate_per_hour=record["event_rate_per_hour"],
                distinct_client_type_count=record["distinct_client_type_count"],
                distinct_user_agent_count=record["distinct_user_agent_count"],
                client_type_entropy=record["client_type_entropy"],
                user_agent_entropy=record["user_agent_entropy"],
                response_time_mean_ms=record["response_time_mean_ms"],
                response_time_std_ms=record["response_time_std_ms"],
            )

        model._artifact = BaselineArtifact(**metadata)
        return model


def _user_schema() -> Any:
    import pyarrow as pa

    string_list = pa.list_(pa.string())
    return pa.schema(
        [
            pa.field("user_id", pa.string(), nullable=False),
            pa.field("event_count", pa.int64(), nullable=False),
            pa.field("known_device_ids", string_list, nullable=False),
            pa.field("known_source_ids", string_list, nullable=False),
            pa.field("known_country_codes", string_list, nullable=False),
            pa.field("known_application_ids", string_list, nullable=False),
            pa.field("known_auth_methods", string_list, nullable=False),
            pa.field("hour_histogram", pa.list_(pa.float64()), nullable=False),
            pa.field("success_rate", pa.float64(), nullable=True),
            pa.field("event_rate_per_hour", pa.float64(), nullable=True),
            pa.field("response_time_mean_ms", pa.float64(), nullable=True),
            pa.field("response_time_std_ms", pa.float64(), nullable=True),
            pa.field("interarrival_median_s", pa.float64(), nullable=True),
            pa.field("interarrival_p90_s", pa.float64(), nullable=True),
            pa.field("centroid_latitude", pa.float64(), nullable=True),
            pa.field("centroid_longitude", pa.float64(), nullable=True),
            pa.field("located_event_count", pa.int64(), nullable=False),
            pa.field("truncated_sets", string_list, nullable=False),
        ]
    )


def _source_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("event_count", pa.int64(), nullable=False),
            pa.field("targeted_user_count", pa.int64(), nullable=False),
            pa.field("success_rate", pa.float64(), nullable=True),
            pa.field("event_rate_per_hour", pa.float64(), nullable=True),
            pa.field("distinct_client_type_count", pa.int64(), nullable=False),
            pa.field("distinct_user_agent_count", pa.int64(), nullable=False),
            pa.field("client_type_entropy", pa.float64(), nullable=True),
            pa.field("user_agent_entropy", pa.float64(), nullable=True),
            pa.field("response_time_mean_ms", pa.float64(), nullable=True),
            pa.field("response_time_std_ms", pa.float64(), nullable=True),
        ]
    )


def fit_baseline(
    events: Sequence[AuthEvent],
    config: FeatureConfig,
    *,
    permitted_event_ids: frozenset[UUID],
    interval: tuple[datetime, datetime],
) -> BehavioralBaselineModel:
    """Convenience wrapper: fit a baseline from a full feature configuration."""
    return BehavioralBaselineModel(config.baseline).fit(
        events, permitted_event_ids=permitted_event_ids, interval=interval
    )
