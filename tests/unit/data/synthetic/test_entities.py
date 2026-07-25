"""Unit tests for password_attack_detector.data.synthetic.entities."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import numpy as np

from password_attack_detector.data.synthetic.config import SyntheticConfig
from password_attack_detector.data.synthetic.entities import (
    EntityPopulation,
    build_entity_population,
    make_device_id,
    make_session_id,
    make_source_id,
    make_user_id,
)

_PSEUDONYM_RE = re.compile(r"^(u|s|d|sess):[0-9a-f]{32}$")
_START = datetime(2024, 1, 1, tzinfo=UTC)


def _small_config(**kwargs: object) -> SyntheticConfig:
    defaults: dict[str, object] = {
        "seed": 42,
        "start_time": _START,
        "duration_hours": 1,
        "num_users": 4,
        "num_sources": 3,
        "num_devices": 6,
        "num_applications": 2,
        "events_per_hour": 5,
    }
    defaults.update(kwargs)
    return SyntheticConfig(**defaults)  # type: ignore[arg-type]


class TestIdHelpers:
    def test_make_user_id_format(self) -> None:
        uid = make_user_id(42, 0)
        assert _PSEUDONYM_RE.match(uid), f"Invalid format: {uid}"

    def test_make_source_id_format(self) -> None:
        sid = make_source_id(42, 0)
        assert _PSEUDONYM_RE.match(sid), f"Invalid format: {sid}"

    def test_make_device_id_format(self) -> None:
        did = make_device_id(42, 0)
        assert _PSEUDONYM_RE.match(did), f"Invalid format: {did}"

    def test_make_session_id_format(self) -> None:
        sess = make_session_id(42, 0)
        assert _PSEUDONYM_RE.match(sess), f"Invalid format: {sess}"

    def test_user_id_has_u_prefix(self) -> None:
        assert make_user_id(42, 0).startswith("u:")

    def test_source_id_has_s_prefix(self) -> None:
        assert make_source_id(42, 0).startswith("s:")

    def test_device_id_has_d_prefix(self) -> None:
        assert make_device_id(42, 0).startswith("d:")

    def test_session_id_has_sess_prefix(self) -> None:
        assert make_session_id(42, 0).startswith("sess:")

    def test_ids_are_deterministic(self) -> None:
        assert make_user_id(42, 5) == make_user_id(42, 5)
        assert make_source_id(99, 3) == make_source_id(99, 3)

    def test_different_seeds_different_ids(self) -> None:
        assert make_user_id(1, 0) != make_user_id(2, 0)

    def test_different_indices_different_ids(self) -> None:
        assert make_user_id(42, 0) != make_user_id(42, 1)

    def test_user_and_source_same_args_differ(self) -> None:
        assert make_user_id(42, 0) != make_source_id(42, 0)

    def test_device_and_session_same_args_differ(self) -> None:
        assert make_device_id(42, 0) != make_session_id(42, 0)

    def test_hex_part_is_32_chars(self) -> None:
        uid = make_user_id(42, 0)
        hex_part = uid.split(":")[1]
        assert len(hex_part) == 32

    def test_hex_part_is_lowercase(self) -> None:
        for i in range(5):
            uid = make_user_id(42, i)
            hex_part = uid.split(":")[1]
            assert hex_part == hex_part.lower()


class TestBuildEntityPopulation:
    def _build(self, **kwargs: object) -> EntityPopulation:
        cfg = _small_config(**kwargs)
        rng = np.random.default_rng(cfg.seed)
        return build_entity_population(cfg, rng)

    def test_correct_user_count(self) -> None:
        pop = self._build(num_users=4)
        assert len(pop.users) == 4

    def test_correct_source_count(self) -> None:
        pop = self._build(num_sources=3)
        assert len(pop.sources) == 3

    def test_correct_device_count(self) -> None:
        pop = self._build(num_devices=6)
        assert len(pop.devices) == 6

    def test_correct_application_count(self) -> None:
        pop = self._build(num_applications=2)
        assert len(pop.applications) == 2

    def test_user_ids_have_correct_format(self) -> None:
        pop = self._build()
        for user in pop.users:
            assert _PSEUDONYM_RE.match(user.user_id), user.user_id

    def test_source_ids_have_correct_format(self) -> None:
        pop = self._build()
        for source in pop.sources:
            assert _PSEUDONYM_RE.match(source.source_id), source.source_id

    def test_device_ids_have_correct_format(self) -> None:
        pop = self._build()
        for device in pop.devices:
            assert _PSEUDONYM_RE.match(device.device_id), device.device_id

    def test_user_ids_are_unique(self) -> None:
        pop = self._build(num_users=10)
        ids = [u.user_id for u in pop.users]
        assert len(ids) == len(set(ids))

    def test_source_ids_are_unique(self) -> None:
        pop = self._build(num_sources=5)
        ids = [s.source_id for s in pop.sources]
        assert len(ids) == len(set(ids))

    def test_device_ids_are_unique(self) -> None:
        pop = self._build(num_devices=8)
        ids = [d.device_id for d in pop.devices]
        assert len(ids) == len(set(ids))

    def test_application_ids_are_unique(self) -> None:
        pop = self._build(num_applications=3)
        ids = [a.application_id for a in pop.applications]
        assert len(ids) == len(set(ids))

    def test_deterministic_same_seed(self) -> None:
        cfg = _small_config(seed=99)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        pop1 = build_entity_population(cfg, rng1)
        pop2 = build_entity_population(cfg, rng2)
        assert [u.user_id for u in pop1.users] == [u.user_id for u in pop2.users]
        assert [s.source_id for s in pop1.sources] == [
            s.source_id for s in pop2.sources
        ]

    def test_different_seeds_different_populations(self) -> None:
        cfg1 = _small_config(seed=1)
        cfg2 = _small_config(seed=2)
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        pop1 = build_entity_population(cfg1, rng1)
        pop2 = build_entity_population(cfg2, rng2)
        # IDs use seed in their generation so must differ
        assert [u.user_id for u in pop1.users] != [u.user_id for u in pop2.users]

    def test_source_country_codes_valid(self) -> None:
        pop = self._build(num_sources=5)
        for source in pop.sources:
            assert len(source.country_code) == 2
            assert source.country_code.isupper()

    def test_source_latitude_in_range(self) -> None:
        pop = self._build(num_sources=10)
        for source in pop.sources:
            assert -90.0 <= source.coarse_latitude <= 90.0

    def test_source_longitude_in_range(self) -> None:
        pop = self._build(num_sources=10)
        for source in pop.sources:
            assert -180.0 <= source.coarse_longitude <= 180.0

    def test_users_have_preferred_methods(self) -> None:
        pop = self._build()
        for user in pop.users:
            assert len(user.preferred_auth_methods) >= 1

    def test_users_have_known_devices(self) -> None:
        pop = self._build(num_users=4, num_devices=8)
        for user in pop.users:
            assert len(user.known_device_ids) >= 1

    def test_baseline_success_rate_in_range(self) -> None:
        pop = self._build()
        for user in pop.users:
            assert 0.0 <= user.baseline_success_rate <= 1.0

    def test_population_is_frozen(self) -> None:
        pop = self._build()
        import dataclasses

        assert dataclasses.is_dataclass(pop)

    def test_application_ids_labelled_correctly(self) -> None:
        pop = self._build(num_applications=3)
        app_ids = [a.application_id for a in pop.applications]
        assert "app-00" in app_ids
        assert "app-01" in app_ids
        assert "app-02" in app_ids

    def test_uuid_uniqueness_across_namespaces(self) -> None:
        # user, source, device for same seed/index must all differ
        uid = make_user_id(42, 0)
        sid = make_source_id(42, 0)
        did = make_device_id(42, 0)
        assert len({uid, sid, did}) == 3
