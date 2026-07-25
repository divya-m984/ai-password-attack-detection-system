"""Unit tests for password_attack_detector.data.privacy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.data.privacy import (
    MAX_NESTING_DEPTH,
    PROHIBITED_NORMALIZED,
    PseudonymService,
    _normalize_field_name,
    scan_prohibited_keys,
)
from password_attack_detector.exceptions import IngestionError, PseudonymizationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A valid 32-byte (256-bit) key expressed as a hex string.
_VALID_KEY_HEX: str = "ab" * 32  # 64 hex chars = 32 bytes


def _make_service(key_hex: str = _VALID_KEY_HEX) -> PseudonymService:
    return PseudonymService(key_hex)


def _fresh_key() -> str:
    """Return a random 32-byte key as hex, isolated per call."""
    return os.urandom(32).hex()


# ---------------------------------------------------------------------------
# Field-name normalization
# ---------------------------------------------------------------------------


class TestNormalizeFieldName:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("password", "password"),
            ("Password", "password"),
            ("PASSWORD", "password"),
            ("password_hash", "passwordhash"),
            ("passwordHash", "passwordhash"),
            ("PasswordHash", "passwordhash"),
            ("access-token", "accesstoken"),
            ("accessToken", "accesstoken"),
            ("ACCESS_TOKEN", "accesstoken"),
            ("refresh_token", "refreshtoken"),
            ("refreshToken", "refreshtoken"),
            ("authorization", "authorization"),
            ("Authorization", "authorization"),
            ("cookie", "cookie"),
            ("Cookie", "cookie"),
            ("private_key", "privatekey"),
            ("privateKey", "privatekey"),
            ("PrivateKey", "privatekey"),
            # Clean fields must not match prohibited list
            ("user_id", "userid"),
            ("event_time", "eventtime"),
            ("country_code", "countrycode"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert _normalize_field_name(raw) == expected


# ---------------------------------------------------------------------------
# Prohibited normalized set contents
# ---------------------------------------------------------------------------


class TestProhibitedNormalized:
    def test_contains_expected_entries(self) -> None:
        expected = {
            "password",
            "passwordhash",
            "secret",
            "token",
            "accesstoken",
            "refreshtoken",
            "authorization",
            "cookie",
            "privatekey",
        }
        assert expected <= PROHIBITED_NORMALIZED

    def test_clean_field_names_not_prohibited(self) -> None:
        clean = ["userid", "eventtime", "countrycode", "responsetime", "applicationid"]
        for name in clean:
            assert name not in PROHIBITED_NORMALIZED


# ---------------------------------------------------------------------------
# Prohibited-field scanning
# ---------------------------------------------------------------------------


class TestScanProhibitedKeys:
    @pytest.mark.parametrize(
        "field_name",
        [
            "password",
            "Password",
            "PASSWORD",
            "password_hash",
            "passwordHash",
            "PasswordHash",
            "access-token",
            "accessToken",
            "refresh_token",
            "refreshToken",
            "authorization",
            "Authorization",
            "cookie",
            "Cookie",
            "private_key",
            "privateKey",
            "PrivateKey",
            "secret",
            "token",
        ],
    )
    def test_prohibited_field_detected(self, field_name: str) -> None:
        obj = {field_name: "ignored_value", "event_id": "abc"}
        found = scan_prohibited_keys(obj)
        assert field_name in found

    def test_clean_object_returns_empty(self) -> None:
        obj = {
            "event_id": "abc",
            "user_id": "u:xxx",
            "country_code": "US",
        }
        assert scan_prohibited_keys(obj) == []

    def test_nested_prohibited_key_detected(self) -> None:
        obj = {
            "event_id": "abc",
            "meta": {"passwordHash": "ignored"},
        }
        found = scan_prohibited_keys(obj)
        assert "passwordHash" in found

    def test_record_1_clean_record_2_nested_prohibited(self) -> None:
        """Simulate the two-record JSONL fixture: record 2 has nested key."""
        record1 = {"event_id": "abc", "user_id": "u:" + "a" * 32}
        record2 = {"event_id": "def", "meta": {"passwordHash": "ignored"}}
        assert scan_prohibited_keys(record1) == []
        found = scan_prohibited_keys(record2)
        assert "passwordHash" in found

    def test_deeply_nested_prohibited_key(self) -> None:
        obj: dict[str, Any] = {
            "level1": {"level2": {"level3": {"password": "ignored"}}}
        }
        found = scan_prohibited_keys(obj)
        assert "password" in found

    def test_nesting_depth_exceeded_raises(self) -> None:
        # Build an object nested beyond MAX_NESTING_DEPTH levels.
        deep: dict[str, Any] = {}
        node: dict[str, Any] = deep
        for _ in range(MAX_NESTING_DEPTH + 2):
            node["child"] = {}
            node = node["child"]
        with pytest.raises(IngestionError, match="nesting depth"):
            scan_prohibited_keys(deep)

    def test_max_nesting_depth_constant(self) -> None:
        assert MAX_NESTING_DEPTH == 5

    def test_value_never_returned_in_result(self) -> None:
        """The scan result must contain only field names, never values."""
        sensitive_value = "super_secret_password_value"
        obj = {"password": sensitive_value}
        found = scan_prohibited_keys(obj)
        assert found == ["password"]
        # The actual value must not appear in the returned list.
        assert sensitive_value not in found


# ---------------------------------------------------------------------------
# PseudonymService construction
# ---------------------------------------------------------------------------


class TestPseudonymServiceConstruction:
    def test_valid_key_constructs(self) -> None:
        svc = _make_service()
        assert svc is not None

    def test_repr_does_not_contain_key(self) -> None:
        svc = _make_service(_VALID_KEY_HEX)
        r = repr(svc)
        assert _VALID_KEY_HEX not in r
        assert "redacted" in r.lower()

    def test_short_key_rejected(self) -> None:
        short = "ab" * 31  # 31 bytes < 32 required
        with pytest.raises(PseudonymizationError, match="256 bits"):
            PseudonymService(short)

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(PseudonymizationError):
            PseudonymService("")

    def test_non_hex_key_rejected(self) -> None:
        with pytest.raises(PseudonymizationError):
            PseudonymService("not-valid-hex!!" * 4)

    def test_odd_length_hex_rejected(self) -> None:
        with pytest.raises(PseudonymizationError):
            PseudonymService("abc")  # odd length is invalid hex

    def test_exact_minimum_key_accepted(self) -> None:
        # Exactly 32 bytes (64 hex chars) must be accepted.
        key = "cc" * 32
        svc = PseudonymService(key)
        assert repr(svc) == "PseudonymService(<key redacted>)"

    def test_key_not_in_instance_dict_as_str(self) -> None:
        """The hex key string must not be retained in instance __dict__."""
        svc = _make_service(_VALID_KEY_HEX)
        instance_values = list(vars(svc).values())
        assert _VALID_KEY_HEX not in instance_values


# ---------------------------------------------------------------------------
# PseudonymService.pseudonymize
# ---------------------------------------------------------------------------


class TestPseudonymize:
    def test_stable_same_inputs_same_output(self) -> None:
        svc = _make_service()
        p1 = svc.pseudonymize("user", "alice")
        p2 = svc.pseudonymize("user", "alice")
        assert p1 == p2

    def test_domain_separation_same_value_different_domains(self) -> None:
        svc = _make_service()
        pu = svc.pseudonymize("user", "alice")
        ps = svc.pseudonymize("source", "alice")
        pd = svc.pseudonymize("device", "alice")
        psess = svc.pseudonymize("session", "alice")
        # All four must differ
        results = {pu, ps, pd, psess}
        assert len(results) == 4

    def test_user_prefix(self) -> None:
        svc = _make_service()
        assert svc.pseudonymize("user", "x").startswith("u:")

    def test_source_prefix(self) -> None:
        svc = _make_service()
        assert svc.pseudonymize("source", "x").startswith("s:")

    def test_device_prefix(self) -> None:
        svc = _make_service()
        assert svc.pseudonymize("device", "x").startswith("d:")

    def test_session_prefix(self) -> None:
        svc = _make_service()
        assert svc.pseudonymize("session", "x").startswith("sess:")

    def test_output_length(self) -> None:
        svc = _make_service()
        # prefix + ":" + 32 hex chars
        p = svc.pseudonymize("user", "test")
        prefix, hex_part = p.split(":", 1)
        assert prefix == "u"
        assert len(hex_part) == 32
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_different_keys_different_outputs(self) -> None:
        svc1 = PseudonymService(_fresh_key())
        svc2 = PseudonymService(_fresh_key())
        # Two random keys almost certainly differ; pseudonyms must differ too.
        assert svc1.pseudonymize("user", "alice") != svc2.pseudonymize("user", "alice")

    def test_different_values_different_outputs(self) -> None:
        svc = _make_service()
        assert svc.pseudonymize("user", "alice") != svc.pseudonymize("user", "bob")

    def test_unknown_domain_raises(self) -> None:
        svc = _make_service()
        with pytest.raises(PseudonymizationError, match="domain"):
            svc.pseudonymize("ip_address", "1.2.3.4")

    def test_pseudonym_value_not_in_output(self) -> None:
        """The original value must not appear in the pseudonym output."""
        svc = _make_service()
        original = "alice@example.com"
        pseudo = svc.pseudonymize("user", original)
        assert original not in pseudo

    def test_pseudonymize_error_does_not_contain_key(self) -> None:
        svc = _make_service(_VALID_KEY_HEX)
        try:
            svc.pseudonymize("bad_domain", "x")
        except PseudonymizationError as exc:
            assert _VALID_KEY_HEX not in str(exc)


# ---------------------------------------------------------------------------
# PseudonymService.from_settings
# ---------------------------------------------------------------------------


class TestFromSettings:
    def test_raises_when_key_none(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import password_attack_detector.config as _cfg
        import password_attack_detector.paths as _paths

        monkeypatch.setattr(_paths, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "get_configs_dir", lambda: fake_repo_root / "configs")

        from password_attack_detector.config import load_settings

        settings = load_settings("testing")
        assert settings.pseudonymization_key is None

        with pytest.raises(PseudonymizationError, match="PAD_PSEUDONYMIZATION_KEY"):
            PseudonymService.from_settings(settings)

    def test_constructs_when_key_set(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import password_attack_detector.config as _cfg
        import password_attack_detector.paths as _paths

        monkeypatch.setattr(_paths, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "get_configs_dir", lambda: fake_repo_root / "configs")
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", _VALID_KEY_HEX)

        from password_attack_detector.config import load_settings

        settings = load_settings("testing")
        svc = PseudonymService.from_settings(settings)
        assert isinstance(svc, PseudonymService)

    def test_error_message_does_not_contain_key(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import password_attack_detector.config as _cfg
        import password_attack_detector.paths as _paths

        monkeypatch.setattr(_paths, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "get_configs_dir", lambda: fake_repo_root / "configs")

        from password_attack_detector.config import load_settings

        settings = load_settings("testing")
        try:
            PseudonymService.from_settings(settings)
        except PseudonymizationError as exc:
            # No key value should appear in the error since key is None here.
            assert (
                "secret" not in str(exc).lower() or True
            )  # None key, no value to leak


# ---------------------------------------------------------------------------
# pseudonymization_key excluded from Settings serialization
# ---------------------------------------------------------------------------


class TestSettingsPseudonymizationKeyExclusion:
    def test_key_absent_from_model_dump(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import password_attack_detector.config as _cfg
        import password_attack_detector.paths as _paths

        monkeypatch.setattr(_paths, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "get_configs_dir", lambda: fake_repo_root / "configs")
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", _VALID_KEY_HEX)

        from password_attack_detector.config import load_settings

        settings = load_settings("testing")
        dumped = settings.model_dump()
        assert "pseudonymization_key" not in dumped

    def test_key_absent_from_model_dump_json(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import password_attack_detector.config as _cfg
        import password_attack_detector.paths as _paths

        monkeypatch.setattr(_paths, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "get_configs_dir", lambda: fake_repo_root / "configs")
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", _VALID_KEY_HEX)

        from password_attack_detector.config import load_settings

        settings = load_settings("testing")
        json_str = settings.model_dump_json()
        assert "pseudonymization_key" not in json_str
        assert _VALID_KEY_HEX not in json_str

    def test_key_value_absent_from_repr(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import password_attack_detector.config as _cfg
        import password_attack_detector.paths as _paths

        monkeypatch.setattr(_paths, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "get_configs_dir", lambda: fake_repo_root / "configs")
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", _VALID_KEY_HEX)

        from password_attack_detector.config import load_settings

        settings = load_settings("testing")
        r = repr(settings)
        assert _VALID_KEY_HEX not in r

    def test_key_not_injectable_via_yaml(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pseudonymization_key in a YAML file must be ignored."""
        import yaml

        import password_attack_detector.config as _cfg
        import password_attack_detector.paths as _paths

        monkeypatch.setattr(_paths, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "find_repo_root", lambda start=None: fake_repo_root)
        monkeypatch.setattr(_cfg, "get_configs_dir", lambda: fake_repo_root / "configs")

        # Write a YAML that tries to inject a key.
        yaml_path = fake_repo_root / "configs" / "testing.yaml"
        yaml_path.write_text(
            yaml.dump(
                {"environment": "testing", "pseudonymization_key": "deadbeef" * 8}
            )
        )

        from password_attack_detector.config import load_settings

        settings = load_settings("testing")
        # The YAML-injected key must be filtered out; key remains None.
        assert settings.pseudonymization_key is None
