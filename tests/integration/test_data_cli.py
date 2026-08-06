"""Integration tests for the ``data`` CLI command group (Milestone 12).

All tests run against the installed Typer app via ``CliRunner``.
The development-size synthetic configuration is never used here.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml
from typer.testing import CliRunner

from password_attack_detector.cli import app

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

runner = CliRunner()


def _tiny_cfg() -> dict[str, Any]:
    """Return a minimal valid synthetic configuration dict."""
    return {
        "generator_version": "1.0.0",
        "seed": 42,
        "start_time": "2024-01-01T00:00:00+00:00",
        "duration_hours": 1,
        "num_users": 3,
        "num_sources": 2,
        "num_devices": 4,
        "num_applications": 1,
        "events_per_hour": 5,
        "enabled_scenarios": {
            "normal": True,
            "brute_force": False,
            "password_spraying": False,
            "credential_stuffing": False,
            "distributed_brute_force": False,
            "account_takeover_indicator": False,
            "impossible_travel": False,
            "bot_activity": False,
            "novel_anomaly_holdout": False,
        },
        "campaign_parameters": {
            "brute_force": {"attempts_per_campaign": 5, "num_campaigns": 1},
            "password_spraying": {"passwords_per_round": 3, "num_campaigns": 1},
            "credential_stuffing": {"credentials_per_batch": 5, "num_campaigns": 1},
            "distributed_brute_force": {
                "attempts_per_source": 3,
                "num_sources": 2,
                "num_campaigns": 1,
            },
            "account_takeover_indicator": {"num_campaigns": 1},
            "impossible_travel": {"num_campaigns": 1},
            "bot_activity": {"events_per_campaign": 5, "num_campaigns": 1},
            "novel_anomaly_holdout": {"num_campaigns": 1},
        },
    }


@pytest.fixture()
def config_yaml(tmp_path: Path) -> Path:
    """Write a tiny synthetic YAML config and return its path."""
    p = tmp_path / "synthetic-tiny.yaml"
    p.write_text(yaml.dump(_tiny_cfg()), encoding="utf-8")
    return p


@pytest.fixture()
def generated_dir(tmp_path: Path, config_yaml: Path) -> Path:
    """Generate a small dataset and return the output directory."""
    out = tmp_path / "ds"
    result = runner.invoke(
        app, ["data", "generate", str(config_yaml), "--output-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    return out


def _make_canonical_event(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid canonical-event dict for CSV/JSONL tests."""
    base: dict[str, Any] = {
        "schema_version": "1.0.0",
        "event_id": str(uuid4()),
        "event_time": "2024-01-01T00:00:00+00:00",
        "user_id": "u:aabbccdd",
        "source_id": "s:aabbccdd",
        "device_id": "d:aabbccdd",
        "session_id": "sess:aabbccdd",
        "application_id": "app-1",
        "authentication_method": "password",
        "authentication_outcome": "success",
    }
    base.update(overrides)
    return base


def _write_canonical_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_exits_zero(self, config_yaml: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = runner.invoke(
            app, ["data", "generate", str(config_yaml), "--output-dir", str(out)]
        )
        assert result.exit_code == 0

    def test_generate_creates_expected_files(self, generated_dir: Path) -> None:
        assert (generated_dir / "events.parquet").exists()
        assert (generated_dir / "labels.parquet").exists()
        assert (generated_dir / "events.jsonl").exists()
        assert (generated_dir / "manifest.json").exists()

    def test_generate_output_contains_event_count(
        self, config_yaml: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "cnt"
        result = runner.invoke(
            app, ["data", "generate", str(config_yaml), "--output-dir", str(out)]
        )
        assert "Events:" in result.output

    def test_generate_output_contains_fingerprint(
        self, config_yaml: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "fp"
        result = runner.invoke(
            app, ["data", "generate", str(config_yaml), "--output-dir", str(out)]
        )
        assert "Content hash:" in result.output


# ---------------------------------------------------------------------------
# 2. validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_passes_on_generated_events(self, generated_dir: Path) -> None:
        result = runner.invoke(
            app, ["data", "validate", str(generated_dir / "events.parquet")]
        )
        assert result.exit_code == 0
        assert "PASS" in result.output or "WARN" in result.output

    def test_validate_fails_on_missing_file(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["data", "validate", str(tmp_path / "nofile.parquet")]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 3. profile
# ---------------------------------------------------------------------------


class TestProfile:
    def test_profile_outputs_json_to_stdout(self, generated_dir: Path) -> None:
        result = runner.invoke(
            app, ["data", "profile", str(generated_dir / "events.parquet")]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "row_count" in parsed

    def test_profile_writes_files_to_output_dir(
        self, generated_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "reports"
        result = runner.invoke(
            app,
            [
                "data",
                "profile",
                str(generated_dir / "events.parquet"),
                "--output-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0
        assert (out / "quality-report.json").exists()
        assert (out / "quality-report.md").exists()

    def test_profile_with_gt_path_populates_gt_section(
        self, generated_dir: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "data",
                "profile",
                str(generated_dir / "events.parquet"),
                "--gt-path",
                str(generated_dir / "labels.parquet"),
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed.get("gt_row_count") is not None
        assert parsed["gt_row_count"] > 0


# ---------------------------------------------------------------------------
# 4. manifest creation
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_creates_manifest_json(self, generated_dir: Path) -> None:
        manifest_file = generated_dir / "manifest.json"
        assert manifest_file.exists()
        data = json.loads(manifest_file.read_text())
        assert data["row_count"] > 0

    def test_manifest_contains_checksums(self, generated_dir: Path) -> None:
        data = json.loads((generated_dir / "manifest.json").read_text())
        assert len(data["artifacts"]) > 0
        for entry in data["artifacts"]:
            assert len(entry["sha256"]) == 64

    def test_manifest_has_no_absolute_paths(self, generated_dir: Path) -> None:
        data = json.loads((generated_dir / "manifest.json").read_text())
        text = json.dumps(data)
        assert "/home/" not in text
        assert "/root/" not in text

    def test_manifest_cmd_creates_manifest_for_existing_dir(
        self, generated_dir: Path, tmp_path: Path
    ) -> None:
        import shutil

        new_dir = tmp_path / "ingested"
        new_dir.mkdir()
        shutil.copy(generated_dir / "events.parquet", new_dir / "events.parquet")

        result = runner.invoke(app, ["data", "manifest", str(new_dir)])
        assert result.exit_code == 0
        assert (new_dir / "manifest.json").exists()

    def test_manifest_cmd_refuses_overwrite_without_force(
        self, generated_dir: Path
    ) -> None:
        result = runner.invoke(app, ["data", "manifest", str(generated_dir)])
        assert result.exit_code != 0

    def test_manifest_cmd_overwrites_with_force(self, generated_dir: Path) -> None:
        result = runner.invoke(app, ["data", "manifest", str(generated_dir), "--force"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 5. manifest verification
# ---------------------------------------------------------------------------


class TestVerifyManifest:
    def test_verify_manifest_passes_on_generated_dir(self, generated_dir: Path) -> None:
        result = runner.invoke(app, ["data", "verify-manifest", str(generated_dir)])
        assert result.exit_code == 0
        assert "Verification passed" in result.output

    def test_verify_manifest_shows_check_table(self, generated_dir: Path) -> None:
        result = runner.invoke(app, ["data", "verify-manifest", str(generated_dir)])
        assert "PASS" in result.output


# ---------------------------------------------------------------------------
# 6. safe overwrite rejection
# ---------------------------------------------------------------------------


class TestOverwriteRejection:
    def test_generate_refuses_overwrite_by_default(
        self, config_yaml: Path, generated_dir: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["data", "generate", str(config_yaml), "--output-dir", str(generated_dir)],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 7. forced overwrite success
# ---------------------------------------------------------------------------


class TestForcedOverwrite:
    def test_generate_force_succeeds_on_existing_dir(
        self, config_yaml: Path, generated_dir: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                str(config_yaml),
                "--output-dir",
                str(generated_dir),
                "--force",
            ],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 8. modified-artifact verification failure
# ---------------------------------------------------------------------------


class TestModifiedArtifactVerificationFailure:
    def test_verify_manifest_fails_when_events_corrupted(
        self, generated_dir: Path
    ) -> None:
        events_file = generated_dir / "events.parquet"
        original = events_file.read_bytes()
        events_file.write_bytes(original + b"\x00CORRUPTED")
        result = runner.invoke(app, ["data", "verify-manifest", str(generated_dir)])
        assert result.exit_code != 0

    def test_verify_manifest_fails_when_manifest_missing(
        self, generated_dir: Path
    ) -> None:
        (generated_dir / "manifest.json").unlink()
        result = runner.invoke(app, ["data", "verify-manifest", str(generated_dir)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 9. canonical and ground-truth separation
# ---------------------------------------------------------------------------


class TestCanonicalGTSeparation:
    def test_events_parquet_has_no_gt_columns(self, generated_dir: Path) -> None:
        import pyarrow.parquet as pq

        from password_attack_detector.data.schemas import PROHIBITED_GT_COLUMNS

        table = pq.read_table(generated_dir / "events.parquet")
        columns = set(table.schema.names)
        assert columns.isdisjoint(PROHIBITED_GT_COLUMNS)

    def test_labels_parquet_joins_via_event_id(self, generated_dir: Path) -> None:
        import pyarrow.parquet as pq

        events = pq.read_table(generated_dir / "events.parquet")
        labels = pq.read_table(generated_dir / "labels.parquet")
        assert "event_id" in events.schema.names
        assert "event_id" in labels.schema.names
        assert events.num_rows == labels.num_rows

    def test_manifest_records_separate_row_counts(self, generated_dir: Path) -> None:
        data = json.loads((generated_dir / "manifest.json").read_text())
        assert "row_count" in data
        assert "ground_truth_row_count" in data
        assert data["row_count"] == data["ground_truth_row_count"]


# ---------------------------------------------------------------------------
# 10. deterministic content fingerprints
# ---------------------------------------------------------------------------


class TestDeterministicFingerprints:
    def test_same_config_produces_same_fingerprint(
        self, config_yaml: Path, tmp_path: Path
    ) -> None:
        dir1 = tmp_path / "ds1"
        dir2 = tmp_path / "ds2"
        runner.invoke(
            app, ["data", "generate", str(config_yaml), "--output-dir", str(dir1)]
        )
        runner.invoke(
            app, ["data", "generate", str(config_yaml), "--output-dir", str(dir2)]
        )
        m1 = json.loads((dir1 / "manifest.json").read_text())
        m2 = json.loads((dir2 / "manifest.json").read_text())
        assert m1["content_fingerprint"] == m2["content_fingerprint"]

    def test_different_seed_produces_different_fingerprint(
        self, tmp_path: Path
    ) -> None:
        cfg1 = _tiny_cfg()
        cfg2 = {**_tiny_cfg(), "seed": 999}
        p1 = tmp_path / "c1.yaml"
        p2 = tmp_path / "c2.yaml"
        p1.write_text(yaml.dump(cfg1))
        p2.write_text(yaml.dump(cfg2))
        dir1 = tmp_path / "d1"
        dir2 = tmp_path / "d2"
        runner.invoke(app, ["data", "generate", str(p1), "--output-dir", str(dir1)])
        runner.invoke(app, ["data", "generate", str(p2), "--output-dir", str(dir2)])
        m1 = json.loads((dir1 / "manifest.json").read_text())
        m2 = json.loads((dir2 / "manifest.json").read_text())
        assert m1["content_fingerprint"] != m2["content_fingerprint"]


# ---------------------------------------------------------------------------
# 11. empty ingestion failure
# ---------------------------------------------------------------------------


class TestEmptyIngestionFailure:
    def test_empty_csv_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "data",
                "ingest",
                str(csv_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "csv",
            ],
        )
        assert result.exit_code != 0

    def test_empty_jsonl_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        result = runner.invoke(
            app,
            ["data", "ingest", str(jsonl_path), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0

    def test_empty_ingestion_does_not_publish(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("", encoding="utf-8")
        out = tmp_path / "out"
        runner.invoke(
            app,
            [
                "data",
                "ingest",
                str(csv_path),
                "--output-dir",
                str(out),
                "--format",
                "csv",
            ],
        )
        assert not (out / "events.parquet").exists()


# ---------------------------------------------------------------------------
# 12. prohibited CSV field rejection
# ---------------------------------------------------------------------------


class TestProhibitedCSVFieldRejection:
    def test_csv_with_password_column_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        csv_path = tmp_path / "bad.csv"
        row = _make_canonical_event()
        row["password"] = "hunter2"
        _write_canonical_csv(csv_path, [row])
        result = runner.invoke(
            app,
            ["data", "ingest", str(csv_path), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0

    def test_csv_with_token_column_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        csv_path = tmp_path / "bad_token.csv"
        row = _make_canonical_event()
        row["token"] = "secret"
        _write_canonical_csv(csv_path, [row])
        result = runner.invoke(
            app,
            ["data", "ingest", str(csv_path), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0

    def test_csv_with_valid_fields_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        csv_path = tmp_path / "good.csv"
        _write_canonical_csv(csv_path, [_make_canonical_event()])
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            ["data", "ingest", str(csv_path), "--output-dir", str(out)],
        )
        assert result.exit_code == 0
        assert (out / "events.parquet").exists()


# ---------------------------------------------------------------------------
# 13. prohibited nested JSONL field rejection
# ---------------------------------------------------------------------------


class TestProhibitedNestedJSONLFieldRejection:
    def test_jsonl_with_nested_password_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        row = _make_canonical_event()
        row["metadata"] = {"password": "s3cr3t"}
        jsonl_path = tmp_path / "bad.jsonl"
        _write_jsonl(jsonl_path, [row])
        result = runner.invoke(
            app,
            ["data", "ingest", str(jsonl_path), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0

    def test_jsonl_without_prohibited_keys_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        jsonl_path = tmp_path / "good.jsonl"
        _write_jsonl(jsonl_path, [_make_canonical_event()])
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            ["data", "ingest", str(jsonl_path), "--output-dir", str(out)],
        )
        assert result.exit_code == 0
        assert (out / "events.parquet").exists()


# ---------------------------------------------------------------------------
# 14. quarantine privacy
# ---------------------------------------------------------------------------


class TestQuarantinePrivacy:
    def test_quarantine_policy_accepts_valid_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        good = _make_canonical_event()
        bad_row: dict[str, Any] = {
            "event_id": str(uuid4()),
            "event_time": "not-a-date",
        }
        jsonl_path = tmp_path / "mixed.jsonl"
        _write_jsonl(jsonl_path, [good, bad_row])
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "data",
                "ingest",
                str(jsonl_path),
                "--output-dir",
                str(out),
                "--policy",
                "quarantine",
            ],
        )
        assert result.exit_code == 0
        assert (out / "events.parquet").exists()
        assert "Quarantined:      1" in result.output

    def test_quarantine_output_contains_no_raw_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAD_PSEUDONYMIZATION_KEY", "a" * 64)
        good = _make_canonical_event()
        bad_row: dict[str, Any] = {
            "event_id": "SENTINEL_IDENTIFIER_VALUE",
            "event_time": "bad",
        }
        jsonl_path = tmp_path / "mixed.jsonl"
        _write_jsonl(jsonl_path, [good, bad_row])
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "data",
                "ingest",
                str(jsonl_path),
                "--output-dir",
                str(out),
                "--policy",
                "quarantine",
            ],
        )
        assert "SENTINEL_IDENTIFIER_VALUE" not in result.output


# ---------------------------------------------------------------------------
# 15. installed CLI entrypoint behavior
# ---------------------------------------------------------------------------


class TestInstalledCLIEntrypoint:
    def test_data_help_lists_all_seven_commands(self) -> None:
        result = runner.invoke(app, ["data", "--help"])
        assert result.exit_code == 0
        for cmd in (
            "generate",
            "ingest",
            "validate",
            "profile",
            "manifest",
            "verify-manifest",
            "schema",
        ):
            assert cmd in result.output

    def test_version_shows_0_3_0(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.4.0" in result.output

    def test_schema_json_format(self) -> None:
        result = runner.invoke(app, ["data", "schema", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "fields" in data
        assert "event_id" in data["fields"]

    def test_root_help_still_shows_phase1_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "version" in result.output
        assert "doctor" in result.output
        assert "show-config" in result.output
        assert "data" in result.output
        assert "features" in result.output
