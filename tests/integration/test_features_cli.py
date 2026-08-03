"""Integration tests for the ``features`` CLI group and the full pipeline."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner, Result

from password_attack_detector.cli import app
from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.data.serialization import (
    write_events_parquet,
    write_labels_parquet,
)
from tests.features.factories import make_event, make_labels

runner = CliRunner()

_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)


def _tiny_config() -> dict[str, object]:
    """A CI-sized feature configuration with short windows and a small purge."""
    return {
        "windows": ["1m", "5m"],
        "cardinality_windows": ["5m"],
        "dispersion_windows": ["5m"],
        "device_session_windows": ["5m"],
        "pair_windows": ["5m"],
        "baseline": {
            "rate_reference_window": "5m",
            "min_events_per_user": 2,
            "min_events_per_source": 2,
            "response_time_min_events": 2,
        },
        "split": {
            "purge": "5m",
            "strict_isolation": True,
            "max_excluded_fraction": 0.6,
        },
    }


def _events(count: int = 120) -> list[AuthEvent]:
    rng = random.Random(864209)
    return [
        make_event(
            t=float(index) * 60.0,
            user=f"u{rng.randint(1, 4)}",
            source=f"s{rng.randint(1, 3)}",
            device=f"d{rng.randint(1, 3)}",
            outcome=rng.choice(["success", "success", "failure"]),
            response_time_ms=rng.randint(30, 400),
            country=rng.choice(["US", "GB"]),
            latitude=rng.choice([37.8, 51.5]),
            longitude=rng.choice([-122.4, -0.1]),
            key=str(index),
        )
        for index in range(count)
    ]


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """A directory holding events, labels, and a feature configuration."""
    events = _events()
    write_events_parquet(events, tmp_path / "events.parquet")
    write_labels_parquet(make_labels(events), tmp_path / "labels.parquet")
    (tmp_path / "features.yaml").write_text(
        yaml.safe_dump(_tiny_config()), encoding="utf-8"
    )
    return tmp_path


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


# --- help ------------------------------------------------------------------


class TestHelp:
    def test_root_help_lists_the_features_group(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "features" in result.output

    def test_features_help_lists_every_command(self) -> None:
        result = runner.invoke(app, ["features", "--help"])
        assert result.exit_code == 0
        for command in (
            "build",
            "split",
            "fit-baseline",
            "transform",
            "audit-leakage",
            "validate",
            "profile",
            "verify-manifest",
            "catalog",
        ):
            assert command in result.output

    @pytest.mark.parametrize(
        "command",
        [
            "build",
            "split",
            "fit-baseline",
            "transform",
            "audit-leakage",
            "validate",
            "profile",
            "verify-manifest",
            "catalog",
        ],
    )
    def test_each_command_has_help(self, command: str) -> None:
        result = runner.invoke(app, ["features", command, "--help"])
        assert result.exit_code == 0

    def test_bare_group_shows_help(self) -> None:
        result = runner.invoke(app, ["features"])
        assert "Usage" in result.output


# --- catalog ---------------------------------------------------------------


class TestCatalogCommand:
    def test_text_output_summarises_the_catalog(self) -> None:
        result = runner.invoke(app, ["features", "catalog"])
        assert result.exit_code == 0
        assert "Catalog fingerprint" in result.output

    def test_json_output_is_parseable(self, tmp_path: Path) -> None:
        target = tmp_path / "catalog.json"
        result = runner.invoke(
            app, ["features", "catalog", "--format", "json", "-o", str(target)]
        )
        assert result.exit_code == 0
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["feature_count"] == len(payload["features"])

    def test_markdown_output_is_written(self, tmp_path: Path) -> None:
        target = tmp_path / "catalog.md"
        result = runner.invoke(
            app, ["features", "catalog", "--format", "markdown", "-o", str(target)]
        )
        assert result.exit_code == 0
        assert target.read_text(encoding="utf-8").startswith("# Feature Catalog")

    def test_unknown_format_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["features", "catalog", "--format", "xml"])
        assert result.exit_code == 1

    def test_invalid_config_exits_nonzero(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text('windows: ["5m", "1m"]\n', encoding="utf-8")
        result = runner.invoke(app, ["features", "catalog", "--config", str(bad)])
        assert result.exit_code == 1


# --- split -----------------------------------------------------------------


class TestSplitCommand:
    def test_reports_split_sizes(self, workspace: Path) -> None:
        result = _invoke(
            "features",
            "split",
            str(workspace / "events.parquet"),
            "--labels",
            str(workspace / "labels.parquet"),
            "--config",
            str(workspace / "features.yaml"),
        )
        assert result.exit_code == 0, result.output
        assert "train" in result.output

    def test_writes_the_split_table_and_manifest(self, workspace: Path) -> None:
        out = workspace / "split-out"
        result = _invoke(
            "features",
            "split",
            str(workspace / "events.parquet"),
            "--labels",
            str(workspace / "labels.parquet"),
            "--config",
            str(workspace / "features.yaml"),
            "-o",
            str(out),
        )
        assert result.exit_code == 0, result.output
        assert (out / "feature_splits.parquet").exists()
        assert (out / "split_manifest.json").exists()

    def test_refuses_to_overwrite_without_force(self, workspace: Path) -> None:
        out = workspace / "split-out"
        args = [
            "features",
            "split",
            str(workspace / "events.parquet"),
            "--labels",
            str(workspace / "labels.parquet"),
            "--config",
            str(workspace / "features.yaml"),
            "-o",
            str(out),
        ]
        assert _invoke(*args).exit_code == 0
        assert _invoke(*args).exit_code == 1
        assert _invoke(*args, "--force").exit_code == 0

    def test_missing_events_exits_nonzero(self, workspace: Path) -> None:
        result = _invoke(
            "features",
            "split",
            str(workspace / "absent.parquet"),
            "--labels",
            str(workspace / "labels.parquet"),
        )
        assert result.exit_code == 1


# --- fit-baseline ----------------------------------------------------------


class TestFitBaselineCommand:
    def test_requires_an_explicit_training_selection(self, workspace: Path) -> None:
        # A baseline is never silently fitted on validation or test data.
        result = _invoke(
            "features",
            "fit-baseline",
            str(workspace / "events.parquet"),
            "-o",
            str(workspace / "baseline"),
        )
        assert result.exit_code == 1
        assert "exactly one" in result.output

    def test_rejects_both_selections_at_once(self, workspace: Path) -> None:
        result = _invoke(
            "features",
            "fit-baseline",
            str(workspace / "events.parquet"),
            "-o",
            str(workspace / "baseline"),
            "--train-end",
            "2024-03-04T13:00:00+00:00",
            "--split-table",
            str(workspace / "feature_splits.parquet"),
        )
        assert result.exit_code == 1

    def test_fits_from_a_train_end_cutoff(self, workspace: Path) -> None:
        result = _invoke(
            "features",
            "fit-baseline",
            str(workspace / "events.parquet"),
            "-o",
            str(workspace / "baseline"),
            "--train-end",
            "2024-03-04T13:00:00+00:00",
        )
        assert result.exit_code == 0, result.output
        assert (workspace / "baseline" / "baseline.json").exists()

    def test_fits_from_a_split_table(self, workspace: Path) -> None:
        out = workspace / "split-out"
        _invoke(
            "features",
            "split",
            str(workspace / "events.parquet"),
            "--labels",
            str(workspace / "labels.parquet"),
            "--config",
            str(workspace / "features.yaml"),
            "-o",
            str(out),
        )
        result = _invoke(
            "features",
            "fit-baseline",
            str(workspace / "events.parquet"),
            "-o",
            str(workspace / "baseline"),
            "--config",
            str(workspace / "features.yaml"),
            "--split-table",
            str(out / "feature_splits.parquet"),
        )
        assert result.exit_code == 0, result.output

    def test_refuses_to_overwrite_without_force(self, workspace: Path) -> None:
        args = [
            "features",
            "fit-baseline",
            str(workspace / "events.parquet"),
            "-o",
            str(workspace / "baseline"),
            "--train-end",
            "2024-03-04T13:00:00+00:00",
        ]
        assert _invoke(*args).exit_code == 0
        assert _invoke(*args).exit_code == 1
        assert _invoke(*args, "--force").exit_code == 0

    def test_summary_shows_no_identifiers(self, workspace: Path) -> None:
        result = _invoke(
            "features",
            "fit-baseline",
            str(workspace / "events.parquet"),
            "-o",
            str(workspace / "baseline"),
            "--train-end",
            "2024-03-04T13:00:00+00:00",
        )
        assert not _PSEUDONYM_RE.search(result.output)
        assert not _UUID_RE.search(result.output)


# --- build -----------------------------------------------------------------


def _build(workspace: Path, *extra: str) -> Result:
    return _invoke(
        "features",
        "build",
        str(workspace / "events.parquet"),
        "--labels",
        str(workspace / "labels.parquet"),
        "--config",
        str(workspace / "features.yaml"),
        "-o",
        str(workspace / "processed"),
        "--reports-dir",
        str(workspace / "reports"),
        *extra,
    )


class TestBuildCommand:
    def test_builds_the_full_dataset(self, workspace: Path) -> None:
        result = _build(workspace)
        assert result.exit_code == 0, result.output
        out = workspace / "processed"
        for name in (
            "feature_snapshots.parquet",
            "feature_labels.parquet",
            "feature_splits.parquet",
            "feature_manifest.json",
            "split_manifest.json",
        ):
            assert (out / name).exists(), name

    def test_writes_all_four_reports(self, workspace: Path) -> None:
        assert _build(workspace).exit_code == 0
        reports = workspace / "reports"
        for name in (
            "feature_quality.json",
            "feature_quality.md",
            "leakage_audit.json",
            "leakage_audit.md",
        ):
            assert (reports / name).exists(), name

    def test_the_leakage_audit_passes(self, workspace: Path) -> None:
        assert _build(workspace).exit_code == 0
        audit = json.loads(
            (workspace / "reports" / "leakage_audit.json").read_text(encoding="utf-8")
        )
        assert audit["status"] == "pass", audit["errors"]

    def test_every_check_ran(self, workspace: Path) -> None:
        _build(workspace)
        audit = json.loads(
            (workspace / "reports" / "leakage_audit.json").read_text(encoding="utf-8")
        )
        assert len(audit["checks"]) == 12
        assert not [c for c in audit["checks"] if c["skipped"]]

    def test_validation_passes(self, workspace: Path) -> None:
        _build(workspace)
        manifest = json.loads(
            (workspace / "processed" / "feature_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["validation_result"]["status"] in {"valid", "warning"}

    def test_snapshots_carry_no_ground_truth(self, workspace: Path) -> None:
        import pyarrow.parquet as pq

        from password_attack_detector.data.schemas import PROHIBITED_GT_COLUMNS

        _build(workspace)
        columns = set(
            pq.read_schema(workspace / "processed" / "feature_snapshots.parquet").names
        )
        assert not columns & PROHIBITED_GT_COLUMNS
        assert "split" not in columns
        assert "campaign_id" not in columns

    def test_labels_and_splits_are_separate_tables(self, workspace: Path) -> None:
        import pyarrow.parquet as pq

        _build(workspace)
        out = workspace / "processed"
        label_columns = set(pq.read_schema(out / "feature_labels.parquet").names)
        split_columns = set(pq.read_schema(out / "feature_splits.parquet").names)
        assert "malicious" in label_columns
        assert "split" in split_columns
        assert label_columns & split_columns == {"event_id"}

    def test_refuses_to_overwrite_without_force(self, workspace: Path) -> None:
        assert _build(workspace).exit_code == 0
        assert _build(workspace).exit_code == 1
        assert _build(workspace, "--force").exit_code == 0

    def test_skip_audit_warns_loudly(self, workspace: Path) -> None:
        result = _build(workspace, "--skip-audit")
        assert result.exit_code == 0
        assert "skipped" in result.output.lower()

    def test_output_shows_no_identifiers(self, workspace: Path) -> None:
        result = _build(workspace)
        assert not _PSEUDONYM_RE.search(result.output)
        assert not _UUID_RE.search(result.output)

    def test_output_shows_no_coordinates(self, workspace: Path) -> None:
        result = _build(workspace)
        for coordinate in ("37.8", "51.5", "-122.4"):
            assert coordinate not in result.output

    def test_missing_labels_exits_nonzero(self, workspace: Path) -> None:
        result = _invoke(
            "features",
            "build",
            str(workspace / "events.parquet"),
            "--labels",
            str(workspace / "absent.parquet"),
            "-o",
            str(workspace / "processed"),
        )
        assert result.exit_code == 1


# --- downstream commands over a built dataset -------------------------------


@pytest.fixture()
def built(workspace: Path) -> Path:
    result = _build(workspace)
    assert result.exit_code == 0, result.output
    return workspace


class TestValidateCommand:
    def test_a_built_dataset_validates(self, built: Path) -> None:
        result = _invoke(
            "features",
            "validate",
            str(built / "processed" / "feature_snapshots.parquet"),
            "--config",
            str(built / "features.yaml"),
        )
        assert result.exit_code == 0, result.output

    def test_relationship_checks_run_with_companions(self, built: Path) -> None:
        out = built / "processed"
        result = _invoke(
            "features",
            "validate",
            str(out / "feature_snapshots.parquet"),
            "--labels",
            str(out / "feature_labels.parquet"),
            "--splits",
            str(out / "feature_splits.parquet"),
            "--config",
            str(built / "features.yaml"),
        )
        assert result.exit_code == 0, result.output

    def test_a_corrupt_table_exits_nonzero(self, built: Path) -> None:
        target = built / "processed" / "feature_snapshots.parquet"
        target.write_bytes(b"not parquet")
        result = _invoke(
            "features",
            "validate",
            str(target),
            "--config",
            str(built / "features.yaml"),
        )
        assert result.exit_code == 1

    def test_findings_show_no_identifiers(self, built: Path) -> None:
        target = built / "processed" / "feature_snapshots.parquet"
        target.write_bytes(b"not parquet")
        result = _invoke(
            "features",
            "validate",
            str(target),
            "--config",
            str(built / "features.yaml"),
        )
        assert not _UUID_RE.search(result.output)


class TestProfileCommand:
    def test_prints_json_by_default(self, built: Path) -> None:
        result = _invoke(
            "features",
            "profile",
            str(built / "events.parquet"),
            "--config",
            str(built / "features.yaml"),
        )
        assert result.exit_code == 0, result.output

    def test_writes_reports_to_a_directory(self, built: Path) -> None:
        out = built / "profile-out"
        result = _invoke(
            "features",
            "profile",
            str(built / "events.parquet"),
            "--splits",
            str(built / "processed" / "feature_splits.parquet"),
            "--config",
            str(built / "features.yaml"),
            "-o",
            str(out),
        )
        assert result.exit_code == 0, result.output
        assert (out / "feature_quality.json").exists()
        assert (out / "feature_quality.md").exists()

    def test_reports_contain_no_identifiers(self, built: Path) -> None:
        out = built / "profile-out"
        _invoke(
            "features",
            "profile",
            str(built / "events.parquet"),
            "--config",
            str(built / "features.yaml"),
            "-o",
            str(out),
        )
        text = (out / "feature_quality.json").read_text(encoding="utf-8")
        text += (out / "feature_quality.md").read_text(encoding="utf-8")
        assert not _UUID_RE.search(text)
        assert not _PSEUDONYM_RE.search(text)


class TestAuditLeakageCommand:
    def test_audits_a_built_dataset(self, built: Path) -> None:
        result = _invoke(
            "features",
            "audit-leakage",
            str(built / "events.parquet"),
            "--labels",
            str(built / "labels.parquet"),
            "--splits",
            str(built / "processed" / "feature_splits.parquet"),
            "--config",
            str(built / "features.yaml"),
        )
        assert result.exit_code == 0, result.output

    def test_json_output_is_parseable(self, built: Path) -> None:
        result = _invoke(
            "features",
            "audit-leakage",
            str(built / "events.parquet"),
            "--labels",
            str(built / "labels.parquet"),
            "--splits",
            str(built / "processed" / "feature_splits.parquet"),
            "--config",
            str(built / "features.yaml"),
            "--format",
            "json",
        )
        assert result.exit_code == 0
        assert '"status"' in result.output

    def test_a_broken_split_table_fails_the_audit(self, built: Path) -> None:
        # Reassign every event to train, which breaks purge isolation.
        import pyarrow as pa
        import pyarrow.parquet as pq

        target = built / "processed" / "feature_splits.parquet"
        rows = pq.read_table(target).to_pylist()
        broken = {
            "event_id": [r["event_id"] for r in rows],
            "split": ["train" if i % 2 else "test" for i in range(len(rows))],
            "exclusion_reason": [None] * len(rows),
        }
        pq.write_table(pa.table(broken), target)

        result = _invoke(
            "features",
            "audit-leakage",
            str(built / "events.parquet"),
            "--labels",
            str(built / "labels.parquet"),
            "--splits",
            str(target),
            "--config",
            str(built / "features.yaml"),
        )
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestVerifyManifestCommand:
    def test_a_built_dataset_verifies(self, built: Path) -> None:
        result = _invoke("features", "verify-manifest", str(built / "processed"))
        assert result.exit_code == 0, result.output

    def test_a_tampered_artifact_fails(self, built: Path) -> None:
        (built / "processed" / "feature_splits.parquet").write_bytes(b"corrupt")
        result = _invoke("features", "verify-manifest", str(built / "processed"))
        assert result.exit_code == 1

    def test_a_directory_without_a_manifest_fails(self, tmp_path: Path) -> None:
        result = _invoke("features", "verify-manifest", str(tmp_path))
        assert result.exit_code == 1


class TestTransformCommand:
    def test_applies_an_existing_baseline(self, built: Path) -> None:
        _invoke(
            "features",
            "fit-baseline",
            str(built / "events.parquet"),
            "-o",
            str(built / "baseline"),
            "--config",
            str(built / "features.yaml"),
            "--train-end",
            "2024-03-04T13:00:00+00:00",
        )
        out = built / "transformed"
        result = _invoke(
            "features",
            "transform",
            str(built / "events.parquet"),
            "-o",
            str(out),
            "--baseline",
            str(built / "baseline"),
            "--config",
            str(built / "features.yaml"),
        )
        assert result.exit_code == 0, result.output
        assert (out / "feature_snapshots.parquet").exists()

    def test_works_without_a_baseline(self, built: Path) -> None:
        out = built / "transformed"
        result = _invoke(
            "features",
            "transform",
            str(built / "events.parquet"),
            "-o",
            str(out),
            "--config",
            str(built / "features.yaml"),
        )
        assert result.exit_code == 0, result.output

    def test_refuses_to_overwrite_without_force(self, built: Path) -> None:
        args = [
            "features",
            "transform",
            str(built / "events.parquet"),
            "-o",
            str(built / "transformed"),
            "--config",
            str(built / "features.yaml"),
        ]
        assert _invoke(*args).exit_code == 0
        assert _invoke(*args).exit_code == 1
        assert _invoke(*args, "--force").exit_code == 0


# --- module invocation -----------------------------------------------------


class TestModuleInvocation:
    def test_features_help_works_via_python_m(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "password_attack_detector", "features", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "build" in result.stdout
