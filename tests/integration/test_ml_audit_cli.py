"""End-to-end tests for ``ml audit-features`` over a real published dataset.

These run the whole chain: generate a micro event stream, publish it through
``features build``, draft an allowlist from the same feature configuration, and
audit the result.  Nothing here is mocked, and nothing is fitted.

The dataset is deliberately tiny -- 160 events over about two and a half hours,
with one-minute and five-minute windows.  The 720-hour ML development dataset is
not built here and never will be: a contract test that needs a month of traffic
to express itself is testing the generator.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner, Result

from password_attack_detector.cli import app
from password_attack_detector.data.enums import CampaignStage, ScenarioType
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.data.serialization import (
    write_events_parquet,
    write_labels_parquet,
)
from password_attack_detector.ml.eligibility import (
    CHECK_NAMES,
    ML_AUDIT_JSON_FILE,
    ML_AUDIT_MD_FILE,
)
from tests.features.factories import make_event

runner = CliRunner()

_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)

#: How many attack campaigns the fixture contains, and how long each runs.
#: Several, spread through the stream, so the validation split carries more than
#: one group and the boundary has somewhere meaningful to fall.
_CAMPAIGN_COUNT = 12
_CAMPAIGN_LENGTH = 4
_EVENT_COUNT = 160


def _repo_root() -> Path:
    """Return the repository root, located from this test file."""
    return Path(__file__).resolve().parents[2]


def _feature_config() -> dict[str, object]:
    """A CI-sized feature configuration: short windows, a small purge."""
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


def _events() -> list[AuthEvent]:
    """Return a deterministic event stream with periodic bursts of failures."""
    events: list[AuthEvent] = []
    for index in range(_EVENT_COUNT):
        attacking = _campaign_of(index) is not None
        events.append(
            make_event(
                t=float(index) * 60.0,
                user=f"u{index % 5 + 1}",
                source=f"s{1 if attacking else index % 3 + 2}",
                device=f"d{index % 3 + 1}",
                outcome="failure" if attacking else "success",
                response_time_ms=40 if attacking else 200 + index % 50,
                country="US" if attacking else "GB",
                latitude=37.8 if attacking else 51.5,
                longitude=-122.4 if attacking else -0.1,
                key=str(index),
            )
        )
    return events


def _campaign_of(index: int) -> str | None:
    """Return the campaign an event index belongs to, or ``None`` if benign.

    Campaigns are spaced evenly through the stream so every split receives
    several, rather than clustering at one end where a chronological split would
    hand them all to one partition.
    """
    stride = _EVENT_COUNT // _CAMPAIGN_COUNT
    position = index % stride
    if position < _CAMPAIGN_LENGTH:
        return f"campaign-{index // stride:02d}"
    return None


def _labels(
    events: Sequence[AuthEvent], *, prefix: str = "campaign-"
) -> list[GroundTruthLabel]:
    """Return ground truth pairing each burst with a campaign identifier.

    Benign rows carry ``normal-<seed>``, exactly as the Phase 2 generator writes
    them: ``GroundTruthLabel.campaign_id`` is required, so background traffic
    gets a placeholder. Campaign rows carry a stage, which is likewise what the
    generator writes on every campaign it runs and on nothing else. The fixture
    reproduces both conventions rather than inventing friendlier ones, because
    the CLI has to work on generator output.

    *prefix* renames the campaigns without changing anything else, so a test can
    hand the resolver an identifier that reads exactly like the benign
    placeholder and check that the metadata still wins.
    """
    labels = []
    for index, event in enumerate(events):
        campaign = _campaign_of(index)
        named = None if campaign is None else campaign.replace("campaign-", prefix)
        labels.append(
            GroundTruthLabel(
                event_id=event.event_id,
                campaign_id=named or "normal-864209",
                scenario=(
                    ScenarioType.BRUTE_FORCE
                    if named is not None
                    else ScenarioType.NORMAL
                ),
                malicious=named is not None,
                supervised_training_eligible=True,
                generator_version="1.0.0",
                campaign_stage=None if named is None else CampaignStage.ACTIVE,
            )
        )
    return labels


def _invoke(*arguments: str) -> Result:
    """Run the CLI with *arguments* and return the result."""
    return runner.invoke(app, list(arguments))


def _publish(tmp_path: Path, *, prefix: str = "campaign-") -> Path:
    """Publish a feature dataset and a drafted allowlist under *tmp_path*."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    events = _events()
    write_events_parquet(events, tmp_path / "events.parquet")
    write_labels_parquet(_labels(events, prefix=prefix), tmp_path / "labels.parquet")
    (tmp_path / "features.yaml").write_text(
        yaml.safe_dump(_feature_config()), encoding="utf-8"
    )

    built = _invoke(
        "features",
        "build",
        str(tmp_path / "events.parquet"),
        "--labels",
        str(tmp_path / "labels.parquet"),
        "--config",
        str(tmp_path / "features.yaml"),
        "-o",
        str(tmp_path / "processed"),
        "--reports-dir",
        str(tmp_path / "reports"),
    )
    assert built.exit_code == 0, built.output

    drafted = _invoke(
        "ml",
        "catalog",
        "--emit-allowlist",
        str(tmp_path / "allowlist.yaml"),
        "--feature-config",
        str(tmp_path / "features.yaml"),
    )
    assert drafted.exit_code == 0, drafted.output
    return tmp_path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Return a published workspace with ordinarily named campaigns."""
    return _publish(tmp_path)


def _audit(workspace: Path, *extra: str, **replace: str) -> Result:
    """Run ``ml audit-features`` over the published workspace."""
    arguments = {
        "--features": str(workspace / "processed" / "feature_snapshots.parquet"),
        "--labels": str(workspace / "processed" / "feature_labels.parquet"),
        "--splits": str(workspace / "processed" / "feature_splits.parquet"),
        "--campaign-labels": str(workspace / "labels.parquet"),
        "--feature-manifest": str(workspace / "processed" / "feature_manifest.json"),
        "--allowlist": str(workspace / "allowlist.yaml"),
        "--feature-config": str(workspace / "features.yaml"),
        "--config": str(_repo_root() / "configs" / "ml" / "model-testing.yaml"),
        "-o": str(workspace / "ml-reports"),
    }
    arguments.update(replace)
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return _invoke("ml", "audit-features", *flat, *extra)


def _report(workspace: Path) -> dict[str, object]:
    """Return the published JSON audit report."""
    path = workspace / "ml-reports" / ML_AUDIT_JSON_FILE
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_the_audit_passes_on_a_published_dataset(workspace: Path) -> None:
    """The whole chain, end to end, on real published artifacts."""
    result = _audit(workspace)
    assert result.exit_code == 0, result.output
    assert "Audit status: pass" in result.stdout


def test_the_audit_publishes_both_reports(workspace: Path) -> None:
    """JSON and Markdown, matching the convention every other report follows."""
    assert _audit(workspace).exit_code == 0
    for name in (ML_AUDIT_JSON_FILE, ML_AUDIT_MD_FILE):
        assert (workspace / "ml-reports" / name).exists(), name


def test_the_json_report_records_every_check_as_passed(workspace: Path) -> None:
    """Fifteen checks, all evaluated, none skipped."""
    assert _audit(workspace).exit_code == 0
    payload = _report(workspace)
    checks = payload["checks"]
    assert isinstance(checks, list)
    assert [check["name"] for check in checks] == list(CHECK_NAMES)
    assert all(check["status"] == "pass" for check in checks), [
        check for check in checks if check["status"] != "pass"
    ]
    assert payload["status"] == "pass"


def test_the_console_output_names_every_check(workspace: Path) -> None:
    """A reader must be able to see what was checked, not only the verdict."""
    result = _audit(workspace)
    for name in CHECK_NAMES:
        assert name in result.stdout, name


def test_the_audit_reports_counts_and_the_partition(workspace: Path) -> None:
    """Aggregate counts, including both validation halves."""
    result = _audit(workspace)
    for label in (
        "Eligible features",
        "Rows",
        "Validation-A rows",
        "Validation-B rows",
        "Validation-A campaigns",
        "Validation-B campaigns",
    ):
        assert label in result.stdout, label


def test_the_partition_puts_campaigns_on_both_sides(workspace: Path) -> None:
    """The fixture is sized so the boundary genuinely divides campaign groups."""
    assert _audit(workspace).exit_code == 0
    partition = _report(workspace)["validation_partition"]
    assert isinstance(partition, dict)
    assert partition["status"] == "partitioned"
    assert partition["partition_a_rows"] > 0
    assert partition["partition_b_rows"] > 0
    assert partition["partition_a_campaigns"] > 0
    assert partition["partition_b_campaigns"] > 0


def test_the_audit_is_deterministic(workspace: Path) -> None:
    """Two runs over one dataset publish byte-identical reports."""
    _audit(workspace)
    first = (workspace / "ml-reports" / ML_AUDIT_JSON_FILE).read_text(encoding="utf-8")
    _audit(workspace)
    second = (workspace / "ml-reports" / ML_AUDIT_JSON_FILE).read_text(encoding="utf-8")
    assert first == second


def test_the_configured_leakage_classes_narrow_the_matrix(workspace: Path) -> None:
    """The CI configuration excludes baseline-derived features, and it shows."""
    assert _audit(workspace).exit_code == 0
    payload = _report(workspace)
    catalog_size = len(
        yaml.safe_load((workspace / "allowlist.yaml").read_text(encoding="utf-8"))[
            "entries"
        ]
    )
    assert isinstance(payload["checked_feature_count"], int)
    assert 0 < payload["checked_feature_count"] < catalog_size


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def test_the_console_output_carries_no_identifier_or_path(workspace: Path) -> None:
    """The sweep every CLI test module in this repository applies."""
    result = _audit(workspace)
    assert not _UUID_RE.search(result.stdout)
    assert not _PSEUDONYM_RE.search(result.stdout)
    assert str(Path.home()) not in result.stdout
    assert "campaign-00" not in result.stdout


@pytest.mark.parametrize("name", [ML_AUDIT_JSON_FILE, ML_AUDIT_MD_FILE])
def test_a_published_report_carries_no_identifier_or_campaign(
    workspace: Path, name: str
) -> None:
    """The same sweep over both published formats."""
    _audit(workspace)
    rendered = (workspace / "ml-reports" / name).read_text(encoding="utf-8")
    assert not _UUID_RE.search(rendered)
    assert not _PSEUDONYM_RE.search(rendered)
    assert str(Path.home()) not in rendered
    assert "campaign-" not in rendered
    assert "c-normal" not in rendered


def test_no_report_carries_a_measured_detection_figure(workspace: Path) -> None:
    """An eligibility audit says nothing about how well anything detects."""
    _audit(workspace)
    rendered = (
        (workspace / "ml-reports" / ML_AUDIT_MD_FILE)
        .read_text(encoding="utf-8")
        .lower()
    )
    for term in ("accuracy", "auc", "f1 score", "recall of", "precision of"):
        assert term not in rendered, term
    assert "no model has been fitted" in rendered


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        "--features",
        "--labels",
        "--splits",
        "--campaign-labels",
        "--feature-manifest",
        "--allowlist",
    ],
)
def test_a_missing_input_exits_non_zero(workspace: Path, option: str) -> None:
    """Every required path is checked before any work begins."""
    result = _audit(workspace, **{option: str(workspace / "absent.parquet")})
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "option",
    ["--campaign-labels", "--feature-manifest"],
)
def test_a_required_input_has_no_default(option: str) -> None:
    """Omitting one must fail rather than run a smaller audit.

    Both feed a mandatory check. Making either optional would let the command
    report fourteen passes and one skip as a clean run, which is exactly the
    reading a skipped check must never support.
    """
    result = _invoke("ml", "audit-features", "--help")
    assert result.exit_code == 0
    assert option in result.stdout


def test_an_allowlist_that_disagrees_with_the_catalog_exits_non_zero(
    workspace: Path,
) -> None:
    """A reviewed contract describing a different system is refused."""
    document = yaml.safe_load(
        (workspace / "allowlist.yaml").read_text(encoding="utf-8")
    )
    document["entries"][0]["leakage_class"] = (
        "baseline_derived"
        if document["entries"][0]["leakage_class"] != "baseline_derived"
        else "prior_only"
    )
    path = workspace / "wrong.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    result = _audit(workspace, **{"--allowlist": str(path)})
    assert result.exit_code != 0
    assert not _UUID_RE.search(result.output)


def test_an_incomplete_allowlist_fails_the_unreviewed_check(workspace: Path) -> None:
    """Dropping an entry makes the catalog feature unreviewed, and the audit fail."""
    document = yaml.safe_load(
        (workspace / "allowlist.yaml").read_text(encoding="utf-8")
    )
    dropped = document["entries"].pop()["name"]
    path = workspace / "incomplete.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    result = _audit(workspace, **{"--allowlist": str(path)})
    assert result.exit_code != 0
    assert "NO_UNREVIEWED_CATALOG_FEATURE" in result.output
    assert dropped in _report(workspace)["checks"][4]["message"]  # type: ignore[index]


def test_a_manifest_from_another_catalog_fails_the_audit(workspace: Path) -> None:
    """Provenance is compared, not assumed."""
    manifest = json.loads(
        (workspace / "processed" / "feature_manifest.json").read_text(encoding="utf-8")
    )
    manifest["feature_catalog_fingerprint"] = "d" * 64
    path = workspace / "wrong-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _audit(workspace, **{"--feature-manifest": str(path)})
    assert result.exit_code != 0
    assert "SCHEMA_AND_CATALOG_FINGERPRINT_MATCH" in result.output


def test_an_unreadable_manifest_exits_non_zero(workspace: Path) -> None:
    """A malformed manifest fails loudly rather than skipping its check."""
    path = workspace / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    result = _audit(workspace, **{"--feature-manifest": str(path)})
    assert result.exit_code != 0


def test_a_manifest_that_is_not_an_object_exits_non_zero(workspace: Path) -> None:
    """A JSON array is not a manifest."""
    path = workspace / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    result = _audit(workspace, **{"--feature-manifest": str(path)})
    assert result.exit_code != 0


def test_a_failing_audit_names_the_failing_checks(workspace: Path) -> None:
    """A non-zero exit must say what failed, in stable coded names."""
    manifest = json.loads(
        (workspace / "processed" / "feature_manifest.json").read_text(encoding="utf-8")
    )
    manifest["feature_schema_version"] = "9.9.9"
    path = workspace / "wrong-schema.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _audit(workspace, **{"--feature-manifest": str(path)})
    assert result.exit_code != 0
    assert "Audit failed" in result.output


# ---------------------------------------------------------------------------
# Drafting the allowlist
# ---------------------------------------------------------------------------


def test_emitting_an_allowlist_says_it_is_a_draft(tmp_path: Path) -> None:
    """The command must not let a generated file be mistaken for a reviewed one."""
    result = _invoke("ml", "catalog", "--emit-allowlist", str(tmp_path / "a.yaml"))
    assert result.exit_code == 0
    assert "draft, not a reviewed contract" in result.stdout
    assert (tmp_path / "a.yaml").exists()


def test_emitting_an_allowlist_reports_a_relative_path(tmp_path: Path) -> None:
    """No absolute path under a personal home directory reaches the terminal."""
    result = _invoke("ml", "catalog", "--emit-allowlist", str(tmp_path / "a.yaml"))
    assert str(Path.home()) not in result.stdout


def test_emitting_with_an_unreadable_feature_config_exits_non_zero(
    tmp_path: Path,
) -> None:
    """A typo in the configuration path fails rather than drafting a default."""
    result = _invoke(
        "ml",
        "catalog",
        "--emit-allowlist",
        str(tmp_path / "a.yaml"),
        "--feature-config",
        str(tmp_path / "absent.yaml"),
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Both entry points
# ---------------------------------------------------------------------------


def test_the_console_script_entry_point_runs_the_audit(workspace: Path) -> None:
    """``password-attack-detector ml audit-features`` as an installed script."""
    completed = subprocess.run(
        [
            "password-attack-detector",
            "ml",
            "audit-features",
            "--features",
            str(workspace / "processed" / "feature_snapshots.parquet"),
            "--labels",
            str(workspace / "processed" / "feature_labels.parquet"),
            "--splits",
            str(workspace / "processed" / "feature_splits.parquet"),
            "--campaign-labels",
            str(workspace / "labels.parquet"),
            "--feature-manifest",
            str(workspace / "processed" / "feature_manifest.json"),
            "--allowlist",
            str(workspace / "allowlist.yaml"),
            "--feature-config",
            str(workspace / "features.yaml"),
            "--config",
            str(_repo_root() / "configs" / "ml" / "model-testing.yaml"),
            "-o",
            str(workspace / "script-reports"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_repo_root(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "Audit status: pass" in completed.stdout


def test_the_module_entry_point_advertises_the_audit() -> None:
    """``python -m password_attack_detector ml --help`` lists both commands."""
    completed = subprocess.run(
        [sys.executable, "-m", "password_attack_detector", "ml", "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=_repo_root(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "audit-features" in completed.stdout
    assert "catalog" in completed.stdout


# ---------------------------------------------------------------------------
# The campaign-group invariant, end to end
# ---------------------------------------------------------------------------


def test_campaigns_stay_whole_across_the_published_partition(workspace: Path) -> None:
    """Every campaign group is indivisible on a real published dataset.

    Group counts exceeding campaign counts proves the benign placeholder rows
    were resolved to singletons rather than fused into one campaign; the audit's
    own disjointness check proves no campaign crossed the boundary.
    """
    assert _audit(workspace).exit_code == 0
    payload = _report(workspace)
    partition = payload["validation_partition"]
    assert isinstance(partition, dict)

    groups = partition["partition_a_groups"] + partition["partition_b_groups"]
    campaigns = partition["partition_a_campaigns"] + partition["partition_b_campaigns"]
    assert campaigns > 0
    assert groups > campaigns, (groups, campaigns)

    raw_checks = payload["checks"]
    assert isinstance(raw_checks, list)
    checks = {check["name"]: check["status"] for check in raw_checks}
    assert checks["VALIDATION_HALVES_CAMPAIGN_DISJOINT"] == "pass"
    assert checks["VALIDATION_SUPPORT_SUFFICIENT"] == "pass"


def test_the_generator_placeholder_does_not_starve_a_half(workspace: Path) -> None:
    """Both halves carry benign rows, which the placeholder-as-campaign reading loses."""
    assert _audit(workspace).exit_code == 0
    partition = _report(workspace)["validation_partition"]
    assert isinstance(partition, dict)
    assert partition["partition_a_benign_rows"] > 0
    assert partition["partition_b_benign_rows"] > 0


def test_campaigns_named_like_the_placeholder_are_still_campaigns(
    tmp_path: Path, workspace: Path
) -> None:
    """End to end, the metadata outranks the spelling.

    The same dataset published twice, differing only in what the campaigns are
    called: ``campaign-00`` in one, ``normal-00`` in the other. The second reads
    exactly like the benign filler and is declared exactly like a campaign, and
    the CLI has to reach the same partition -- same group counts, same campaign
    counts, same benign support in both halves. A resolver that let the shape
    decide would dissolve every campaign in the second run into singletons.
    """
    renamed = _publish(tmp_path / "renamed", prefix="normal-")

    assert _audit(workspace).exit_code == 0
    assert _audit(renamed).exit_code == 0
    ordinary = _report(workspace)["validation_partition"]
    lookalike = _report(renamed)["validation_partition"]
    assert isinstance(ordinary, dict)
    assert isinstance(lookalike, dict)

    for key in (
        "partition_a_groups",
        "partition_b_groups",
        "partition_a_campaigns",
        "partition_b_campaigns",
        "partition_a_benign_rows",
        "partition_b_benign_rows",
    ):
        assert lookalike[key] == ordinary[key], key
    assert lookalike["partition_a_campaigns"] > 0
