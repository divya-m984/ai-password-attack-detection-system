"""A published feature dataset and the artifact root trained from it.

Shared by every command suite that needs real Milestone 6 artifacts to work
against. The dataset is deliberately tiny -- 160 events over about two and a
half hours -- and the 720-hour development workflow is never run here: a
contract test that needs a month of traffic to express itself is testing the
generator rather than the command.

Nothing in this module is a test. It is the fixture the tests share, kept in one
place so the Milestone 6 training suite and the Milestone 7 selection and freeze
suites cannot drift into disagreeing about what a published experiment looks
like.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import yaml
from typer.testing import CliRunner, Result

from password_attack_detector.cli import app
from password_attack_detector.data.enums import CampaignStage, ScenarioType
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.data.serialization import (
    write_events_parquet,
    write_labels_parquet,
)
from tests.features.factories import make_event

runner = CliRunner()

#: Any pseudonym the CLI might print, in the exact shape the feature layer
#: emits them. Asserted absent rather than merely unlikely.
PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")

#: The fixture stream. Several campaigns, spread through it, so the validation
#: split carries more than one group and the boundary has somewhere to fall.
CAMPAIGN_COUNT = 12
CAMPAIGN_LENGTH = 4
EVENT_COUNT = 160


def repo_root() -> Path:
    """Return the repository root, located from this file."""
    return Path(__file__).resolve().parents[2]


def feature_config() -> dict[str, object]:
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


def campaign_of(index: int) -> str | None:
    """Return the campaign an event index belongs to, or ``None`` if benign."""
    stride = EVENT_COUNT // CAMPAIGN_COUNT
    position = index % stride
    if position < CAMPAIGN_LENGTH:
        return f"campaign-{index // stride:02d}"
    return None


def events() -> list[AuthEvent]:
    """Return a deterministic event stream with periodic bursts of failures."""
    return [
        make_event(
            t=float(index) * 60.0,
            user=f"u{index % 5 + 1}",
            source=f"s{1 if campaign_of(index) else index % 3 + 2}",
            device=f"d{index % 3 + 1}",
            outcome="failure" if campaign_of(index) else "success",
            response_time_ms=40 if campaign_of(index) else 200 + index % 50,
            country="US" if campaign_of(index) else "GB",
            latitude=37.8 if campaign_of(index) else 51.5,
            longitude=-122.4 if campaign_of(index) else -0.1,
            key=str(index),
        )
        for index in range(EVENT_COUNT)
    ]


def labels(events: Sequence[AuthEvent]) -> list[GroundTruthLabel]:
    """Return ground truth pairing each burst with a campaign identifier.

    Two scenarios alternate across campaigns so the known-category head has more
    than one class to learn. Benign rows carry the generator's ``normal-<seed>``
    placeholder, exactly as real generator output does.
    """
    scenarios = (ScenarioType.BRUTE_FORCE, ScenarioType.PASSWORD_SPRAYING)
    labels = []
    for index, event in enumerate(events):
        campaign = campaign_of(index)
        ordinal = 0 if campaign is None else int(campaign.split("-")[1])
        labels.append(
            GroundTruthLabel(
                event_id=event.event_id,
                campaign_id=campaign or "normal-864209",
                scenario=(
                    scenarios[ordinal % len(scenarios)]
                    if campaign is not None
                    else ScenarioType.NORMAL
                ),
                malicious=campaign is not None,
                supervised_training_eligible=True,
                generator_version="1.0.0",
                campaign_stage=None if campaign is None else CampaignStage.ACTIVE,
            )
        )
    return labels


def invoke(*arguments: str) -> Result:
    """Run the CLI with *arguments* and return the result."""
    return runner.invoke(app, list(arguments))


def build_workspace(root: Path) -> Path:
    """Publish a feature dataset and a drafted allowlist under *root*."""
    stream = events()
    write_events_parquet(stream, root / "events.parquet")
    write_labels_parquet(labels(stream), root / "labels.parquet")
    (root / "features.yaml").write_text(
        yaml.safe_dump(feature_config()), encoding="utf-8"
    )

    built = invoke(
        "features",
        "build",
        str(root / "events.parquet"),
        "--labels",
        str(root / "labels.parquet"),
        "--config",
        str(root / "features.yaml"),
        "-o",
        str(root / "processed"),
        "--reports-dir",
        str(root / "reports"),
    )
    assert built.exit_code == 0, built.output

    drafted = invoke(
        "ml",
        "catalog",
        "--emit-allowlist",
        str(root / "allowlist.yaml"),
        "--feature-config",
        str(root / "features.yaml"),
    )
    assert drafted.exit_code == 0, drafted.output
    return root


def train(workspace: Path, output_root: Path, **replace: str) -> Result:
    """Run ``ml train`` over the published workspace."""
    arguments = {
        "--features": str(workspace / "processed" / "feature_snapshots.parquet"),
        "--labels": str(workspace / "processed" / "feature_labels.parquet"),
        "--splits": str(workspace / "processed" / "feature_splits.parquet"),
        "--campaign-labels": str(workspace / "labels.parquet"),
        "--feature-manifest": str(workspace / "processed" / "feature_manifest.json"),
        "--allowlist": str(workspace / "allowlist.yaml"),
        "--feature-config": str(workspace / "features.yaml"),
        "--config": str(repo_root() / "configs" / "ml" / "model-testing.yaml"),
        "--output-root": str(output_root),
    }
    arguments.update(replace)
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return invoke("ml", "train", *flat)
