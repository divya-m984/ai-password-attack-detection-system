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


#: Campaigns reassigned to the novel-anomaly holdout.  Two of twelve, and
#: deliberately *early* ones: the splits are chronological, so carving out late
#: campaigns would starve validation and TEST of positives and the eligibility
#: audit would refuse the run -- correctly. Nothing is weakened to accommodate
#: the holdout; the campaigns are chosen so every supervised split keeps the
#: support it already required.
HOLDOUT_CAMPAIGNS = ("campaign-01", "campaign-02")


def labels_with_holdout(events: Sequence[AuthEvent]) -> list[GroundTruthLabel]:
    """Return ground truth in which two campaigns are genuine novel anomalies.

    A novel-anomaly row is one the supervised pipeline was never allowed to fit
    on: it carries the ``NOVEL_ANOMALY_HOLDOUT`` scenario and
    ``supervised_training_eligible=False``, which is what routes it out of
    TRAIN, validation, and TEST alike. The rows are otherwise ordinary, so the
    holdout is a real population rather than a marker column.
    """
    rewritten = []
    for label in labels(events):
        if label.campaign_id in HOLDOUT_CAMPAIGNS:
            rewritten.append(
                label.model_copy(
                    update={
                        "scenario": ScenarioType.NOVEL_ANOMALY_HOLDOUT,
                        "supervised_training_eligible": False,
                    }
                )
            )
        else:
            rewritten.append(label)
    return rewritten


def invoke(*arguments: str) -> Result:
    """Run the CLI with *arguments* and return the result."""
    return runner.invoke(app, list(arguments))


def build_workspace(
    root: Path,
    *,
    config: dict[str, object] | None = None,
    with_holdout: bool = False,
    stream: Sequence[AuthEvent] | None = None,
) -> Path:
    """Publish a feature dataset and a drafted allowlist under *root*.

    *config* overrides the feature configuration. The comparison suites need a
    catalog wide enough for the Phase 4 rules as well as for the model, and
    widening the default would silently retrain every other suite.

    *stream* overrides the event stream itself, for the reproducibility audit's
    positive control: proving a lineage is live needs a genuinely different
    training population, not a differently configured view of the same one.
    """
    stream = list(stream) if stream is not None else events()
    write_events_parquet(stream, root / "events.parquet")
    ground_truth = labels_with_holdout(stream) if with_holdout else labels(stream)
    write_labels_parquet(ground_truth, root / "labels.parquet")
    (root / "features.yaml").write_text(
        yaml.safe_dump(config if config is not None else feature_config()),
        encoding="utf-8",
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


#: The configuration every command in these suites is run under.
ML_CONFIG = str(repo_root() / "configs" / "ml" / "model-testing.yaml")


def freeze(workspace: Path, output_root: Path, reports: Path) -> Path:
    """Train, select, and freeze a champion, returning the artifact root.

    The prediction suites need a *frozen champion*, and a frozen champion is the
    end of a real pipeline. Building one by hand would let those suites pass
    against a shape the commands never produce.
    """
    trained = train(workspace, output_root)
    assert trained.exit_code == 0, trained.output
    selected = invoke(
        "ml",
        "select",
        "--output-root",
        str(output_root),
        "--config",
        ML_CONFIG,
        "--reports-dir",
        str(reports),
    )
    assert selected.exit_code == 0, selected.output
    frozen = invoke(
        "ml",
        "freeze-champion",
        "--output-root",
        str(output_root),
        "--config",
        ML_CONFIG,
    )
    assert frozen.exit_code == 0, frozen.output
    return output_root


def predict(
    workspace: Path, output_root: Path, *, split: str = "test", **replace: str
) -> Result:
    """Run ``ml predict`` over the frozen champion under *output_root*.

    Notice what is absent from the argument map: there is no ``--labels`` and no
    ``--campaign-labels``. Prediction is carried out on feature-side inputs
    alone, and the fixture cannot supply ground truth even by mistake.
    """
    arguments = {
        "--features": str(workspace / "processed" / "feature_snapshots.parquet"),
        "--splits": str(workspace / "processed" / "feature_splits.parquet"),
        "--feature-manifest": str(workspace / "processed" / "feature_manifest.json"),
        "--allowlist": str(workspace / "allowlist.yaml"),
        "--feature-config": str(workspace / "features.yaml"),
        "--config": ML_CONFIG,
        "--output-root": str(output_root),
        "--split": split,
    }
    arguments.update(replace)
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return invoke("ml", "predict", *flat)


def write_rule_config(path: Path) -> Path:
    """Write the Phase 4 configuration these suites run under."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rule_config()), encoding="utf-8")
    return path


def detect(workspace: Path, output_dir: Path) -> Result:
    """Run the Phase 4 engine over the same feature snapshots.

    The comparison suites need a *published* rule run, not a recomputed one: an
    evaluation whose rule arm was produced on the fly under whatever
    configuration happened to be in scope would be an evaluation against a
    moving target.
    """
    config = write_rule_config(output_dir / "rules.yaml")
    return invoke(
        "detection",
        "run",
        "--config",
        str(config),
        "--features",
        str(workspace / "processed" / "feature_snapshots.parquet"),
        "--feature-manifest",
        str(workspace / "processed" / "feature_manifest.json"),
        "--feature-config",
        str(workspace / "features.yaml"),
        "-o",
        str(output_dir),
        "--reports-dir",
        str(output_dir / "reports"),
    )


def prediction_ids(output_root: Path) -> dict[str, str]:
    """Return every published prediction identifier, keyed by the split it scored."""
    import json

    from password_attack_detector.ml.prediction_manifest import (
        PREDICTION_MANIFEST_FILE,
        PREDICTIONS_DIR,
    )

    found: dict[str, str] = {}
    for item in sorted((output_root / PREDICTIONS_DIR).iterdir()):
        manifest = item / PREDICTION_MANIFEST_FILE
        if manifest.is_file():
            scope = json.loads(manifest.read_text(encoding="utf-8"))["scope"]
            found[str(scope)] = item.name
    return found


def evaluate(
    workspace: Path,
    output_root: Path,
    detection_dir: Path,
    reports: Path,
    **replace: str,
) -> Result:
    """Run ``ml evaluate`` over a frozen champion and a published TEST prediction."""
    from password_attack_detector.detection.serialization import RISK_FILE

    arguments = {
        "--features": str(workspace / "processed" / "feature_snapshots.parquet"),
        "--labels": str(workspace / "processed" / "feature_labels.parquet"),
        "--splits": str(workspace / "processed" / "feature_splits.parquet"),
        "--campaign-labels": str(workspace / "labels.parquet"),
        "--allowlist": str(workspace / "allowlist.yaml"),
        "--feature-config": str(workspace / "features.yaml"),
        "--config": ML_CONFIG,
        "--risk-assessments": str(detection_dir / RISK_FILE),
        "--output-root": str(output_root),
        "--reports-dir": str(reports),
    }
    arguments.update(replace)
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return invoke("ml", "evaluate", *flat)


def materialize(
    workspace: Path,
    output_root: Path,
    detection_dir: Path,
    *,
    validation_prediction: str,
    **replace: str,
) -> Result:
    """Run ``deploy materialize`` over the frozen champion and its lineage.

    Notice what is absent from the argument map: there is no ``--prediction`` for
    a TEST publication and no risk file scoped to TEST anchors. The
    reconstruction reproduces a decision that was frozen before TEST was opened,
    and the fixture cannot hand it a TEST quantity even by mistake.

    ``--rule-config`` is absent too, and deliberately: :func:`evaluate` does not
    pass one either, so the frozen selection was made against the *default*
    detection configuration's fingerprint. Passing the workspace's rules here
    would be a genuinely different upstream input, and the materializer would
    correctly refuse -- which is what
    ``test_a_changed_rule_configuration_is_refused`` asserts on purpose.
    """
    from password_attack_detector.detection.serialization import RISK_FILE

    arguments = {
        "--features": str(workspace / "processed" / "feature_snapshots.parquet"),
        "--labels": str(workspace / "processed" / "feature_labels.parquet"),
        "--splits": str(workspace / "processed" / "feature_splits.parquet"),
        "--campaign-labels": str(workspace / "labels.parquet"),
        "--allowlist": str(workspace / "allowlist.yaml"),
        "--feature-config": str(workspace / "features.yaml"),
        "--config": ML_CONFIG,
        "--risk-assessments": str(detection_dir / RISK_FILE),
        "--validation-prediction": validation_prediction,
        "--output-root": str(output_root),
    }
    arguments.update(replace)
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return invoke("deploy", "materialize", *flat)


def rule_config() -> dict[str, object]:
    """A Phase 4 configuration whose rules read the CI catalog's own windows.

    The rule catalog's defaults reach for 15-minute and 1-hour windows. A
    CI-sized stream cannot declare those and still leave a supervised split
    with any support, so the *rules* are pointed at the windows this workspace
    publishes rather than the workspace being widened to suit the rules.

    Every rule stays enabled. Disabling the ones whose default windows do not
    fit would hand the comparison a rule arm weaker than the real engine, which
    is exactly the unfairness the comparison exists to avoid.
    """
    windows = {
        "PAD-BF-001": {"window": "5m", "cardinality_window": "5m"},
        "PAD-BOT-001": {"dispersion_window": "5m", "cardinality_window": "5m"},
        "PAD-CS-001": {"window": "5m", "cardinality_window": "5m"},
        "PAD-DBF-001": {"window": "5m", "cardinality_window": "5m"},
        "PAD-MFA-001": {"window": "5m"},
        "PAD-PS-001": {"window": "5m", "cardinality_window": "5m"},
    }
    return {
        "rules": {
            rule_id: {"parameters": parameters}
            for rule_id, parameters in windows.items()
        }
    }


def explain(
    workspace: Path,
    output_root: Path,
    *,
    prediction_id: str,
    split: str = "validation",
    **replace: str,
) -> Result:
    """Run ``ml explain`` over a frozen champion and a published prediction.

    Notice what is absent from the argument map, again: there is no
    ``--labels``. Attribution decomposes what a model said, and the fixture
    cannot hand it an answer even by mistake.
    """
    arguments = {
        "--features": str(workspace / "processed" / "feature_snapshots.parquet"),
        "--splits": str(workspace / "processed" / "feature_splits.parquet"),
        "--feature-manifest": str(workspace / "processed" / "feature_manifest.json"),
        "--allowlist": str(workspace / "allowlist.yaml"),
        "--feature-config": str(workspace / "features.yaml"),
        "--config": ML_CONFIG,
        "--output-root": str(output_root),
        "--split": split,
        "--prediction": prediction_id,
    }
    arguments.update(replace)
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return invoke("ml", "explain", *flat)


def drift(
    workspace: Path,
    output_root: Path,
    reports: Path,
    *,
    incoming_split: str = "validation",
    reference_prediction: str | None = None,
    incoming_prediction: str | None = None,
    **replace: str,
) -> Result:
    """Run ``ml drift`` against the frozen training reference profile.

    The prediction options come as a pair or not at all, matching the command:
    comparing an output distribution against no baseline measures nothing.
    """
    arguments = {
        "--features": str(workspace / "processed" / "feature_snapshots.parquet"),
        "--splits": str(workspace / "processed" / "feature_splits.parquet"),
        "--feature-manifest": str(workspace / "processed" / "feature_manifest.json"),
        "--allowlist": str(workspace / "allowlist.yaml"),
        "--feature-config": str(workspace / "features.yaml"),
        "--config": ML_CONFIG,
        "--output-root": str(output_root),
        "--incoming-split": incoming_split,
        "--reports-dir": str(reports),
    }
    if reference_prediction is not None:
        arguments["--reference-prediction"] = reference_prediction
    if incoming_prediction is not None:
        arguments["--incoming-prediction"] = incoming_prediction
    arguments.update(replace)
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return invoke("ml", "drift", *flat)


def explanation_directory(output_root: Path) -> Path:
    """Return the single published explanation directory under *output_root*."""
    published = sorted(
        item for item in (output_root / "explanations").iterdir() if item.is_dir()
    )
    assert len(published) == 1, [item.name for item in published]
    return published[0]
