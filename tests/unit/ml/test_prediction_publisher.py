"""Publishing predictions: the happy path once, and the rollbacks repeatedly.

The interesting behaviour is not that a valid publication lands. It is that a
write that fails halfway, a validation that refuses, a semantically different
publication at the same identity, and a corrupt destination all stop -- with the
destination and every previously published artifact exactly as they were, and no
flag that lets any of them through.

The fault-injection tests monkeypatch the publisher's own call sites rather than
the filesystem, so each one names the *stage* that failed rather than hoping a
generic error lands somewhere useful.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import ExperimentPublicationError
from password_attack_detector.ml import prediction_publisher as publisher
from password_attack_detector.ml.enums import AuditStatus, MLSplit
from password_attack_detector.ml.prediction_manifest import (
    BINARY_PREDICTION_FILE,
    CATEGORY_PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    PREDICTIONS_DIR,
    QUALITY_REPORT_JSON_FILE,
    QUALITY_REPORT_MD_FILE,
    VALIDATION_RESULT_FILE,
    PredictionManifest,
)
from password_attack_detector.ml.prediction_publisher import publish_predictions
from password_attack_detector.ml.prediction_validation import validate_publication
from tests.ml import predictions as px
from tests.ml import runs as rx


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> rx.Experiment:
    """Publish one experiment, shared by every test in this module."""
    return rx.publish_experiment(tmp_path_factory.mktemp("publisher-source"))


@pytest.fixture
def prepared(source: rx.Experiment, tmp_path: Path) -> px.Prepared:
    """Return a writable frozen champion, one per test."""
    return px.prepare(tmp_path / "root", source=source)


def publish(prepared: px.Prepared, **overrides: Any) -> Any:
    """Publish predictions for the test split under the prepared champion."""
    settings: dict[str, Any] = {
        "champion": prepared.champion(),
        "dataset": px.inference_dataset(prepared),
        "root": prepared.root,
    }
    settings.update(overrides)
    return publish_predictions(**settings)


def directory_of(prepared: px.Prepared, publication: Any) -> Path:
    """Return the published directory."""
    return Path(prepared.root / PREDICTIONS_DIR / str(publication.prediction_id))


def fail_at(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make the publisher's *name* stage raise."""

    def refuse(*_: Any, **__: Any) -> None:
        raise RuntimeError(f"injected failure in {name}")

    monkeypatch.setattr(publisher, name, refuse)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_publication_lands_complete_and_validated(prepared: px.Prepared) -> None:
    """Every declared artifact present, and the whole thing validating."""
    publication = publish(prepared)
    assert publication.created is True
    assert publication.validation_status is AuditStatus.PASS
    directory = directory_of(prepared, publication)
    present = {item.name for item in directory.iterdir()}
    assert BINARY_PREDICTION_FILE in present
    assert VALIDATION_RESULT_FILE in present
    assert QUALITY_REPORT_JSON_FILE in present
    assert QUALITY_REPORT_MD_FILE in present
    assert PREDICTION_MANIFEST_FILE in present
    assert validate_publication(directory).passed


def test_the_category_head_is_published_for_the_flagged_rows_only(
    prepared: px.Prepared,
) -> None:
    """Beside the binary artifact, never instead of it, and only for triage.

    The category head was fitted on known-malicious rows, so the artifact covers
    the rows the binary champion routed to it and no others.
    """
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
    )

    publication = publish(prepared)
    directory = directory_of(prepared, publication)
    assert (directory / CATEGORY_PREDICTION_FILE).is_file()

    binary = read_binary_predictions(directory / BINARY_PREDICTION_FILE)
    flagged = sum(1 for row in binary if row.flagged_malicious)
    assert publication.category_row_count == flagged
    assert publication.category_row_count < publication.row_count


def test_no_category_artifact_is_fabricated_when_none_was_asked_for(
    prepared: px.Prepared,
) -> None:
    """An all-unknown stand-in would be a model output nothing produced."""
    publication = publish(prepared, include_category=False)
    assert publication.category_row_count is None
    assert not (
        directory_of(prepared, publication) / CATEGORY_PREDICTION_FILE
    ).is_file()


def test_the_manifest_is_written_last(
    prepared: px.Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its presence means everything it covers is already there.

    Asserted by failing the publication *after* the row artifacts are staged and
    checking that no manifest exists anywhere -- not in the destination, and not
    in a staging directory somebody could later mistake for one.
    """
    fail_at(monkeypatch, "build_quality_report")
    with pytest.raises(ExperimentPublicationError):
        publish(prepared)
    assert not list((prepared.root / PREDICTIONS_DIR).rglob(PREDICTION_MANIFEST_FILE))


def test_the_publication_names_the_champion_it_came_from(
    prepared: px.Prepared,
) -> None:
    """Every lineage fingerprint, once, on the manifest."""
    publication = publish(prepared)
    champion = prepared.champion()
    manifest = PredictionManifest.from_json(
        (directory_of(prepared, publication) / PREDICTION_MANIFEST_FILE).read_text(
            encoding="utf-8"
        )
    )
    assert manifest.lineage.champion_lock_fingerprint == champion.lock.lock_fingerprint
    assert manifest.lineage.model_id == champion.lock.model_id
    assert manifest.lineage.decision_threshold == champion.decision_threshold
    assert manifest.lineage.binary_score_kind is champion.score_kind
    assert manifest.inference_input_fingerprint == (
        px.inference_dataset(prepared).inference_input_fingerprint
    )


# ---------------------------------------------------------------------------
# Idempotency and conflict
# ---------------------------------------------------------------------------


def test_publishing_the_same_predictions_twice_is_idempotent(
    prepared: px.Prepared,
) -> None:
    """The same evidence is the same publication, not a second one."""
    first = publish(prepared)
    second = publish(prepared)
    assert second.prediction_id == first.prediction_id
    assert second.created is False
    assert len(list((prepared.root / PREDICTIONS_DIR).iterdir())) == 1


def test_a_semantically_different_publication_gets_its_own_identity(
    prepared: px.Prepared,
) -> None:
    """Different predictions are a different publication, never an overwrite."""
    first = publish(prepared)
    second = publish(
        prepared, dataset=px.inference_dataset(prepared, scope=MLSplit.VALIDATION)
    )
    assert second.prediction_id != first.prediction_id
    assert (directory_of(prepared, first) / PREDICTION_MANIFEST_FILE).is_file()
    assert (directory_of(prepared, second) / PREDICTION_MANIFEST_FILE).is_file()


def test_a_contradicting_publication_at_one_identity_is_refused(
    prepared: px.Prepared,
) -> None:
    """A published prediction is evidence, and is never rewritten."""
    publication = publish(prepared)
    directory = directory_of(prepared, publication)
    manifest = json.loads(
        (directory / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    manifest["split_membership_fingerprint"] = "9" * 64
    (directory / PREDICTION_MANIFEST_FILE).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ExperimentPublicationError, match=r"not readable|different"):
        publish(prepared)


def test_a_destination_without_a_manifest_is_never_completed_in_place(
    prepared: px.Prepared,
) -> None:
    """Its two halves would come from two runs."""
    publication = publish(prepared)
    (directory_of(prepared, publication) / PREDICTION_MANIFEST_FILE).unlink()
    with pytest.raises(ExperimentPublicationError, match="never completed in place"):
        publish(prepared)


def test_a_corrupt_destination_is_a_fault_not_an_idempotent_republication(
    prepared: px.Prepared,
) -> None:
    """Corruption is not idempotency."""
    publication = publish(prepared)
    directory = directory_of(prepared, publication)
    (directory / QUALITY_REPORT_MD_FILE).write_text("tampered", encoding="utf-8")
    with pytest.raises(ExperimentPublicationError, match="does not validate"):
        publish(prepared)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage",
    [
        "write_binary_predictions",
        "read_binary_predictions",
        "validate_staged_predictions",
        "build_quality_report",
        "quality_report_to_markdown",
    ],
)
def test_a_failure_at_any_stage_leaves_no_partial_publication(
    prepared: px.Prepared, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """No staging directory, no destination, nothing half-written."""
    fail_at(monkeypatch, stage)
    with pytest.raises(ExperimentPublicationError):
        publish(prepared)
    predictions_root = prepared.root / PREDICTIONS_DIR
    assert not any(predictions_root.iterdir())


def test_a_validation_failure_promotes_nothing(
    prepared: px.Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publication that would fail 'ml validate' the moment it landed."""

    class _Failed:
        """A validation outcome that refuses, however sound the artifact is."""

        passed = False
        failures = ("M009",)
        status = AuditStatus.FAIL

    monkeypatch.setattr(publisher, "validate_publication", lambda _directory: _Failed())
    with pytest.raises(ExperimentPublicationError, match="did not validate"):
        publish(prepared)
    assert not any((prepared.root / PREDICTIONS_DIR).iterdir())


def test_an_existing_publication_survives_a_later_failure(
    prepared: px.Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one property that makes a publication evidence."""
    publication = publish(prepared)
    directory = directory_of(prepared, publication)
    before = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }

    fail_at(monkeypatch, "write_binary_predictions")
    with pytest.raises(ExperimentPublicationError):
        publish(
            prepared,
            dataset=px.inference_dataset(prepared, scope=MLSplit.VALIDATION),
        )

    after = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }
    assert after == before


def test_a_leftover_staging_directory_is_replaced_not_reused(
    prepared: px.Prepared,
) -> None:
    """A previous crash leaves nothing a later run could adopt."""
    predictions_root = prepared.root / PREDICTIONS_DIR
    predictions_root.mkdir(parents=True, exist_ok=True)
    dataset = px.inference_dataset(prepared)
    stale = predictions_root / f".staging-{dataset.inference_input_fingerprint[:16]}"
    stale.mkdir()
    (stale / "left-over.txt").write_text("from an earlier crash", encoding="utf-8")

    publication = publish(prepared, dataset=dataset)
    assert not stale.exists()
    assert "left-over.txt" not in {
        item.name for item in directory_of(prepared, publication).iterdir()
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_roots_produce_the_same_publication(
    source: rx.Experiment, tmp_path: Path
) -> None:
    """Same identity, same manifest bytes, same reports, in two directories."""
    first = px.prepare(tmp_path / "one", source=source)
    second = px.prepare(tmp_path / "two", source=source)
    left = publish(first)
    right = publish(second)

    assert left.prediction_id == right.prediction_id
    assert left.prediction_content_fingerprint == right.prediction_content_fingerprint
    for name in (
        BINARY_PREDICTION_FILE,
        CATEGORY_PREDICTION_FILE,
        VALIDATION_RESULT_FILE,
        QUALITY_REPORT_JSON_FILE,
        QUALITY_REPORT_MD_FILE,
        PREDICTION_MANIFEST_FILE,
    ):
        assert (directory_of(first, left) / name).read_bytes() == (
            directory_of(second, right) / name
        ).read_bytes(), name


def test_a_shuffled_source_table_produces_the_same_publication(
    prepared: px.Prepared,
) -> None:
    """Physical row order reaches neither the bytes nor the identity.

    Asserted through idempotency, which is the strongest available form of it:
    the shuffled publication is recognised as the *same* publication and writes
    nothing, rather than landing beside it as a near-identical twin.
    """
    ordered = publish(prepared)
    shuffled = publish(prepared, dataset=px.shuffled_inference_dataset(prepared))
    assert shuffled.prediction_id == ordered.prediction_id
    assert shuffled.created is False
    assert len(list((prepared.root / PREDICTIONS_DIR).iterdir())) == 1


def test_the_holdout_is_a_separate_publication(prepared: px.Prepared) -> None:
    """Never merged into the test artifact, and marked as a probe."""
    test = publish(prepared)
    holdout = publish(
        prepared,
        dataset=px.inference_dataset(prepared, scope=MLSplit.NOVEL_ANOMALY_HOLDOUT),
    )
    assert holdout.prediction_id != test.prediction_id
    assert holdout.scope_role == "generalisation_probe"
    assert test.scope_role == "supervised_prediction"
    assert (directory_of(prepared, test) / PREDICTION_MANIFEST_FILE).is_file()


def test_publishing_the_holdout_does_not_touch_the_test_artifact(
    prepared: px.Prepared,
) -> None:
    """A generalisation probe changes nothing about a supervised publication."""
    test = publish(prepared)
    directory = directory_of(prepared, test)
    before = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }
    publish(
        prepared,
        dataset=px.inference_dataset(prepared, scope=MLSplit.NOVEL_ANOMALY_HOLDOUT),
    )
    after = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }
    assert after == before


def test_publishing_writes_no_ledger_record(prepared: px.Prepared) -> None:
    """Prediction identity lives on the manifest, not in the experiment ledger."""
    before = sorted(path.name for path in (prepared.root / "ledger").rglob("*.json"))
    publish(prepared)
    after = sorted(path.name for path in (prepared.root / "ledger").rglob("*.json"))
    assert after == before
    assert not (prepared.root / "ledger" / "test_evaluation").exists()


def test_publication_leaves_the_frozen_artifacts_untouched(
    prepared: px.Prepared,
) -> None:
    """Prediction reads the champion; it never writes to it."""
    champion_root = prepared.root / "champion"
    runs_root = prepared.root / "runs"
    before = {
        str(path.relative_to(prepared.root)): path.read_bytes()
        for path in [*champion_root.rglob("*"), *runs_root.rglob("*")]
        if path.is_file()
    }
    publish(prepared)
    after = {
        str(path.relative_to(prepared.root)): path.read_bytes()
        for path in [*champion_root.rglob("*"), *runs_root.rglob("*")]
        if path.is_file()
    }
    assert after == before


def test_a_publication_directory_is_never_copied_over(
    prepared: px.Prepared, tmp_path: Path
) -> None:
    """A destination is promoted by rename, so a partial copy cannot exist."""
    publication = publish(prepared)
    copied = tmp_path / "copy"
    shutil.copytree(directory_of(prepared, publication), copied)
    assert validate_publication(copied).passed
