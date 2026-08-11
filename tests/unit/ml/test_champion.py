"""Freezing a champion, and every state in which freezing is refused.

The refusals are most of this file, and deliberately so. A freeze is the last
decision made before the test split is opened, so the interesting behaviour is
not that a valid champion freezes -- it is that a selection which found nothing,
a run whose artifact no longer verifies, a comparator that vanished from the
ledger, and a contradicting second freeze all stop, with no flag that lets them
through.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import ExperimentPublicationError
from password_attack_detector.ml.champion import (
    CHAMPION_LOCK_FILE,
    ChampionLock,
    build_champion_lock,
    freeze_champion,
    reconcile_freezes,
    scope_key_for,
)
from password_attack_detector.ml.enums import ChampionStatus, MLTask, ModelFamily
from password_attack_detector.ml.ledger import ExperimentLedger
from password_attack_detector.ml.selection import (
    CandidateEvidence,
    publish_selection,
    reconcile_selections,
    select_binary_champion,
    select_category_head,
)
from tests.ml import runs as rx

BINARY = MLTask.BINARY_MALICIOUS


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> rx.Experiment:
    """Return one published experiment, shared by every test in this module."""
    return rx.publish_experiment(tmp_path_factory.mktemp("experiment"))


@dataclasses.dataclass(frozen=True, slots=True)
class Frozen:
    """One experiment copied into a writable root, with its two selections."""

    root: Path
    ledger: ExperimentLedger
    evidence: dict[str, CandidateEvidence]
    selection: Any
    category: Any


@pytest.fixture
def prepared(experiment: rx.Experiment, tmp_path: Path) -> Frozen:
    """Return a writable copy of the experiment with both selections published.

    Copied rather than shared: freezing writes, and a test that corrupted a run
    directory would otherwise decide what every later test sees.
    """
    root = tmp_path / "root"
    shutil.copytree(experiment.root, root)
    ledger = ExperimentLedger(root / "ledger")

    from password_attack_detector.ml.selection import load_candidate_evidence

    evidence = load_candidate_evidence(root, ledger=ledger)
    config = rx.config()
    binary = select_binary_champion(evidence, config=config)
    category = select_category_head(evidence, config=config)
    publish_selection(binary, root=root, ledger=ledger)
    publish_selection(category, root=root, ledger=ledger)
    return Frozen(
        root=root,
        ledger=ledger,
        evidence={item.run_id: item for item in evidence},
        selection=binary.record,
        category=category.record,
    )


def freeze(prepared: Frozen, **overrides: Any) -> Any:
    """Freeze the prepared experiment's champion."""
    settings: dict[str, Any] = {
        "selection": prepared.selection,
        "evidence": prepared.evidence,
        "config": rx.config(),
        "root": prepared.root,
        "ledger": prepared.ledger,
        "category": prepared.category,
    }
    settings.update(overrides)
    selection = settings.pop("selection")
    return freeze_champion(selection, **settings)


def lock_of(prepared: Frozen) -> ChampionLock:
    """Return the lock the prepared experiment would freeze."""
    return build_champion_lock(
        prepared.selection,
        evidence=prepared.evidence,
        config=rx.config(),
        root=prepared.root,
        category=prepared.category,
    )


# ---------------------------------------------------------------------------
# The scope
# ---------------------------------------------------------------------------


def test_the_scope_is_the_experiment_not_the_candidate_set(prepared: Frozen) -> None:
    """Two selections over different candidates answer the same question."""
    fewer = select_binary_champion(
        [
            item
            for item in prepared.evidence.values()
            if item.catalog_model_id != "M-001"
        ],
        config=rx.config(),
    ).record
    assert fewer.record_id != prepared.selection.record_id
    assert scope_key_for(fewer) == scope_key_for(prepared.selection)


def test_different_acceptance_criteria_are_a_different_scope(
    prepared: Frozen,
) -> None:
    """A champion chosen under other gates does not replace this one."""
    other = select_binary_champion(
        list(prepared.evidence.values()),
        config=rx.strict_gates(min_detection_rate=0.999),
    ).record
    assert scope_key_for(other) != scope_key_for(prepared.selection)


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------


def test_the_lock_names_the_whole_subject_of_a_later_evaluation(
    prepared: Frozen,
) -> None:
    """Everything a test evaluation may load, fixed before any of it is run."""
    lock = lock_of(prepared)
    assert lock.task is BINARY
    assert lock.training_run_id == prepared.selection.selected_run_id
    assert lock.model_content_fingerprint
    assert lock.model_manifest_fingerprint
    assert lock.preprocessor_fingerprint
    assert lock.binary_threshold_fingerprint
    assert lock.feature_catalog_fingerprint
    assert lock.eligible_feature_list_fingerprint
    assert lock.validation_partition_fingerprint
    assert lock.gate_config_fingerprint
    assert lock.serializer_id and lock.serializer_version >= 1
    assert lock.dependency_contract_fingerprint


def test_the_lock_carries_no_metric_at_all(prepared: Frozen) -> None:
    """Which model was chosen; the selection record beside it says why."""
    fields = set(type(lock_of(prepared)).model_fields)
    for forbidden in (
        "detection_rate",
        "false_positive_rate",
        "precision",
        "expected_calibration_error",
        "test_metrics",
    ):
        assert forbidden not in fields


def test_the_lock_is_identical_wherever_it_is_frozen(
    prepared: Frozen, tmp_path: Path
) -> None:
    """No path, no host, no clock: two freezes produce the same bytes."""
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(prepared.root, elsewhere)
    other = build_champion_lock(
        prepared.selection,
        evidence=prepared.evidence,
        config=rx.config(),
        root=elsewhere,
        category=prepared.category,
    )
    assert other.to_json() == lock_of(prepared).to_json()


def test_the_lock_reseals_itself_on_the_way_back_in(prepared: Frozen) -> None:
    """A tampered lock does not load; the digest is checked, not stored."""
    payload = json.loads(lock_of(prepared).to_json())
    payload["catalog_model_id"] = "M-020"
    with pytest.raises(Exception, match="not valid"):
        ChampionLock.from_json(json.dumps(payload))


def test_the_category_head_is_bound_alongside_not_instead(prepared: Frozen) -> None:
    """The binary champion is the champion; the head rides with it."""
    lock = lock_of(prepared)
    assert lock.category_head is not None
    assert lock.category_head.training_run_id == prepared.category.selected_run_id
    assert len(lock.category_head.class_order) >= 2
    assert lock.category_head.training_run_id != lock.training_run_id


def test_a_freeze_without_a_category_selection_binds_no_head(
    prepared: Frozen,
) -> None:
    """Absence is represented rather than filled in."""
    lock = build_champion_lock(
        prepared.selection,
        evidence=prepared.evidence,
        config=rx.config(),
        root=prepared.root,
        category=None,
    )
    assert lock.category_head is None


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_selection_that_found_nothing_freezes_nothing(prepared: Frozen) -> None:
    """Neither a measured negative nor an unresolved question is a promotion."""
    negative = select_binary_champion(
        list(prepared.evidence.values()),
        config=rx.strict_gates(min_detection_rate=0.999),
    ).record
    assert negative.status is not ChampionStatus.ELIGIBLE
    with pytest.raises(ExperimentPublicationError, match="found one"):
        freeze(prepared, selection=negative)


def test_an_unverifiable_artifact_freezes_nothing(prepared: Frozen) -> None:
    """Re-verified at freeze time, not trusted because selection checked it."""
    run_id = prepared.selection.selected_run_id
    directory = prepared.evidence[run_id].directory / "model"
    target = next(path for path in sorted(directory.iterdir()) if path.is_file())
    target.write_bytes(target.read_bytes() + b"\x00")

    with pytest.raises(ExperimentPublicationError, match="does not verify"):
        freeze(prepared)


def test_a_candidate_missing_from_the_ledger_freezes_nothing(
    prepared: Frozen, tmp_path: Path
) -> None:
    """A frozen comparison must still be the comparison the ledger records."""
    empty = ExperimentLedger(tmp_path / "empty-ledger")
    with pytest.raises(ExperimentPublicationError, match="absent from the ledger"):
        freeze(prepared, ledger=empty)


def test_a_run_the_selection_never_judged_freezes_nothing(
    prepared: Frozen,
) -> None:
    """The chosen candidate has to be one of the candidates."""
    stranger = next(
        item
        for item in prepared.evidence.values()
        if item.run_id not in prepared.selection.candidate_run_ids
    )
    evidence = dict(prepared.evidence)
    del evidence[prepared.selection.selected_run_id]
    with pytest.raises(ExperimentPublicationError, match="could not be read back"):
        freeze(prepared, evidence=evidence)
    assert stranger.run_id in prepared.evidence


def test_a_candidate_with_no_operating_point_freezes_nothing(
    prepared: Frozen,
) -> None:
    """A champion with no threshold decides nothing at inference time."""
    run_id = prepared.selection.selected_run_id
    evidence = dict(prepared.evidence)
    evidence[run_id] = dataclasses.replace(evidence[run_id], threshold=None)
    with pytest.raises(ExperimentPublicationError, match="no frozen operating point"):
        freeze(prepared, evidence=evidence)


def test_a_candidate_with_no_calibration_evidence_freezes_nothing(
    prepared: Frozen,
) -> None:
    """The configured protocol requires a calibrator; absence is not a pass."""
    run_id = prepared.selection.selected_run_id
    evidence = dict(prepared.evidence)
    evidence[run_id] = dataclasses.replace(evidence[run_id], calibration_quality=None)
    with pytest.raises(ExperimentPublicationError, match="calibration report"):
        freeze(prepared, evidence=evidence)


def test_the_reference_baseline_is_never_frozen(prepared: Frozen) -> None:
    """A model cannot qualify by beating itself."""
    lock = lock_of(prepared)
    assert lock.model_family is not ModelFamily.PRIOR_BASELINE
    with pytest.raises(ValueError, match="reference baseline"):
        ChampionLock.seal(
            **{
                **lock.model_dump(),
                "category_head": lock.category_head,
                "model_family": ModelFamily.PRIOR_BASELINE,
            }
        )


def test_there_is_no_override(prepared: Frozen) -> None:
    """Asserted on the signature: no force, no allow, no bypass."""
    import inspect

    for function in (freeze_champion, build_champion_lock):
        names = set(inspect.signature(function).parameters)
        assert not any(
            token in name
            for name in names
            for token in ("force", "allow", "override", "skip", "test")
        )


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def test_freezing_writes_the_lock_and_indexes_the_receipt(prepared: Frozen) -> None:
    """The lock on disk, the receipt in the ledger, in that order."""
    published = freeze(prepared)
    lock_path = prepared.root / "champion" / published.scope_key / CHAMPION_LOCK_FILE

    assert published.created
    assert lock_path.is_file()
    stored = ChampionLock.from_json(lock_path.read_text(encoding="utf-8"))
    assert stored.lock_fingerprint == published.lock_fingerprint
    receipts = prepared.ledger.champion_freezes()
    assert [item.record_id for item in receipts] == [published.record_id]
    assert receipts[0].scope_key == published.scope_key


def test_the_receipt_names_the_selection_it_came_from(prepared: Frozen) -> None:
    """A freeze is traceable to the comparison that justified it."""
    published = freeze(prepared)
    receipt = prepared.ledger.read_freeze(published.record_id)
    assert receipt.validation_selection_id == prepared.selection.record_id
    assert receipt.selected_run_id == prepared.selection.selected_run_id
    assert receipt.category_selection_id == prepared.category.record_id


def test_refreezing_the_same_champion_changes_nothing(prepared: Frozen) -> None:
    """Idempotent, so a retried command is not a second promotion."""
    first = freeze(prepared)
    second = freeze(prepared)
    assert first.created and not second.created
    assert second.lock_fingerprint == first.lock_fingerprint
    assert len(prepared.ledger.champion_freezes()) == 1


def test_a_contradicting_freeze_in_one_scope_is_refused(prepared: Frozen) -> None:
    """One champion per experiment, and no way to replace it in place."""
    freeze(prepared)
    without_head = dataclasses.replace(prepared, category=None)
    with pytest.raises(ExperimentPublicationError, match="already frozen"):
        freeze(without_head)


def test_an_incomplete_freeze_is_never_completed_in_place(prepared: Frozen) -> None:
    """A champion directory with no lock is wreckage, not a destination."""
    published = freeze(prepared)
    (prepared.root / "champion" / published.scope_key / CHAMPION_LOCK_FILE).unlink()
    with pytest.raises(ExperimentPublicationError, match="incomplete freeze"):
        freeze(prepared)


def test_an_unindexed_freeze_is_recovered_by_reading_it(
    prepared: Frozen, tmp_path: Path
) -> None:
    """Reconcile reads the frozen lock; it never rebuilds one.

    Recovered into a fresh ledger that has itself been reconciled from the
    published selections, which is the state a machine repairing an interrupted
    publication is actually in -- and the receipt it derives is byte-identical
    to the one the original freeze appended.
    """
    published = freeze(prepared)
    fresh = ExperimentLedger(tmp_path / "fresh-ledger")
    reconcile_selections(root=prepared.root, ledger=fresh)

    appended = reconcile_freezes(root=prepared.root, ledger=fresh)
    assert appended == (published.record_id,)
    assert reconcile_freezes(root=prepared.root, ledger=fresh) == ()

    recovered = fresh.read_freeze(appended[0])
    assert recovered.scope_key == scope_key_for(prepared.selection)
    assert recovered.to_json() == (
        prepared.ledger.read_freeze(published.record_id).to_json()
    )
