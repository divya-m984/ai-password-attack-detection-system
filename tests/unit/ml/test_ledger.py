"""The experiment ledger: append-only, idempotent, and impossible to amend.

Most of this suite is about what the ledger *refuses*. A ledger that only ever
appends is easy to write and worth very little unless the refusals are real: a
stored record that can be edited, a duplicate that silently overwrites, or a
truncated file that reads back as a shorter history would each undo the
guarantee the append-only design exists to provide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from password_attack_detector.exceptions import (
    DataValidationError,
    LedgerConflictError,
)
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    CalibrationStatus,
    ExperimentRecordType,
    MLTask,
    ModelFamily,
    SelectionStatus,
    TrainingRunStatus,
)
from password_attack_detector.ml.ledger import (
    LEDGER_FILE,
    LEDGER_SCHEMA_VERSION,
    RECORD_TYPE_DIRECTORIES,
    ExperimentLedger,
    TrainingRunRecord,
)
from password_attack_detector.ml.schemas import ExperimentRecordIdentity

DIGEST = "a" * 64


def identity(**overrides: Any) -> ExperimentRecordIdentity:
    """Return a derived training-run identity."""
    fields: dict[str, Any] = {
        "record_type": ExperimentRecordType.TRAINING_RUN,
        "model_catalog_version": "1.0.0",
        "required_feature_schema_version": "1.0.0",
        "task": MLTask.BINARY_MALICIOUS,
        "model_family": ModelFamily.LOGISTIC_REGRESSION,
        "catalog_model_id": "M-010",
        "seed": 7,
        "ml_config_fingerprint": DIGEST,
        "model_catalog_fingerprint": "b" * 64,
        "training_data_fingerprint": "c" * 64,
    }
    fields.update(overrides)
    return ExperimentRecordIdentity.derive(**fields)


def record(**overrides: Any) -> TrainingRunRecord:
    """Return a valid training-run record."""
    fields: dict[str, Any] = {
        "record_type": ExperimentRecordType.TRAINING_RUN,
        "identity": identity(),
        "status": TrainingRunStatus.COMPLETED,
        "task": MLTask.BINARY_MALICIOUS,
        "model_family": ModelFamily.LOGISTIC_REGRESSION,
        "catalog_model_id": "M-010",
        "champion_eligible": True,
        "reference_baseline": False,
        "experimental": False,
        "model_id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
        "train_row_count": 120,
        "validation_a_row_count": 60,
        "validation_b_row_count": 60,
        "calibration_method": CalibrationMethod.PLATT,
        "calibration_status": CalibrationStatus.FITTED,
        "calibration_quality_admissible": True,
        "binary_threshold_status": SelectionStatus.SELECTED,
    }
    fields.update(overrides)
    return TrainingRunRecord.seal(**fields)


@pytest.fixture
def ledger(tmp_path: Path) -> ExperimentLedger:
    """Return a ledger rooted in a temporary directory."""
    return ExperimentLedger(tmp_path / "ledger")


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------


def test_a_training_run_record_carries_no_test_metric() -> None:
    """The field that must not exist, checked by name rather than by review.

    A training run with somewhere to put a test metric is a training run that
    eventually holds one. A test evaluation is its own record type, written
    after a champion is frozen.
    """
    fields = set(TrainingRunRecord.model_fields)
    for forbidden in (
        "test_metrics",
        "test_score",
        "test_evaluation",
        "holdout_metrics",
        "metrics",
        "validation_metrics",
        "brier_score",
        "expected_calibration_error",
    ):
        assert forbidden not in fields


def test_a_training_run_record_declares_no_champion_verdict() -> None:
    """Milestone 6 ranks nothing, so there is nowhere to record a ranking."""
    fields = set(TrainingRunRecord.model_fields)
    for forbidden in ("champion", "is_champion", "champion_status", "rank", "ranking"):
        assert forbidden not in fields
    # ``champion_eligible`` is a property of the *family*, copied from the
    # reviewed catalog, and says nothing about this run.
    assert "champion_eligible" in fields


def test_a_training_run_record_declares_nothing_observational() -> None:
    """No path, no host, no user, no timestamp: identity would move with them."""
    fields = set(TrainingRunRecord.model_fields) | set(
        ExperimentRecordIdentity.model_fields
    )
    for forbidden in (
        "output_dir",
        "run_directory",
        "hostname",
        "username",
        "temp_dir",
        "created_at",
        "published_at",
        "timestamp",
    ):
        assert forbidden not in fields


def test_a_record_identifier_is_derived_and_cannot_be_assigned() -> None:
    """An identifier a caller could choose would let two runs claim to be one."""
    derived = identity()
    assert derived.run_id == derived.derived_run_id()
    with pytest.raises(ValueError, match="cannot be supplied"):
        ExperimentRecordIdentity.derive(
            record_type=ExperimentRecordType.TRAINING_RUN,
            run_id="3f2504e0-4f89-41d3-9a0c-0305e82c3301",
            model_catalog_version="1.0.0",
            required_feature_schema_version="1.0.0",
            task=MLTask.BINARY_MALICIOUS,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            seed=7,
            ml_config_fingerprint=DIGEST,
            model_catalog_fingerprint=DIGEST,
        )


def test_a_record_whose_identifier_does_not_recompute_is_refused() -> None:
    """The identity is checked against its own content at construction."""
    tampered = identity().model_copy(
        update={"run_id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301"}
    )
    with pytest.raises(ValidationError, match="not the one this identity"):
        record(identity=tampered)


def test_every_lineage_input_changes_the_identifier() -> None:
    """Each fingerprint is part of identity, so each one moves it.

    Swept rather than spot-checked: a field added to the identity joins the
    digest automatically, and this asserts that the ones already there did.
    """
    baseline = identity().run_id
    other = "d" * 64
    for field in (
        "ml_config_fingerprint",
        "model_catalog_fingerprint",
        "training_data_fingerprint",
        "allowlist_fingerprint",
        "eligible_feature_list_fingerprint",
        "preprocessor_fingerprint",
        "class_weight_fingerprint",
        "validation_partition_fingerprint",
        "candidate_fingerprint",
        "model_content_fingerprint",
        "calibration_state_fingerprint",
        "threshold_selection_fingerprint",
        "category_abstention_fingerprint",
        "anomaly_threshold_fingerprint",
        "dependency_contract_fingerprint",
    ):
        assert identity(**{field: other}).run_id != baseline, field
    assert identity(seed=8).run_id != baseline
    assert identity(catalog_model_id="M-020").run_id != baseline


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"record_type": ExperimentRecordType.CHAMPION_FREEZE}, "record type"),
        ({"task": MLTask.ATTACK_CATEGORY}, "different tasks"),
        ({"model_family": ModelFamily.RANDOM_FOREST}, "different families"),
        ({"catalog_model_id": "M-020"}, "different catalog entries"),
        (
            {"status": TrainingRunStatus.COMPLETED, "failing_requirements": ("x",)},
            "names no failing requirement",
        ),
        ({"model_id": None}, "names the model it published"),
    ],
)
def test_a_record_enforces_its_internal_agreement(
    overrides: dict[str, Any], expected: str
) -> None:
    """Identity, task, family, and status must describe one coherent run."""
    with pytest.raises(ValidationError, match=expected):
        record(**overrides)


def test_the_three_tracks_publish_different_things() -> None:
    """A record cannot claim an artifact its task does not have."""
    with pytest.raises(ValidationError, match="carries no calibrator"):
        _anomaly_record(calibration_method=CalibrationMethod.PLATT)
    with pytest.raises(ValidationError, match="no binary operating point"):
        _anomaly_record(binary_threshold_status=SelectionStatus.SELECTED)
    with pytest.raises(ValidationError, match="neither a category abstention"):
        record(
            category_abstention_status=SelectionStatus.SELECTED,
            category_threshold_data_selected=True,
        )
    with pytest.raises(ValidationError, match="never inherits the binary"):
        _category_record(binary_threshold_status=SelectionStatus.SELECTED)


def _anomaly_record(**overrides: Any) -> TrainingRunRecord:
    """Return an anomaly training-run record."""
    fields: dict[str, Any] = {
        "identity": identity(
            task=MLTask.ANOMALY,
            model_family=ModelFamily.ISOLATION_FOREST,
            catalog_model_id="M-030",
        ),
        "task": MLTask.ANOMALY,
        "model_family": ModelFamily.ISOLATION_FOREST,
        "catalog_model_id": "M-030",
        "champion_eligible": False,
        "experimental": True,
        "calibration_method": CalibrationMethod.NONE,
        "calibration_status": None,
        "calibration_quality_admissible": None,
        "binary_threshold_status": None,
        "anomaly_threshold_status": SelectionStatus.SELECTED,
    }
    fields.update(overrides)
    return record(**fields)


def _category_record(**overrides: Any) -> TrainingRunRecord:
    """Return a category training-run record."""
    fields: dict[str, Any] = {
        "identity": identity(task=MLTask.ATTACK_CATEGORY),
        "task": MLTask.ATTACK_CATEGORY,
        "calibration_method": CalibrationMethod.NONE,
        "calibration_status": None,
        "calibration_quality_admissible": None,
        "binary_threshold_status": None,
        "category_abstention_status": SelectionStatus.SELECTED,
        "category_threshold_data_selected": True,
    }
    fields.update(overrides)
    return record(**fields)


def test_the_abstention_provenance_flag_travels_with_the_abstention_point() -> None:
    """A fallback threshold must stay distinguishable from a selected one."""
    with pytest.raises(ValidationError, match="exists exactly when"):
        _category_record(category_threshold_data_selected=None)


def test_artifact_digests_are_sorted_and_unique() -> None:
    """Deterministically ordered and uniquely named."""
    with pytest.raises(ValidationError, match="sorted path order"):
        record(artifact_digests=(("model/model.json", DIGEST), ("a.json", DIGEST)))
    with pytest.raises(ValidationError, match="more than once"):
        record(artifact_digests=(("a.json", DIGEST), ("a.json", DIGEST)))


def test_artifact_digests_reject_traversal_and_absolute_paths() -> None:
    """A run record names files inside its own directory and nowhere else."""
    for path in ("/etc/passwd", "../../elsewhere.json"):
        with pytest.raises(ValidationError, match="relative path only"):
            record(artifact_digests=((path, DIGEST),))


# ---------------------------------------------------------------------------
# Append semantics
# ---------------------------------------------------------------------------


def test_appending_writes_one_file_per_record(ledger: ExperimentLedger) -> None:
    """One record, one file, named by the identifier it derived."""
    item = record()
    result = ledger.append(item)
    assert result.created is True
    assert result.record_id == item.run_id
    stored = ledger.directory_for(ExperimentRecordType.TRAINING_RUN) / (
        f"{item.run_id}.json"
    )
    assert stored.is_file()
    assert stored.read_text(encoding="utf-8") == item.to_json() + "\n"


def test_appending_the_same_record_twice_is_idempotent(
    ledger: ExperimentLedger,
) -> None:
    """Appending the same fact twice is not a change, so it is not a failure."""
    item = record()
    first = ledger.append(item)
    second = ledger.append(item)
    assert first.created is True
    assert second.created is False
    assert second.record_id == first.record_id
    assert len(ledger.training_runs()) == 1


def test_the_same_identity_with_different_content_is_a_conflict(
    ledger: ExperimentLedger,
) -> None:
    """Two runs claiming to be one run. Neither is overwritten, neither merged."""
    original = record()
    ledger.append(original)
    stored = ledger.directory_for(ExperimentRecordType.TRAINING_RUN) / (
        f"{original.run_id}.json"
    )
    before = stored.read_text(encoding="utf-8")

    # Same identity -- the status is not part of it -- different content.
    conflicting = record(
        status=TrainingRunStatus.THRESHOLD_UNAVAILABLE,
        failing_requirements=("max_false_positive_rate",),
        binary_threshold_status=SelectionStatus.NO_FEASIBLE_THRESHOLD,
    )
    assert conflicting.run_id == original.run_id
    with pytest.raises(LedgerConflictError, match="append-only"):
        ledger.append(conflicting)
    assert stored.read_text(encoding="utf-8") == before


def test_the_ledger_offers_no_way_to_change_a_record(
    ledger: ExperimentLedger,
) -> None:
    """No update, no upsert, no delete -- not even privately.

    Asserted over the public surface *and* the private one: a helper that
    rewrote a record would be reachable from the next feature that wanted it.
    """
    forbidden = ("update", "upsert", "replace", "delete", "remove", "amend", "patch")
    for name in dir(ExperimentLedger):
        assert not any(name.startswith(verb) for verb in forbidden), name


def test_records_are_returned_in_identifier_order(ledger: ExperimentLedger) -> None:
    """Two ledgers holding the same records enumerate them identically."""
    items = [
        record(),
        _category_record(),
        _anomaly_record(),
    ]
    for item in items:
        ledger.append(item)
    listed = [item.run_id for item in ledger.training_runs()]
    assert listed == sorted(listed)
    assert set(listed) == {item.run_id for item in items}
    assert [item.run_id for item in ledger.iter_training_runs()] == listed


def test_the_ledger_contract_file_is_written_once(ledger: ExperimentLedger) -> None:
    """Idempotent, and refused if something else wrote a different ledger here."""
    ledger.initialize()
    contract = ledger.root / LEDGER_FILE
    payload = contract.read_text(encoding="utf-8")
    assert json.loads(payload)["ledger_schema_version"] == LEDGER_SCHEMA_VERSION

    ledger.initialize()
    assert contract.read_text(encoding="utf-8") == payload

    contract.write_text('{"ledger_schema_version": "9.9.9"}', encoding="utf-8")
    with pytest.raises(LedgerConflictError, match="different ledger contract"):
        ledger.initialize()


def test_only_the_directory_being_written_is_created(
    ledger: ExperimentLedger,
) -> None:
    """An empty champion-freeze directory would advertise a record kind M6 has not."""
    ledger.append(record())
    present = {path.name for path in ledger.root.iterdir() if path.is_dir()}
    assert present == {"training_run"}
    for record_type, name in RECORD_TYPE_DIRECTORIES.items():
        if record_type is not ExperimentRecordType.TRAINING_RUN:
            assert not (ledger.root / name).exists()


def test_a_record_type_with_no_declared_location_is_refused(
    ledger: ExperimentLedger,
) -> None:
    """Dispatch is a closed mapping, so an unmapped type has nowhere to go."""
    assert set(RECORD_TYPE_DIRECTORIES) == set(ExperimentRecordType)
    assert (
        ledger.contains("nope", record_type=ExperimentRecordType.TRAINING_RUN) is False
    )


# ---------------------------------------------------------------------------
# Reading and integrity
# ---------------------------------------------------------------------------


def test_a_truncated_record_is_refused(ledger: ExperimentLedger) -> None:
    """A short file is not a short record, and the reader will not guess."""
    item = record()
    ledger.append(item)
    stored = ledger.directory_for(ExperimentRecordType.TRAINING_RUN) / (
        f"{item.run_id}.json"
    )
    payload = stored.read_text(encoding="utf-8")
    stored.write_text(payload[: len(payload) // 2], encoding="utf-8")
    with pytest.raises(DataValidationError, match="not valid"):
        ledger.training_runs()


def test_a_tampered_record_is_refused(ledger: ExperimentLedger) -> None:
    """Every field is covered by the record's own digest."""
    item = record()
    ledger.append(item)
    stored = ledger.directory_for(ExperimentRecordType.TRAINING_RUN) / (
        f"{item.run_id}.json"
    )
    payload = json.loads(stored.read_text(encoding="utf-8"))
    payload["train_row_count"] = 999_999
    stored.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataValidationError, match="not valid"):
        ledger.read(item.run_id)


def test_a_record_from_an_unsupported_contract_is_refused(
    ledger: ExperimentLedger,
) -> None:
    """Version checked before validation, so nothing is partially understood."""
    item = record()
    ledger.append(item)
    stored = ledger.directory_for(ExperimentRecordType.TRAINING_RUN) / (
        f"{item.run_id}.json"
    )
    payload = json.loads(stored.read_text(encoding="utf-8"))
    payload["ledger_schema_version"] = "9.9.9"
    stored.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataValidationError, match="not valid"):
        ledger.read(item.run_id)


def test_an_oversized_record_is_refused(ledger: ExperimentLedger) -> None:
    """A record is kilobytes of fingerprints; anything vastly larger is not one."""
    item = record()
    ledger.append(item)
    stored = ledger.directory_for(ExperimentRecordType.TRAINING_RUN) / (
        f"{item.run_id}.json"
    )
    stored.write_text("x" * (1 << 21), encoding="utf-8")
    with pytest.raises(DataValidationError, match="larger than any record"):
        ledger.training_runs()


def test_reading_an_absent_record_says_so(ledger: ExperimentLedger) -> None:
    """A missing record is missing, not empty."""
    with pytest.raises(DataValidationError, match="no such training-run record"):
        ledger.read("3f2504e0-4f89-41d3-9a0c-0305e82c3301")


def test_an_empty_ledger_lists_nothing(ledger: ExperimentLedger) -> None:
    """Reading a ledger nobody has written to is not an error."""
    assert ledger.training_runs() == ()


def test_a_stored_record_round_trips_byte_identically(
    ledger: ExperimentLedger,
) -> None:
    """Deserialize, reserialize, compare bytes."""
    item = record()
    ledger.append(item)
    assert ledger.read(item.run_id).to_json() == item.to_json()


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "banned", ["e0000", "campaign", "u:", "/home/", "anchor_event_id", "/tmp"]
)
def test_no_identity_reaches_a_ledger_record(banned: str) -> None:
    """Fingerprints and counts go in; nothing else does."""
    assert banned not in record().to_json()
    assert banned not in _anomaly_record().to_json()
