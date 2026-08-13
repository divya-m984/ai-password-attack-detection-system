"""Validating a published prediction artifact, one stable failure code at a time.

The pattern throughout: publish a valid artifact, break exactly one thing, and
assert both that validation fails *and* that it fails with the code reserved for
that thing. A validator that reported a generic failure would be true and
useless -- somebody investigating a broken publication needs to know which of
twenty-eight checks caught it.

The last section is the firewall: validation reads no label, computes no metric,
and takes no parameter through which either could arrive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.enums import AuditCheckStatus, AuditStatus, MLSplit
from password_attack_detector.ml.prediction_manifest import (
    BINARY_PREDICTION_FILE,
    CATEGORY_PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    PREDICTIONS_DIR,
    QUALITY_REPORT_JSON_FILE,
    VALIDATION_RESULT_FILE,
)
from password_attack_detector.ml.prediction_publisher import publish_predictions
from password_attack_detector.ml.prediction_serialization import (
    BINARY_PREDICTION_SCHEMA,
    write_binary_predictions,
)
from password_attack_detector.ml.prediction_validation import (
    PUBLISHED_CHECKS,
    STAGED_CHECKS,
    VALIDATION_SCHEMA_VERSION,
    MLValidationResult,
    ValidationCheck,
    validate_publication,
    validate_staged_predictions,
)
from tests.ml import predictions as px
from tests.ml import runs as rx
from tests.unit.ml.test_prediction_serialization import binary_rows, write_raw


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> rx.Experiment:
    """Publish one experiment, shared by every test in this module."""
    return rx.publish_experiment(tmp_path_factory.mktemp("validation-source"))


@pytest.fixture
def published(source: rx.Experiment, tmp_path: Path) -> Path:
    """Return one valid published prediction directory, one per test."""
    prepared = px.prepare(tmp_path / "root", source=source)
    publication = publish_predictions(
        champion=prepared.champion(),
        dataset=px.inference_dataset(prepared),
        root=prepared.root,
    )
    return prepared.root / PREDICTIONS_DIR / publication.prediction_id


def codes(result: MLValidationResult) -> dict[str, AuditCheckStatus]:
    """Return the outcome of every check, keyed by code."""
    return {check.code: check.status for check in result.checks}


def assert_fails_with(result: MLValidationResult, code: str) -> None:
    """Assert the aggregate failed and *code* is among the reasons."""
    assert result.status is AuditStatus.FAIL
    assert code in result.failures, (code, result.failures)


# ---------------------------------------------------------------------------
# The valid case
# ---------------------------------------------------------------------------


def test_a_valid_publication_passes_every_mandatory_check(published: Path) -> None:
    """Twenty-eight checks, all of them run, all of them passing."""
    result = validate_publication(published)
    assert result.passed
    assert result.failures == ()
    assert set(codes(result)) == set(PUBLISHED_CHECKS)
    assert all(status is AuditCheckStatus.PASS for status in codes(result).values())


def test_the_result_is_sealed_and_reproducible(published: Path) -> None:
    """Two validations of one artifact are the same record."""
    first = validate_publication(published)
    second = validate_publication(published)
    assert first.to_json() == second.to_json()
    assert first.result_fingerprint == first.recomputed_fingerprint()


def test_the_stored_result_records_the_staged_stage(published: Path) -> None:
    """A staged result can never read as evidence a manifest was checked."""
    stored = MLValidationResult.from_json(
        (published / VALIDATION_RESULT_FILE).read_text(encoding="utf-8")
    )
    assert stored.stage.stage == "staged"
    assert {check.code for check in stored.checks} == set(STAGED_CHECKS)
    assert stored.passed


# ---------------------------------------------------------------------------
# One broken thing at a time
# ---------------------------------------------------------------------------


def test_a_missing_manifest_fails_and_skips_the_rest(published: Path) -> None:
    """A skipped mandatory check is not a passed check."""
    (published / PREDICTION_MANIFEST_FILE).unlink()
    result = validate_publication(published)
    assert_fails_with(result, "M001")
    assert result.scope is None
    skipped = [
        code
        for code, status in codes(result).items()
        if status is AuditCheckStatus.SKIPPED
    ]
    assert len(skipped) == len(PUBLISHED_CHECKS) - 1


def test_an_unparsable_manifest_fails(published: Path) -> None:
    """Malformed JSON is a failure, not an absence."""
    (published / PREDICTION_MANIFEST_FILE).write_text("{ not json", encoding="utf-8")
    assert_fails_with(validate_publication(published), "M001")


def test_a_manifest_from_another_contract_version_fails(published: Path) -> None:
    """The version is checked before the payload is understood."""
    payload = json.loads(
        (published / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    payload["prediction_manifest_schema_version"] = "9.9.9"
    (published / PREDICTION_MANIFEST_FILE).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert_fails_with(validate_publication(published), "M001")


def test_an_unexpected_file_fails(published: Path) -> None:
    """An extra file in a verified directory is a mistake or an attempt."""
    (published / "surprise.txt").write_text("hello", encoding="utf-8")
    result = validate_publication(published)
    assert_fails_with(result, "M004")
    assert_fails_with(result, "M005")


def test_a_symbolic_link_fails(published: Path) -> None:
    """A member pointing outside the publication is refused."""
    (published / QUALITY_REPORT_JSON_FILE).unlink()
    (published / QUALITY_REPORT_JSON_FILE).symlink_to(
        published / VALIDATION_RESULT_FILE
    )
    assert_fails_with(validate_publication(published), "M005")


def test_a_tampered_file_fails_its_checksum(published: Path) -> None:
    """Bytes are digested, and the declared digest is not believed."""
    path = published / QUALITY_REPORT_JSON_FILE
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    result = validate_publication(published)
    assert_fails_with(result, "M006")
    assert_fails_with(result, "M007")


def test_a_missing_declared_file_fails(published: Path) -> None:
    """A declared artifact that is not there is a failure at several checks."""
    (published / VALIDATION_RESULT_FILE).unlink()
    result = validate_publication(published)
    assert_fails_with(result, "M004")
    assert_fails_with(result, "M006")


def test_a_replaced_row_table_fails_the_content_fingerprint(published: Path) -> None:
    """Rewriting the rows with different predictions moves the fingerprint."""
    write_binary_predictions(binary_rows(4), published / BINARY_PREDICTION_FILE)
    result = validate_publication(published)
    assert_fails_with(result, "M006")
    assert_fails_with(result, "M008")
    assert_fails_with(result, "M009")


def test_a_row_contradicting_the_frozen_threshold_fails(published: Path) -> None:
    """The decision is recomputed from the row's own numbers, and not repaired."""
    write_raw(
        published / BINARY_PREDICTION_FILE,
        BINARY_PREDICTION_SCHEMA,
        {
            "anchor_event_id": ["00000000-0000-5000-8000-000000000001"],
            "anchor_event_time": [_epoch()],
            "malicious_decision_score": [0.1],
            "malicious_probability": [None],
            "decision_threshold": [0.9],
            "flagged_malicious": [True],
        },
    )
    assert_fails_with(validate_publication(published), "M010")


def test_a_duplicate_anchor_fails(published: Path) -> None:
    """Each anchor event contributes exactly one prediction."""
    rows = binary_rows(3)
    duplicated = [*rows, rows[0]]
    result = validate_staged_predictions(
        binary=duplicated,
        category=None,
        anomaly=None,
        scope=MLSplit.TEST,
        binary_schema_matches=True,
    )
    assert_fails_with(result, "M012")


def test_an_unsorted_table_fails() -> None:
    """Rows are sorted by anchor time, then by anchor identifier."""
    rows = list(reversed(binary_rows(4)))
    result = validate_staged_predictions(
        binary=rows,
        category=None,
        anomaly=None,
        scope=MLSplit.TEST,
        binary_schema_matches=True,
    )
    assert_fails_with(result, "M013")


def test_two_thresholds_in_one_table_fail() -> None:
    """A publication applies one frozen operating point to every row."""
    rows = binary_rows(3)
    mixed = [
        *rows[:2],
        rows[2].model_copy(
            update={"decision_threshold": 0.9, "flagged_malicious": False}
        ),
    ]
    result = validate_staged_predictions(
        binary=mixed,
        category=None,
        anomaly=None,
        scope=MLSplit.TEST,
        binary_schema_matches=True,
    )
    assert_fails_with(result, "M018")


def test_a_declared_schema_mismatch_fails() -> None:
    """The reader's verdict is carried into the staged result, not re-derived."""
    result = validate_staged_predictions(
        binary=binary_rows(3),
        category=None,
        anomaly=None,
        scope=MLSplit.TEST,
        binary_schema_matches=False,
    )
    assert_fails_with(result, "M010")


def test_an_empty_table_fails() -> None:
    """A publication with no rows is not a publication of no findings."""
    result = validate_staged_predictions(
        binary=[],
        category=None,
        anomaly=None,
        scope=MLSplit.TEST,
        binary_schema_matches=True,
    )
    assert_fails_with(result, "M011")


def test_a_missing_triage_row_is_a_stable_failure(published: Path) -> None:
    """A flagged row that was not categorised is a triage result that was lost."""
    from password_attack_detector.ml.prediction_serialization import (
        read_category_predictions,
        write_category_predictions,
    )

    rows = read_category_predictions(published / CATEGORY_PREDICTION_FILE)
    assert rows, "the fixture routes rows to triage"
    write_category_predictions(list(rows[:-1]), published / CATEGORY_PREDICTION_FILE)
    result = validate_publication(published)
    assert_fails_with(result, "M021")
    detail = next(check for check in result.checks if check.code == "M021").detail
    assert "binary-positive" in detail


def test_categorising_a_row_the_binary_head_cleared_is_a_stable_failure(
    published: Path,
) -> None:
    """Contradictory applicability is refused, never silently repaired.

    A triage row for an anchor the binary head did not flag is a category result
    for an event nobody routed to triage -- the exact confusion the artifact
    boundary exists to prevent.
    """
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
        read_category_predictions,
        write_category_predictions,
    )

    binary = read_binary_predictions(published / BINARY_PREDICTION_FILE)
    rows = list(read_category_predictions(published / CATEGORY_PREDICTION_FILE))
    cleared = next(row for row in binary if not row.flagged_malicious)
    smuggled = rows[0].model_copy(
        update={
            "anchor_event_id": cleared.anchor_event_id,
            "anchor_event_time": cleared.anchor_event_time,
        }
    )
    write_category_predictions(
        sorted(
            [*rows[1:], smuggled],
            key=lambda row: (row.anchor_event_time, row.anchor_event_id),
        ),
        published / CATEGORY_PREDICTION_FILE,
    )
    result = validate_publication(published)
    assert_fails_with(result, "M021")


def test_a_malformed_category_payload_fails(published: Path) -> None:
    """The one free-form field on a row is parsed, and a broken one is refused.

    This is the input that makes ``ml profile`` refuse: a publication whose
    class map will not parse never reaches a summary.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from password_attack_detector.ml.prediction_serialization import (
        CATEGORY_PREDICTION_SCHEMA,
    )

    path = published / CATEGORY_PREDICTION_FILE
    table = pq.read_table(path)
    broken = table.set_column(
        table.schema.get_field_index("category_scores_json"),
        "category_scores_json",
        pa.array(["{not json" for _ in range(table.num_rows)], type=pa.string()),
    )
    pq.write_table(broken.cast(CATEGORY_PREDICTION_SCHEMA), path)
    result = validate_publication(published)
    assert result.status is AuditStatus.FAIL
    assert result.failures


def test_the_triage_table_covers_exactly_the_flagged_rows(published: Path) -> None:
    """The positive case, so the two refusals above are not vacuous."""
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
        read_category_predictions,
    )

    binary = read_binary_predictions(published / BINARY_PREDICTION_FILE)
    rows = read_category_predictions(published / CATEGORY_PREDICTION_FILE)
    flagged = [row.anchor_event_id for row in binary if row.flagged_malicious]
    assert [row.anchor_event_id for row in rows] == flagged
    assert len(flagged) < len(binary)
    assert validate_publication(published).passed


def test_a_bound_report_swapped_for_another_fails(published: Path) -> None:
    """A report describing different predictions has different bytes."""
    payload = json.loads(
        (published / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    payload["quality_report_fingerprint"] = "a" * 64
    (published / PREDICTION_MANIFEST_FILE).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    # The seal refuses the edit outright, which is the stronger outcome.
    assert_fails_with(validate_publication(published), "M001")


def test_a_manifest_whose_scope_role_was_rewritten_fails(published: Path) -> None:
    """A generalisation probe is never republished as a supervised scope."""
    payload = json.loads(
        (published / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    payload["scope"] = "novel_anomaly_holdout"
    (published / PREDICTION_MANIFEST_FILE).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert_fails_with(validate_publication(published), "M001")


def test_a_directory_that_does_not_exist_is_an_error(tmp_path: Path) -> None:
    """A caller asked about a publication; there is not one to answer about."""
    with pytest.raises(DataValidationError, match="not present"):
        validate_publication(tmp_path / "nothing")


# ---------------------------------------------------------------------------
# The check sets
# ---------------------------------------------------------------------------


def test_the_published_set_strictly_contains_the_staged_one() -> None:
    """A promoted publication is checked at least as thoroughly as a staged one."""
    assert set(STAGED_CHECKS) < set(PUBLISHED_CHECKS)


def test_every_code_is_declared_once() -> None:
    """A stable code cannot name two things."""
    for codes_ in (STAGED_CHECKS, PUBLISHED_CHECKS):
        assert len(set(codes_)) == len(codes_)


def test_every_code_is_in_the_reserved_namespace() -> None:
    """``M0xx``, distinct from the ``MLD`` dataset-assembly namespace."""
    for code in PUBLISHED_CHECKS:
        assert code.startswith("M0")
        assert code[1:].isdigit()


def test_a_result_missing_a_check_is_refused() -> None:
    """A check that quietly stopped being emitted fails the result."""
    valid = validate_staged_predictions(
        binary=binary_rows(3),
        category=None,
        anomaly=None,
        scope=MLSplit.TEST,
        binary_schema_matches=True,
    )
    with pytest.raises(ValueError, match="is not a result that ran it"):
        MLValidationResult.seal(
            stage=valid.stage,
            status=valid.status,
            scope=valid.scope,
            checks=valid.checks[:-1],
            binary_row_count=valid.binary_row_count,
            category_row_count=None,
            anomaly_row_count=None,
        )


def test_a_check_reporting_the_wrong_name_is_refused() -> None:
    """The code and the name it was reserved for are one statement."""
    with pytest.raises(ValueError, match="reports name"):
        ValidationCheck(
            code="M012",
            name="something_else",
            status=AuditCheckStatus.PASS,
            mandatory=True,
            detail="ok",
        )


def test_an_unknown_code_is_refused() -> None:
    """The namespace is closed."""
    with pytest.raises(ValueError, match="unknown validation check code"):
        ValidationCheck(
            code="M999",
            name="whatever",
            status=AuditCheckStatus.PASS,
            mandatory=True,
            detail="ok",
        )


def test_a_pass_with_a_failing_mandatory_check_is_refused() -> None:
    """The aggregate grade is derived, not asserted."""
    valid = validate_staged_predictions(
        binary=binary_rows(3),
        category=None,
        anomaly=None,
        scope=MLSplit.TEST,
        binary_schema_matches=True,
    )
    broken = tuple(
        check.model_copy(update={"status": AuditCheckStatus.FAIL})
        if check.code == "M012"
        else check
        for check in valid.checks
    )
    with pytest.raises(ValueError, match="skipped mandatory check is not a passed one"):
        MLValidationResult.seal(
            stage=valid.stage,
            status=AuditStatus.PASS,
            scope=valid.scope,
            checks=broken,
            binary_row_count=valid.binary_row_count,
            category_row_count=None,
            anomaly_row_count=None,
        )


def test_the_validation_contract_version_is_pinned() -> None:
    """A change to what validation checks is a visible edit."""
    assert VALIDATION_SCHEMA_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# The firewall
# ---------------------------------------------------------------------------


def test_validation_takes_no_label_argument() -> None:
    """Stated as a signature, so a reviewer does not have to remember it."""
    import inspect

    assert set(inspect.signature(validate_publication).parameters) == {"directory"}
    staged = set(inspect.signature(validate_staged_predictions).parameters)
    assert staged == {
        "binary",
        "category",
        "anomaly",
        "scope",
        "binary_schema_matches",
    }
    for absent in ("labels", "truth", "ground_truth", "targets"):
        assert absent not in staged, absent


def test_a_validation_result_declares_no_metric_field() -> None:
    """There is nothing outcome-dependent here to report."""
    forbidden = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "pr_auc",
        "brier_score",
        "confusion_matrix",
        "label_fingerprint",
        "anchor_event_id",
    }
    for model in (MLValidationResult, ValidationCheck):
        assert not set(model.model_fields) & forbidden, model.__name__


def test_no_check_detail_names_an_anchor(published: Path) -> None:
    """A failure report is handed to whoever is investigating it."""
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
    )

    result = validate_publication(published)
    rendered = " ".join(check.detail for check in result.checks)
    for row in read_binary_predictions(published / BINARY_PREDICTION_FILE):
        assert row.anchor_event_id not in rendered


def test_no_check_detail_names_a_path(published: Path) -> None:
    """Nor where on a machine the publication happens to live."""
    result = validate_publication(published)
    rendered = " ".join(check.detail for check in result.checks)
    assert str(published) not in rendered
    assert "/tmp" not in rendered


def test_validating_a_publication_writes_nothing(published: Path) -> None:
    """Reading an artifact does not modify it."""
    before = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in published.iterdir()
    }
    validate_publication(published)
    after = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in published.iterdir()
    }
    assert after == before


def _epoch() -> Any:
    """Return the literal instant the row fixtures count from."""
    from tests.unit.ml.test_prediction_serialization import EPOCH

    return EPOCH
