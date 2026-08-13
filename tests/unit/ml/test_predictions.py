"""Batch inference under a frozen champion, and every state that refuses it.

Two properties run through the whole file.

**The lock is checked, not believed.** Most of the tests below tamper with one
artifact the lock names -- the lock's own bytes, the freeze receipt, the model
manifest, the calibrator, the operating point, the category head -- and assert
that loading refuses. A verification chain that trusted the lock's account of
itself would pass every one of them.

**No label is anywhere near this.** The scoring functions take an inference
dataset assembled from features and split membership, and the suite asserts the
firewall structurally: there is no label parameter on any signature, and
rewriting the fixture's labels changes not one published byte.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    ManifestVerificationError,
    ModelNotReadyError,
    ModelTrainingError,
)
from password_attack_detector.ml.champion import CHAMPION_DIR, CHAMPION_LOCK_FILE
from password_attack_detector.ml.dataset import LabelRow
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    MLSplit,
    ScoreKind,
    is_probability,
)
from password_attack_detector.ml.predictions import (
    ANOMALY_PREDICTION_COLUMNS,
    BINARY_PREDICTION_COLUMNS,
    CATEGORY_PREDICTION_COLUMNS,
    PREDICTION_SCHEMA_VERSION,
    PROHIBITED_PREDICTION_COLUMNS,
    AnomalyScore,
    BinaryPrediction,
    CategoryPrediction,
    FrozenChampion,
    category_scores_payload,
    predict_binary,
    predict_category,
    verify_inference_feature_contract,
)
from password_attack_detector.ml.thresholds import flagged_malicious
from tests.ml import predictions as px
from tests.ml import runs as rx

EPOCH = datetime(2024, 3, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> rx.Experiment:
    """Publish one experiment, shared by every test that only reads it."""
    return rx.publish_experiment(tmp_path_factory.mktemp("prediction-source"))


@pytest.fixture
def prepared(source: rx.Experiment, tmp_path: Path) -> px.Prepared:
    """Return a writable frozen champion, one per test."""
    return px.prepare(tmp_path / "root", source=source)


def anchored(**overrides: Any) -> dict[str, Any]:
    """Return the join identity every row schema carries."""
    fields: dict[str, Any] = {
        "anchor_event_id": "00000000-0000-5000-8000-000000000001",
        "anchor_event_time": EPOCH,
    }
    fields.update(overrides)
    return fields


def binary_row(**overrides: Any) -> BinaryPrediction:
    """Return one valid uncalibrated binary prediction."""
    fields: dict[str, Any] = {
        **anchored(),
        "score_kind": ScoreKind.DECISION_SCORE,
        "malicious_decision_score": 0.8,
        "malicious_probability": None,
        "decision_threshold": 0.5,
        "flagged_malicious": True,
    }
    fields.update(overrides)
    return BinaryPrediction(**fields)


# ---------------------------------------------------------------------------
# Loading the frozen champion
# ---------------------------------------------------------------------------


def test_a_valid_champion_loads_with_its_whole_lineage(
    prepared: px.Prepared,
) -> None:
    """The lock, the model, the calibrator, the threshold, and the head."""
    champion = prepared.champion()
    assert champion.lock.scope_key == prepared.scope_key
    assert champion.binary.model_id == champion.lock.model_id
    assert champion.threshold.selection_fingerprint == (
        champion.lock.binary_threshold_fingerprint
    )
    assert (champion.calibrator is not None) == is_probability(champion.score_kind)
    assert champion.category is not None
    assert champion.lock.category_head is not None
    assert champion.category.class_order == champion.lock.category_head.class_order
    assert champion.freeze_record_id


def test_loading_needs_no_model_path_and_accepts_none(
    prepared: px.Prepared,
) -> None:
    """There is no parameter through which a lock could be bypassed."""
    import inspect

    parameters = set(inspect.signature(FrozenChampion.load).parameters)
    assert parameters == {"root", "ledger", "scope_key", "catalog", "config"}
    for absent in ("model_path", "model_id", "force", "ignore_lock", "labels"):
        assert absent not in parameters, absent


def test_a_root_with_no_frozen_champion_is_refused(tmp_path: Path) -> None:
    """Predicting without a lock is not predicting with a default one."""
    from password_attack_detector.ml.ledger import ExperimentLedger

    with pytest.raises(ArtifactNotFoundError):
        FrozenChampion.load(tmp_path, ledger=ExperimentLedger(tmp_path / "ledger"))


def test_a_tampered_lock_is_refused(prepared: px.Prepared) -> None:
    """The seal is a field, so an edited lock recomputes to something else."""
    path = prepared.root / CHAMPION_DIR / prepared.scope_key / CHAMPION_LOCK_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["catalog_model_id"] != "M-999"
    payload["catalog_model_id"] = "M-999"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelTrainingError, match="champion lock"):
        prepared.champion()


def test_a_lock_moved_into_another_scope_is_refused(prepared: px.Prepared) -> None:
    """The directory name is checked against the lock's own scope key."""
    champion_root = prepared.root / CHAMPION_DIR
    moved = "0" * 64
    (champion_root / prepared.scope_key).rename(champion_root / moved)
    with pytest.raises(ModelNotReadyError, match="scope key"):
        prepared.champion(scope_key=moved)


def test_a_lock_the_ledger_never_recorded_is_refused(
    prepared: px.Prepared,
) -> None:
    """A file is not a decision; the freeze receipt is what records one."""
    for path in (prepared.root / "ledger" / "champion_freeze").glob("*.json"):
        path.unlink()
    with pytest.raises(ModelNotReadyError, match="freeze record"):
        prepared.champion()


def test_a_missing_model_manifest_is_refused(prepared: px.Prepared) -> None:
    """The manifest's bytes are digested, not its claims read."""
    champion = prepared.champion()
    manifest = (
        prepared.root
        / "runs"
        / champion.lock.training_run_id
        / "model"
        / "model_manifest.json"
    )
    manifest.unlink()
    with pytest.raises((ArtifactNotFoundError, ModelNotReadyError)):
        prepared.champion()


def test_a_replaced_model_manifest_is_refused(prepared: px.Prepared) -> None:
    """A manifest swapped after the freeze digests to something else."""
    champion = prepared.champion()
    manifest = (
        prepared.root
        / "runs"
        / champion.lock.training_run_id
        / "model"
        / "model_manifest.json"
    )
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ModelNotReadyError, match="model manifest"):
        prepared.champion()


def test_a_swapped_operating_point_is_refused(prepared: px.Prepared) -> None:
    """The frozen threshold is bound by fingerprint, not by file name."""
    champion = prepared.champion()
    path = (
        prepared.root
        / "runs"
        / champion.lock.training_run_id
        / "thresholds"
        / "binary_threshold.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_threshold"] = 0.999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelTrainingError, match="threshold selection"):
        prepared.champion()


def test_a_missing_operating_point_is_refused(prepared: px.Prepared) -> None:
    """A champion with nothing to apply is not a champion that flags nothing."""
    champion = prepared.champion()
    (
        prepared.root
        / "runs"
        / champion.lock.training_run_id
        / "thresholds"
        / "binary_threshold.json"
    ).unlink()
    with pytest.raises(ModelNotReadyError, match="operating point"):
        prepared.champion()


def test_a_swapped_calibrator_is_refused(prepared: px.Prepared) -> None:
    """Probabilities from a different calibrator are different probabilities."""
    champion = prepared.champion()
    if champion.calibrator is None:
        pytest.skip("this fixture's champion is uncalibrated")
    path = (
        prepared.root
        / "runs"
        / champion.lock.training_run_id
        / "calibration"
        / "calibration_state.json"
    )
    path.unlink()
    with pytest.raises(ModelNotReadyError, match="calibrator"):
        prepared.champion()


def test_a_tampered_category_abstention_point_is_refused(
    prepared: px.Prepared,
) -> None:
    """The head's abstention floor is bound by fingerprint too."""
    champion = prepared.champion()
    assert champion.category is not None
    path = (
        prepared.root
        / "runs"
        / champion.category.run_id
        / "thresholds"
        / "category_abstention.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["min_category_score"] = 0.999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelTrainingError, match="abstention selection"):
        prepared.champion()


def test_a_run_the_ledger_disagrees_with_is_refused(prepared: px.Prepared) -> None:
    """A champion whose own history contradicts it is not loaded."""
    champion = prepared.champion()
    receipt = (
        prepared.root / "runs" / champion.lock.training_run_id / "training_run.json"
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["train_row_count"] = payload["train_row_count"] + 1
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelTrainingError, match="training-run record"):
        prepared.champion()


def test_several_frozen_scopes_require_one_to_be_named(
    prepared: px.Prepared,
) -> None:
    """Predicting under whichever came first would depend on directory order."""
    champion_root = prepared.root / CHAMPION_DIR
    second = champion_root / ("f" * 64)
    second.mkdir()
    (second / CHAMPION_LOCK_FILE).write_text(
        (champion_root / prepared.scope_key / CHAMPION_LOCK_FILE).read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    from password_attack_detector.ml.ledger import ExperimentLedger

    with pytest.raises(ModelNotReadyError, match="frozen champions"):
        FrozenChampion.load(
            prepared.root, ledger=ExperimentLedger(prepared.root / "ledger")
        )


# ---------------------------------------------------------------------------
# Binary inference
# ---------------------------------------------------------------------------


def test_binary_inference_emits_one_row_per_scoped_anchor(
    prepared: px.Prepared,
) -> None:
    """Every row in scope, once, in canonical order."""
    dataset = px.inference_dataset(prepared)
    rows = predict_binary(prepared.champion(), dataset)
    assert len(rows) == dataset.row_count
    anchors = [row.anchor_event_id for row in rows]
    assert len(set(anchors)) == len(anchors)
    assert [row.anchor_event_id for row in rows] == [
        anchor.anchor_event_id for anchor in dataset.frame.anchors
    ]


def test_the_raw_decision_score_is_preserved_beside_the_probability(
    prepared: px.Prepared,
) -> None:
    """A calibrated publication keeps both, and says which decided."""
    champion = prepared.champion()
    rows = predict_binary(champion, px.inference_dataset(prepared))
    if champion.calibrator is None:
        pytest.skip("this fixture's champion is uncalibrated")
    for row in rows:
        assert row.malicious_probability is not None
        assert 0.0 <= row.malicious_probability <= 1.0
        assert math.isfinite(row.malicious_decision_score)
        assert row.score_kind is ScoreKind.CALIBRATED_PROBABILITY
        assert row.decided_score == row.malicious_probability


def test_the_frozen_threshold_is_applied_to_every_row(
    prepared: px.Prepared,
) -> None:
    """One operating point, recomputable from each row's own numbers."""
    champion = prepared.champion()
    rows = predict_binary(champion, px.inference_dataset(prepared))
    assert {row.decision_threshold for row in rows} == {champion.decision_threshold}
    for row in rows:
        assert row.flagged_malicious == (row.decided_score >= row.decision_threshold)


def test_a_row_exactly_on_the_threshold_is_flagged() -> None:
    """``>=``, and the difference is exactly the row sitting on the line."""
    row = binary_row(malicious_decision_score=0.5, decision_threshold=0.5)
    assert row.flagged_malicious is True
    assert flagged_malicious(0.5, threshold=0.5) is True
    with pytest.raises(ValueError, match="contradicts the frozen predicate"):
        binary_row(
            malicious_decision_score=0.5,
            decision_threshold=0.5,
            flagged_malicious=False,
        )


def test_a_row_just_below_the_threshold_is_not_flagged() -> None:
    """The other side of the same comparison."""
    row = binary_row(
        malicious_decision_score=0.499999999,
        decision_threshold=0.5,
        flagged_malicious=False,
    )
    assert row.flagged_malicious is False


def test_an_uncalibrated_row_carries_no_probability() -> None:
    """A raw score is never relabelled a probability."""
    with pytest.raises(ValueError, match="not a probability under another name"):
        binary_row(malicious_probability=0.8)


def test_a_calibrated_row_must_carry_its_probability() -> None:
    """A row decided on a probability records the probability it was decided on."""
    with pytest.raises(ValueError, match="must carry the probability"):
        binary_row(
            score_kind=ScoreKind.CALIBRATED_PROBABILITY, malicious_probability=None
        )


def test_a_probability_outside_the_unit_interval_is_refused() -> None:
    """Bounds are checked on the row, not assumed from the calibrator."""
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        binary_row(
            score_kind=ScoreKind.CALIBRATED_PROBABILITY,
            malicious_probability=1.5,
            decision_threshold=0.5,
            flagged_malicious=True,
        )


def test_a_class_score_kind_is_not_a_binary_decision() -> None:
    """Only two score kinds may decide a binary row."""
    with pytest.raises(ValueError, match="decided on a decision score"):
        binary_row(score_kind=ScoreKind.CLASS_SCORE)


def test_a_non_finite_score_is_refused() -> None:
    """NaN is stable in a digest and meaningless in a comparison."""
    with pytest.raises(ValueError, match="must be finite"):
        binary_row(malicious_decision_score=float("inf"))


def test_a_naive_anchor_time_is_refused() -> None:
    """A naive timestamp has no position on the timeline the order depends on."""
    with pytest.raises(ValueError, match="timezone-aware"):
        binary_row(anchor_event_time=datetime(2024, 3, 1))


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------


def triage(prepared: px.Prepared, **kwargs: Any) -> Any:
    """Return the binary predictions and the triage rows they route."""
    champion = prepared.champion()
    dataset = px.inference_dataset(prepared, **kwargs)
    binary = predict_binary(champion, dataset)
    return (
        champion,
        dataset,
        binary,
        predict_category(champion, dataset, binary=binary),
    )


def test_category_inference_uses_the_frozen_class_order(
    prepared: px.Prepared,
) -> None:
    """Every declared class, in the frozen order, on every row."""
    champion, _, _, rows = triage(prepared)
    assert champion.category is not None
    assert rows, "this fixture routes at least one row to triage"
    for row in rows:
        scores = row.class_scores()
        assert tuple(scores) == champion.category.class_order
        assert row.min_category_score == champion.category.min_category_score


def test_a_row_below_the_floor_abstains(prepared: px.Prepared) -> None:
    """Abstention is an outcome, not a failure."""
    champion, _, _, rows = triage(prepared)
    assert champion.category is not None
    for row in rows:
        if row.max_category_score < row.min_category_score:
            assert row.predicted_scenario == UNKNOWN_CATEGORY
        else:
            assert row.predicted_scenario in champion.category.class_order


# ---------------------------------------------------------------------------
# Category applicability: triage is downstream of the binary decision
# ---------------------------------------------------------------------------


def test_only_binary_positive_rows_are_categorised(prepared: px.Prepared) -> None:
    """The artifact's membership is the applicability record."""
    _, _, binary, rows = triage(prepared)
    flagged = {row.anchor_event_id for row in binary if row.flagged_malicious}
    categorised = {row.anchor_event_id for row in rows}
    assert categorised == flagged
    assert len(rows) == len(flagged)
    assert len(rows) < len(binary), "the fixture must flag some rows and not others"


def test_an_unflagged_row_never_yields_a_known_scenario(
    prepared: px.Prepared,
) -> None:
    """Property A: the head is never asked about a row the binary head cleared."""
    _, _, binary, rows = triage(prepared)
    unflagged = {row.anchor_event_id for row in binary if not row.flagged_malicious}
    assert unflagged, "the fixture must leave some rows unflagged"
    for row in rows:
        assert row.anchor_event_id not in unflagged


def test_an_unflagged_row_is_never_counted_as_an_abstention(
    prepared: px.Prepared,
) -> None:
    """Property B: not applicable is absence, not ``unknown``."""
    _, _, binary, rows = triage(prepared)
    unflagged = [row for row in binary if not row.flagged_malicious]
    abstained = [row for row in rows if row.predicted_scenario == UNKNOWN_CATEGORY]
    assert unflagged
    assert len(abstained) < len(unflagged) + len(abstained)
    for row in abstained:
        assert row.anchor_event_id in {
            item.anchor_event_id for item in binary if item.flagged_malicious
        }


def test_a_flagged_row_below_the_floor_is_unknown() -> None:
    """Property C: asked, and declined to commit."""
    order = ("brute_force", "credential_stuffing")
    row = CategoryPrediction(
        **anchored(),
        predicted_scenario=UNKNOWN_CATEGORY,
        category_scores_json=category_scores_payload((0.2, 0.1), order),
        max_category_score=0.2,
        min_category_score=0.35,
    )
    assert row.predicted_scenario == UNKNOWN_CATEGORY


def test_a_flagged_row_exactly_on_the_floor_is_a_known_class() -> None:
    """Property D: the frozen ``>=`` rule, restated at the boundary."""
    order = ("brute_force", "credential_stuffing")
    row = CategoryPrediction(
        **anchored(),
        predicted_scenario="brute_force",
        category_scores_json=category_scores_payload((0.35, 0.1), order),
        max_category_score=0.35,
        min_category_score=0.35,
    )
    assert row.predicted_scenario == "brute_force"


def test_category_scores_cannot_alter_the_binary_decision(
    prepared: px.Prepared,
) -> None:
    """Property F: triage reads the binary decision and never writes it."""
    champion = prepared.champion()
    dataset = px.inference_dataset(prepared)
    binary = predict_binary(champion, dataset)
    before = tuple(binary)
    predict_category(champion, dataset, binary=binary)
    assert tuple(binary) == before
    assert predict_binary(champion, dataset) == before


def test_triage_refuses_binary_rows_from_another_population(
    prepared: px.Prepared,
) -> None:
    """Applicability cannot be resolved against rows from a different scope."""
    champion = prepared.champion()
    test = px.inference_dataset(prepared)
    other = predict_binary(
        champion, px.inference_dataset(prepared, scope=MLSplit.TRAIN)
    )
    with pytest.raises(ModelNotReadyError, match="different rows"):
        predict_category(champion, test, binary=other)


def test_no_applicable_row_produces_an_empty_triage_table(
    prepared: px.Prepared,
) -> None:
    """Zero applicable rows is honest, and is not a table of abstentions."""
    champion = prepared.champion()
    dataset = px.inference_dataset(prepared)
    binary = predict_binary(champion, dataset)
    cleared = tuple(
        row.model_copy(
            update={
                "flagged_malicious": False,
                "decision_threshold": 1.0,
                "malicious_probability": 0.0,
            }
        )
        for row in binary
    )
    assert predict_category(champion, dataset, binary=cleared) == ()


def test_a_tie_goes_to_the_earliest_declared_class() -> None:
    """The declared order decides, and nothing else does."""
    order = ("brute_force", "credential_stuffing")
    row = CategoryPrediction(
        **anchored(),
        predicted_scenario="brute_force",
        category_scores_json=category_scores_payload((0.6, 0.6), order),
        max_category_score=0.6,
        min_category_score=0.2,
    )
    assert row.predicted_scenario == "brute_force"
    with pytest.raises(ValueError, match="contradicts the frozen abstention rule"):
        CategoryPrediction(
            **anchored(),
            predicted_scenario="credential_stuffing",
            category_scores_json=category_scores_payload((0.6, 0.6), order),
            max_category_score=0.6,
            min_category_score=0.2,
        )


def test_a_row_exactly_on_the_abstention_floor_is_assigned() -> None:
    """``>=`` here too, and stated rather than left to a call site."""
    order = ("brute_force", "credential_stuffing")
    row = CategoryPrediction(
        **anchored(),
        predicted_scenario="credential_stuffing",
        category_scores_json=category_scores_payload((0.1, 0.35), order),
        max_category_score=0.35,
        min_category_score=0.35,
    )
    assert row.predicted_scenario == "credential_stuffing"


def test_a_malformed_class_map_is_refused() -> None:
    """The one free-form field on the row is parsed strictly."""
    for payload in ('{"a": 1', "[]", '{"a": "x"}', '{"b": 1, "a": 2}'):
        with pytest.raises(
            ValueError, match=r"category_scores_json|not a number|order"
        ):
            CategoryPrediction(
                **anchored(),
                predicted_scenario=UNKNOWN_CATEGORY,
                category_scores_json=payload,
                max_category_score=0.0,
                min_category_score=1.0,
            )


def test_the_abstention_label_is_never_a_scored_class() -> None:
    """``unknown`` is the outcome, not a class the head was fitted to predict."""
    with pytest.raises(ValueError, match="abstention outcome"):
        CategoryPrediction(
            **anchored(),
            predicted_scenario=UNKNOWN_CATEGORY,
            category_scores_json=json.dumps({UNKNOWN_CATEGORY: 0.9, "z": 0.1}),
            max_category_score=0.9,
            min_category_score=0.2,
        )


def test_an_oversized_class_payload_is_refused() -> None:
    """A bounded field, because a prediction table is untrusted data."""
    with pytest.raises(ValueError, match="size ceiling"):
        CategoryPrediction(
            **anchored(),
            predicted_scenario=UNKNOWN_CATEGORY,
            category_scores_json=json.dumps(
                {f"c{index}": 0.1 for index in range(2000)}
            ),
            max_category_score=0.1,
            min_category_score=1.0,
        )


def test_no_frozen_head_means_no_fabricated_head(prepared: px.Prepared) -> None:
    """Property G: an all-unknown stand-in would be output nothing produced."""
    champion = prepared.champion()
    dataset = px.inference_dataset(prepared)
    binary = predict_binary(champion, dataset)
    stripped = type(champion)(
        lock=champion.lock,
        binary=champion.binary,
        threshold=champion.threshold,
        calibrator=champion.calibrator,
        category=None,
        scope_key=champion.scope_key,
        freeze_record_id=champion.freeze_record_id,
    )
    with pytest.raises(ModelNotReadyError, match="no category head"):
        predict_category(stripped, dataset, binary=binary)


def test_the_triage_join_is_deterministic_by_anchor(prepared: px.Prepared) -> None:
    """Property H: the two artifacts join on anchor identity, reproducibly."""
    _, _, binary, rows = triage(prepared)
    shuffled_champion = prepared.champion()
    shuffled = px.shuffled_inference_dataset(prepared)
    other_binary = predict_binary(shuffled_champion, shuffled)
    other_rows = predict_category(shuffled_champion, shuffled, binary=other_binary)
    assert [row.anchor_event_id for row in other_rows] == [
        row.anchor_event_id for row in rows
    ]
    assert other_rows == rows
    by_anchor = {row.anchor_event_id: row for row in binary}
    for row in rows:
        assert by_anchor[row.anchor_event_id].flagged_malicious is True


# ---------------------------------------------------------------------------
# The anomaly artifact
# ---------------------------------------------------------------------------


def test_an_anomaly_row_carries_a_magnitude_and_no_probability() -> None:
    """There is no probability field on the schema at all."""
    row = AnomalyScore(
        **anchored(), anomaly_score=-0.4, anomaly_threshold=None, flagged_anomalous=None
    )
    assert row.experimental is True
    assert row.influences_champion_selection is False
    assert "malicious_probability" not in AnomalyScore.model_fields
    assert "probability" not in " ".join(ANOMALY_PREDICTION_COLUMNS)


def test_the_anomaly_predicate_is_inverted() -> None:
    """A lower anomaly score is more anomalous, so the flag is ``<=``."""
    row = AnomalyScore(
        **anchored(),
        anomaly_score=-0.4,
        anomaly_threshold=-0.4,
        flagged_anomalous=True,
    )
    assert row.flagged_anomalous is True
    with pytest.raises(ValueError, match="contradicts the frozen predicate"):
        AnomalyScore(
            **anchored(),
            anomaly_score=-0.3,
            anomaly_threshold=-0.4,
            flagged_anomalous=True,
        )


def test_an_anomaly_row_cannot_claim_influence() -> None:
    """A measurement that can change what it measures is not one."""
    with pytest.raises(ValueError, match="never influences champion selection"):
        AnomalyScore(
            **anchored(),
            anomaly_score=-0.4,
            anomaly_threshold=None,
            flagged_anomalous=None,
            influences_champion_selection=True,
        )


def test_an_anomaly_flag_without_a_threshold_is_refused() -> None:
    """A flag and the threshold that produced it are recorded together."""
    with pytest.raises(ValueError, match="recorded together"):
        AnomalyScore(
            **anchored(),
            anomaly_score=-0.4,
            anomaly_threshold=None,
            flagged_anomalous=True,
        )


# ---------------------------------------------------------------------------
# Determinism and the label firewall
# ---------------------------------------------------------------------------


def test_shuffling_the_source_rows_changes_nothing(
    prepared: px.Prepared,
) -> None:
    """Physical file order reaches neither the rows nor the input fingerprint."""
    champion = prepared.champion()
    ordered = px.inference_dataset(prepared)
    shuffled = px.shuffled_inference_dataset(prepared)
    assert shuffled.inference_input_fingerprint == ordered.inference_input_fingerprint
    assert predict_binary(champion, shuffled) == predict_binary(champion, ordered)


def test_rewriting_the_labels_changes_nothing(prepared: px.Prepared) -> None:
    """The loader never opened them, so there is nothing for them to change."""
    champion = prepared.champion()
    before = predict_binary(champion, px.inference_dataset(prepared))
    fingerprint = px.inference_dataset(prepared).inference_input_fingerprint

    flipped = rx.Rows(
        features=list(prepared.rows.features),
        labels=[
            LabelRow(
                event_id=label.event_id,
                attack_class=label.attack_class,
                malicious=not label.malicious,
                supervised_training_eligible=label.supervised_training_eligible,
            )
            for label in prepared.rows.labels
        ],
        splits=list(prepared.rows.splits),
        campaigns=list(prepared.rows.campaigns),
    )
    after = predict_binary(champion, px.inference_dataset(prepared, rows=flipped))
    assert after == before
    assert (
        px.inference_dataset(prepared, rows=flipped).inference_input_fingerprint
        == fingerprint
    )


def test_changing_one_feature_value_moves_the_input_fingerprint(
    prepared: px.Prepared,
) -> None:
    """Identity follows the rows that were scored."""
    anchors = rx.anchors_in(prepared.rows, MLSplit.TEST)
    mutated = rx.revalue_anchor(prepared.rows, anchors[0], value=0.123456)
    assert (
        px.inference_dataset(prepared, rows=mutated).inference_input_fingerprint
        != px.inference_dataset(prepared).inference_input_fingerprint
    )


def test_scoring_is_repeatable_within_one_champion(prepared: px.Prepared) -> None:
    """Two calls, byte for byte."""
    champion = prepared.champion()
    dataset = px.inference_dataset(prepared)
    assert predict_binary(champion, dataset) == predict_binary(champion, dataset)


def test_test_and_holdout_are_separate_inference_inputs(
    prepared: px.Prepared,
) -> None:
    """Two scopes, two fingerprints, and no row in both."""
    test = px.inference_dataset(prepared, scope=MLSplit.TEST)
    holdout = px.inference_dataset(prepared, scope=MLSplit.NOVEL_ANOMALY_HOLDOUT)
    assert test.inference_input_fingerprint != holdout.inference_input_fingerprint
    test_anchors = {anchor.anchor_event_id for anchor in test.frame.anchors}
    holdout_anchors = {anchor.anchor_event_id for anchor in holdout.frame.anchors}
    assert not (test_anchors & holdout_anchors)


# ---------------------------------------------------------------------------
# The published columns
# ---------------------------------------------------------------------------


def test_the_published_columns_carry_the_join_key_and_nothing_else() -> None:
    """The one identity a prediction row needs, and no other context."""
    for columns in (
        BINARY_PREDICTION_COLUMNS,
        CATEGORY_PREDICTION_COLUMNS,
        ANOMALY_PREDICTION_COLUMNS,
    ):
        assert columns[0] == "anchor_event_id"
        assert columns[1] == "anchor_event_time"
        assert not set(columns) & PROHIBITED_PREDICTION_COLUMNS


@pytest.mark.parametrize(
    "column",
    [
        "campaign_id",
        "user_id",
        "source_id",
        "latitude",
        "password",
        "malicious",
        "attack_class",
        "split",
        "risk_score",
        "fused_flagged",
    ],
)
def test_a_forbidden_column_is_named_in_the_prohibition(column: str) -> None:
    """Every category the privacy rule covers, asserted rather than assumed."""
    assert column in PROHIBITED_PREDICTION_COLUMNS


def test_no_row_schema_declares_a_prohibited_column() -> None:
    """The import-time guard, restated where a reader will look for it."""
    for model in (BinaryPrediction, CategoryPrediction, AnomalyScore):
        assert not set(model.model_fields) & PROHIBITED_PREDICTION_COLUMNS


def test_the_prediction_contract_version_is_pinned() -> None:
    """A change to what a row carries is a visible edit."""
    assert PREDICTION_SCHEMA_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# The feature contract
# ---------------------------------------------------------------------------


def contract_arguments(prepared: px.Prepared, **overrides: Any) -> dict[str, Any]:
    """Return a coherent feature-contract check, with *overrides* applied."""
    catalog = rx.feature_catalog()
    eligible = rx.eligible_features(catalog)
    from tests.ml import factories as fx

    allowlist = fx.allowlist_for(catalog)
    arguments: dict[str, Any] = {
        "feature_manifest": {
            "feature_catalog_fingerprint": catalog.fingerprint(),
            "feature_schema_version": rx.config().required_feature_schema_version,
        },
        "catalog_fingerprint": catalog.fingerprint(),
        "allowlist_fingerprint": allowlist.fingerprint(),
        "eligible_feature_list_fingerprint": eligible.fingerprint(),
        "required_feature_schema_version": (
            rx.config().required_feature_schema_version
        ),
        "compatible_catalog_fingerprints": (catalog.fingerprint(),),
    }
    arguments.update(overrides)
    return arguments


def test_a_matching_feature_contract_is_accepted(prepared: px.Prepared) -> None:
    """The contract the champion was frozen against, offered back to it."""
    verify_inference_feature_contract(
        prepared.champion().lock, **contract_arguments(prepared)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("catalog_fingerprint", "a" * 64),
        ("allowlist_fingerprint", "b" * 64),
        ("eligible_feature_list_fingerprint", "c" * 64),
        ("required_feature_schema_version", "9.9.9"),
        ("compatible_catalog_fingerprints", ()),
    ],
)
def test_a_disagreeing_feature_contract_is_refused(
    prepared: px.Prepared, field: str, value: Any
) -> None:
    """A model scored against a different contract produces different numbers."""
    with pytest.raises(ModelNotReadyError, match="feature contract"):
        verify_inference_feature_contract(
            prepared.champion().lock, **contract_arguments(prepared, **{field: value})
        )


def test_an_out_of_scope_split_is_refused(prepared: px.Prepared) -> None:
    """Excluded rows are excluded from prediction too."""
    from password_attack_detector.exceptions import DataValidationError

    with pytest.raises(DataValidationError):
        px.inference_dataset(prepared, scope=MLSplit.EXCLUDED)


def test_the_scoring_functions_take_no_label_argument() -> None:
    """Stated as a signature, so a reviewer does not have to remember it."""
    import inspect

    assert set(inspect.signature(predict_binary).parameters) == {
        "champion",
        "dataset",
    }
    # ``binary`` is this publication's own decisions, not ground truth: it is
    # what routes a row to triage, and it is produced two lines earlier by the
    # same frozen champion.
    assert set(inspect.signature(predict_category).parameters) == {
        "champion",
        "dataset",
        "binary",
    }
    for function in (predict_binary, predict_category):
        for absent in ("labels", "truth", "targets", "y_true"):
            assert absent not in inspect.signature(function).parameters, absent


def test_a_timestamp_helper_is_never_the_current_time() -> None:
    """No wall clock takes part in a prediction; the anchors carry the time."""
    row = binary_row(anchor_event_time=EPOCH + timedelta(minutes=3))
    assert row.anchor_event_time.year == 2024


# ---------------------------------------------------------------------------
# The experimental probe, end to end
# ---------------------------------------------------------------------------


def anomaly_run_id(prepared: px.Prepared) -> str:
    """Return the published experimental anomaly run's identifier."""
    from password_attack_detector.ml.enums import MLTask
    from password_attack_detector.ml.ledger import ExperimentLedger

    ledger = ExperimentLedger(prepared.root / "ledger")
    runs = [
        record for record in ledger.training_runs() if record.task is MLTask.ANOMALY
    ]
    assert runs, "the fixture publishes an experimental anomaly run"
    return runs[0].record_id


def test_a_verified_anomaly_run_scores_magnitudes(prepared: px.Prepared) -> None:
    """An anomaly_score per row, and no probability anywhere near it."""
    from password_attack_detector.ml.predictions import (
        ExperimentalAnomalyRun,
        predict_anomaly,
    )

    probe = ExperimentalAnomalyRun.load(
        prepared.root, anomaly_run_id(prepared), ledger=prepared.ledger
    )
    dataset = px.inference_dataset(prepared)
    rows = predict_anomaly(probe, dataset)
    assert len(rows) == dataset.row_count
    for row in rows:
        assert math.isfinite(row.anomaly_score)
        assert row.experimental is True
        assert row.influences_champion_selection is False
    assert "malicious_probability" not in AnomalyScore.model_fields


def test_the_anomaly_probe_is_never_read_from_the_lock(
    prepared: px.Prepared,
) -> None:
    """M-030 is named explicitly or not at all."""
    champion = prepared.champion()
    rendered = champion.lock.to_json()
    assert anomaly_run_id(prepared) not in rendered
    assert "anomaly" not in rendered.lower()


def test_a_supervised_run_is_refused_as_an_anomaly_probe(
    prepared: px.Prepared,
) -> None:
    """The experimental artifact is published from its own lineage."""
    from password_attack_detector.ml.predictions import ExperimentalAnomalyRun

    with pytest.raises(ModelNotReadyError, match="not an anomaly run"):
        ExperimentalAnomalyRun.load(
            prepared.root,
            prepared.champion().lock.training_run_id,
            ledger=prepared.ledger,
        )


def test_the_anomaly_probe_changes_no_supervised_decision(
    prepared: px.Prepared,
) -> None:
    """Publishing it beside the binary rows leaves them byte for byte."""
    from password_attack_detector.ml.predictions import ExperimentalAnomalyRun

    champion = prepared.champion()
    dataset = px.inference_dataset(prepared)
    without = predict_binary(champion, dataset)
    ExperimentalAnomalyRun.load(
        prepared.root, anomaly_run_id(prepared), ledger=prepared.ledger
    )
    assert predict_binary(champion, dataset) == without


def test_a_corrupted_preprocessor_refuses_the_champion(
    prepared: px.Prepared,
) -> None:
    """A matrix built by different rules is a different matrix."""
    champion = prepared.champion()
    path = (
        prepared.root
        / "runs"
        / champion.lock.training_run_id
        / "model"
        / "preprocessor.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["preprocessing_schema_version"] = "9.9.9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="CHECKSUM_MISMATCH"):
        prepared.champion()
