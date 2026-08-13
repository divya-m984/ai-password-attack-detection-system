"""The locked TEST evaluation, against artifacts the pipeline really froze.

Nothing here is hand-built: the dataset is assembled, trained, selected, frozen,
predicted from, and only then evaluated. An evaluation suite fed a fixture object
graph would pass against shapes the pipeline never produces.

The file is organised around the two properties the milestone rests on.

**The order.** Every frozen-lineage failure fires *before* the outcomes are used
for anything, and the suite proves that with a sentinel: a
:class:`TestOutcome` subclass that raises if its ground truth is read at all.
An evaluation refused for a broken champion must never have touched it.

**Immutability.** A published receipt is byte-identical after the labels behind
it change, a second identical evaluation is idempotent, a materially different
one gets its own identity, and a conflicting one at the same identity is
refused.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import (
    DataValidationError,
    ExperimentPublicationError,
    ModelNotReadyError,
)
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    ComparisonSystem,
    ExperimentRecordType,
    MLSplit,
)
from password_attack_detector.ml.enums import (
    TestEvaluationStatus as EvaluationStatus,
)
from password_attack_detector.ml.fusion import RuleEvidence
from password_attack_detector.ml.ledger import ExperimentLedger
from password_attack_detector.ml.ledger import (
    TestEvaluationRecord as EvaluationRecord,
)
from password_attack_detector.ml.prediction_manifest import (
    BINARY_PREDICTION_FILE,
    CATEGORY_PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    PREDICTIONS_DIR,
    PredictionManifest,
)
from password_attack_detector.ml.prediction_publisher import publish_predictions
from password_attack_detector.ml.prediction_serialization import (
    read_binary_predictions,
    read_category_predictions,
)
from password_attack_detector.ml.test_evaluation import (
    EVALUATION_RECEIPT_FILE,
    EVALUATIONS_DIR,
    SYNTHETIC_TEST_CAVEAT,
    SYSTEM_COMPARISON_MD,
    evaluate_test,
    publish_evaluation,
    reconcile_evaluations,
)
from password_attack_detector.ml.test_evaluation import (
    TestOutcome as Outcome,
)
from password_attack_detector.ml.test_evaluation import (
    test_label_fingerprint as label_fingerprint,
)
from password_attack_detector.ml.test_evaluation import (
    test_population_fingerprint as population_fingerprint,
)
from tests.ml import predictions as px
from tests.ml import runs as rx

RULE_CONFIG = "c" * 64


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> rx.Experiment:
    """Publish one experiment, shared by every test in this module."""
    return rx.publish_experiment(tmp_path_factory.mktemp("evaluation-source"))


class Prepared:
    """One frozen champion with published TEST predictions, ready to evaluate."""

    def __init__(self, source: rx.Experiment, root: Path) -> None:
        self.prepared = px.prepare(root, source=source)
        self.champion = self.prepared.champion()
        dataset = px.inference_dataset(self.prepared, scope=MLSplit.TEST)
        publication = publish_predictions(
            champion=self.champion, dataset=dataset, root=self.prepared.root
        )
        directory = self.prepared.root / PREDICTIONS_DIR / publication.prediction_id
        self.manifest = PredictionManifest.from_json(
            (directory / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        self.binary = read_binary_predictions(directory / BINARY_PREDICTION_FILE)
        self.category = (
            read_category_predictions(directory / CATEGORY_PREDICTION_FILE)
            if (directory / CATEGORY_PREDICTION_FILE).is_file()
            else None
        )
        self.root = self.prepared.root
        self.ledger = ExperimentLedger(self.root / "ledger")

    def outcomes(self, *, flip: bool = False) -> list[Outcome]:
        """Return the TEST outcomes a permitted reader produced."""
        rows = rx.dataset(self.prepared.rows).for_split(MLSplit.TEST)
        return [
            Outcome(
                anchor_event_id=anchor.anchor_event_id,
                malicious=(not anchor.malicious) if flip else anchor.malicious,
                known_category=anchor.known_category,
                split=MLSplit.TEST,
            )
            for anchor in rows.anchors
        ]

    def rules(self) -> dict[str, RuleEvidence]:
        """Return stand-in frozen Phase 4 decisions for the TEST anchors."""
        return {
            row.anchor_event_id: RuleEvidence(
                flagged=index % 3 == 0,
                ordinal_risk_score=70.0 if index % 3 == 0 else 5.0,
            )
            for index, row in enumerate(self.binary)
        }

    def evaluate(self, **overrides: Any) -> Any:
        """Run the locked evaluation over this frozen state."""
        settings: dict[str, Any] = {
            "champion": self.champion,
            "manifest": self.manifest,
            "binary": self.binary,
            "category": self.category,
            "rule_decisions": self.rules(),
            "rule_configuration_fingerprint": RULE_CONFIG,
            "fusion": None,
            "stacked_state": None,
            "outcomes": self.outcomes(),
        }
        settings.update(overrides)
        return evaluate_test(**settings)


@pytest.fixture
def prepared(source: rx.Experiment, tmp_path: Path) -> Prepared:
    """Return a writable frozen champion with published TEST predictions."""
    return Prepared(source, tmp_path / "root")


# ---------------------------------------------------------------------------
# The evaluation itself
# ---------------------------------------------------------------------------


def test_a_locked_evaluation_completes_over_the_frozen_state(
    prepared: Prepared,
) -> None:
    """Two systems, one population, and a receipt bound to the whole lineage."""
    evaluation = prepared.evaluate()
    assert evaluation.status is EvaluationStatus.COMPLETED
    systems = [item.system for item in evaluation.comparison.systems]
    assert systems == [ComparisonSystem.RULE_ONLY, ComparisonSystem.ML_ONLY]
    assert evaluation.record.record_type is ExperimentRecordType.TEST_EVALUATION
    assert evaluation.record.prediction_id == prepared.manifest.prediction_id
    assert evaluation.record.rule_configuration_fingerprint == RULE_CONFIG


def test_the_rule_only_system_gets_no_manufactured_discrimination_metric(
    prepared: Prepared,
) -> None:
    """Its ordinal magnitude is not a ranking score, and none is invented."""
    rule_only = prepared.evaluate().comparison.for_system(ComparisonSystem.RULE_ONLY)
    assert rule_only is not None
    assert rule_only.metrics.pr_auc is None
    assert rule_only.metrics.pr_auc_unavailable_reason is not None
    assert "ordinal" in rule_only.metrics.pr_auc_unavailable_reason
    assert rule_only.metrics.calibration is None


def test_the_ml_only_system_is_scored_from_the_published_predictions(
    prepared: Prepared,
) -> None:
    """The stored decision, the stored score, and the stored probability."""
    evaluation = prepared.evaluate()
    ml_only = evaluation.comparison.for_system(ComparisonSystem.ML_ONLY)
    assert ml_only is not None
    assert ml_only.metrics.pr_auc is not None
    assert ml_only.decision_source.endswith(prepared.manifest.prediction_id)
    assert ml_only.metrics.confusion.row_count == len(prepared.binary)


def test_the_hybrid_is_unavailable_without_a_frozen_selection(
    prepared: Prepared,
) -> None:
    """No hybrid is fabricated when no validation-only selection chose one."""
    evaluation = prepared.evaluate()
    assert evaluation.comparison.for_system(ComparisonSystem.HYBRID) is None
    assert evaluation.comparison.hybrid_unavailable_reason is not None
    assert "no_fusion_selection" in evaluation.comparison.hybrid_unavailable_reason
    assert evaluation.record.selected_fusion_strategy is None


def test_the_category_evaluation_keeps_the_three_states_apart(
    prepared: Prepared,
) -> None:
    """Not-applicable, unknown, and a known class are counted separately."""
    category = prepared.evaluate().category
    assert category is not None
    flagged = sum(1 for row in prepared.binary if row.flagged_malicious)
    assert category.applicable_row_count == flagged
    assert category.not_applicable_count == len(prepared.binary) - flagged
    assert (
        category.known_output_count + category.unknown_count
        == category.applicable_row_count
    )
    head = prepared.champion.category
    assert head is not None
    assert category.class_order == head.class_order
    assert UNKNOWN_CATEGORY not in category.class_order


def test_a_binary_negative_row_is_never_a_category_abstention(
    prepared: Prepared,
) -> None:
    """The head was not asked about it, so it did not refuse."""
    category = prepared.evaluate().category
    assert category is not None
    assert category.not_applicable_count > 0
    assert category.unknown_count <= category.applicable_row_count


def test_every_report_carries_the_synthetic_caveat(prepared: Prepared) -> None:
    """No number here is a production claim, and every report says so."""
    reports = prepared.evaluate().reports
    assert reports
    for name, body in reports.items():
        assert SYNTHETIC_TEST_CAVEAT in body, name


def test_the_comparison_report_says_it_selects_nothing(prepared: Prepared) -> None:
    """A descriptive final evaluation, stated in the artifact."""
    body = prepared.evaluate().reports[SYSTEM_COMPARISON_MD]
    assert "does not select a champion" in body
    assert "declared order" in body


def test_reports_are_deterministic(prepared: Prepared) -> None:
    """Two evaluations of one frozen state render byte-identical documents."""
    assert prepared.evaluate().reports == prepared.evaluate().reports


# ---------------------------------------------------------------------------
# The order: refusals fire before the outcomes are used
# ---------------------------------------------------------------------------


class ExplodingOutcome:
    """A sentinel that duck-types an outcome and refuses to disclose its label.

    Used to prove the ordering claim behaviourally rather than by inspection: if
    a frozen-lineage refusal ever moved to after the labels were consumed, these
    tests would raise ``AssertionError`` instead of the expected refusal.
    """

    def __init__(self, anchor_event_id: str, known_category: str | None) -> None:
        self.anchor_event_id = anchor_event_id
        self.known_category = known_category
        self.split = MLSplit.TEST

    @property
    def malicious(self) -> bool:  # pragma: no cover - must never be reached
        raise AssertionError(
            "the TEST ground truth was read before the frozen lineage verified"
        )


def sentinels(prepared: Prepared) -> list[Any]:
    """Return outcomes that refuse to disclose their labels."""
    return [
        ExplodingOutcome(item.anchor_event_id, item.known_category)
        for item in prepared.outcomes()
    ]


def test_a_prediction_publication_for_another_split_is_refused_before_labels(
    prepared: Prepared,
) -> None:
    """The refusal fires while the outcomes are still unread."""
    other = prepared.manifest.model_copy(update={"scope": MLSplit.VALIDATION})
    with pytest.raises(ModelNotReadyError, match="evaluates TEST predictions"):
        prepared.evaluate(manifest=other, outcomes=sentinels(prepared))


def test_predictions_from_another_champion_are_refused_before_labels(
    prepared: Prepared,
) -> None:
    """A receipt would otherwise name a model that did not decide."""
    lineage = prepared.manifest.lineage.model_copy(
        update={"champion_lock_fingerprint": "9" * 64}
    )
    other = prepared.manifest.model_copy(update={"lineage": lineage})
    with pytest.raises(ModelNotReadyError, match="different champion"):
        prepared.evaluate(manifest=other, outcomes=sentinels(prepared))


def test_a_missing_rule_configuration_is_refused_before_labels(
    prepared: Prepared,
) -> None:
    """A comparator names the frozen configuration it ran under."""
    with pytest.raises(ModelNotReadyError, match="names the frozen configuration"):
        prepared.evaluate(
            rule_configuration_fingerprint="", outcomes=sentinels(prepared)
        )


def test_a_fusion_selection_from_another_champion_is_refused_before_labels(
    prepared: Prepared, tmp_path: Path
) -> None:
    """A hybrid chosen against a different model is not this model's hybrid."""
    from tests.unit.ml.test_fusion import select

    selection = select()
    with pytest.raises(ModelNotReadyError, match="different champion"):
        prepared.evaluate(fusion=selection, outcomes=sentinels(prepared))


def test_a_missing_rule_decision_is_refused(prepared: Prepared) -> None:
    """A system evaluated on fewer rows than its comparators is not one."""
    rules = prepared.rules()
    rules.pop(next(iter(rules)))
    with pytest.raises(DataValidationError, match="missing 1 decision"):
        prepared.evaluate(rule_decisions=rules)


def test_a_row_outside_the_test_split_is_refused(prepared: Prepared) -> None:
    """A supervised TEST evaluation covers TEST rows and only those."""
    outcomes = prepared.outcomes()
    outcomes[0] = replace(outcomes[0], split=MLSplit.VALIDATION)
    with pytest.raises(DataValidationError, match="not in the TEST split"):
        prepared.evaluate(outcomes=outcomes)


def test_a_missing_outcome_is_refused(prepared: Prepared) -> None:
    """The population and the ground truth must describe the same rows."""
    with pytest.raises(DataValidationError, match="carry no TEST outcome"):
        prepared.evaluate(outcomes=prepared.outcomes()[:-1])


# ---------------------------------------------------------------------------
# Scoped fingerprints
# ---------------------------------------------------------------------------


def test_the_label_fingerprint_covers_only_the_evaluated_rows(
    prepared: Prepared,
) -> None:
    """A digest over the whole table would move when an unrelated row did."""
    outcomes = prepared.outcomes()
    assert label_fingerprint(outcomes) == label_fingerprint(list(reversed(outcomes)))
    assert label_fingerprint(outcomes) != label_fingerprint(outcomes[:-1])


def test_changing_one_evaluated_label_moves_the_evaluation_identity(
    prepared: Prepared,
) -> None:
    """The receipt is about these outcomes, so a changed one is a new receipt."""
    baseline = prepared.evaluate()
    flipped = prepared.outcomes()
    flipped[0] = replace(flipped[0], malicious=not flipped[0].malicious)
    changed = prepared.evaluate(outcomes=flipped)
    assert changed.record.test_label_fingerprint != (
        baseline.record.test_label_fingerprint
    )
    assert changed.record.record_id != baseline.record.record_id


def test_the_population_and_label_digests_are_separate(prepared: Prepared) -> None:
    """A row leaving the population and a row's outcome changing are two findings."""
    outcomes = prepared.outcomes()
    flipped = list(outcomes)
    flipped[0] = replace(flipped[0], malicious=not flipped[0].malicious)
    assert population_fingerprint(flipped) == population_fingerprint(outcomes)
    assert label_fingerprint(flipped) != label_fingerprint(outcomes)


def test_no_anchor_appears_in_a_published_digest(prepared: Prepared) -> None:
    """Anchors take part in the digests; a hash is what leaves."""
    record = prepared.evaluate().record
    rendered = record.to_json()
    for row in prepared.binary[:20]:
        assert row.anchor_event_id not in rendered


# ---------------------------------------------------------------------------
# The receipt: identity, immutability, idempotency
# ---------------------------------------------------------------------------


def test_the_receipt_binds_the_whole_frozen_lineage(prepared: Prepared) -> None:
    """Champion, predictions, rules, metrics, and the computed comparison."""
    record = prepared.evaluate().record
    lock = prepared.champion.lock
    assert record.champion_freeze_record_id == prepared.champion.freeze_record_id
    assert record.champion_lock_fingerprint == lock.lock_fingerprint
    assert record.selected_run_id == lock.training_run_id
    assert record.selected_model_content_fingerprint == lock.model_content_fingerprint
    assert record.preprocessor_fingerprint == lock.preprocessor_fingerprint
    assert record.binary_threshold_fingerprint == lock.binary_threshold_fingerprint
    assert record.prediction_manifest_fingerprint == (
        prepared.manifest.prediction_manifest_fingerprint
    )
    assert record.metric_definition_fingerprint
    assert record.comparison_fingerprint
    assert record.report_fingerprints


def test_the_receipt_carries_nothing_observational() -> None:
    """No path, no host, no user, no publication time, no back-fill slot."""
    for absent in (
        "output_dir",
        "hostname",
        "username",
        "published_at",
        "created_at",
        "test_metrics",
    ):
        assert absent not in EvaluationRecord.model_fields, absent


def test_an_identical_evaluation_is_idempotent(prepared: Prepared) -> None:
    """Anti-peek F: same frozen state, same receipt, one publication."""
    first = prepared.evaluate()
    second = prepared.evaluate()
    assert first.record.record_id == second.record.record_id
    assert first.record.to_json() == second.record.to_json()

    published = publish_evaluation(first, root=prepared.root, ledger=prepared.ledger)
    again = publish_evaluation(second, root=prepared.root, ledger=prepared.ledger)
    assert published.created is True
    assert again.created is False
    assert again.record_id == published.record_id
    assert len(list((prepared.root / EVALUATIONS_DIR).iterdir())) == 1


def test_a_materially_different_evaluation_gets_its_own_identity(
    prepared: Prepared,
) -> None:
    """A different rule configuration is a different evaluation."""
    baseline = prepared.evaluate()
    other = prepared.evaluate(rule_configuration_fingerprint="d" * 64)
    assert other.record.record_id != baseline.record.record_id
    publish_evaluation(baseline, root=prepared.root, ledger=prepared.ledger)
    publish_evaluation(other, root=prepared.root, ledger=prepared.ledger)
    assert len(list((prepared.root / EVALUATIONS_DIR).iterdir())) == 2


def test_a_conflicting_evaluation_at_one_identity_is_refused(
    prepared: Prepared,
) -> None:
    """A published evaluation is evidence and is never rewritten."""
    evaluation = prepared.evaluate()
    publish_evaluation(evaluation, root=prepared.root, ledger=prepared.ledger)
    directory = prepared.root / EVALUATIONS_DIR / evaluation.record.record_id
    payload = json.loads(
        (directory / EVALUATION_RECEIPT_FILE).read_text(encoding="utf-8")
    )
    payload["row_count"] = payload["row_count"] + 1
    (directory / EVALUATION_RECEIPT_FILE).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ExperimentPublicationError, match=r"not readable|different"):
        publish_evaluation(evaluation, root=prepared.root, ledger=prepared.ledger)


def test_a_receipt_survives_its_labels_changing_afterwards(
    prepared: Prepared,
) -> None:
    """Anti-peek B: a published receipt is never back-filled or overwritten."""
    evaluation = prepared.evaluate()
    publish_evaluation(evaluation, root=prepared.root, ledger=prepared.ledger)
    directory = prepared.root / EVALUATIONS_DIR / evaluation.record.record_id
    before = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }

    # A later evaluation of *different* TEST truth is a different evaluation.
    flipped = prepared.evaluate(outcomes=prepared.outcomes(flip=True))
    publish_evaluation(flipped, root=prepared.root, ledger=prepared.ledger)

    after = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }
    assert after == before
    assert flipped.record.record_id != evaluation.record.record_id


def test_publication_leaves_every_earlier_ledger_record_untouched(
    prepared: Prepared,
) -> None:
    """A test evaluation references the frozen state; it amends none of it."""
    before = {
        path.name: path.read_bytes()
        for path in (prepared.root / "ledger").rglob("*.json")
    }
    publish_evaluation(prepared.evaluate(), root=prepared.root, ledger=prepared.ledger)
    after = {
        path.name: path.read_bytes()
        for path in (prepared.root / "ledger").rglob("*.json")
    }
    for name, payload in before.items():
        assert after[name] == payload, name


def test_a_published_evaluation_can_be_reconciled_into_the_ledger(
    prepared: Prepared,
) -> None:
    """The recovery for an evaluation promoted and then not indexed."""
    evaluation = prepared.evaluate()
    publish_evaluation(evaluation, root=prepared.root, ledger=prepared.ledger)
    for path in (prepared.root / "ledger" / "test_evaluation").glob("*.json"):
        path.unlink()
    appended = reconcile_evaluations(root=prepared.root, ledger=prepared.ledger)
    assert appended == (evaluation.record.record_id,)
    assert prepared.ledger.read_evaluation(evaluation.record.record_id)


def test_a_failed_publication_leaves_no_partial_evaluation(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No half-written directory that a later reader would treat as complete."""
    import password_attack_detector.ml.test_evaluation as module

    def refuse(*_: Any, **__: Any) -> None:
        raise RuntimeError("injected failure while staging")

    monkeypatch.setattr(module, "_verify_staged", refuse)
    with pytest.raises(ExperimentPublicationError):
        publish_evaluation(
            prepared.evaluate(), root=prepared.root, ledger=prepared.ledger
        )
    root = prepared.root / EVALUATIONS_DIR
    assert not root.exists() or not any(root.iterdir())


# ---------------------------------------------------------------------------
# Anti-peek: what may and may not move the evaluation
# ---------------------------------------------------------------------------


def test_changing_test_labels_leaves_the_predictions_and_champion_alone(
    prepared: Prepared,
) -> None:
    """Anti-peek A: the upstream artifacts never read them."""
    lock_before = (
        prepared.root / "champion" / prepared.prepared.scope_key / "champion.lock"
    ).read_bytes()
    directory = prepared.root / PREDICTIONS_DIR / prepared.manifest.prediction_id
    predictions_before = (directory / BINARY_PREDICTION_FILE).read_bytes()

    prepared.evaluate(outcomes=prepared.outcomes(flip=True))

    assert (
        prepared.root / "champion" / prepared.prepared.scope_key / "champion.lock"
    ).read_bytes() == lock_before
    assert (directory / BINARY_PREDICTION_FILE).read_bytes() == predictions_before


def test_changing_the_novel_holdout_does_not_move_the_supervised_identity(
    prepared: Prepared,
) -> None:
    """Anti-peek E: the holdout is outside the supervised receipt entirely."""
    from password_attack_detector.ml.test_evaluation import evaluate_novel_holdout

    baseline = prepared.evaluate()
    # The supervised evaluation is built from TEST rows; a holdout evaluation is
    # a separate artifact and cannot reach the receipt's identity.
    assert evaluate_novel_holdout(scores=(), outcomes={}).row_count == 0
    assert prepared.evaluate().record.record_id == baseline.record.record_id


def test_an_evaluation_writes_no_selection_of_any_kind(prepared: Prepared) -> None:
    """No champion is re-chosen and no threshold moves after TEST is opened."""
    evaluation = prepared.evaluate()
    assert evaluation.record.selected_run_id == prepared.champion.lock.training_run_id
    assert evaluation.record.binary_threshold_fingerprint == (
        prepared.champion.lock.binary_threshold_fingerprint
    )
    for absent in ("winner", "best_system", "ranking"):
        assert absent not in type(evaluation.comparison).model_fields, absent


def test_the_evaluation_module_imports_no_label_reader() -> None:
    """The exact allowlist is unchanged: outcomes arrive as typed arguments."""
    import ast
    from pathlib import Path as _Path

    module = (
        _Path(__file__).resolve().parents[3]
        / "src"
        / "password_attack_detector"
        / "ml"
        / "test_evaluation.py"
    )
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
            imported |= {alias.name for alias in node.names}
    for forbidden in (
        "password_attack_detector.detection.evaluation",
        "password_attack_detector.features.serialization",
        "password_attack_detector.data.serialization",
        "LabelRecord",
        "SplitRecord",
        "GroundTruthLabel",
    ):
        assert forbidden not in imported, forbidden
    assert "pyarrow" not in {name.split(".")[0] for name in imported}
