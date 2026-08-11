"""Validation-only champion selection.

Every test here starts from a *published* Milestone 6 experiment rather than a
hand-built object graph. Selection reads artifacts off disk and revalidates them
against their own digests, so a stub would let this suite pass against a shape
the training pipeline never produces.

The experiment is published once for the module. Nothing in this file republishes
it: the negative outcomes are built by withholding evidence or by judging the
same evidence under a different acceptance criterion, which is what actually
happens in the field, and which keeps the suite CI-sized.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import (
    DataValidationError,
    ExperimentPublicationError,
    ModelTrainingError,
)
from password_attack_detector.ml.catalog import MODEL_CATALOG
from password_attack_detector.ml.enums import (
    ChampionStatus,
    GateStatus,
    MLTask,
)
from password_attack_detector.ml.ledger import (
    ExperimentLedger,
    ValidationSelectionRecord,
)
from password_attack_detector.ml.ranking import (
    DISCRIMINATION_SCORE_KIND,
    PR_AUC_INTEGRATION,
    RANKING_METRIC_NAME,
)
from password_attack_detector.ml.selection import (
    SELECTION_FILE,
    CandidateEvidence,
    SelectionOutcome,
    champion_candidate_model_ids,
    load_candidate_evidence,
    publish_selection,
    reconcile_selections,
    select_binary_champion,
    select_category_head,
    selection_report,
    selection_report_markdown,
)
from tests.ml import runs as rx

BINARY = MLTask.BINARY_MALICIOUS
CATEGORY = MLTask.ATTACK_CATEGORY


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> rx.Experiment:
    """Return one published experiment, shared by every test in this module."""
    return rx.publish_experiment(tmp_path_factory.mktemp("experiment"))


@pytest.fixture
def evidence(experiment: rx.Experiment) -> tuple[CandidateEvidence, ...]:
    """Return the published evidence, in the order selection loaded it."""
    return experiment.evidence


def binary(
    evidence: tuple[CandidateEvidence, ...], **overrides: Any
) -> SelectionOutcome:
    """Return the binary selection over *evidence* under a CI-sized config."""
    return select_binary_champion(evidence, config=rx.config(**overrides))


def gate_of(outcome: SelectionOutcome, run_id: str, gate_id: str) -> Any:
    """Return one candidate's verdict on one gate."""
    result = next(item for item in outcome.results if item.run_id == run_id)
    return next(item for item in result.gates if item.gate_id == gate_id)


def without(
    evidence: tuple[CandidateEvidence, ...], catalog_model_id: str
) -> tuple[CandidateEvidence, ...]:
    """Return *evidence* with every run of one catalog entry withheld."""
    return tuple(item for item in evidence if item.catalog_model_id != catalog_model_id)


# ---------------------------------------------------------------------------
# The candidate universe
# ---------------------------------------------------------------------------


def test_the_candidate_universe_is_derived_from_the_catalog() -> None:
    """Not written down a second time, where it could quietly disagree."""
    assert champion_candidate_model_ids(BINARY) == ("M-001", "M-010", "M-020")
    assert champion_candidate_model_ids(CATEGORY) == ("M-010", "M-020")


def test_the_reference_baseline_is_not_in_the_candidate_universe() -> None:
    """M-000 supports the binary task and is still never promotable."""
    spec = MODEL_CATALOG.get("M-000")
    assert BINARY in spec.supported_tasks
    assert not spec.champion_eligible
    assert "M-000" not in champion_candidate_model_ids(BINARY)


def test_an_unproven_serializer_is_not_a_candidate() -> None:
    """M-021 round-trips nothing anybody verified, so it promotes nothing."""
    assert not MODEL_CATALOG.get("M-021").champion_eligible
    assert "M-021" not in champion_candidate_model_ids(BINARY)


def test_the_anomaly_task_has_no_champion_candidates() -> None:
    """The anomaly track is experimental; Milestone 7 promotes nothing from it."""
    assert champion_candidate_model_ids(MLTask.ANOMALY) == ()
    assert "M-030" not in champion_candidate_model_ids(BINARY)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_is_ordered_by_identifier(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Two machines enumerating the same runs produce the same sequence."""
    run_ids = [item.run_id for item in evidence]
    assert run_ids == sorted(run_ids)
    assert len(run_ids) == len(set(run_ids))


def test_every_published_run_is_read_back_verified(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Nothing is trusted because it was written by this process."""
    assert evidence
    assert all(item.artifact_verified for item in evidence)


def test_a_directory_the_ledger_does_not_hold_is_not_evidence(
    experiment: rx.Experiment, tmp_path: Path
) -> None:
    """The ledger is the record of what was published; a stray copy is not."""
    root = tmp_path / "root"
    shutil.copytree(experiment.root, root)
    original = next(iter(sorted((root / "runs").iterdir())))
    shutil.copytree(original, root / "runs" / ("0" * 32))

    ledger = ExperimentLedger(root / "ledger")
    loaded = load_candidate_evidence(root, ledger=ledger)
    assert [item.run_id for item in loaded] == [
        item.run_id for item in experiment.evidence
    ]


def test_a_run_that_contradicts_its_ledger_record_is_refused(
    experiment: rx.Experiment, tmp_path: Path
) -> None:
    """Evidence whose own history contradicts it is a fault, not a candidate."""
    root = tmp_path / "root"
    shutil.copytree(experiment.root, root)
    directories = sorted((root / "runs").iterdir())
    swapped = (directories[0] / "training_run.json").read_text(encoding="utf-8")
    (directories[1] / "training_run.json").write_text(swapped, encoding="utf-8")

    with pytest.raises(DataValidationError, match="disagree"):
        load_candidate_evidence(root, ledger=ExperimentLedger(root / "ledger"))


# ---------------------------------------------------------------------------
# The positive outcome
# ---------------------------------------------------------------------------


def test_a_champion_is_selected_on_validation_evidence(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """The whole point, once: gates clear and one run is chosen."""
    outcome = binary(evidence)
    assert outcome.status is ChampionStatus.ELIGIBLE
    assert outcome.selected_run_id is not None
    assert outcome.record.selected_model_content_fingerprint is not None


def test_every_candidate_is_recorded_whatever_the_verdict(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """A record of only the winner would be an assertion, not a comparison."""
    outcome = binary(evidence)
    judged = {item.catalog_model_id for item in outcome.results}
    published = {
        item.catalog_model_id
        for item in evidence
        if item.run.task is BINARY and item.catalog_model_id != "M-000"
    }
    assert judged == published
    assert all(item.gates for item in outcome.results)


def test_the_reference_baseline_is_never_judged_and_never_selected(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """It is the comparator, and a comparator that could win is not one."""
    outcome = binary(evidence)
    reference = next(
        item
        for item in evidence
        if item.catalog_model_id == "M-000" and item.run.task is BINARY
    )
    assert outcome.reference_run_id == reference.run_id
    assert reference.run_id not in [item.run_id for item in outcome.results]
    assert reference.run_id not in outcome.record.candidate_run_ids
    assert outcome.selected_run_id != reference.run_id


def test_the_caller_order_cannot_reach_the_answer(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Sorted internally, so a shuffled argument decides nothing."""
    forwards = binary(evidence)
    backwards = binary(tuple(reversed(evidence)))
    assert backwards.record.to_json() == forwards.record.to_json()


def test_the_selection_is_reproducible(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Same evidence, same criteria, same record identity."""
    first = binary(evidence)
    second = binary(evidence)
    assert second.record.record_id == first.record.record_id
    assert second.record.to_json() == first.record.to_json()


def test_the_ranking_follows_the_predeclared_objective(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Declared before the evidence was seen, and applied as declared."""
    outcome = binary(evidence)
    detection = [
        gate_of(outcome, run_id, "detection_rate_floor").observed
        for run_id in outcome.ranking
    ]
    assert detection == sorted(detection, reverse=True)
    assert outcome.ranking[0] == outcome.selected_run_id
    assert outcome.rationale[0] == "objective=max_detection_rate"
    assert outcome.rationale[1].startswith("tie_break=min_false_positive_rate")
    assert any(line.startswith("rank=1 model=") for line in outcome.rationale)


def test_only_eligible_candidates_are_ranked(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """A ranking over blocked candidates would imply a runner-up that could run."""
    outcome = binary(evidence)
    eligible = {
        item.run_id
        for item in outcome.results
        if item.status is ChampionStatus.ELIGIBLE
    }
    assert set(outcome.ranking) == eligible


# ---------------------------------------------------------------------------
# The mandatory reference comparison
# ---------------------------------------------------------------------------


def test_a_missing_reference_baseline_resolves_nothing(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Without the comparator every candidate's gain is unmeasurable, not zero."""
    outcome = binary(without(evidence, "M-000"))
    assert outcome.status is ChampionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert outcome.reference_run_id is None
    assert outcome.record.reference_run_id is None
    for result in outcome.results:
        gate = gate_of(outcome, result.run_id, "baseline_pr_auc_gain")
        assert gate.status is GateStatus.INCONCLUSIVE
        assert gate.reason == "reference_baseline_run_missing"
        assert "baseline_pr_auc_gain" in result.blocking_gates


def test_two_reference_baselines_make_the_comparator_ambiguous(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Picking whichever came first would be comparing against a coin toss."""
    reference = next(
        item
        for item in evidence
        if item.catalog_model_id == "M-000" and item.run.task is BINARY
    )
    outcome = binary((*evidence, reference))
    assert outcome.status is ChampionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert outcome.reference_run_id is None
    reasons = {
        gate_of(outcome, item.run_id, "baseline_pr_auc_gain").reason
        for item in outcome.results
    }
    assert reasons == {"reference_baseline_ambiguous"}


def test_the_comparison_is_decided_on_exact_ranking_evidence(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Every candidate and the comparator carry the exact curve, not the grid's.

    The operating-point curve may have been bounded by ``search_grid_size``;
    the discrimination evidence is built from every distinct score level, so the
    two are separate artifacts and only the exact one decides the gate.
    """
    outcome = binary(evidence)
    reference = next(
        item for item in evidence if item.run_id == outcome.reference_run_id
    )
    assert reference.ranking is not None
    assert reference.ranking.metric_name == RANKING_METRIC_NAME
    assert reference.ranking.integration == PR_AUC_INTEGRATION
    assert reference.ranking.score_kind is DISCRIMINATION_SCORE_KIND
    for result in outcome.results:
        candidate = next(item for item in evidence if item.run_id == result.run_id)
        assert candidate.ranking is not None
        assert candidate.ranking.score_kind is reference.ranking.score_kind
        gate = gate_of(outcome, result.run_id, "baseline_pr_auc_gain")
        assert gate.metric == f"{RANKING_METRIC_NAME}_gain"
        assert "sampled" not in (gate.reason or "")


def test_a_candidate_without_exact_evidence_cannot_pass_the_gate(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Withholding the exact curve blocks; it never falls back to the grid's."""
    stripped = tuple(
        dataclasses.replace(item, ranking=None)
        if item.catalog_model_id != "M-000"
        else item
        for item in evidence
    )
    outcome = binary(stripped)
    for result in outcome.results:
        gate = gate_of(outcome, result.run_id, "baseline_pr_auc_gain")
        assert gate.status is GateStatus.INCONCLUSIVE
        assert gate.reason == "ranking_evidence_unavailable"
    assert outcome.status is ChampionStatus.INSUFFICIENT_VALIDATION_SUPPORT


def test_a_comparator_without_exact_evidence_fails_closed(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """A malformed M-000 is not an excuse to skip the mandatory comparison."""
    blinded = tuple(
        dataclasses.replace(item, ranking=None)
        if item.catalog_model_id == "M-000"
        else item
        for item in evidence
    )
    outcome = binary(blinded)
    assert outcome.reference_run_id is None
    for result in outcome.results:
        gate = gate_of(outcome, result.run_id, "baseline_pr_auc_gain")
        assert gate.status is GateStatus.INCONCLUSIVE
        assert gate.reason == "reference_ranking_evidence_unavailable"
    assert outcome.status is ChampionStatus.INSUFFICIENT_VALIDATION_SUPPORT


def test_the_gain_over_the_baseline_is_measured_not_assumed(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Every candidate carries a number and the criterion it was judged by."""
    outcome = binary(evidence)
    for result in outcome.results:
        gate = gate_of(outcome, result.run_id, "baseline_pr_auc_gain")
        assert gate.status is not GateStatus.INCONCLUSIVE
        assert gate.observed is not None
        assert gate.required is not None


# ---------------------------------------------------------------------------
# The negative outcomes
# ---------------------------------------------------------------------------


def test_an_unreachable_criterion_is_a_measured_negative(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """No candidate is promoted, and nothing is relaxed to find one."""
    strict = rx.strict_gates(min_detection_rate=0.999)
    outcome = select_binary_champion(evidence, config=strict)
    assert outcome.status is ChampionStatus.NO_ELIGIBLE_CHAMPION
    assert outcome.ranking == ()
    assert outcome.selected_run_id is None
    assert outcome.record.selected_model_id is None
    for result in outcome.results:
        assert result.status is ChampionStatus.NOT_ELIGIBLE
        assert "detection_rate_floor" in result.blocking_gates


def test_an_empty_candidate_universe_is_a_measured_negative(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Nothing promotable was configured, which is a finding, not thin data."""
    reference_only = tuple(
        item for item in evidence if item.catalog_model_id == "M-000"
    )
    outcome = binary(reference_only)
    assert outcome.results == ()
    assert outcome.status is ChampionStatus.NO_ELIGIBLE_CHAMPION


def test_unresolved_dominates_failed(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """One candidate that might have qualified leaves the question open.

    A candidate whose operating point was never published is not a candidate
    that lost; the evidence to judge it does not exist. Reported alongside a
    candidate that *did* measurably fail, the selection is unresolved rather
    than decided.
    """
    withheld, *rest = [
        item
        for item in evidence
        if item.run.task is BINARY and item.catalog_model_id != "M-000"
    ]
    assert rest, "the precedence rule needs a second candidate to fail measurably"
    thin = dataclasses.replace(withheld, threshold=None, calibration_quality=None)
    outcome = select_binary_champion(
        (
            thin,
            *rest,
            *(item for item in evidence if item.run.task is not BINARY),
            *(
                item
                for item in evidence
                if item.catalog_model_id == "M-000" and item.run.task is BINARY
            ),
        ),
        config=rx.strict_gates(min_detection_rate=0.999),
    )

    unresolved = next(item for item in outcome.results if item.run_id == thin.run_id)
    measured = next(item for item in outcome.results if item.run_id == rest[0].run_id)
    assert all(
        gate.status is GateStatus.INCONCLUSIVE
        for gate in unresolved.gates
        if gate.blocking
    )
    assert any(gate.status is GateStatus.FAIL for gate in measured.gates)
    assert outcome.status is ChampionStatus.INSUFFICIENT_VALIDATION_SUPPORT


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_the_record_reseals_itself_on_the_way_back_in(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """A digest carried as a field is checked, not merely stored."""
    record = binary(evidence).record
    payload = json.loads(record.to_json())
    payload["status"] = str(ChampionStatus.NO_ELIGIBLE_CHAMPION)
    with pytest.raises(ModelTrainingError, match="not valid"):
        ValidationSelectionRecord.from_json(json.dumps(payload))


def test_the_record_names_its_whole_comparison(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Candidates, comparator, criteria, and the data they were measured on."""
    record = binary(evidence).record
    assert set(record.candidate_run_ids) == {
        item.run_id for item in binary(evidence).results
    }
    assert record.reference_run_id is not None
    assert record.gate_config_fingerprint
    assert record.ml_config_fingerprint
    assert record.model_catalog_fingerprint
    assert record.validation_partition_fingerprint
    assert record.readable_training_data_fingerprint
    assert record.readable_label_fingerprint
    assert record.readable_split_fingerprint


def test_the_record_carries_no_test_evidence(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Milestone 7 has not read the test split, and cannot claim to have."""
    record = binary(evidence).record
    text = record.to_json().lower()
    for forbidden in ("test_metrics", "test_evaluation", "holdout"):
        assert forbidden not in text
    assert not any(name.startswith("test_") for name in type(record).model_fields)


def test_a_selection_that_found_nothing_names_nothing(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """The selected fields exist exactly when a champion does."""
    record = select_binary_champion(
        evidence, config=rx.strict_gates(min_detection_rate=0.999)
    ).record
    assert record.selected_run_id is None
    assert record.selected_model_id is None
    assert record.selected_model_content_fingerprint is None
    assert record.ranking == ()


def test_different_criteria_are_a_different_selection(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Same candidates judged differently is not the same finding."""
    lenient = binary(evidence).record
    strict = select_binary_champion(
        evidence, config=rx.strict_gates(min_detection_rate=0.999)
    ).record
    assert strict.gate_config_fingerprint != lenient.gate_config_fingerprint
    assert strict.record_id != lenient.record_id


# ---------------------------------------------------------------------------
# The category head
# ---------------------------------------------------------------------------


def test_the_category_head_is_a_separate_question(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Separate evidence, separate record, and no reference baseline."""
    outcome = select_category_head(evidence, config=rx.config())
    assert outcome.task is CATEGORY
    assert outcome.reference_run_id is None
    assert outcome.record.reference_run_id is None
    assert outcome.status is ChampionStatus.ELIGIBLE
    assert outcome.record.record_id != binary(evidence).record.record_id


def test_the_category_head_declares_no_discrimination_metric(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """It has no reference comparison, so it names no metric for one."""
    record = select_category_head(evidence, config=rx.config()).record
    assert record.discrimination_metric is None
    assert record.discrimination_integration is None
    assert record.discrimination_score_kind is None
    assert record.reference_run_id is None


def test_no_binary_reference_requirement_leaks_into_the_category_task(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Withholding M-000 blocks the binary selection and not the category one."""
    without_baseline = without(evidence, "M-000")
    assert (
        binary(without_baseline).status
        is ChampionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    )

    category = select_category_head(without_baseline, config=rx.config())
    assert category.status is ChampionStatus.ELIGIBLE
    gate_ids = {gate.gate_id for gate in category.results[0].gates}
    assert "baseline_pr_auc_gain" not in gate_ids


def test_the_category_precision_gate_is_not_tautological(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """The floor in force decides it, not the floor the artifact was made under.

    The measured failure itself is pinned in the gate suite, on a head whose
    precision is below a floor; here the point is that the *criterion* moves
    with the configuration rather than with the artifact.
    """
    from password_attack_detector.ml.config import CategoryConfig

    base = rx.config()
    strict = rx.config(
        category=CategoryConfig(
            **{
                **base.category.model_dump(),
                "min_known_category_precision": 0.999,
            }
        )
    )
    outcome = select_category_head(evidence, config=strict)
    relaxed = select_category_head(evidence, config=base)
    precision = next(
        gate
        for gate in outcome.results[0].gates
        if gate.gate_id == "category_precision_floor"
    )
    before = next(
        gate
        for gate in relaxed.results[0].gates
        if gate.gate_id == "category_precision_floor"
    )
    # The artifact was selected under the loose floor and still records it. The
    # gate applies the one in force, so tightening the configuration tightens
    # the criterion -- which is the whole difference between a check and a
    # restatement of what the producer already decided.
    assert before.required == base.category.min_known_category_precision
    assert precision.required == 0.999
    assert precision.observed == before.observed


def test_the_category_head_is_judged_by_the_category_gates(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Coverage and per-class support, not a binary operating point."""
    outcome = select_category_head(evidence, config=rx.config())
    gate_ids = {gate.gate_id for gate in outcome.results[0].gates}
    assert "category_precision_floor" in gate_ids
    assert "false_positive_ceiling" not in gate_ids


def test_a_binary_champion_is_no_reason_to_invent_a_category_head(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Withholding the category runs leaves the category question unanswered."""
    binary_only = tuple(item for item in evidence if item.run.task is not CATEGORY)
    outcome = select_category_head(binary_only, config=rx.config())
    assert outcome.results == ()
    assert outcome.status is ChampionStatus.NO_ELIGIBLE_CHAMPION
    assert outcome.selected_run_id is None


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def test_publishing_a_selection_indexes_it_last(
    evidence: tuple[CandidateEvidence, ...], tmp_path: Path
) -> None:
    """Promote the directory, then index it; never the other way round."""
    ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = binary(evidence)
    published = publish_selection(outcome, root=tmp_path, ledger=ledger)

    directory = tmp_path / "selections" / published.record_id
    assert published.created
    assert (directory / SELECTION_FILE).is_file()
    assert (directory / "ml_gates.json").is_file()
    assert (directory / "ml_gates.md").is_file()
    assert [item.record_id for item in ledger.validation_selections()] == [
        published.record_id
    ]


def test_republishing_the_same_selection_changes_nothing(
    evidence: tuple[CandidateEvidence, ...], tmp_path: Path
) -> None:
    """Idempotent, so a retried command is not a second finding."""
    ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = binary(evidence)
    first = publish_selection(outcome, root=tmp_path, ledger=ledger)
    second = publish_selection(outcome, root=tmp_path, ledger=ledger)

    assert first.created and not second.created
    assert second.record_id == first.record_id
    assert len(ledger.validation_selections()) == 1


def test_a_negative_outcome_is_published_like_any_other(
    evidence: tuple[CandidateEvidence, ...], tmp_path: Path
) -> None:
    """A history of successes would be a history of successes, not of what happened."""
    ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = select_binary_champion(
        evidence, config=rx.strict_gates(min_detection_rate=0.999)
    )
    published = publish_selection(outcome, root=tmp_path, ledger=ledger)
    assert published.status is ChampionStatus.NO_ELIGIBLE_CHAMPION
    stored = ledger.read_selection(published.record_id)
    assert stored.status is ChampionStatus.NO_ELIGIBLE_CHAMPION


def test_an_incomplete_selection_is_never_completed_in_place(
    evidence: tuple[CandidateEvidence, ...], tmp_path: Path
) -> None:
    """A directory without its record is wreckage, not a destination."""
    ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = binary(evidence)
    published = publish_selection(outcome, root=tmp_path, ledger=ledger)
    (tmp_path / "selections" / published.record_id / SELECTION_FILE).unlink()

    with pytest.raises(ExperimentPublicationError, match="incomplete"):
        publish_selection(outcome, root=tmp_path, ledger=ledger)


def test_a_published_selection_is_never_overwritten(
    evidence: tuple[CandidateEvidence, ...], tmp_path: Path
) -> None:
    """Published evidence is evidence, whatever a later run believes."""
    ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = binary(evidence)
    published = publish_selection(outcome, root=tmp_path, ledger=ledger)
    receipt = tmp_path / "selections" / published.record_id / SELECTION_FILE
    receipt.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ExperimentPublicationError, match="not readable"):
        publish_selection(outcome, root=tmp_path, ledger=ledger)


def test_an_unindexed_selection_is_recovered_by_reading_it(
    evidence: tuple[CandidateEvidence, ...], tmp_path: Path
) -> None:
    """Reconcile reads the published record; it never rebuilds one."""
    published_ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = binary(evidence)
    published = publish_selection(outcome, root=tmp_path, ledger=published_ledger)

    fresh = ExperimentLedger(tmp_path / "fresh-ledger")
    assert reconcile_selections(root=tmp_path, ledger=fresh) == (published.record_id,)
    assert reconcile_selections(root=tmp_path, ledger=fresh) == ()
    assert fresh.read_selection(published.record_id).to_json() == (
        outcome.record.to_json()
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def test_the_report_states_what_it_does_not_know(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Validation-only, said in the artifact a reader actually opens."""
    markdown = selection_report_markdown(binary(evidence))
    assert "Validation-only" in markdown
    assert "no figure here describes performance on unseen data" in markdown


def test_the_report_carries_every_candidate_and_every_gate(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Including the ones that failed, and the reason each one gives."""
    outcome = binary(evidence)
    report = selection_report(outcome)
    assert len(report["candidates"]) == len(outcome.results)
    for entry, result in zip(report["candidates"], outcome.results, strict=True):
        assert len(entry["gates"]) == len(result.gates)
        assert all(gate["reason"] for gate in entry["gates"])


def test_the_report_carries_no_identity_and_no_path(
    evidence: tuple[CandidateEvidence, ...],
) -> None:
    """Reports travel further than the run directory they came from."""
    outcome = binary(evidence)
    text = json.dumps(selection_report(outcome)) + selection_report_markdown(outcome)
    assert "/home" not in text
    assert "e00000" not in text
    assert str(Path.home()) not in text


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


def test_selection_reads_no_rows_labels_or_splits() -> None:
    """Asserted on the module's syntax tree, not on its behaviour today."""
    import ast

    import password_attack_detector.ml.selection as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    assert "pyarrow" not in roots
    assert "pandas" not in roots
    assert "sklearn" not in roots
    assert "password_attack_detector.ml.dataset" not in imported
    assert "password_attack_detector.detection.evaluation" not in imported


def test_no_selector_argument_accepts_test_data() -> None:
    """There is no bypass to find, because there is no parameter to pass one to."""
    import inspect

    for function in (select_binary_champion, select_category_head):
        names = set(inspect.signature(function).parameters)
        assert not any(
            token in name
            for name in names
            for token in ("test", "holdout", "force", "allow")
        )
