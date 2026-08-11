"""Exact validation-B ranking evidence, and the PR-AUC it publishes.

Every number in this file is worked by hand. A discrimination metric checked
against a library's implementation of the same idea would agree with whatever
convention that library chose, which is exactly the thing that has to be pinned
here: the metric name, the integration rule, and the treatment of ties are
contract, not implementation detail.

The other half of the suite is about what must *not* move the number: the row
order, the identifiers the rows carry, the operating-threshold grid, and the
size of that grid. A ranking metric that shifted with any of them would be
measuring something other than the model's ordering.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.calibration import BinaryScoreSample
from password_attack_detector.ml.config import ThresholdConfig
from password_attack_detector.ml.enums import (
    MLSplit,
    ScoreKind,
    ValidationPartition,
)
from password_attack_detector.ml.ranking import (
    DISCRIMINATION_SCORE_KIND,
    PR_AUC_INTEGRATION,
    RANKING_METRIC_NAME,
    RankingEvidence,
    ScoreLevel,
    build_ranking_evidence,
    pr_auc,
)
from password_attack_detector.ml.thresholds import select_binary_threshold
from tests.ml import runs as rx
from tests.ml import selection as sx


def evidence(
    scores: Sequence[float], malicious: Sequence[bool], **extra: Any
) -> RankingEvidence:
    """Return exact ranking evidence over hand-written validation-B scores."""
    built = build_ranking_evidence(
        sample(scores, malicious, **extra),
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert built is not None
    return built


def sample(
    scores: Sequence[float], malicious: Sequence[bool], **extra: Any
) -> BinaryScoreSample:
    """Return a canonical validation-B score sample."""
    return sx.binary_sample(
        list(scores),
        list(malicious),
        partition=ValidationPartition.VALIDATION_B,
        **extra,
    )


def anchored(
    scores: Sequence[float], malicious: Sequence[bool], anchors: Any
) -> BinaryScoreSample:
    """Return a validation-B sample over explicit anchors."""
    return BinaryScoreSample(
        source=sx.source(ValidationPartition.VALIDATION_B),
        anchors=anchors,
        scores=tuple(scores),
        malicious=tuple(malicious),
        score_kind=ScoreKind.DECISION_SCORE,
        model_id="model-under-test",
        model_content_fingerprint=sx.MODEL_FINGERPRINT,
        preprocessor_fingerprint=sx.PREPROCESSOR_FINGERPRINT,
    )


def descending(
    scores: Sequence[float], malicious: Sequence[bool]
) -> tuple[list[float], list[bool]]:
    """Return the rows sorted by score, descending, keeping pairs together."""
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    return ([scores[index] for index in order], [malicious[index] for index in order])


# ---------------------------------------------------------------------------
# The metric is named, and the name means one thing
# ---------------------------------------------------------------------------


def test_the_metric_name_and_convention_are_pinned() -> None:
    """A refactor that swapped the arithmetic would have to change these.

    The gate configuration names ``min_pr_auc_gain_over_baseline``, so the
    metric is PR-AUC and the record says so. It is not published under one name
    and computed as another.
    """
    assert RANKING_METRIC_NAME == "pr_auc"
    assert PR_AUC_INTEGRATION == "stepwise"
    assert DISCRIMINATION_SCORE_KIND is ScoreKind.DECISION_SCORE

    record = evidence([0.9, 0.1], [True, False])
    assert record.metric_name == RANKING_METRIC_NAME
    assert record.integration == PR_AUC_INTEGRATION
    assert record.score_kind is DISCRIMINATION_SCORE_KIND
    assert json.loads(record.to_json())["metric_name"] == "pr_auc"


def test_the_integration_rule_is_the_declared_step_sum() -> None:
    """Worked by hand, level by level, with no interpolation anywhere.

    Four rows: scores 0.9 (malicious), 0.8 (benign), 0.7 (malicious),
    0.6 (benign).

    ==========  ====  ====  =========  ======  ======
    level       TP    FP    precision  recall  delta
    ==========  ====  ====  =========  ======  ======
    0.9         1     0     1          0.5     0.5
    0.8         1     1     1/2        0.5     0
    0.7         2     1     2/3        1.0     0.5
    0.6         2     2     1/2        1.0     0
    ==========  ====  ====  =========  ======  ======

    PR-AUC = 1 * 0.5 + 2/3 * 0.5 = 0.833333333.

    Trapezoidal integration would give 0.5 * (1 + 1/2) / 2 + ... -- a different
    number, which is why the convention is declared rather than assumed.
    """
    record = evidence([0.9, 0.8, 0.7, 0.6], [True, False, True, False])
    assert record.distinct_score_count == 4
    assert record.pr_auc == pytest.approx(0.5 + (2 / 3) * 0.5, abs=1e-9)
    assert [level.precision for level in record.levels] == [
        1.0,
        0.5,
        pytest.approx(2 / 3, abs=1e-9),
        0.5,
    ]
    assert [level.recall for level in record.levels] == [0.5, 0.5, 1.0, 1.0]


def test_perfect_ranking_is_one() -> None:
    """Every positive above every benign row: precision one at every recall."""
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    record = evidence(scores, [True, True, True, False, False])
    assert record.pr_auc == 1.0


def test_completely_reversed_ranking_is_the_hand_computed_floor() -> None:
    """Every benign row above every positive one, worked out level by level.

    Two positives below two benign rows. The positives are reached at levels
    three and four:

        precision = 1/3 at recall 0.5, precision = 2/4 at recall 1.0

    PR-AUC = (1/3) * 0.5 + (1/2) * 0.5 = 0.416666667, well under the 0.5
    prevalence a constant scorer would earn -- an ordering can be worse than no
    ordering at all.
    """
    record = evidence([0.9, 0.8, 0.2, 0.1], [False, False, True, True])
    assert record.pr_auc == pytest.approx((1 / 3) * 0.5 + 0.5 * 0.5, abs=1e-9)
    assert record.pr_auc < 0.5


def test_a_constant_score_reduces_to_the_positive_prevalence() -> None:
    """The M-000 case: one score group, and the metric is the class prior.

    Three positives and seven benign rows all scored alike. The single level
    carries TP=3, FP=7, precision 0.3, recall 1.0, so PR-AUC = 0.3 -- the
    positive prevalence, computed without any operating threshold and without
    inventing a second score level to make a curve out of.
    """
    record = evidence([0.5] * 10, [True] * 3 + [False] * 7)
    assert record.distinct_score_count == 1
    assert record.pr_auc == 0.3
    assert record.positive_count / record.row_count == 0.3
    assert record.levels[0].cumulative_true_positives == 3
    assert record.levels[0].cumulative_false_positives == 7


# ---------------------------------------------------------------------------
# Ties, order, and identity
# ---------------------------------------------------------------------------


def test_rows_sharing_a_score_are_one_level() -> None:
    """A tie is a group, not a sequence to be resolved by something else."""
    record = evidence([0.5, 0.5, 0.5, 0.1], [True, False, True, False])
    assert record.distinct_score_count == 2
    top = record.levels[0]
    assert (top.positive_count, top.negative_count) == (2, 1)
    assert (top.cumulative_true_positives, top.cumulative_false_positives) == (2, 1)


def test_reordering_tied_rows_changes_nothing() -> None:
    """Ordering inside a tie is ranking performance nobody measured."""
    scores = [0.5, 0.5, 0.5, 0.5, 0.1]
    positives_first = evidence(scores, [True, True, False, False, False])
    interleaved = evidence(scores, [True, False, True, False, False])
    benign_first = evidence(scores, [False, False, True, True, False])

    assert positives_first.pr_auc == interleaved.pr_auc == benign_first.pr_auc
    assert positives_first.levels == interleaved.levels == benign_first.levels


def test_shuffling_the_rows_produces_byte_identical_evidence() -> None:
    """The curve is grouped by score, so arrival order cannot reach it."""
    scores = [0.9, 0.7, 0.7, 0.4, 0.4, 0.1]
    malicious = [True, True, False, True, False, False]
    forwards = evidence(*descending(scores, malicious))
    backwards = evidence(*descending(scores[::-1], malicious[::-1]))
    assert backwards.to_json() == forwards.to_json()


def test_the_anchors_never_reach_the_evidence() -> None:
    """Same scores, different rows: identical evidence, and no identifier in it."""
    scores = [0.9, 0.5, 0.5, 0.1]
    malicious = [True, True, False, False]
    first = evidence(scores, malicious)
    later = build_ranking_evidence(
        anchored(scores, malicious, sx.anchors(len(scores), start=500)),
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert later is not None
    assert later.to_json() == first.to_json()
    assert "e00" not in first.to_json()
    assert "anchor" not in first.to_json()


def test_two_score_scales_with_one_ordering_measure_the_same() -> None:
    """Discrimination is the ordering, not the units it is expressed in."""
    malicious = [True, True, False, True, False, False]
    small = evidence([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], malicious)
    large = evidence([900.0, 800.0, 700.0, 600.0, 500.0, 400.0], malicious)
    negative = evidence([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], malicious)
    assert small.pr_auc == large.pr_auc == negative.pr_auc
    assert [level.precision for level in small.levels] == [
        level.precision for level in large.levels
    ]


# ---------------------------------------------------------------------------
# Independence from the operating-threshold search
# ---------------------------------------------------------------------------


def _threshold_grid(size: int, sample_: BinaryScoreSample) -> Any:
    """Return the operating-point selection under a grid of *size* candidates."""
    base = rx.config().thresholds
    return select_binary_threshold(
        sample_,
        config=ThresholdConfig(
            **{**base.model_dump(), "search_grid_size": size},
        ),
        support=rx.support(),
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )


def test_the_threshold_grid_does_not_reach_the_ranking_metric() -> None:
    """The two curves are built by different code from the same scores.

    The operating-point search is bounded to four candidates and then to forty;
    its curve changes and its truncation flag changes, and the exact ranking
    metric is the same number both times, because it never came from that curve.
    """
    scores = [round(0.99 - index / 100.0, 9) for index in range(40)]
    malicious = [index % 3 == 0 for index in range(40)]
    rows = sample(scores, malicious)

    narrow = _threshold_grid(4, rows)
    wide = _threshold_grid(40, rows)
    assert narrow.candidates_truncated is True
    assert wide.candidates_truncated is False
    assert len(narrow.curve) < len(wide.curve)

    exact = evidence(scores, malicious)
    rebuilt = build_ranking_evidence(rows, ml_config_fingerprint=sx.CONFIG_FINGERPRINT)
    assert rebuilt is not None
    assert exact.distinct_score_count == 40
    assert exact.pr_auc == rebuilt.pr_auc


def test_the_evidence_carries_no_grid_setting_to_move_with() -> None:
    """Asserted on the record's own field set, not on one computed pair."""
    fields = set(type(evidence([0.9, 0.1], [True, False])).model_fields)
    for absent in (
        "search_grid_size",
        "candidate_count",
        "candidates_truncated",
        "selected_threshold",
    ):
        assert absent not in fields, absent


# ---------------------------------------------------------------------------
# Absence, refusal, and the firewall
# ---------------------------------------------------------------------------


def test_one_class_alone_yields_no_evidence() -> None:
    """No positive row is no recall; no benign row is no precision to lose."""
    assert (
        build_ranking_evidence(
            sample([0.9, 0.8], [False, False]),
            ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
        )
        is None
    )
    assert (
        build_ranking_evidence(
            sample([0.9, 0.8], [True, True]),
            ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
        )
        is None
    )


def test_pr_auc_refuses_an_empty_recall_denominator() -> None:
    """An area under a curve with no positive row has no meaning to publish."""
    with pytest.raises(ModelTrainingError):
        pr_auc(
            (
                ScoreLevel(
                    score=0.5,
                    positive_count=0,
                    negative_count=1,
                    cumulative_true_positives=0,
                    cumulative_false_positives=1,
                    precision=0.0,
                    recall=0.0,
                ),
            ),
            positive_count=0,
        )


@pytest.mark.parametrize(
    ("split", "partition"),
    [
        (MLSplit.TEST, None),
        (MLSplit.NOVEL_ANOMALY_HOLDOUT, None),
        (MLSplit.VALIDATION, ValidationPartition.VALIDATION_A),
        (MLSplit.TRAIN, None),
    ],
)
def test_evidence_is_measured_on_validation_b_alone(
    split: MLSplit, partition: ValidationPartition | None
) -> None:
    """Every other split is refused at the door, including both firewalled ones."""
    rows = sx.binary_sample([0.9, 0.1], [True, False], split=split, partition=partition)
    with pytest.raises(ModelTrainingError):
        build_ranking_evidence(rows, ml_config_fingerprint=sx.CONFIG_FINGERPRINT)


def test_adding_test_rows_cannot_change_the_evidence() -> None:
    """The evidence is a function of the validation-B rows and nothing else.

    Rows are appended to a *test* sample and the validation-B evidence is
    rebuilt: byte-identical, because the test sample never reached it and there
    is no argument through which it could have.
    """
    scores = [0.9, 0.6, 0.6, 0.2]
    malicious = [True, True, False, False]
    before = evidence(scores, malicious)

    mutated_test_rows = sx.binary_sample(
        [*scores, 0.95, 0.05],
        [*malicious, True, False],
        split=MLSplit.TEST,
        partition=None,
    )
    with pytest.raises(ModelTrainingError):
        build_ranking_evidence(
            mutated_test_rows, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
        )
    assert evidence(scores, malicious).to_json() == before.to_json()


def test_uncanonical_rows_are_refused() -> None:
    """The order guard is re-asserted here, not inherited on trust."""
    rows = anchored([0.9, 0.1], [True, False], tuple(reversed(sx.anchors(2))))
    with pytest.raises(ModelTrainingError):
        build_ranking_evidence(rows, ml_config_fingerprint=sx.CONFIG_FINGERPRINT)


# ---------------------------------------------------------------------------
# The sealed record
# ---------------------------------------------------------------------------


def test_the_record_recomputes_its_metric_from_its_own_curve() -> None:
    """A published metric that disagreed with its levels would not load."""
    record = evidence([0.9, 0.5, 0.5, 0.1], [True, True, False, False])
    payload = json.loads(record.to_json())
    payload["pr_auc"] = 0.999999999
    with pytest.raises(ModelTrainingError, match="not valid"):
        RankingEvidence.from_json(json.dumps(payload))


def test_the_record_refuses_a_split_score_group() -> None:
    """One score, two levels, would be an ordering invented inside a tie."""
    record = evidence([0.5, 0.5, 0.1], [True, False, False])
    payload = json.loads(record.to_json())
    payload["levels"] = [payload["levels"][0], payload["levels"][0]]
    payload["distinct_score_count"] = 2
    with pytest.raises(ModelTrainingError, match="not valid"):
        RankingEvidence.from_json(json.dumps(payload))


def test_the_record_publishes_counts_and_no_rows() -> None:
    """A level is a score and four counts; there is nowhere for a row to sit."""
    record = evidence([0.9, 0.5, 0.5, 0.1], [True, True, False, False])
    published = json.loads(record.to_json())
    assert set(published["levels"][0]) == {
        "score",
        "positive_count",
        "negative_count",
        "cumulative_true_positives",
        "cumulative_false_positives",
        "precision",
        "recall",
    }
    assert record.row_count == 4
    assert record.positive_count == 2
    assert record.negative_count == 2
