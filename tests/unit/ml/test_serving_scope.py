"""Tests for the live-serving scope and the batch that carries it.

The property under test is negative and structural: a request scored by the
deployed service is **not** filed as an experimental population.  That cannot be
demonstrated by a value being right, only by a type being separate and a field
being absent -- so most of what follows is about what does not exist.

The numeric half of the claim -- that a live batch scores identically to the
published-split path under the same frozen champion -- needs a real champion and
lives in ``tests/integration/test_serving_bundle.py``.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any

import pytest

from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.dataset import (
    ServingAnchor,
    ServingBatch,
    ServingFrame,
    assemble_inference_dataset,
    assemble_serving_batch,
)
from password_attack_detector.ml.enums import FIT_ELIGIBLE_SPLITS, MLSplit, ServingScope
from password_attack_detector.ml.features import resolve_eligible_features
from password_attack_detector.ml.preprocessing import FeatureFrame, TransformableFrame
from tests.ml import factories as fx

ALL_CLASSES = ("prior_only", "current_event_context", "baseline_derived")


@pytest.fixture()
def eligible() -> Any:
    """Return the resolved feature contract every batch here is built over."""
    catalog = fx.small_catalog()
    return resolve_eligible_features(
        catalog, fx.allowlist_for(catalog), include_leakage_classes=ALL_CLASSES
    )


def rows(eligible: Any) -> list[dict[str, Any]]:
    """Return three feature rows, deliberately not in canonical order."""
    names = eligible.feature_names
    return [
        fx.feature_row(3, names=names, minutes=3),
        fx.feature_row(1, names=names, minutes=1),
        fx.feature_row(2, names=names, minutes=2),
    ]


def splits(eligible: Any) -> list[Any]:
    """Return a TEST split assignment for each of :func:`rows`."""
    return [fx.split_row(index, str(MLSplit.TEST)) for index in (3, 1, 2)]


def batch(eligible: Any) -> ServingBatch:
    """Return a serving batch over :func:`rows`."""
    return assemble_serving_batch(feature_rows=rows(eligible), eligible=eligible)


# ---------------------------------------------------------------------------
# The scope is a separate type from a split
# ---------------------------------------------------------------------------


def test_the_serving_scope_has_exactly_one_member() -> None:
    """Serving rows are all in one scope, so a caller cannot select another."""
    assert list(ServingScope) == [ServingScope.LIVE]
    assert str(ServingScope.LIVE) == "live_serving"


def test_the_serving_scope_is_not_a_split() -> None:
    """Two types, deliberately, so one cannot be substituted for the other."""
    assert not issubclass(ServingScope, MLSplit)
    assert not isinstance(ServingScope.LIVE, MLSplit)
    assert {item.value for item in ServingScope}.isdisjoint(
        {item.value for item in MLSplit}
    )


def test_the_serving_scope_is_not_fit_eligible() -> None:
    """Nothing is ever fitted on live traffic, and it cannot claim to be.

    Compared by *value* rather than by membership: the two enums are disjoint
    types, so a membership test is one mypy correctly reports as statically
    impossible -- which is the property, stated where a type checker can see it.
    """
    assert ServingScope.LIVE.value not in {split.value for split in FIT_ELIGIBLE_SPLITS}


# ---------------------------------------------------------------------------
# The batch carries no split, and there is no way to give it one
# ---------------------------------------------------------------------------


def test_the_assembler_takes_neither_a_split_table_nor_a_scope() -> None:
    """The firewall stated as a signature: scoring this request needs neither."""
    parameters = set(inspect.signature(assemble_serving_batch).parameters)
    assert parameters == {"feature_rows", "eligible", "feature_catalog_fingerprint"}
    assert "splits" not in parameters
    assert "scope" not in parameters
    assert "labels" not in parameters


def test_the_serving_frame_declares_no_split() -> None:
    """A live frame has nowhere to record membership of a population."""
    assert "split" not in ServingFrame.__annotations__
    assert "split" not in ServingAnchor.__annotations__
    assert set(ServingFrame.__annotations__) == {
        "feature_names",
        "anchors",
        "feature_matrix",
    }


def test_a_serving_batch_declares_its_scope_as_a_serving_scope() -> None:
    """Typed, not stringly: the annotation is what the import guard reads."""
    assert ServingBatch.__annotations__["scope"] == "ServingScope"


def test_a_serving_frame_is_transformable_but_is_not_a_feature_frame(
    eligible: Any,
) -> None:
    """Applying a frozen state needs no split; fitting one does."""
    frame = batch(eligible).frame
    assert isinstance(frame, TransformableFrame)
    assert not isinstance(frame, FeatureFrame)


def test_a_split_scoped_frame_is_still_a_feature_frame(eligible: Any) -> None:
    """Splitting the protocol did not narrow what the fit path accepts."""
    dataset = assemble_inference_dataset(
        feature_rows=rows(eligible),
        splits=splits(eligible),
        eligible=eligible,
        scope=MLSplit.TEST,
    )
    assert isinstance(dataset.frame, FeatureFrame)
    assert isinstance(dataset.frame, TransformableFrame)


# ---------------------------------------------------------------------------
# The batch itself
# ---------------------------------------------------------------------------


def test_a_batch_names_the_live_scope(eligible: Any) -> None:
    """What the rows are, in the ML layer's own vocabulary."""
    built = batch(eligible)
    assert built.scope is ServingScope.LIVE
    assert built.row_count == 3


def test_a_batch_is_canonically_ordered(eligible: Any) -> None:
    """Request order does not reach a verdict; anchor time and identity do."""
    built = batch(eligible)
    times = [anchor.anchor_event_time for anchor in built.frame.anchors]
    assert times == sorted(times)


def test_a_batch_carries_the_reviewed_feature_contract(eligible: Any) -> None:
    """The same contract the frozen champion was fitted under, in its own order."""
    built = batch(eligible)
    assert built.feature_names == eligible.feature_names
    assert built.frame.feature_names == eligible.feature_names
    assert built.eligible_feature_list_fingerprint == eligible.fingerprint()
    assert built.allowlist_id == eligible.allowlist_id


def test_identical_feature_state_derives_an_identical_digest(eligible: Any) -> None:
    """What makes "the same window scores the same way" a checkable claim."""
    first = batch(eligible)
    second = assemble_serving_batch(
        feature_rows=list(reversed(rows(eligible))), eligible=eligible
    )
    assert first.serving_input_fingerprint == second.serving_input_fingerprint


def test_a_changed_cell_moves_the_digest(eligible: Any) -> None:
    """A digest that did not move when the input did would prove nothing."""
    first = batch(eligible)
    altered = rows(eligible)
    altered[0] = altered[0] | {eligible.feature_names[0]: 999.0}
    second = assemble_serving_batch(feature_rows=altered, eligible=eligible)
    assert first.serving_input_fingerprint != second.serving_input_fingerprint


# ---------------------------------------------------------------------------
# Nothing is relaxed because the caller is a request
# ---------------------------------------------------------------------------


def test_an_empty_batch_is_refused(eligible: Any) -> None:
    """There is nothing to score, and no row to answer about."""
    with pytest.raises(DataValidationError, match="No feature rows"):
        assemble_serving_batch(feature_rows=[], eligible=eligible)


def test_a_duplicate_anchor_is_refused(eligible: Any) -> None:
    """Two rows for one anchor is two answers to one question."""
    row = fx.feature_row(1, names=eligible.feature_names, minutes=1)
    with pytest.raises(DataValidationError):
        assemble_serving_batch(feature_rows=[row, dict(row)], eligible=eligible)


def test_a_missing_admitted_feature_is_refused(eligible: Any) -> None:
    """A frozen model is applied to its whole contract or to nothing."""
    dropped = eligible.feature_names[0]
    altered = rows(eligible)
    altered[0] = {key: value for key, value in altered[0].items() if key != dropped}
    with pytest.raises(DataValidationError, match="missing"):
        assemble_serving_batch(feature_rows=altered, eligible=eligible)


def test_a_prohibited_column_is_refused(eligible: Any) -> None:
    """A label-shaped column in a request is refused exactly as in a file."""
    altered = rows(eligible)
    altered[0] = altered[0] | {"malicious": True}
    with pytest.raises(DataValidationError):
        assemble_serving_batch(feature_rows=altered, eligible=eligible)


def test_a_naive_timestamp_is_refused(eligible: Any) -> None:
    """Point-in-time semantics need an instant, not a local reading of one."""
    altered = rows(eligible)
    altered[0] = altered[0] | {"anchor_event_time": datetime(2026, 3, 4, 12, 0)}
    with pytest.raises(DataValidationError):
        assemble_serving_batch(feature_rows=altered, eligible=eligible)


def test_a_non_finite_cell_is_refused(eligible: Any) -> None:
    """A matrix a frozen preprocessor cannot encode is refused before it tries."""
    altered = rows(eligible)
    altered[0] = altered[0] | {eligible.feature_names[0]: float("nan")}
    with pytest.raises(DataValidationError):
        assemble_serving_batch(feature_rows=altered, eligible=eligible)


def test_a_batch_and_a_dataset_over_one_population_agree_on_the_matrix(
    eligible: Any,
) -> None:
    """Same rows, same contract, same cells in the same canonical order.

    The structural half of "serving still produces identical model scores": the
    two assemblers hand a frozen preprocessor exactly the same matrix.  The
    numeric half is asserted against a real champion in the integration suite.
    """
    built = batch(eligible)
    dataset = assemble_inference_dataset(
        feature_rows=rows(eligible),
        splits=splits(eligible),
        eligible=eligible,
        scope=MLSplit.TEST,
    )
    assert built.frame.feature_matrix == dataset.frame.feature_matrix
    assert [anchor.anchor_event_id for anchor in built.frame.anchors] == [
        anchor.anchor_event_id for anchor in dataset.frame.anchors
    ]
    assert built.eligible_feature_list_fingerprint == (
        dataset.eligible_feature_list_fingerprint
    )
