"""Unit tests for the frozen drift reference profile.

Two properties are swept across the file. **The reference comes from training
rows and nowhere else** -- the eligible-split set has one member, the capture
refuses anything outside it, and the profile schema refuses to be constructed
naming an evaluation split. And **the partition is complete before any incoming
row exists** -- every feature carries a null cell whether or not the reference
was ever null, categorical vocabularies come from the frozen preprocessor rather
than from the data, and a degenerate reference gets a deliberately degenerate
partition instead of a grid that collapses to nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from password_attack_detector.exceptions import (
    ModelNotReadyError,
    ModelTrainingError,
)
from password_attack_detector.ml.config import DriftConfig
from password_attack_detector.ml.enums import (
    DRIFT_REFERENCE_ELIGIBLE_SPLITS,
    MLSplit,
    ReferenceFeatureKind,
)
from password_attack_detector.ml.reference import (
    NULL_BIN,
    UNKNOWN_BIN,
    FeatureReference,
    MLReferenceProfile,
    ReferenceBin,
    build_reference_profile,
    feature_reference_index,
    numeric_bin_index,
    quantile_edges,
    reference_profile_to_markdown,
)
from password_attack_detector.ml.schemas import PROHIBITED_METADATA_FIELDS
from tests.ml.models import prepare


class _Frame:
    """The narrow shape a capture reads: names, anchors, and a value matrix."""

    def __init__(self, names: tuple[str, ...], rows: tuple[tuple[Any, ...], ...]):
        self.feature_names = names
        self.feature_matrix = rows
        self.anchors = tuple(range(len(rows)))

    @property
    def row_count(self) -> int:
        """Return the number of reference rows."""
        return len(self.feature_matrix)


class _Reference:
    """A stand-in for the loaded inference dataset the capture consumes."""

    def __init__(self, frame: _Frame, scope: MLSplit = MLSplit.TRAIN):
        self.frame = frame
        self.scope = scope
        self.inference_input_fingerprint = "f" * 64


class _Lock:
    """The champion lock fields a profile binds, and nothing else."""

    def __init__(self, preprocessor_fingerprint: str):
        self.lock_fingerprint = "a" * 64
        self.scope_key = "b" * 64
        self.catalog_model_id = "M-010"
        self.model_id = "model-1"
        self.model_content_fingerprint = "c" * 64
        self.preprocessor_fingerprint = preprocessor_fingerprint
        self.feature_catalog_fingerprint = "d" * 64
        self.allowlist_fingerprint = "e" * 64
        self.eligible_feature_list_fingerprint = "0" * 64


@pytest.fixture
def captured() -> tuple[MLReferenceProfile, Any, _Frame]:
    """A profile captured from a real fitted preprocessor and its own frame."""
    prepared = prepare()
    frame = _Frame(
        prepared.preprocessor.raw_feature_names,
        tuple(tuple(row) for row in prepared.frame.feature_matrix),
    )
    profile = build_reference_profile(
        lock=_Lock(prepared.preprocessor.fingerprint()),
        preprocessor=prepared.preprocessor,
        reference=_Reference(frame),
        reference_split=MLSplit.TRAIN,
        required_feature_schema_version="1.0.0",
        drift_config=DriftConfig(quantile_count=4, min_reference_rows=1),
        drift_config_fingerprint="1" * 64,
    )
    return profile, prepared.preprocessor, frame


# ---------------------------------------------------------------------------
# The reference source
# ---------------------------------------------------------------------------


def test_only_training_rows_may_be_a_reference() -> None:
    """One member, and the absence of the others is the enforcement."""
    assert {MLSplit.TRAIN} == DRIFT_REFERENCE_ELIGIBLE_SPLITS
    for split in (
        MLSplit.VALIDATION,
        MLSplit.TEST,
        MLSplit.NOVEL_ANOMALY_HOLDOUT,
        MLSplit.EXCLUDED,
    ):
        assert split not in DRIFT_REFERENCE_ELIGIBLE_SPLITS


@pytest.mark.parametrize(
    "split", [MLSplit.TEST, MLSplit.NOVEL_ANOMALY_HOLDOUT, MLSplit.VALIDATION]
)
def test_capturing_from_an_ineligible_split_is_refused(split: MLSplit) -> None:
    """The command is not the only guard; the library refuses too."""
    prepared = prepare()
    frame = _Frame(
        prepared.preprocessor.raw_feature_names,
        tuple(tuple(row) for row in prepared.frame.feature_matrix),
    )
    with pytest.raises(ModelNotReadyError, match="reviewed source"):
        build_reference_profile(
            lock=_Lock(prepared.preprocessor.fingerprint()),
            preprocessor=prepared.preprocessor,
            reference=_Reference(frame, scope=split),
            reference_split=split,
            required_feature_schema_version="1.0.0",
            drift_config=DriftConfig(min_reference_rows=1),
            drift_config_fingerprint="1" * 64,
        )


def test_a_profile_naming_an_evaluation_split_is_refused_by_the_schema(
    captured: tuple[MLReferenceProfile, Any, _Frame],
) -> None:
    """Even a hand-built payload cannot name test as its baseline."""
    profile, _, _ = captured
    payload = profile.to_dict()
    payload["reference_split"] = str(MLSplit.TEST)
    with pytest.raises(ModelTrainingError, match="not valid"):
        MLReferenceProfile.from_dict(payload)


def test_a_preprocessor_the_champion_was_not_frozen_with_is_refused() -> None:
    """A profile cut with a different encoding describes a matrix nobody scores."""
    prepared = prepare()
    frame = _Frame(
        prepared.preprocessor.raw_feature_names,
        tuple(tuple(row) for row in prepared.frame.feature_matrix),
    )
    with pytest.raises(ModelNotReadyError, match="not the one the champion"):
        build_reference_profile(
            lock=_Lock("9" * 64),
            preprocessor=prepared.preprocessor,
            reference=_Reference(frame),
            reference_split=MLSplit.TRAIN,
            required_feature_schema_version="1.0.0",
            drift_config=DriftConfig(min_reference_rows=1),
            drift_config_fingerprint="1" * 64,
        )


def test_a_reference_below_the_configured_floor_is_refused() -> None:
    """A baseline cut from too few rows makes every later comparison look sure."""
    prepared = prepare()
    frame = _Frame(
        prepared.preprocessor.raw_feature_names,
        tuple(tuple(row) for row in prepared.frame.feature_matrix),
    )
    with pytest.raises(ModelNotReadyError, match="below the configured floor"):
        build_reference_profile(
            lock=_Lock(prepared.preprocessor.fingerprint()),
            preprocessor=prepared.preprocessor,
            reference=_Reference(frame),
            reference_split=MLSplit.TRAIN,
            required_feature_schema_version="1.0.0",
            drift_config=DriftConfig(min_reference_rows=100_000),
            drift_config_fingerprint="1" * 64,
        )


# ---------------------------------------------------------------------------
# Frozen partitions
# ---------------------------------------------------------------------------


def test_every_feature_carries_a_null_cell(
    captured: tuple[MLReferenceProfile, Any, _Frame],
) -> None:
    """Including one the reference was never null for.

    A cell that only appears once something lands in it is a cell the incoming
    data defined, which is exactly what a frozen partition must not permit.
    """
    profile, _, _ = captured
    for feature in profile.features:
        assert NULL_BIN in {item.cell for item in feature.bins}


def test_the_feature_kind_comes_from_the_preprocessor(
    captured: tuple[MLReferenceProfile, Any, _Frame],
) -> None:
    """A profile cannot decide a column is categorical when the model did not."""
    profile, preprocessor, _ = captured
    expected = {
        **{
            item.feature: ReferenceFeatureKind.NUMERIC
            for item in preprocessor.numeric_imputations
        },
        **{
            item.feature: ReferenceFeatureKind.BOOLEAN
            for item in preprocessor.boolean_encodings
        },
        **{
            item.feature: ReferenceFeatureKind.CATEGORICAL
            for item in preprocessor.categorical_encodings
        },
    }
    for feature in profile.features:
        assert feature.kind is expected[feature.feature]


def test_a_categorical_vocabulary_is_the_preprocessors_not_the_datas(
    captured: tuple[MLReferenceProfile, Any, _Frame],
) -> None:
    """A retained category the reference never carried is still a cell.

    The model has a column for it; a population that starts using it has
    drifted, and a partition that omitted the cell could not say so.
    """
    profile, preprocessor, _ = captured
    index = feature_reference_index(profile)
    for encoding in preprocessor.categorical_encodings:
        cells = {item.cell for item in index[encoding.feature].bins}
        assert set(encoding.categories) <= cells
        assert UNKNOWN_BIN in cells
        assert NULL_BIN in cells


def test_numeric_partitions_carry_interior_edges_only(
    captured: tuple[MLReferenceProfile, Any, _Frame],
) -> None:
    """The outermost cells are unbounded, so nothing out of range is dropped."""
    profile, _, _ = captured
    numeric = [
        feature
        for feature in profile.features
        if feature.kind is ReferenceFeatureKind.NUMERIC
        and feature.partition_kind == "quantile"
    ]
    assert numeric
    for feature in numeric:
        value_cells = [item for item in feature.bins if item.cell != NULL_BIN]
        assert value_cells[0].lower is None
        assert value_cells[-1].upper is None


def test_the_reference_proportions_sum_to_one(
    captured: tuple[MLReferenceProfile, Any, _Frame],
) -> None:
    """Every reference row lands somewhere; a partition that lost one is wrong."""
    profile, _, _ = captured
    for feature in profile.features:
        total = sum(item.reference_proportion for item in feature.bins)
        assert abs(total - 1.0) < 1e-6, feature.feature
        counts = sum(item.reference_count for item in feature.bins)
        assert counts == feature.reference_row_count, feature.feature


# ---------------------------------------------------------------------------
# Degenerate references
# ---------------------------------------------------------------------------


def test_quantile_edges_are_interior_and_deduplicated() -> None:
    """A heavily tied column produces repeated quantiles; a zero-width cell is not one."""
    assert quantile_edges([1.0] * 10, count=4) == (1.0,)
    assert quantile_edges([], count=4) == ()
    edges = quantile_edges([float(value) for value in range(100)], count=4)
    assert len(edges) == 3
    assert list(edges) == sorted(edges)
    assert edges[0] > 0.0 and edges[-1] < 99.0


def test_a_partition_needs_at_least_two_cells() -> None:
    """Nothing can move within a partition that has nowhere else to go."""
    with pytest.raises(ModelNotReadyError, match="at least two cells"):
        quantile_edges([1.0, 2.0], count=1)


def test_the_right_closed_convention_is_the_one_used_elsewhere() -> None:
    """A value equal to an edge belongs to the cell below it."""
    edges = (1.0, 2.0)
    assert numeric_bin_index(0.5, edges) == 0
    assert numeric_bin_index(1.0, edges) == 0
    assert numeric_bin_index(1.5, edges) == 1
    assert numeric_bin_index(2.0, edges) == 1
    assert numeric_bin_index(99.0, edges) == 2


def test_an_all_null_reference_still_gets_a_usable_partition() -> None:
    """A later population that starts observing values has drifted."""
    from password_attack_detector.ml.reference import _numeric_reference

    reference = _numeric_reference("f", [None] * 10, quantile_count=4)
    assert reference.partition_kind == "all_null"
    assert reference.reference_null_rate == 1.0
    cells = {item.cell for item in reference.bins}
    assert cells == {NULL_BIN, UNKNOWN_BIN}


def test_a_constant_reference_gets_a_degenerate_two_way_partition() -> None:
    """A quantile grid would collapse; this still detects the thing worth detecting."""
    from password_attack_detector.ml.reference import _numeric_reference

    reference = _numeric_reference("f", [3.0] * 10, quantile_count=4)
    assert reference.partition_kind == "constant"
    cells = {item.cell for item in reference.bins}
    assert NULL_BIN in cells
    assert UNKNOWN_BIN in cells
    assert any(cell.startswith("equals_") for cell in cells)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_the_profile_identity_is_stable_across_captures() -> None:
    """Two captures of the same population must agree byte for byte."""
    prepared = prepare()
    frame = _Frame(
        prepared.preprocessor.raw_feature_names,
        tuple(tuple(row) for row in prepared.frame.feature_matrix),
    )

    def _capture() -> MLReferenceProfile:
        return build_reference_profile(
            lock=_Lock(prepared.preprocessor.fingerprint()),
            preprocessor=prepared.preprocessor,
            reference=_Reference(frame),
            reference_split=MLSplit.TRAIN,
            required_feature_schema_version="1.0.0",
            drift_config=DriftConfig(quantile_count=4, min_reference_rows=1),
            drift_config_fingerprint="1" * 64,
        )

    assert _capture().to_json() == _capture().to_json()


def test_the_profile_identity_moves_with_the_cell_count() -> None:
    """The bin count decides what a comparison means, so it must bind."""
    prepared = prepare()
    frame = _Frame(
        prepared.preprocessor.raw_feature_names,
        tuple(tuple(row) for row in prepared.frame.feature_matrix),
    )

    def _capture(cells: int) -> MLReferenceProfile:
        return build_reference_profile(
            lock=_Lock(prepared.preprocessor.fingerprint()),
            preprocessor=prepared.preprocessor,
            reference=_Reference(frame),
            reference_split=MLSplit.TRAIN,
            required_feature_schema_version="1.0.0",
            drift_config=DriftConfig(quantile_count=cells, min_reference_rows=1),
            drift_config_fingerprint=f"{cells}" * 64,
        )

    assert _capture(4).reference_profile_id != _capture(8).reference_profile_id


def test_a_tampered_profile_is_refused_rather_than_repaired(
    captured: tuple[MLReferenceProfile, Any, _Frame],
) -> None:
    """The seal is a field, recomputed on every read."""
    profile, _, _ = captured
    payload = profile.to_dict()
    payload["reference_row_count"] = payload["reference_row_count"] + 1
    with pytest.raises(ModelTrainingError, match="not valid"):
        MLReferenceProfile.from_dict(payload)


def test_the_profile_is_frozen(
    captured: tuple[MLReferenceProfile, Any, _Frame],
) -> None:
    """There is no field on it a comparison could write back to."""
    profile, _, _ = captured
    with pytest.raises(ValueError):
        profile.reference_row_count = 1


# ---------------------------------------------------------------------------
# Privacy and rendering
# ---------------------------------------------------------------------------


def test_no_schema_here_declares_a_prohibited_field() -> None:
    """A partition is a description of a population, not of a row."""
    for model in (ReferenceBin, FeatureReference, MLReferenceProfile):
        assert not set(model.model_fields) & PROHIBITED_METADATA_FIELDS


def test_the_rendered_profile_omits_the_expected_masses(
    captured: tuple[MLReferenceProfile, Any, _Frame],
) -> None:
    """Identity and shape are review material; the distribution is not.

    The masses stay in the JSON artifact. Rendering them would put the reference
    distribution into a document that travels.
    """
    profile, _, _ = captured
    rendered = reference_profile_to_markdown(profile)
    assert "Partition shape" in rendered
    assert profile.reference_profile_fingerprint in rendered
    assert "reference_proportion" not in rendered
    assert "no label was read" in rendered.lower()


def test_a_bin_with_a_reversed_interval_is_refused() -> None:
    """An upper edge below its lower edge describes no cell at all."""
    with pytest.raises(ValueError, match="lower edge sits below"):
        ReferenceBin(
            cell="bin_00",
            lower=2.0,
            upper=1.0,
            reference_count=0,
            reference_proportion=0.0,
        )


def test_a_feature_partition_without_a_null_cell_is_refused() -> None:
    """A missing observation must have somewhere deterministic to land."""
    with pytest.raises(ValueError, match="no null bin"):
        FeatureReference(
            feature="f",
            kind=ReferenceFeatureKind.NUMERIC,
            partition_kind="quantile",
            bins=(
                ReferenceBin(cell="a", reference_count=1, reference_proportion=0.5),
                ReferenceBin(cell="b", reference_count=1, reference_proportion=0.5),
            ),
            reference_row_count=2,
            reference_null_count=0,
            reference_null_rate=0.0,
        )
