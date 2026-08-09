"""The reviewed feature allowlist: opt-in eligibility and its fingerprint.

The property under test throughout is that **a feature reaches a model because
somebody wrote it down**, and that the written-down version and the executable
catalog are checked against each other rather than one being trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from password_attack_detector.exceptions import ConfigurationError, MLConfigurationError
from password_attack_detector.features.catalog import (
    PROHIBITED_FEATURE_COLUMNS,
    build_catalog,
)
from password_attack_detector.features.config import FeatureConfig
from password_attack_detector.ml.features import (
    ALLOWLIST_SCHEMA_VERSION,
    ML_OUTPUT_COLUMNS,
    ML_PERMITTED_INTENDED_USE,
    FeatureAdmission,
    emit_allowlist_document,
    load_feature_allowlist,
    resolve_eligible_features,
    unreviewed_catalog_features,
)
from tests.ml import factories as fx

ALL_CLASSES = ("prior_only", "current_event_context", "baseline_derived")

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAMPION_ALLOWLIST = REPO_ROOT / "configs" / "ml" / "features-allowlist-v1.yaml"
ABLATION_ALLOWLIST = (
    REPO_ROOT / "configs" / "ml" / "features-allowlist-prior-only-v1.yaml"
)


def _resolve(catalog: Any, allowlist: Any, **kwargs: Any) -> Any:
    """Resolve with every eligible leakage class unless a test narrows it."""
    kwargs.setdefault("include_leakage_classes", ALL_CLASSES)
    return resolve_eligible_features(catalog, allowlist, **kwargs)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_the_allowlist_fingerprint_is_deterministic() -> None:
    """Two identical allowlists digest identically."""
    catalog = fx.small_catalog()
    first = fx.allowlist_for(catalog)
    second = fx.allowlist_for(catalog)
    assert first.fingerprint() == second.fingerprint()


def test_entry_order_does_not_change_the_fingerprint() -> None:
    """The contract is which features are admitted, not the order they were typed."""
    catalog = fx.small_catalog()
    entries = [fx.admission_for(item) for item in catalog.specs]
    forward = fx.allowlist(entries)
    backward = fx.allowlist(list(reversed(entries)))
    assert forward.fingerprint() == backward.fingerprint()


def test_the_fingerprint_ignores_paths_and_prose(tmp_path: Path) -> None:
    """The same reviewed allowlist in two directories digests the same.

    Also covers the rationale: correcting the wording of a review note must not
    invalidate every model that recorded the digest, exactly as the Phase 3
    catalog excludes ``description`` from its own fingerprint.
    """
    catalog = fx.small_catalog()
    entries = [fx.admission_for(item) for item in catalog.specs]
    document = {
        "allowlist_schema_version": ALLOWLIST_SCHEMA_VERSION,
        "allowlist_id": "test_champion",
        "allowlist_version": "1.0.0",
        "required_feature_schema_version": "1.0.0",
        "compatible_feature_catalog_fingerprints": [catalog.fingerprint()],
        "entries": entries,
    }

    first = tmp_path / "a" / "allowlist.yaml"
    second = tmp_path / "b" / "different-name.yaml"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(document), encoding="utf-8")

    reworded = dict(document)
    reworded["entries"] = [
        {**entry, "rationale": "A completely different justification was written."}
        for entry in entries
    ]
    third = tmp_path / "c" / "allowlist.yaml"
    third.parent.mkdir(parents=True)
    third.write_text(yaml.safe_dump(reworded), encoding="utf-8")

    digests = {
        load_feature_allowlist(path).fingerprint() for path in (first, second, third)
    }
    assert len(digests) == 1


def test_the_allowlist_identity_is_part_of_the_fingerprint() -> None:
    """Two allowlists admitting identical features are still distinguishable."""
    catalog = fx.small_catalog()
    entries = [fx.admission_for(item) for item in catalog.specs]
    champion = fx.allowlist(entries, allowlist_id="champion")
    other = fx.allowlist(entries, allowlist_id="experiment")
    assert champion.fingerprint() != other.fingerprint()


def test_a_version_bump_changes_the_fingerprint() -> None:
    """Re-reviewing an unchanged feature set is still a new contract."""
    catalog = fx.small_catalog()
    entries = [fx.admission_for(item) for item in catalog.specs]
    assert (
        fx.allowlist(entries, allowlist_version="1.0.0").fingerprint()
        != fx.allowlist(entries, allowlist_version="1.1.0").fingerprint()
    )


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_a_duplicate_entry_is_rejected() -> None:
    """A feature admitted twice is a review that was not read."""
    entry = fx.admission("user_failure_rate")
    with pytest.raises(ValueError, match="more than once"):
        fx.allowlist([entry, dict(entry)])


def test_an_empty_allowlist_is_rejected() -> None:
    """An allowlist admitting nothing is a configuration mistake, not a contract."""
    with pytest.raises(ValueError, match="at least one feature"):
        fx.allowlist([])


def test_an_unknown_feature_name_fails_resolution() -> None:
    """An admitted feature the catalog does not declare is a disagreement."""
    catalog = fx.small_catalog()
    allowlist = fx.allowlist(
        [fx.admission_for(item) for item in catalog.specs]
        + [fx.admission("a_feature_that_was_removed")],
        catalog_fingerprints=(catalog.fingerprint(),),
    )
    with pytest.raises(
        MLConfigurationError, match="not declared by the feature catalog"
    ):
        _resolve(catalog, allowlist)


def test_a_catalog_removal_fails_rather_than_shrinking_the_matrix() -> None:
    """Losing a feature must be loud; a quietly narrower matrix is a new model."""
    full = fx.small_catalog()
    allowlist = fx.allowlist_for(full)
    reduced = fx.catalog(
        [item for item in full.specs if item.name != "user_failure_rate"]
    )
    with pytest.raises(MLConfigurationError, match="user_failure_rate"):
        _resolve(reduced, allowlist)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("leakage_class", "current_event_context", "leakage class"),
        ("feature_group", "source_history", "feature group"),
        ("decision_point", "requires_fitted_baseline", "decision point"),
    ],
)
def test_metadata_disagreement_fails(field: str, value: str, expected: str) -> None:
    """A review that describes a different system is refused, not reconciled."""
    catalog = fx.small_catalog()
    entries = [fx.admission_for(item) for item in catalog.specs]
    entries[0][field] = value
    allowlist = fx.allowlist(entries, catalog_fingerprints=(catalog.fingerprint(),))
    with pytest.raises(MLConfigurationError, match=expected):
        _resolve(catalog, allowlist)


def test_a_deprecated_feature_fails_resolution() -> None:
    """Deprecation in the catalog withdraws a feature from every model."""
    catalog = fx.catalog(
        [fx.spec("user_failure_rate", deprecated=True), fx.spec("source_failure_rate")]
    )
    allowlist = fx.allowlist_for(catalog)
    with pytest.raises(MLConfigurationError, match="deprecated"):
        _resolve(catalog, allowlist)


def test_a_feature_whose_intended_use_forbids_modelling_fails() -> None:
    """The catalog's own statement of purpose is a necessary screen."""
    catalog = fx.catalog(
        [
            fx.spec("user_failure_rate", intended_use="join key; never a model input"),
            fx.spec("source_failure_rate"),
        ]
    )
    allowlist = fx.allowlist_for(catalog)
    with pytest.raises(MLConfigurationError, match="intended_use"):
        _resolve(catalog, allowlist)


def test_a_key_class_feature_cannot_be_admitted() -> None:
    """The key leakage class is refused at the admission, before any resolution."""
    with pytest.raises(ValueError, match="never a model input"):
        FeatureAdmission(**fx.admission("some_join_column", leakage_class="key"))


def test_the_key_feature_group_cannot_be_admitted() -> None:
    """The group is checked independently of the leakage class."""
    with pytest.raises(ValueError, match="never an input"):
        FeatureAdmission(**fx.admission("some_join_column", feature_group="key"))


@pytest.mark.parametrize("name", ["anchor_event_id", "anchor_event_time", "event_id"])
def test_an_identifier_column_cannot_be_admitted(name: str) -> None:
    """Join and provenance columns are refused by name, whatever else they claim."""
    with pytest.raises(ValueError, match="join, label, or split column"):
        FeatureAdmission(**fx.admission(name))


@pytest.mark.parametrize("name", sorted(ML_OUTPUT_COLUMNS))
def test_a_model_output_name_cannot_be_admitted(name: str) -> None:
    """Every Phase 5 output name is refused as a feature input."""
    with pytest.raises(ValueError):
        FeatureAdmission(**fx.admission(name))


@pytest.mark.parametrize(
    "name",
    sorted(n for n in PROHIBITED_FEATURE_COLUMNS if n.islower() and n.isidentifier()),
)
def test_a_prohibited_column_cannot_be_admitted(name: str) -> None:
    """Ground truth, splits, and campaign metadata are refused by name."""
    with pytest.raises(ValueError):
        FeatureAdmission(**fx.admission(name))


def test_an_admission_needs_a_rationale() -> None:
    """An entry nobody could justify in one line is an entry nobody reviewed."""
    with pytest.raises(ValueError):
        FeatureAdmission(**fx.admission("user_failure_rate", rationale="x"))


def test_pending_review_and_entries_must_be_disjoint() -> None:
    """A feature is admitted or deferred, never recorded as both."""
    with pytest.raises(ValueError, match="both entries and pending_review"):
        fx.allowlist(
            [fx.admission("user_failure_rate")], pending_review=["user_failure_rate"]
        )


def test_an_allowlist_must_pin_a_catalog() -> None:
    """An allowlist that pins no catalog records no review."""
    with pytest.raises(ValueError, match="at least one reviewed catalog"):
        fx.allowlist([fx.admission("user_failure_rate")], catalog_fingerprints=())


def test_a_schema_version_mismatch_fails_resolution() -> None:
    """The allowlist and the catalog must consume the same feature contract."""
    catalog = fx.small_catalog()
    allowlist = fx.allowlist_for(catalog)
    with pytest.raises(MLConfigurationError, match="requires feature schema"):
        _resolve(catalog, allowlist, feature_schema_version="2.0.0")


# ---------------------------------------------------------------------------
# Opt-in behaviour
# ---------------------------------------------------------------------------


def test_a_catalog_addition_is_not_automatically_admitted() -> None:
    """The central property: a new catalog feature is inert until reviewed."""
    original = fx.small_catalog()
    allowlist = fx.allowlist_for(original)
    extended = fx.catalog([*original.specs, fx.spec("user_new_signal_rate")])

    unreviewed = unreviewed_catalog_features(extended, allowlist)
    assert unreviewed == ("user_new_signal_rate",)

    resolved = _resolve(extended, allowlist)
    assert "user_new_signal_rate" not in resolved.feature_names


def test_deferring_a_feature_clears_the_review_queue() -> None:
    """``pending_review`` is a decision; omission is an oversight."""
    original = fx.small_catalog()
    extended = fx.catalog([*original.specs, fx.spec("user_new_signal_rate")])
    allowlist = fx.allowlist(
        [fx.admission_for(item) for item in original.specs],
        catalog_fingerprints=(extended.fingerprint(),),
        pending_review=["user_new_signal_rate"],
    )
    assert unreviewed_catalog_features(extended, allowlist) == ()


def test_narrowing_a_run_cannot_hide_an_unreviewed_feature() -> None:
    """The review queue is configuration-independent, on purpose.

    If narrowing to prior-only features emptied the queue, an unreviewed
    baseline feature could be hidden by choosing a narrower run -- exactly the
    escape hatch this check exists to close.
    """
    original = fx.small_catalog()
    extended = fx.catalog(
        [
            *original.specs,
            fx.spec(
                "user_new_baseline_deviation",
                group="baseline",
                leakage_class="baseline_derived",
                requires_baseline=True,
            ),
        ]
    )
    allowlist = fx.allowlist_for(original)
    _resolve(extended, allowlist, include_leakage_classes=("prior_only",))
    assert "user_new_baseline_deviation" in unreviewed_catalog_features(
        extended, allowlist
    )


def test_a_scoped_allowlist_is_not_responsible_for_classes_it_excludes() -> None:
    """A declared scope distinguishes a deliberate omission from an oversight."""
    catalog = fx.small_catalog()
    prior_only = [
        fx.admission_for(item)
        for item in catalog.specs
        if str(item.leakage_class) == "prior_only"
    ]
    scoped = fx.allowlist(
        prior_only,
        allowlist_id="prior_only_ablation",
        catalog_fingerprints=(catalog.fingerprint(),),
        governed_leakage_classes=["prior_only"],
    )
    assert unreviewed_catalog_features(catalog, scoped) == ()

    unscoped = fx.allowlist(prior_only, catalog_fingerprints=(catalog.fingerprint(),))
    assert unreviewed_catalog_features(catalog, unscoped)


def test_an_admission_outside_the_declared_scope_is_rejected() -> None:
    """A scope that its own entries contradict is not a scope."""
    with pytest.raises(ValueError, match="outside the declared"):
        fx.allowlist(
            [
                fx.admission(
                    "login_hour_deviation",
                    leakage_class="baseline_derived",
                    feature_group="baseline",
                    decision_point="requires_fitted_baseline",
                )
            ],
            governed_leakage_classes=["prior_only"],
        )


# ---------------------------------------------------------------------------
# Resolution output
# ---------------------------------------------------------------------------


def test_feature_order_follows_the_catalog_not_the_file() -> None:
    """The catalog is the authority on column order for every artifact."""
    catalog = fx.small_catalog()
    entries = list(reversed([fx.admission_for(item) for item in catalog.specs]))
    resolved = _resolve(
        catalog, fx.allowlist(entries, catalog_fingerprints=(catalog.fingerprint(),))
    )
    assert resolved.feature_names == catalog.column_order()


def test_configuration_narrowing_is_recorded_not_silent() -> None:
    """A shrunken matrix must be explicable from the resolution itself."""
    catalog = fx.small_catalog()
    resolved = _resolve(
        catalog, fx.allowlist_for(catalog), include_leakage_classes=("prior_only",)
    )
    assert resolved.feature_names == ("user_failure_rate", "source_failure_rate")
    assert resolved.excluded_by_configuration == (
        "current_authentication_outcome",
        "login_hour_deviation",
    )


def test_narrowing_changes_the_eligible_feature_fingerprint() -> None:
    """A narrowed run is a different model identity, not the same one smaller."""
    catalog = fx.small_catalog()
    allowlist = fx.allowlist_for(catalog)
    full = _resolve(catalog, allowlist)
    narrowed = _resolve(catalog, allowlist, include_leakage_classes=("prior_only",))
    assert full.fingerprint() != narrowed.fingerprint()


def test_a_group_allowlist_narrows_the_matrix() -> None:
    """Feature groups are a configured allowlist too."""
    catalog = fx.small_catalog()
    resolved = _resolve(
        catalog,
        fx.allowlist_for(catalog),
        include_feature_groups=("user_history",),
    )
    assert resolved.feature_names == ("user_failure_rate",)


def test_a_configuration_that_excludes_everything_fails_loudly() -> None:
    """An empty matrix is a configuration error, not a zero-column model."""
    catalog = fx.catalog([fx.spec("user_failure_rate")])
    with pytest.raises(MLConfigurationError, match="No admitted feature survived"):
        _resolve(
            catalog,
            fx.allowlist_for(catalog),
            include_leakage_classes=("baseline_derived",),
        )


def test_the_resolved_list_reports_its_length() -> None:
    """``len`` on the eligible list is the feature count."""
    catalog = fx.small_catalog()
    assert len(_resolve(catalog, fx.allowlist_for(catalog))) == 4


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_loading_a_missing_file_reports_no_path(tmp_path: Path) -> None:
    """The failure names the problem, never the absolute path it looked in."""
    missing = tmp_path / "personal" / "allowlist.yaml"
    with pytest.raises(ConfigurationError) as caught:
        load_feature_allowlist(missing)
    assert str(missing) not in str(caught.value)


def test_loading_invalid_yaml_fails(tmp_path: Path) -> None:
    """A malformed file fails loudly rather than resolving to an empty contract."""
    path = tmp_path / "allowlist.yaml"
    path.write_text("entries: [oops\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid YAML"):
        load_feature_allowlist(path)


def test_loading_a_non_mapping_fails(tmp_path: Path) -> None:
    """A YAML list is not an allowlist."""
    path = tmp_path / "allowlist.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be a YAML mapping"):
        load_feature_allowlist(path)


def test_loading_a_yaml_file_cannot_construct_a_python_object(tmp_path: Path) -> None:
    """``safe_load`` constructs scalars, lists, and mappings, and nothing else."""
    path = tmp_path / "allowlist.yaml"
    path.write_text(
        "!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8"
    )
    with pytest.raises(ConfigurationError):
        load_feature_allowlist(path)


# ---------------------------------------------------------------------------
# The committed allowlists
# ---------------------------------------------------------------------------


def test_the_committed_champion_allowlist_resolves_against_the_real_catalog() -> None:
    """The shipped contract and the executable catalog agree, feature by feature."""
    catalog = build_catalog(FeatureConfig())
    allowlist = load_feature_allowlist(CHAMPION_ALLOWLIST)
    resolved = _resolve(catalog, allowlist)

    assert allowlist.allowlist_id == "champion"
    assert len(allowlist.entries) == 198
    assert len(resolved) == 198
    assert unreviewed_catalog_features(catalog, allowlist) == ()


def test_the_committed_allowlist_admits_no_key_column() -> None:
    """The three key columns are absent by construction, not by omission."""
    catalog = build_catalog(FeatureConfig())
    allowlist = load_feature_allowlist(CHAMPION_ALLOWLIST)
    keys = {item.name for item in catalog.specs if str(item.leakage_class) == "key"}
    assert keys
    assert not keys & allowlist.names


def test_the_committed_allowlist_pins_the_shipped_catalogs() -> None:
    """Every feature configuration this repository ships is a reviewed catalog."""
    catalog = build_catalog(FeatureConfig())
    allowlist = load_feature_allowlist(CHAMPION_ALLOWLIST)
    assert catalog.fingerprint() in allowlist.compatible_feature_catalog_fingerprints


def test_every_committed_entry_carries_a_real_rationale() -> None:
    """The shipped file is reviewed, so no entry may keep the drafted placeholder."""
    allowlist = load_feature_allowlist(CHAMPION_ALLOWLIST)
    for entry in allowlist.entries:
        assert "Review before admission" not in entry.rationale, entry.name
        assert len(entry.rationale) > 40, entry.name


def test_the_ablation_is_a_separate_contract_from_the_champion() -> None:
    """The ablation cannot silently become the champion's feature contract."""
    catalog = build_catalog(FeatureConfig())
    champion = load_feature_allowlist(CHAMPION_ALLOWLIST)
    ablation = load_feature_allowlist(ABLATION_ALLOWLIST)

    assert ablation.allowlist_id == "prior_only_ablation"
    assert ablation.fingerprint() != champion.fingerprint()

    champion_features = _resolve(catalog, champion)
    ablation_features = _resolve(
        catalog, ablation, include_leakage_classes=("prior_only",)
    )
    assert ablation_features.fingerprint() != champion_features.fingerprint()
    assert set(ablation_features.feature_names) < set(champion_features.feature_names)


def test_the_ablation_admits_only_prior_only_features() -> None:
    """A prior-only experiment that admitted a baseline feature would not be one."""
    ablation = load_feature_allowlist(ABLATION_ALLOWLIST)
    assert ablation.governed_leakage_classes == ("prior_only",)
    assert {str(entry.leakage_class) for entry in ablation.entries} == {"prior_only"}
    assert len(ablation.entries) == 169


def test_the_ablation_has_no_outstanding_review_queue() -> None:
    """Within its declared scope, the ablation is complete."""
    catalog = build_catalog(FeatureConfig())
    ablation = load_feature_allowlist(ABLATION_ALLOWLIST)
    assert unreviewed_catalog_features(catalog, ablation) == ()


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------


def test_a_drafted_allowlist_loads_and_says_it_is_a_draft(tmp_path: Path) -> None:
    """The generator produces a valid file that announces its own status."""
    catalog = fx.small_catalog()
    document = emit_allowlist_document(
        catalog,
        compatible_feature_catalog_fingerprints=[catalog.fingerprint()],
        admitted_in="draft",
    )
    path = tmp_path / "draft.yaml"
    path.write_text(document, encoding="utf-8")

    loaded = load_feature_allowlist(path)
    assert len(loaded.entries) == 4
    assert "Review before admission" in document
    assert "reviewed and committed by a person" in document


def test_drafting_is_deterministic() -> None:
    """Two drafts of the same catalog are byte-identical."""
    catalog = fx.small_catalog()
    kwargs: dict[str, Any] = {
        "compatible_feature_catalog_fingerprints": [catalog.fingerprint()],
        "admitted_in": "draft",
    }
    assert emit_allowlist_document(catalog, **kwargs) == emit_allowlist_document(
        catalog, **kwargs
    )


def test_the_permitted_intended_use_set_is_narrow() -> None:
    """A screen that admitted several phrasings would admit a typo too."""
    assert len(ML_PERMITTED_INTENDED_USE) == 1


# ---------------------------------------------------------------------------
# Prohibited *inputs* are not prohibited *names*
#
# PROHIBITED_FEATURE_COLUMNS governs one direction of travel: what may be
# accepted into a design matrix. The same names are the intended field names of
# Phase 5 prediction, evaluation, and model artifacts. Both directions are
# asserted here so neither can be weakened by appeal to the other.
# ---------------------------------------------------------------------------

#: The Phase 5 output vocabulary, as it appears in both constants.
PHASE_5_OUTPUT_NAMES = (
    "malicious_probability",
    "malicious_decision_score",
    "flagged_malicious",
    "predicted_scenario",
    "category_scores_json",
    "anomaly_score",
    "fused_flagged",
    "decision_threshold",
    "min_category_score",
)


def test_every_phase_five_output_name_is_a_prohibited_input() -> None:
    """Direction one: none of them may be accepted as a feature."""
    for name in PHASE_5_OUTPUT_NAMES:
        assert name in PROHIBITED_FEATURE_COLUMNS, name
        assert name in ML_OUTPUT_COLUMNS, name


@pytest.mark.parametrize("name", PHASE_5_OUTPUT_NAMES)
def test_a_feature_snapshot_carrying_an_output_name_is_rejected(
    tmp_path: Path, name: str
) -> None:
    """Direction one, at the Phase 3 validator: code ``F006`` on a real table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from password_attack_detector.features.validation import FeatureValidator

    catalog = fx.small_catalog()
    columns: dict[str, list[Any]] = {
        feature: [0.0] for feature in catalog.column_order()
    }
    columns[name] = [0.5]
    path = tmp_path / "feature_snapshots.parquet"
    pq.write_table(pa.table(columns), path)

    result = FeatureValidator(catalog).validate_parquet(path)
    assert not result.passed
    # Rejected as an undeclared column (``F003``) before the dedicated
    # prohibited-column code can fire: a name the catalog never declared cannot
    # reach the later check. Asserting rejection rather than a particular code
    # keeps this a test of the boundary rather than of the order two checks run
    # in.
    assert {error.code for error in result.errors} <= {"F003", "F004", "F006"}
    assert name in " ".join(
        f"{error.code} {error.message} {error.column}" for error in result.errors
    )


@pytest.mark.parametrize("name", PHASE_5_OUTPUT_NAMES)
def test_a_feature_row_carrying_an_output_name_is_rejected(name: str) -> None:
    """Direction one, at the ML reader: a snapshot carrying one is refused."""
    from password_attack_detector.exceptions import DataValidationError
    from password_attack_detector.ml.dataset import assemble_ml_dataset

    catalog = fx.small_catalog()
    eligible = _resolve(catalog, fx.allowlist_for(catalog))
    row = fx.feature_row(0, names=eligible.feature_names)
    row[name] = 0.5

    with pytest.raises(DataValidationError, match="must never reach a model"):
        assemble_ml_dataset(
            feature_rows=[row],
            labels=[fx.label_row(0)],
            splits=[fx.split_row(0, "train")],
            eligible=eligible,
        )


@pytest.mark.parametrize("name", PHASE_5_OUTPUT_NAMES)
def test_an_output_name_cannot_be_admitted_to_the_allowlist(name: str) -> None:
    """Direction one, at the review boundary: it cannot even be written down."""
    with pytest.raises(ValueError):
        FeatureAdmission(**fx.admission(name))


def test_a_phase_five_output_schema_may_declare_these_names() -> None:
    """Direction two: the same names are legitimate on a typed output schema.

    A prediction row is exactly where ``decision_threshold`` and
    ``malicious_probability`` belong. If the prohibition leaked into the output
    side, every Phase 5 artifact would have to be renamed around a rule that was
    never about naming -- so the ML layer's own schema guard is asserted to
    permit them.
    """
    from pydantic import BaseModel, ConfigDict

    from password_attack_detector.ml.schemas import prohibited_metadata_fields

    class MalicousPredictionRow(BaseModel):
        """A stand-in for the Milestone 8 prediction schema."""

        model_config = ConfigDict(extra="forbid", frozen=True)

        malicious_probability: float | None
        malicious_decision_score: float
        flagged_malicious: bool
        predicted_scenario: str
        category_scores_json: str
        anomaly_score: float | None
        fused_flagged: bool
        decision_threshold: float
        min_category_score: float

    # Declared without error, and the layer's field guard permits every one.
    assert prohibited_metadata_fields(list(MalicousPredictionRow.model_fields)) == ()

    row = MalicousPredictionRow(
        malicious_probability=0.75,
        malicious_decision_score=1.5,
        flagged_malicious=True,
        predicted_scenario="brute_force",
        category_scores_json="{}",
        anomaly_score=None,
        fused_flagged=True,
        decision_threshold=0.5,
        min_category_score=0.2,
    )
    assert row.decision_threshold == 0.5
    assert row.malicious_probability == 0.75


def test_the_two_directions_do_not_contradict_each_other() -> None:
    """The same name is a prohibited input and a permitted output field.

    Stated as one assertion so the distinction cannot be read as an accident of
    two independently maintained lists.
    """
    from password_attack_detector.ml.schemas import PROHIBITED_METADATA_FIELDS

    for name in PHASE_5_OUTPUT_NAMES:
        assert name in PROHIBITED_FEATURE_COLUMNS, name
        assert name not in PROHIBITED_METADATA_FIELDS, name
