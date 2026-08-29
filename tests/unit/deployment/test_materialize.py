"""Tests for the materializer's refusals and its structural guarantees.

The *reconstruction* itself needs a full frozen pipeline and is exercised in
``tests/integration/test_serving_bundle.py``, where it is compared against a real
sealed fingerprint. What is worth testing cheaply is everything that happens
before a single fold is refitted: whether the frozen lineage can be found, and
whether it verifies.

Plus the guards. Three properties this module is supposed to have are asserted
here rather than trusted:

* it cannot read a TEST label -- no TEST outcome reader is in its namespace;
* it cannot be handed one -- no entry point takes a TEST-shaped argument;
* the operator command cannot be given a TEST publication either.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from password_attack_detector.deployment.materialize import (
    MaterializationOutcome,
    materialize_serving_bundle,
    read_frozen_fusion,
    reconstruct_stacked_state,
)
from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    ManifestVerificationError,
)
from password_attack_detector.ml.enums import FusionStrategy

LOCK = "b" * 64


# ---------------------------------------------------------------------------
# Finding the frozen selection
# ---------------------------------------------------------------------------


def test_a_root_with_no_evaluation_has_nothing_to_deploy(tmp_path: Path) -> None:
    """Nothing has read TEST here, so nothing has frozen a hybrid."""
    with pytest.raises(ArtifactNotFoundError, match="no locked TEST evaluation"):
        read_frozen_fusion(tmp_path, champion_lock_fingerprint=LOCK)


def test_an_evaluations_directory_with_no_receipts_has_nothing_to_deploy(
    tmp_path: Path,
) -> None:
    """An empty directory is an absence, not a corruption."""
    (tmp_path / "evaluations" / "incomplete").mkdir(parents=True)
    with pytest.raises(ArtifactNotFoundError, match="selected a hybrid strategy"):
        read_frozen_fusion(tmp_path, champion_lock_fingerprint=LOCK)


def test_an_unreadable_receipt_stops_the_materialization(tmp_path: Path) -> None:
    """A receipt that does not verify is refused, never partially believed.

    Skipping it would be worse than failing: the unreadable receipt may be the
    one that selected the hybrid, and a bundle published without it would deploy
    whatever the *other* receipts happened to say.
    """
    directory = tmp_path / "evaluations" / "one"
    directory.mkdir(parents=True)
    (directory / "test_evaluation.json").write_text(
        '{"not": "a receipt"}', encoding="utf-8"
    )
    with pytest.raises(ManifestVerificationError, match="not readable"):
        read_frozen_fusion(tmp_path, champion_lock_fingerprint=LOCK)


# ---------------------------------------------------------------------------
# The outcome type reports both digests
# ---------------------------------------------------------------------------


def test_a_refusal_reports_both_digests_so_an_operator_can_see_which_moved() -> None:
    """A mismatch is a diagnosis, not just a failure."""
    outcome = MaterializationOutcome(
        directory=None,
        created=False,
        strategy=FusionStrategy.STACKED,
        frozen_state_fingerprint="a" * 64,
        reconstructed_state_fingerprint="f" * 64,
        manifest=None,
        refusal="stacked_state_fingerprint_mismatch",
    )
    assert outcome.published is False
    assert outcome.fingerprints_agree is False


def test_agreement_requires_a_frozen_digest_to_agree_with() -> None:
    """Two absent digests are not an agreement."""
    outcome = MaterializationOutcome(
        directory=None,
        created=False,
        strategy=FusionStrategy.OR_GATE,
        frozen_state_fingerprint=None,
        reconstructed_state_fingerprint=None,
        manifest=None,
        refusal=None,
    )
    assert outcome.fingerprints_agree is False


# ---------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------


def test_the_materializer_holds_no_test_outcome_reader() -> None:
    """The import-time guard, asserted so it cannot be quietly deleted."""
    from password_attack_detector.deployment import materialize

    namespace = vars(materialize)
    for forbidden in (
        "TestOutcome",
        "evaluate_test",
        "publish_evaluation",
        "select_fusion_strategy",
        "select_binary_threshold",
        "freeze_champion",
    ):
        assert forbidden not in namespace


def test_no_entry_point_takes_a_test_shaped_argument() -> None:
    """The firewall stated as a signature rather than as a promise."""
    forbidden = {"test", "test_labels", "test_split", "holdout", "novel_holdout"}
    for function in (reconstruct_stacked_state, materialize_serving_bundle):
        assert not forbidden & set(inspect.signature(function).parameters)


def test_the_reconstruction_reads_pre_test_lineage_only() -> None:
    """Every parameter is TRAIN-side, validation-side, or frozen lineage."""
    parameters = set(inspect.signature(reconstruct_stacked_state).parameters)
    assert parameters == {
        "dataset",
        "config",
        "eligible",
        "feature_catalog",
        "champion",
        "rule",
        "rule_configuration_fingerprint",
        "validation_predictions",
    }


def test_the_operator_command_cannot_be_given_a_test_publication() -> None:
    """``deploy materialize`` has a validation prediction option and no other.

    A ``--prediction`` for a TEST publication would be the one keyword that turns
    a reproduction of a pre-TEST decision into something downstream of TEST.
    """
    from password_attack_detector.deployment.cli import materialize as command

    parameters = set(inspect.signature(command).parameters)
    assert "validation_prediction" in parameters
    assert "prediction" not in parameters
    assert "prediction_id" not in parameters
    assert "holdout_prediction" not in parameters


def test_the_deployment_cli_exposes_exactly_two_commands() -> None:
    """One command writes an artifact and one reads it. Nothing trains."""
    from password_attack_detector.deployment.cli import deployment_app

    names = {
        command.name or getattr(command.callback, "__name__", None)
        for command in deployment_app.registered_commands
    }
    assert names == {"materialize", "inspect"}
