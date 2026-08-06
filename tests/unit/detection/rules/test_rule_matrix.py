"""The contract every registered rule must satisfy, applied to all nine.

These are the invariants that are the same question for every rule --
determinism, boundedness, privacy, the detection-time boundary -- so they are
written once and parametrized rather than copied nine times.  A tenth rule
added to the registry is covered by this module the moment it is registered,
which is the point: a new rule cannot arrive without a firing fixture and
without passing the whole matrix.

Rule-specific thresholds, discriminators, and false-positive controls live in
the per-rule modules beside this one.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.enums import CorrelationGroup, RuleStatus
from password_attack_detector.detection.rules import ALL_RULES, RULE_IMPLEMENTATIONS
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.detection.schemas import FiredDetection
from password_attack_detector.exceptions import RuleEvaluationError
from password_attack_detector.features.catalog import (
    PROHIBITED_FEATURE_COLUMNS,
    FeatureCatalog,
)
from tests.unit.detection import factories

#: Every rule paired with the builder that produces a snapshot firing it.
FIRING_ROWS: dict[str, Callable[..., dict[str, Any]]] = {
    "PAD-ATO-001": factories.takeover_row,
    "PAD-BF-001": factories.brute_force_row,
    "PAD-BF-002": factories.success_after_burst_row,
    "PAD-BOT-001": factories.bot_row,
    "PAD-CS-001": factories.stuffing_row,
    "PAD-DBF-001": factories.distributed_row,
    "PAD-GEO-001": factories.impossible_travel_row,
    "PAD-MFA-001": factories.mfa_row,
    "PAD-PS-001": factories.spraying_row,
}

RULE_IDS: list[str] = sorted(FIRING_ROWS)

#: Shapes that must never appear in evidence: a UUID, a Phase 2 pseudonym, or
#: a bare hexadecimal identifier.
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_PSEUDONYM_RE = re.compile(r"(u|s|d|sess):[0-9a-f]{32}")

#: Columns a rule must never consult.  Injected into a snapshot to prove that
#: reading one cannot change a verdict.
_FORBIDDEN_INJECTIONS: dict[str, Any] = {
    "label": "malicious",
    "attack_class": "brute_force",
    "malicious": True,
    "scenario": "novel_anomaly_holdout",
    "campaign_id": "campaign-0001",
    "split": "test",
    "supervised_training_eligible": False,
    "model_probability": 0.99,
    "user_id": "u:" + "a" * 32,
    "source_id": "s:" + "b" * 32,
}


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


def prepared_rule(
    rule_id: str, catalog: FeatureCatalog, **config_kwargs: Any
) -> PreparedRule:
    """Prepare *rule_id* against the default detection configuration."""
    return RULE_IMPLEMENTATIONS[rule_id].prepare(
        DetectionConfig(**config_kwargs), catalog
    )


def firing_row(
    rule_id: str, catalog: FeatureCatalog, **overrides: Any
) -> dict[str, Any]:
    """Return a snapshot that fires *rule_id* at its declared defaults."""
    return FIRING_ROWS[rule_id](catalog, **overrides)


# ---------------------------------------------------------------------------
# Registration and catalog consistency
# ---------------------------------------------------------------------------


def test_every_catalogued_rule_has_an_implementation() -> None:
    assert sorted(RULE_IMPLEMENTATIONS) == sorted(RULE_CATALOG.rule_ids)


def test_every_registered_rule_has_a_firing_fixture() -> None:
    """A rule with no fixture would silently skip the whole matrix."""
    assert sorted(FIRING_ROWS) == sorted(RULE_CATALOG.rule_ids)


def test_registration_is_a_static_tuple_without_duplicates() -> None:
    identifiers = [rule.spec.rule_id for rule in ALL_RULES]
    assert sorted(identifiers) == sorted(set(identifiers))
    assert len(ALL_RULES) == len(RULE_CATALOG)


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_implementation_carries_the_registered_specification(rule_id: str) -> None:
    """Not merely an equal spec: the catalog's own object."""
    assert RULE_IMPLEMENTATIONS[rule_id].spec is RULE_CATALOG.get(rule_id)


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_result_metadata_matches_the_catalog(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    spec = RULE_CATALOG.get(rule_id)
    result = prepared_rule(rule_id, catalog).evaluate(firing_row(rule_id, catalog))
    assert result.rule_id == spec.rule_id
    assert result.rule_version == spec.rule_version
    assert result.rule_family is spec.family
    assert result.attack_category is spec.attack_category
    assert result.correlation_group is spec.correlation_group
    assert result.severity is spec.default_severity


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_emitted_evidence_codes_are_all_declared(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    spec = RULE_CATALOG.get(rule_id)
    declared = {item.evidence_code for item in spec.evidence}
    result = prepared_rule(rule_id, catalog).evaluate(firing_row(rule_id, catalog))
    assert {item.evidence_code for item in result.evidence} <= declared


def test_correlation_groups_match_the_approved_design() -> None:
    """The groups are load-bearing: they stop one behaviour scoring twice."""
    grouped: dict[CorrelationGroup, list[str]] = {}
    for spec in RULE_CATALOG.specs:
        grouped.setdefault(spec.correlation_group, []).append(spec.rule_id)
    assert grouped == {
        CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET: [
            "PAD-BF-001",
            "PAD-BF-002",
            "PAD-DBF-001",
        ],
        CorrelationGroup.SOURCE_FANOUT: ["PAD-CS-001", "PAD-PS-001"],
        CorrelationGroup.SESSION_ANOMALY: ["PAD-ATO-001", "PAD-MFA-001"],
        CorrelationGroup.LOCATION_MOVEMENT: ["PAD-GEO-001"],
        CorrelationGroup.AUTOMATION_TIMING: ["PAD-BOT-001"],
    }


# ---------------------------------------------------------------------------
# Firing, evidence, and signal strength
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_positive_case_fires_with_evidence_and_reasons(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    result = prepared_rule(rule_id, catalog).evaluate(firing_row(rule_id, catalog))
    assert result.status is RuleStatus.FIRED
    assert result.evidence
    assert result.reason_codes
    assert 0.0 < result.signal_strength <= 1.0


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_signal_strength_stays_bounded_and_finite(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    result = prepared_rule(rule_id, catalog).evaluate(firing_row(rule_id, catalog))
    assert result.signal_strength == result.signal_strength  # not NaN
    assert 0.0 <= result.signal_strength <= 1.0


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_a_neutral_snapshot_does_not_fire(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    """A row with observed but unremarkable history must not fire anything."""
    result = prepared_rule(rule_id, catalog).evaluate(factories.quiet_row(catalog))
    assert result.status is not RuleStatus.FIRED
    assert result.evidence == ()
    assert result.signal_strength == 0.0


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_evaluation_is_deterministic_across_repeats(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    prepared = prepared_rule(rule_id, catalog)
    row = firing_row(rule_id, catalog)
    first = prepared.evaluate(row)
    second = prepared.evaluate(row)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_two_preparations_agree(rule_id: str, catalog: FeatureCatalog) -> None:
    """Preparation carries no state that could drift between runs."""
    row = firing_row(rule_id, catalog)
    assert prepared_rule(rule_id, catalog).evaluate(row) == prepared_rule(
        rule_id, catalog
    ).evaluate(row)


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_input_mapping_order_does_not_change_the_result(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    """Snapshot columns are read by key, so build order cannot matter."""
    prepared = prepared_rule(rule_id, catalog)
    row = firing_row(rule_id, catalog)
    reversed_row = dict(reversed(list(row.items())))
    assert prepared.evaluate(reversed_row) == prepared.evaluate(row)


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_the_rule_does_not_mutate_the_snapshot(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    prepared = prepared_rule(rule_id, catalog)
    row = firing_row(rule_id, catalog)
    before = dict(row)
    prepared.evaluate(row)
    assert row == before


# ---------------------------------------------------------------------------
# Privacy and the detection-time boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_evidence_carries_no_identifier(rule_id: str, catalog: FeatureCatalog) -> None:
    result = prepared_rule(rule_id, catalog).evaluate(firing_row(rule_id, catalog))
    for item in result.evidence:
        rendered = item.model_dump_json()
        assert not _UUID_RE.search(rendered), item.evidence_code
        assert not _PSEUDONYM_RE.search(rendered), item.evidence_code
        assert factories.ANCHOR not in rendered


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_evidence_avoids_claim_language(rule_id: str, catalog: FeatureCatalog) -> None:
    """Evidence describes an observation, never proof or a probability."""
    forbidden = {
        "proof",
        "proves",
        "proven",
        "confirms",
        "confirmed",
        "guarantees",
        "certainly",
        "definitely",
        "probability",
        "likelihood",
    }
    result = prepared_rule(rule_id, catalog).evaluate(firing_row(rule_id, catalog))
    for item in result.evidence:
        tokens = {
            token.strip(".,;:!?()[]'\"") for token in item.message.lower().split()
        }
        assert not tokens & forbidden, item.evidence_code


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_rule_declares_no_prohibited_feature(rule_id: str) -> None:
    spec = RULE_CATALOG.get(rule_id)
    for template in (*spec.required_features, *spec.optional_features):
        base = template.split("__", maxsplit=1)[0]
        assert template not in PROHIBITED_FEATURE_COLUMNS
        assert base not in PROHIBITED_FEATURE_COLUMNS


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_label_split_and_campaign_columns_cannot_change_a_verdict(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    """Injecting ground truth into the snapshot must change nothing at all."""
    prepared = prepared_rule(rule_id, catalog)
    row = firing_row(rule_id, catalog)
    baseline = prepared.evaluate(row)

    for column, value in _FORBIDDEN_INJECTIONS.items():
        assert prepared.evaluate({**row, column: value}) == baseline

    inverted = {
        **row,
        **{
            column: (not value if isinstance(value, bool) else "inverted")
            for column, value in _FORBIDDEN_INJECTIONS.items()
        },
    }
    assert prepared.evaluate(inverted) == baseline


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_rule_reads_only_columns_it_resolved(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    """A snapshot trimmed to the resolved columns must still evaluate.

    If a rule reached for anything it did not declare, this row would not
    carry it and the evaluation would raise.
    """
    prepared = prepared_rule(rule_id, catalog)
    row = firing_row(rule_id, catalog)
    trimmed = {
        column: row[column]
        for column in (
            "anchor_event_id",
            "anchor_event_time",
            *prepared.preparation.features.values(),
        )
    }
    assert prepared.evaluate(trimmed) == prepared.evaluate(row)


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_a_missing_required_column_is_refused(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    prepared = prepared_rule(rule_id, catalog)
    row = firing_row(rule_id, catalog)
    dropped = next(iter(prepared.preparation.features.values()))
    incomplete = {key: value for key, value in row.items() if key != dropped}
    with pytest.raises(RuleEvaluationError, match="does not carry column"):
        prepared.evaluate(incomplete)


@pytest.mark.parametrize("column", ["anchor_event_id", "anchor_event_time"])
def test_a_missing_anchor_key_is_refused(column: str, catalog: FeatureCatalog) -> None:
    prepared = prepared_rule("PAD-BF-001", catalog)
    row = factories.brute_force_row(catalog)
    del row[column]
    with pytest.raises(RuleEvaluationError, match="anchor column"):
        prepared.evaluate(row)


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_a_fired_result_publishes_with_a_deterministic_identifier(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    result = prepared_rule(rule_id, catalog).evaluate(firing_row(rule_id, catalog))
    first = FiredDetection.from_result(result)
    second = FiredDetection.from_result(result)
    assert first.detection_id == second.detection_id
    assert first == second


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_detection_identifiers_differ_between_rules_on_one_anchor(
    rule_id: str, catalog: FeatureCatalog
) -> None:
    identifiers = set()
    for other in RULE_IDS:
        result = prepared_rule(other, catalog).evaluate(
            firing_row(other, catalog, anchor_event_id=factories.ANCHOR)
        )
        if result.status is RuleStatus.FIRED:
            identifiers.add(FiredDetection.from_result(result).detection_id)
    assert len(identifiers) == len(RULE_IDS)
