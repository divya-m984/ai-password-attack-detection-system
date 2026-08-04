"""Static registration of the concrete detection rules.

Registration is a literal tuple in this module.  There is no discovery, no
plugin path, no entry point, and no configuration-supplied import: adding a rule
means editing reviewed Python and the catalog together, and a rule whose
specification is not in :data:`RULE_CATALOG` cannot be registered at all.

The concrete rule implementations land in a later milestone.  :data:`ALL_RULES`
is empty until then, and :func:`build_rule_index` is already the gate every
future entry must pass.
"""

from __future__ import annotations

from collections.abc import Sequence

from password_attack_detector.detection.catalog import RULE_CATALOG, RuleCatalog
from password_attack_detector.detection.rules.base import (
    BasePreparedRule,
    BaseRule,
    PreparedRule,
    Rule,
    RulePreparation,
    SignalComponent,
    SnapshotView,
    build_evidence,
    clamp,
    insufficient_history_reason_code,
    safe_ratio,
    saturate,
    saturate_inverse,
    weighted_strength,
)
from password_attack_detector.exceptions import DetectionConfigurationError

__all__ = [
    "ALL_RULES",
    "RULE_IMPLEMENTATIONS",
    "BasePreparedRule",
    "BaseRule",
    "PreparedRule",
    "Rule",
    "RulePreparation",
    "SignalComponent",
    "SnapshotView",
    "build_evidence",
    "build_rule_index",
    "clamp",
    "insufficient_history_reason_code",
    "safe_ratio",
    "saturate",
    "saturate_inverse",
    "weighted_strength",
]


def build_rule_index(
    rules: Sequence[Rule], *, catalog: RuleCatalog = RULE_CATALOG
) -> dict[str, Rule]:
    """Index *rules* by identifier, rejecting anything unregistered.

    Every implementation must correspond to a catalog entry, and the entry it
    carries must be the catalog's own specification -- not a lookalike built
    elsewhere.  That is what keeps the catalog the single source of rule
    metadata rather than a parallel description that can drift.

    Raises:
        DetectionConfigurationError: on a duplicate identifier, an identifier
            absent from *catalog*, or a specification that is not the catalog's.
    """
    index: dict[str, Rule] = {}
    for rule in rules:
        rule_id = rule.spec.rule_id
        if rule_id in index:
            raise DetectionConfigurationError(
                f"Duplicate rule implementation registered for {rule_id!r}"
            )
        if not catalog.has(rule_id):
            raise DetectionConfigurationError(
                f"Rule implementation {rule_id!r} has no catalog entry"
            )
        if catalog.get(rule_id) is not rule.spec:
            raise DetectionConfigurationError(
                f"Rule implementation {rule_id!r} carries a specification that "
                f"is not the registered catalog entry"
            )
        index[rule_id] = rule
    return index


#: Every registered rule implementation, in a fixed order.  Concrete rules
#: arrive in the next milestone; the registry gate above is already in force.
ALL_RULES: tuple[Rule, ...] = ()

#: The implementations indexed by rule identifier.
RULE_IMPLEMENTATIONS: dict[str, Rule] = build_rule_index(ALL_RULES)
