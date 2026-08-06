"""Static registration of the concrete detection rules.

Registration is a literal tuple in this module.  There is no discovery, no
plugin path, no entry point, and no configuration-supplied import: adding a rule
means editing reviewed Python and the catalog together, and a rule whose
specification is not in :data:`RULE_CATALOG` cannot be registered at all.

:func:`build_rule_index` is the gate every entry passes.  It rejects a duplicate
identifier, an identifier with no catalog entry, and -- the case that matters
most -- an implementation carrying a specification that merely *looks* like the
registered one.  That keeps the catalog the single source of rule metadata
rather than a parallel description free to drift.
"""

from __future__ import annotations

from collections.abc import Sequence

from password_attack_detector.detection.catalog import RULE_CATALOG, RuleCatalog
from password_attack_detector.detection.rules.account_takeover import (
    AccountTakeoverRule,
)
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
from password_attack_detector.detection.rules.bot_activity import BotActivityRule
from password_attack_detector.detection.rules.brute_force import (
    ConcentratedBruteForceRule,
    SuccessAfterFailureBurstRule,
)
from password_attack_detector.detection.rules.credential_stuffing import (
    CredentialStuffingRule,
)
from password_attack_detector.detection.rules.distributed_brute_force import (
    DistributedBruteForceRule,
)
from password_attack_detector.detection.rules.impossible_travel import (
    ImpossibleTravelRule,
)
from password_attack_detector.detection.rules.mfa_anomaly import MFASequenceAnomalyRule
from password_attack_detector.detection.rules.password_spraying import (
    PasswordSprayingRule,
)
from password_attack_detector.exceptions import DetectionConfigurationError

__all__ = [
    "ALL_RULES",
    "RULE_IMPLEMENTATIONS",
    "AccountTakeoverRule",
    "BasePreparedRule",
    "BaseRule",
    "BotActivityRule",
    "ConcentratedBruteForceRule",
    "CredentialStuffingRule",
    "DistributedBruteForceRule",
    "ImpossibleTravelRule",
    "MFASequenceAnomalyRule",
    "PasswordSprayingRule",
    "PreparedRule",
    "Rule",
    "RulePreparation",
    "SignalComponent",
    "SnapshotView",
    "SuccessAfterFailureBurstRule",
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


#: Every registered rule implementation, in a fixed order.
#:
#: The order here is documentation only.  The engine iterates the catalog,
#: which is sorted by rule identifier, so no output depends on how this tuple
#: happens to be written.
ALL_RULES: tuple[Rule, ...] = (
    ConcentratedBruteForceRule(),
    SuccessAfterFailureBurstRule(),
    PasswordSprayingRule(),
    CredentialStuffingRule(),
    DistributedBruteForceRule(),
    AccountTakeoverRule(),
    ImpossibleTravelRule(),
    BotActivityRule(),
    MFASequenceAnomalyRule(),
)

#: The implementations indexed by rule identifier.
RULE_IMPLEMENTATIONS: dict[str, Rule] = build_rule_index(ALL_RULES)
