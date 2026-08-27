"""How values are rendered, and the vocabulary the rendering is allowed to use.

Kept out of :mod:`~password_attack_detector.dashboard.api_client` for the
ordinary reason -- a transport that formatted would be a transport nobody could
test -- and kept out of the page modules for a sharper one: the rules about what
a number may be *called* are enforced here, once, where they can be tested.

Three of those rules do real work:

* **A risk score is never a percentage.**  The rule layer's ``risk_score`` is an
  ordinal 0-100 severity magnitude.  :func:`format_risk_score` renders it as
  ``72.4 / 100`` and there is no code path that appends a ``%``.
* **A decision score is never a probability.**  :func:`format_model_score` reads
  the ``score_kind`` the service declared and labels the number accordingly.  An
  uncalibrated score is shown under its own name; it is never relabelled, and its
  absence is never filled in with the other one.
* **Nothing user-entered is rendered as markup.**  :func:`escape_text` is applied
  to every string that reaches an HTML block.  The only HTML this package emits
  is the static stylesheet, and it interpolates nothing.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from datetime import datetime
from typing import Final

__all__ = [
    "SEVERITY_COLORS",
    "SEVERITY_ORDER",
    "STATE_COLORS",
    "escape_text",
    "format_component_state",
    "format_fusion_strategy",
    "format_fusion_strategy_short",
    "format_model_family",
    "format_model_score",
    "format_reason_code",
    "format_risk_score",
    "format_timestamp",
    "format_verdict",
    "severity_rank",
    "truncate_identifier",
]

#: The Phase 4 ordinal severity scale, weakest first.  The order the dashboard
#: sorts and charts by; it is the detection layer's scale, not one invented here.
SEVERITY_ORDER: Final[tuple[str, ...]] = ("none", "low", "medium", "high", "critical")

#: Accent colour per severity.  Green reads "nothing here", amber "look at this",
#: red "act on this" -- and nothing below ``high`` is ever rendered red, so a red
#: cell on a page always means the same thing.
SEVERITY_COLORS: Final[dict[str, str]] = {
    "none": "#3fb950",
    "low": "#58a6ff",
    "medium": "#d29922",
    "high": "#f0883e",
    "critical": "#f85149",
}

#: Accent colour per component readiness state.
STATE_COLORS: Final[dict[str, str]] = {
    "ready": "#3fb950",
    "unavailable": "#f85149",
    "disabled": "#8b949e",
}

#: Placeholder for a value the service did not report.  One string everywhere,
#: so an absent number never renders as an empty cell that reads like a zero.
_ABSENT: Final[str] = "—"


def escape_text(value: object) -> str:
    """Return *value* as text that is safe to place inside an HTML block.

    Applied to everything that reaches a styled block, including values the API
    returned: the service sanitizes what it publishes, but "the other end is
    careful" is not an escaping strategy, and a rule description is a string
    somebody may one day write a ``<`` into.
    """
    return html.escape(str(value), quote=True)


def severity_rank(severity: str) -> int:
    """Return a sortable rank for *severity*, unknown values sorting first."""
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return -1


def format_risk_score(value: float | None) -> str:
    """Return the rule layer's ordinal magnitude, on the scale it is measured on.

    Rendered ``72.4 / 100`` rather than ``72.4%``. The denominator is stated
    because the number is a bounded ordinal rather than a fraction of anything,
    and a percent sign would invite exactly the comparison against the model's
    probability that the whole system refuses to make.
    """
    if value is None:
        return _ABSENT
    return f"{value:.1f} / 100"


def format_model_score(
    *,
    score_kind: str | None,
    decision_score: float | None,
    probability: float | None,
) -> tuple[str, str]:
    """Return the label and the value for the model's score.

    The label comes from the ``score_kind`` the service declared, never from
    which field happens to be populated. A lineage with no calibrator publishes
    a decision score and no probability, and calling that number a probability
    -- or rendering it as a percentage -- would state a property of the model
    that nobody established.
    """
    if score_kind == "calibrated_probability" and probability is not None:
        return ("Calibrated probability", f"{probability:.6f}")
    if decision_score is not None:
        return ("Decision score (uncalibrated)", f"{decision_score:.6f}")
    return ("Model score", _ABSENT)


def format_verdict(flagged: bool | None) -> str:
    """Return a verdict as words, with absence distinct from a negative."""
    if flagged is None:
        return _ABSENT
    return "FLAGGED" if flagged else "not flagged"


def format_fusion_strategy(strategy: str | None) -> str:
    """Return a fusion strategy in the terms the project's documents use."""
    if strategy is None:
        return _ABSENT
    return {
        "or_gate": "OR gate",
        "and_gate": "AND gate",
        "stacked": "Stacked (fitted meta-learner)",
    }.get(strategy, strategy)


def format_fusion_strategy_short(strategy: str | None) -> str:
    """Return a fusion strategy as a concise human-readable label for cards."""
    if strategy is None:
        return _ABSENT
    return {
        "or_gate": "OR Gate",
        "and_gate": "AND Gate",
        "stacked": "Stacked",
    }.get(strategy, strategy)


def format_model_family(family: str | None) -> str:
    """Return a model family as a human-readable label for cards.

    Converts internal identifiers like ``logistic_regression`` to title-cased
    prose like ``Logistic Regression`` for prominent display.  The internal
    value is unchanged in technical details.
    """
    if not family:
        return _ABSENT
    return {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }.get(family, family.replace("_", " ").title())


def format_component_state(state: str) -> str:
    """Return a readiness state as a word a viewer can act on."""
    return {
        "ready": "Ready",
        "unavailable": "Unavailable",
        "disabled": "Disabled",
    }.get(state, state)


def format_reason_code(reason: str | None) -> str:
    """Return a stable reason code as readable prose, code intact.

    The code is kept alongside the prose rather than replaced by it: the code is
    what an operator greps a log for, and the prose is what makes the page
    legible. Underscores become spaces and nothing else changes, so a code this
    build has never seen still renders sensibly.
    """
    if not reason:
        return _ABSENT
    return f"{reason.replace('_', ' ')} ({reason})"


def format_timestamp(value: datetime | None) -> str:
    """Return an instant as an unambiguous UTC-offset string, seconds resolution."""
    if value is None:
        return _ABSENT
    return value.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def truncate_identifier(value: str, *, keep: int = 12) -> str:
    """Return a long opaque identifier shortened for display.

    Applied to digests and event identifiers, which are long enough to break a
    table layout and are read for recognition rather than for content. The
    original is never discarded -- callers put it in a tooltip or a code block --
    so this shortens a display, not a value.
    """
    if len(value) <= keep:
        return value
    return f"{value[:keep]}…"


def format_detail(detail: Mapping[str, int | str] | None) -> str:
    """Return an error envelope's aggregate detail as one readable clause."""
    if not detail:
        return ""
    return ", ".join(f"{key}: {value}" for key, value in sorted(detail.items()))
