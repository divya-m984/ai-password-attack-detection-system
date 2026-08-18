"""The wire shapes the dashboard is prepared to read.

Declared here rather than imported from
:mod:`password_attack_detector.api.schemas`, which is the more obvious thing to
do and the wrong one.

Importing the server's models would make the dashboard a second consumer of the
*server's objects* instead of a consumer of its **published contract**, and it
would pull the entire detection and ML stack into this process transitively --
so a dashboard that is supposed to be unable to score anything would have a
scorer, a preprocessor and a fusion function one import away.  Re-declaring the
handful of fields the pages actually render keeps the boundary real: this package
can talk to the API over a socket and could not compute a verdict if it tried.

Three consequences, all deliberate:

* **Forward compatible.**  Every model ignores unknown fields, so a serving
  release that adds one does not break a running dashboard.
* **Narrow.**  Only fields a page displays are declared.  A field nobody renders
  is a field nobody has to keep safe.
* **Tolerant of absence.**  The API omits nothing, but a *misconfigured* endpoint
  might return something else entirely, and a client that raises on a missing key
  turns a backend problem into a traceback on the page.  Defaults are supplied so
  a partial document degrades to a partial display.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AnchorDetection",
    "BatchDetectionDocument",
    "ComponentReport",
    "DetectionDocument",
    "EvidenceItem",
    "ExplanationContribution",
    "ExplanationDocument",
    "HealthDocument",
    "HybridLayer",
    "MLLayer",
    "ModelInfoDocument",
    "ReadinessDocument",
    "RuleCatalogDocument",
    "RuleLayer",
    "RuleSummary",
    "SystemStatusDocument",
    "VersionDocument",
    "WindowSummary",
]


class _Document(BaseModel):
    """Base for every wire shape: frozen, and forgiving of new fields."""

    # ``extra="ignore"`` rather than the ``"forbid"`` the server uses. The server
    # forbids because an undeclared key on the way *in* is an attack surface; a
    # client that forbade would simply refuse to run against a newer service.
    model_config = ConfigDict(extra="ignore", frozen=True)


# ---------------------------------------------------------------------------
# Health, readiness, version
# ---------------------------------------------------------------------------


class HealthDocument(_Document):
    """``GET /health``."""

    status: str = "unknown"
    service: str = ""
    version: str = ""


class ComponentReport(_Document):
    """One runtime component's readiness, from ``GET /ready``."""

    component: str
    state: str
    reason: str | None = None
    required: bool = True


class ReadinessDocument(_Document):
    """``GET /ready``."""

    status: str = "not_ready"
    service: str = ""
    version: str = ""
    components: tuple[ComponentReport, ...] = ()

    @property
    def is_ready(self) -> bool:
        """Return whether the service reported itself ready.

        Read off the status the service published rather than recomputed from
        the component list: readiness is the service's judgement, and a client
        that re-derived it could disagree with the thing it is reporting on.
        """
        return self.status == "ready"

    @property
    def blocking(self) -> tuple[ComponentReport, ...]:
        """Return the required components that are not ready."""
        return tuple(
            item for item in self.components if item.required and item.state != "ready"
        )


class VersionDocument(_Document):
    """``GET /version``."""

    service: str = ""
    package_version: str = ""
    api_schema_version: str = ""
    event_schema_version: str = ""
    feature_schema_version: str = ""
    detection_schema_version: str = ""
    scoring_version: str = ""
    ml_schema_version: str = ""
    fusion_schema_version: str = ""


# ---------------------------------------------------------------------------
# System, model, rules
# ---------------------------------------------------------------------------


class SystemStatusDocument(_Document):
    """``GET /api/v1/system/status``."""

    service: str = ""
    status: str = "not_ready"
    api_schema_version: str = ""
    package_version: str = ""
    rule_detection_enabled: bool = False
    ml_detection_enabled: bool = False
    hybrid_detection_enabled: bool = False
    fusion_strategy: str | None = None
    frozen_fusion_strategy: str | None = None
    hybrid_required: bool = False
    stacked_state_fingerprint: str | None = None
    fusion_unavailable_reason: str | None = None
    champion_model_family: str | None = None
    enabled_rule_count: int = 0
    registered_rule_count: int = 0
    max_batch_events: int = 0


class ModelInfoDocument(_Document):
    """``GET /api/v1/model/info``."""

    # The server clears ``protected_namespaces`` for the same reason: this
    # document's subject *is* a model, and renaming its fields to dodge a
    # Pydantic warning would make the client disagree with the contract.
    model_config = ConfigDict(extra="ignore", frozen=True, protected_namespaces=())

    available: bool = False
    unavailable_reason: str | None = None
    model_family: str | None = None
    catalog_model_id: str | None = None
    model_id: str | None = None
    task: str | None = None
    score_kind: str | None = None
    calibrated: bool | None = None
    decision_threshold: float | None = None
    champion_scope_key: str | None = None
    freeze_record_id: str | None = None
    training_run_id: str | None = None
    validation_selection_id: str | None = None
    ml_schema_version: str | None = None
    required_feature_schema_version: str | None = None
    category_head_available: bool = False


class RuleSummary(_Document):
    """One rule from ``GET /api/v1/rules``."""

    rule_id: str
    rule_version: str = ""
    name: str = ""
    description: str = ""
    family: str = ""
    attack_category: str = ""
    default_severity: str = ""
    enabled: bool = False
    deprecated: bool = False


class RuleCatalogDocument(_Document):
    """``GET /api/v1/rules``."""

    detection_schema_version: str = ""
    rule_count: int = 0
    enabled_rule_count: int = 0
    rules: tuple[RuleSummary, ...] = ()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class EvidenceItem(_Document):
    """One sanitized behavioral observation behind a rule verdict."""

    evidence_code: str = ""
    message: str = ""
    observed_value: bool | int | float | str | None = None


class RuleLayer(_Document):
    """The rule layer's verdict for one anchor.

    ``risk_score`` is an **ordinal 0-100 severity magnitude**, not a probability.
    The dashboard never renders it with a percent sign and never compares it
    against the model's probability.
    """

    flagged: bool = False
    risk_score: float = 0.0
    severity: str = "none"
    primary_attack_category: str | None = None
    contributing_categories: tuple[str, ...] = ()
    fired_rule_ids: tuple[str, ...] = ()
    fired_rule_count: int = 0
    insufficient_data_count: int = 0
    scoring_version: str = ""
    evidence: tuple[EvidenceItem, ...] = ()


class MLLayer(_Document):
    """The frozen champion's verdict for one anchor.

    ``probability`` is present only when the frozen operating point was selected
    against a calibrated probability.  Where it is ``None`` the dashboard shows
    the decision score under its own name and never calls it a probability.
    """

    available: bool = False
    unavailable_reason: str | None = None
    flagged: bool | None = None
    score_kind: str | None = None
    decision_score: float | None = None
    probability: float | None = None
    decision_threshold: float | None = None


class HybridLayer(_Document):
    """The fused verdict for one anchor, under the frozen strategy."""

    available: bool = False
    unavailable_reason: str | None = None
    flagged: bool | None = None
    strategy: str | None = None


class AnchorDetection(_Document):
    """Everything the three layers said about one anchor."""

    anchor_event_id: str
    anchor_event_time: datetime
    rule: RuleLayer = Field(default_factory=RuleLayer)
    ml: MLLayer = Field(default_factory=MLLayer)
    hybrid: HybridLayer = Field(default_factory=HybridLayer)
    severity: str = "none"


class WindowSummary(_Document):
    """Aggregate facts about the window that was evaluated."""

    event_count: int = 0
    anchor_count: int = 0
    feature_schema_version: str = ""
    detection_schema_version: str = ""
    enabled_rule_count: int = 0
    evaluated_snapshot_count: int = 0


class DetectionDocument(_Document):
    """``POST /api/v1/detect``."""

    api_schema_version: str = ""
    window: WindowSummary = Field(default_factory=WindowSummary)
    anchor: AnchorDetection


class BatchDetectionDocument(_Document):
    """``POST /api/v1/detect/batch``."""

    api_schema_version: str = ""
    window: WindowSummary = Field(default_factory=WindowSummary)
    anchors: tuple[AnchorDetection, ...] = ()


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


class ExplanationContribution(_Document):
    """One transformed column's signed contribution to one decision."""

    transformed_feature: str
    contribution: float = 0.0


class ExplanationDocument(_Document):
    """``POST /api/v1/explain``.

    ``decision_value`` is the decision function's own quantity -- a logit, a mean
    leaf score, a threshold step.  The dashboard labels it as such and never
    renders it as a probability or a percentage.
    """

    model_config = ConfigDict(extra="ignore", frozen=True, protected_namespaces=())

    api_schema_version: str = ""
    anchor_event_id: str = ""
    anchor_event_time: datetime | None = None
    available: bool = False
    unavailable_reason: str | None = None
    method: str | None = None
    model_family: str | None = None
    score_kind: str | None = None
    decision_value: float | None = None
    baseline_value: float | None = None
    contributions: tuple[ExplanationContribution, ...] = ()
    transformed_feature_count: int = 0
    omitted_contribution_count: int = 0
    reconstruction_residual: float | None = None
