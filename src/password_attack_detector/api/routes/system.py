"""Aggregate system, model, and rule information.

Everything here is *identity and configuration*, never mechanism.  A reader
learns which layers are running, which model family was frozen, what its
operating point is, and which rules are enabled.  A reader does not learn a
coefficient, a tree array, a feature value, an artifact path, an entity
pseudonym, or anything else these documents' schemas do not declare -- and the
schemas forbid extra fields, so a future field cannot arrive by accident.
"""

from __future__ import annotations

from fastapi import APIRouter

from password_attack_detector.api.dependencies import Runtime
from password_attack_detector.api.schemas import (
    ModelInfoResponse,
    RuleCatalogResponse,
    SystemStatusResponse,
)
from password_attack_detector.api.services import (
    model_info_document,
    rule_catalog_document,
    system_status_document,
)

__all__ = ["system_router"]

system_router = APIRouter(prefix="/api/v1", tags=["System"])


@system_router.get(
    "/system/status",
    response_model=SystemStatusResponse,
    summary="Which detection layers this deployment runs",
    description=(
        "Reports the enabled detection layers, the frozen fusion strategy when "
        "one was selected, the champion's model family, and the request "
        "ceilings this deployment enforces."
    ),
)
def system_status(runtime: Runtime) -> SystemStatusResponse:
    """Return the deployment's layer and configuration summary."""
    return system_status_document(runtime)


@system_router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Frozen champion identity and operating point",
    description=(
        "Reports the frozen champion's family, identifiers, task, score kind, "
        "calibration state, decision threshold, and freeze lineage. Carries no "
        "model parameters and no artifact location. The threshold is reported "
        "for transparency and cannot be changed through this API."
    ),
)
def model_info(runtime: Runtime) -> ModelInfoResponse:
    """Return the champion's public-safe identity, or why there is none."""
    return model_info_document(runtime)


@system_router.get(
    "/rules",
    response_model=RuleCatalogResponse,
    summary="The public rule catalog",
    description=(
        "Reports every registered detection rule with its identifier, version, "
        "name, description, family, attack category, default severity, and "
        "whether this deployment enabled it. Carries no thresholds and no "
        "feature names."
    ),
)
def rules(runtime: Runtime) -> RuleCatalogResponse:
    """Return the public rule catalog and this deployment's enabled set."""
    return rule_catalog_document(runtime)
