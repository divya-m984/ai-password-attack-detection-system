"""The HTTP surface, split by what each group of routes is for.

Routes are deliberately thin.  Each one validates nothing itself -- the request
schemas do that -- decides nothing itself, and calls exactly one typed service
function.  A handler with an ``if`` in it is a handler where a security or
scientific decision has started to live outside the layer that owns it.
"""

from __future__ import annotations

from password_attack_detector.api.routes.detection import detection_router
from password_attack_detector.api.routes.explain import explain_router
from password_attack_detector.api.routes.health import health_router
from password_attack_detector.api.routes.replay import replay_router
from password_attack_detector.api.routes.system import system_router

__all__ = [
    "detection_router",
    "explain_router",
    "health_router",
    "replay_router",
    "system_router",
]
