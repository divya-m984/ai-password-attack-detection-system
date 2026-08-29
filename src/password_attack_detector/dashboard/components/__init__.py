"""Reusable rendering pieces, shared across the console's views.

Split from :mod:`password_attack_detector.dashboard.pages` so a widget that
appears on four pages is written once and behaves identically on all four --
which is what makes "the API is offline" look the same everywhere.

No module here performs a network call.  A component is handed documents a page
already fetched, or the client itself in the one case where probing *is* the
component (:func:`~password_attack_detector.dashboard.components.status
.connectivity`).  Keeping the fetch out of the render is what lets a page make
one round of calls instead of one per panel.
"""

from __future__ import annotations

__all__: list[str] = []
