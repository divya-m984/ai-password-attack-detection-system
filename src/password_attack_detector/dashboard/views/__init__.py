"""The nine views the sidebar navigates between.

Named ``views`` rather than ``pages`` on purpose.  Streamlit treats a directory
called ``pages/`` sitting beside the entrypoint script as an *automatic*
multipage app: every module in it becomes a navigation entry, ordered by
filename, whether or not it was meant to be one.  This console drives its own
navigation from :data:`~password_attack_detector.dashboard.components.header
.PAGES`, so the magic directory would produce a second navigation beside the
real one, listing the same views under filenames and calling their render
functions with no arguments.

Every module here exposes one ``render(...)`` function and performs its own API
reads through the client it is handed.  None imports the detection, feature, or
ML packages: what a view can display is exactly what the service publishes.
"""

from __future__ import annotations

__all__: list[str] = []
