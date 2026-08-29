"""The SOC analyst console: a client of the serving API, and only a client.

This package renders what the detection system decided.  It does not decide
anything, and it structurally cannot:

* **No detection capability is importable from here.**  Nothing in this package
  imports :mod:`password_attack_detector.ml`,
  :mod:`password_attack_detector.detection`, or
  :mod:`password_attack_detector.features`.  There is no rule engine, no
  preprocessor, no model adapter, no calibrator, no threshold, no fusion
  function, and no serving-bundle reader in this process.  A test walks every
  module's syntax tree and enforces it.
* **There is one door to the backend.**
  :class:`~password_attack_detector.dashboard.api_client.DashboardAPIClient` makes
  every request.  Pages call it; they do not construct HTTP.
* **The wire contract is re-declared, not imported.**  The shapes in
  :mod:`~password_attack_detector.dashboard.contracts` describe the API's
  published documents.  Importing the server's own response models would have
  pulled the whole detection stack in transitively and made the boundary above a
  matter of discipline rather than of fact.
* **No scientific setting exists.**  The configuration answers *which service to
  ask and how to render the answer*.  A model picker or a threshold slider would
  produce screenshots of a system nobody deployed, so
  :data:`~password_attack_detector.dashboard.config.PROHIBITED_SETTING_NAMES`
  names them and an import-time guard refuses them.

The console's history is **this browser session only**.  There is no persistent
alert store, no event database, and no live stream yet, and every page that shows
session data says so on the page rather than in a footnote.
"""

from __future__ import annotations

from password_attack_detector.dashboard.config import (
    DashboardSettings,
    load_dashboard_settings,
)

__all__ = ["DashboardSettings", "load_dashboard_settings"]
