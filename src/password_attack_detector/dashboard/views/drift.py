"""Drift monitoring: what the project measures, and what is not deployed yet.

**There is no serving drift endpoint.**  The Phase 5 drift hook captures a
reference profile at training time and compares a later batch against it, offline,
through the ``ml drift`` command.  Nothing in the serving layer publishes a drift
report, so this page has no live figure to show -- and it shows none.  It does not
compute PSI from the session's own windows either: a handful of hand-built demo
requests is not a population, and a number derived from one would be a
measurement of the demo.

What it does show is the contract: the statistic, the thresholds the reviewed
configuration declares, and the honest current state.

The thresholds below are the project's documented defaults, restated here because
this package does not import :mod:`password_attack_detector.ml.config`.  A test
asserts them against ``DriftConfig``'s own field defaults, so a value changed
there is a failing test rather than a page quietly showing the old one.
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.status import Connectivity
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import card, section_title

__all__ = ["PSI_ALERT_THRESHOLD", "PSI_WARN_THRESHOLD", "render"]

#: Population Stability Index above which a column is reported as *warning*.
PSI_WARN_THRESHOLD: Final[float] = 0.10

#: PSI above which a column is reported as *alert*.  Strictly above the warning
#: threshold, which ``DriftConfig`` enforces.
PSI_ALERT_THRESHOLD: Final[float] = 0.25

#: The reference population a profile is captured from.  Training rows only:
#: capturing it from validation or test would make the comparison a measurement
#: against data the model was selected or evaluated on.
REFERENCE_SOURCE: Final[str] = "train"


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the drift monitoring view."""
    st.markdown(section_title("Serving drift status"), unsafe_allow_html=True)
    st.info(
        "**No serving drift report loaded.** The serving API publishes no drift "
        "endpoint in this milestone, so there is no live figure to display and "
        "none is estimated here.",
        icon="📉",
    )
    st.caption(
        "Drift is computed offline by the `ml drift` command against a "
        "reference profile captured at training time. Wiring a deployed drift "
        "workflow into the serving layer is later-milestone work."
    )

    columns = st.columns(3)
    with columns[0]:
        st.markdown(
            card(
                "Warning threshold",
                f"PSI ≥ {PSI_WARN_THRESHOLD:.2f}",
                accent="#d29922",
                note="a column's distribution has moved noticeably",
            ),
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            card(
                "Alert threshold",
                f"PSI ≥ {PSI_ALERT_THRESHOLD:.2f}",
                accent="#f85149",
                note="a column's distribution has moved materially",
            ),
            unsafe_allow_html=True,
        )
    with columns[2]:
        st.markdown(
            card(
                "Reference population",
                REFERENCE_SOURCE,
                accent="#58a6ff",
                note="captured at training time",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(section_title("What is monitored"), unsafe_allow_html=True)
    st.markdown(
        """
A drift check compares the distribution of each **transformed model input**
against a reference profile captured from the training partition, and reports a
Population Stability Index per column.

* **PSI** is a symmetric measure of how far two binned distributions have moved
  apart. It is computed per column and never summed into one score for the model:
  a single headline number would hide which input moved.
* The reference is captured from **training rows only**. Capturing it from
  validation or test would make every later comparison a measurement against
  data the model was selected or evaluated on.
* Crossing a threshold is a **signal to investigate**, not a trigger. Nothing in
  this project retrains, re-thresholds, or re-selects on a drift reading, and no
  part of the serving layer acts on one.
* Drift is a statement about **inputs**, not about accuracy. A stable input
  distribution does not imply the model is still right, and a moved one does not
  imply it is wrong.
        """.strip()
    )

    st.markdown(section_title("Why nothing is shown above"), unsafe_allow_html=True)
    st.markdown(
        '<div class="pad-note">'
        "This console will not derive a drift figure from the windows submitted "
        "in this session. A few hand-built demonstration requests are not a "
        "population, and a PSI computed over them would measure the demo rather "
        "than the deployment. When a serving drift report is published, this "
        "page reads it from the API like every other value on this console."
        "</div>",
        unsafe_allow_html=True,
    )
