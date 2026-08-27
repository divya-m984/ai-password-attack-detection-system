"""The console's visual language, in one static stylesheet.

This module is the *only* place the dashboard emits HTML with
``unsafe_allow_html``, and :data:`STYLESHEET` is a module constant that
interpolates nothing.  Every other block a page renders is built by
:func:`card`, :func:`badge`, or :func:`chip`, each of which escapes every value
it is handed through
:func:`~password_attack_detector.dashboard.formatting.escape_text`.  There is no
function here that accepts pre-built markup, so a value an analyst typed cannot
reach the page as anything but text.

The palette is chosen to make one thing legible at a glance -- *is this normal* --
and the colours are load-bearing rather than decorative:

* **Green** is a healthy state or an unflagged verdict.  Never a threat.
* **Cyan** is neutral structure: headings, borders, identifiers.
* **Amber** is a caution: medium severity, a warning threshold, a degraded but
  working component.
* **Orange and red** are reserved for ``high`` and ``critical`` and for a failed
  required component.  Nothing below that ever renders red, so a red cell on any
  page means the same thing on every page.
"""

from __future__ import annotations

from typing import Final

from password_attack_detector.dashboard.formatting import escape_text

__all__ = [
    "ACCENT",
    "BACKGROUND",
    "BORDER",
    "CRITICAL",
    "STYLESHEET",
    "SUCCESS",
    "SURFACE",
    "SURFACE_ELEVATED",
    "TEXT_MUTED",
    "TEXT_PRIMARY",
    "WARNING",
    "badge",
    "card",
    "chip",
    "section_title",
]

BACKGROUND: Final[str] = "#0b0f19"
SURFACE: Final[str] = "#131a29"
SURFACE_ELEVATED: Final[str] = "#182030"
BORDER: Final[str] = "#1f2a3d"
ACCENT: Final[str] = "#58a6ff"
TEXT_PRIMARY: Final[str] = "#e6edf3"
TEXT_MUTED: Final[str] = "#8b949e"
SUCCESS: Final[str] = "#3fb950"
WARNING: Final[str] = "#d29922"
CRITICAL: Final[str] = "#f85149"

#: The console's stylesheet.  A constant: no value is interpolated into it, so
#: there is no path by which page data could become CSS.
STYLESHEET: Final[str] = """
<style>
  .stApp { background-color: #0b0f19; }
  section[data-testid="stSidebar"] {
    background-color: #0d1424;
    border-right: 1px solid #1f2a3d;
  }
  .pad-header {
    border-bottom: 1px solid #1f2a3d;
    padding: 0.25rem 0 1rem 0;
    margin-bottom: 1.25rem;
  }
  .pad-title {
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    color: #e6edf3;
    margin: 0;
  }
  .pad-subtitle {
    font-size: 0.88rem;
    color: #8b949e;
    letter-spacing: 0.01em;
    margin: 0.25rem 0 0 0;
  }
  .pad-card {
    background: #131a29;
    border: 1px solid #1f2a3d;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
  }
  .pad-card-accent {
    border-top: 2px solid #58a6ff;
  }
  .pad-card-label {
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.35rem;
  }
  .pad-card-value {
    font-size: 1.18rem;
    font-weight: 600;
    color: #e6edf3;
    font-variant-numeric: tabular-nums;
  }
  .pad-card-note {
    font-size: 0.76rem;
    color: #8b949e;
    margin-top: 0.3rem;
  }
  .pad-badge {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .pad-badge-plain {
    text-transform: none;
    letter-spacing: 0.02em;
  }
  .pad-chip {
    display: inline-block;
    padding: 0.12rem 0.5rem;
    margin: 0.1rem 0.25rem 0.1rem 0;
    border-radius: 4px;
    border: 1px solid #1f2a3d;
    background: #0f1626;
    color: #c9d1d9;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.74rem;
  }
  .pad-section {
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #8b949e;
    border-bottom: 1px solid #1f2a3d;
    padding-bottom: 0.35rem;
    margin: 1.4rem 0 0.8rem 0;
  }
  .pad-mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.8rem;
    color: #c9d1d9;
    word-break: break-all;
  }
  .pad-note {
    font-size: 0.82rem;
    color: #8b949e;
    line-height: 1.55;
  }
  .pad-intro {
    font-size: 0.92rem;
    color: #c9d1d9;
    line-height: 1.6;
    max-width: 48rem;
  }
  .pad-sidebar-label {
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #58a6ff;
    margin: 0.8rem 0 0.3rem 0;
    font-weight: 600;
  }
  /* Sidebar navigation grouping.
     One radio still drives navigation: the group headings are drawn above
     fixed option positions rather than by splitting the control into three,
     so there remains exactly one navigation widget and one navigation value.
     Anchored on Streamlit's own ``stRadioGroup``/``stRadioOption`` test ids
     and scoped to the sidebar, so the radios inside a view -- Analytics has
     one -- are untouched.  The ordinals are the first option of each group,
     and a test pins them to PRIMARY_PAGES and ADVANCED_PAGES so a reordered
     navigation cannot leave a heading floating over the wrong item. */
  [data-testid="stSidebar"] [data-testid=stRadioOption] {
    position: relative;
  }
  [data-testid="stSidebar"] [data-testid=stRadioOption]::before {
    position: absolute;
    top: -1.15rem;
    left: 0;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    color: #8b949e;
    white-space: nowrap;
  }
  [data-testid="stSidebar"] [data-testid=stRadioGroup]
    > [data-testid=stRadioOption]:nth-child(1) {
    margin-top: 1.4rem;
  }
  [data-testid="stSidebar"] [data-testid=stRadioGroup]
    > [data-testid=stRadioOption]:nth-child(1)::before {
    content: "PRIMARY";
  }
  [data-testid="stSidebar"] [data-testid=stRadioGroup]
    > [data-testid=stRadioOption]:nth-child(7) {
    margin-top: 1.9rem;
  }
  [data-testid="stSidebar"] [data-testid=stRadioGroup]
    > [data-testid=stRadioOption]:nth-child(7)::before {
    content: "ADVANCED";
  }
  [data-testid="stSidebar"] [data-testid=stRadioGroup]
    > [data-testid=stRadioOption]:nth-child(11) {
    margin-top: 1.9rem;
  }
  [data-testid="stSidebar"] [data-testid=stRadioGroup]
    > [data-testid=stRadioOption]:nth-child(11)::before {
    content: "ABOUT";
  }
  div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
</style>
"""


def card(label: str, value: str, *, accent: str = ACCENT, note: str = "") -> str:
    """Return one compact metric card.

    Every argument is escaped. *accent* is a colour the caller chose from this
    module's own palette or from
    :data:`~password_attack_detector.dashboard.formatting.SEVERITY_COLORS`, and
    it is escaped too rather than trusted -- a caller passing a value through
    from a response would otherwise be writing into a style attribute.
    """
    note_block = f'<div class="pad-card-note">{escape_text(note)}</div>' if note else ""
    return (
        f'<div class="pad-card" style="border-top:2px solid {escape_text(accent)}">'
        f'<div class="pad-card-label">{escape_text(label)}</div>'
        f'<div class="pad-card-value">{escape_text(value)}</div>'
        f"{note_block}</div>"
    )


def badge(text: str, *, color: str, caps: bool = True) -> str:
    """Return one status badge in the given accent colour.

    *caps* is the presentation of the label only, never its value. Severity and
    verdict badges keep the shouted form -- ``CRITICAL`` is meant to be read
    across a room -- while the header's connectivity badges are ordinary words
    an operator reads at their desk, and are rendered ``Online`` rather than
    ``ONLINE``. Neither choice touches the string a caller passed in.
    """
    classes = "pad-badge" if caps else "pad-badge pad-badge-plain"
    return (
        f'<span class="{classes}" style="background:{escape_text(color)}22;'
        f'color:{escape_text(color)};border:1px solid {escape_text(color)}55">'
        f"{escape_text(text)}</span>"
    )


def chip(text: str) -> str:
    """Return one monospaced chip, for a rule identifier or a short code."""
    return f'<span class="pad-chip">{escape_text(text)}</span>'


def section_title(text: str) -> str:
    """Return a section heading in the console's rule."""
    return f'<div class="pad-section">{escape_text(text)}</div>'
