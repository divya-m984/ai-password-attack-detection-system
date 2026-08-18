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
    "STYLESHEET",
    "SURFACE",
    "TEXT_MUTED",
    "badge",
    "card",
    "chip",
    "section_title",
]

BACKGROUND: Final[str] = "#0b0f19"
SURFACE: Final[str] = "#131a29"
ACCENT: Final[str] = "#58a6ff"
TEXT_MUTED: Final[str] = "#8b949e"

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
    padding: 0.25rem 0 0.85rem 0;
    margin-bottom: 1.1rem;
  }
  .pad-title {
    font-size: 1.45rem;
    font-weight: 650;
    letter-spacing: 0.01em;
    color: #e6edf3;
    margin: 0;
  }
  .pad-subtitle {
    font-size: 0.86rem;
    color: #8b949e;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin: 0.2rem 0 0 0;
  }
  .pad-card {
    background: #131a29;
    border: 1px solid #1f2a3d;
    border-left: 3px solid #58a6ff;
    border-radius: 4px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
  }
  .pad-card-label {
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.3rem;
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
    margin-top: 0.25rem;
  }
  .pad-badge {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .pad-chip {
    display: inline-block;
    padding: 0.1rem 0.5rem;
    margin: 0.1rem 0.25rem 0.1rem 0;
    border-radius: 3px;
    border: 1px solid #1f2a3d;
    background: #0f1626;
    color: #c9d1d9;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.74rem;
  }
  .pad-section {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #58a6ff;
    border-bottom: 1px solid #1f2a3d;
    padding-bottom: 0.3rem;
    margin: 1.1rem 0 0.7rem 0;
  }
  .pad-mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.8rem;
    color: #c9d1d9;
    word-break: break-all;
  }
  .pad-note {
    font-size: 0.8rem;
    color: #8b949e;
    line-height: 1.5;
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
        f'<div class="pad-card" style="border-left-color:{escape_text(accent)}">'
        f'<div class="pad-card-label">{escape_text(label)}</div>'
        f'<div class="pad-card-value">{escape_text(value)}</div>'
        f"{note_block}</div>"
    )


def badge(text: str, *, color: str) -> str:
    """Return one status badge in the given accent colour."""
    return (
        f'<span class="pad-badge" style="background:{escape_text(color)}22;'
        f'color:{escape_text(color)};border:1px solid {escape_text(color)}55">'
        f"{escape_text(text)}</span>"
    )


def chip(text: str) -> str:
    """Return one monospaced chip, for a rule identifier or a short code."""
    return f'<span class="pad-chip">{escape_text(text)}</span>'


def section_title(text: str) -> str:
    """Return a section heading in the console's rule."""
    return f'<div class="pad-section">{escape_text(text)}</div>'
