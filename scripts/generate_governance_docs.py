"""Regenerate the tracked Phase 5 governance documents.

Two files, both generated and neither hand-edited::

    docs/model-card.md         the Phase 5 model card
    docs/phase5-acceptance.md  the final Phase 5 acceptance report

Run from the repository root::

    uv run python scripts/generate_governance_docs.py

A test asserts the tracked copies are byte-identical to what this produces, so
editing one by hand fails the build rather than drifting quietly.

The acceptance report written here is the **contract-derived** one: it carries
no pipeline artifacts, so every requirement that needs one is recorded as
``inconclusive``.  That is deliberate.  Baking a particular run's fingerprints
into a tracked document would make the document unregenerable by anybody who did
not have that run, and the integration suite already proves that a real
CI-sized pipeline turns each of those requirements into a pass or a genuine
not-applicable.

This is a maintenance script, not a command.  The public Phase 5 command surface
is fixed, and a governance document is not something a user runs.
"""

from __future__ import annotations

from pathlib import Path

from password_attack_detector import __version__
from password_attack_detector.ml.governance import (
    acceptance_report_to_markdown,
    build_acceptance_report,
    model_card_to_markdown,
)


def main() -> None:
    """Write both governance documents under ``docs/``."""
    docs = Path(__file__).resolve().parents[1] / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    card = docs / "model-card.md"
    card.write_text(model_card_to_markdown(), encoding="utf-8")
    print(f"Wrote {card.relative_to(docs.parent)}")

    report = build_acceptance_report(package_version=__version__)
    acceptance = docs / "phase5-acceptance.md"
    acceptance.write_text(acceptance_report_to_markdown(report), encoding="utf-8")
    print(f"Wrote {acceptance.relative_to(docs.parent)}")
    print(
        f"  {report.passed} pass, {report.failed} fail, "
        f"{report.inconclusive} inconclusive, "
        f"{report.not_applicable} not applicable"
    )


if __name__ == "__main__":
    main()
