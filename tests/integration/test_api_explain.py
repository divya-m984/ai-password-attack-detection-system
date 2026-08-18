"""Attribution over HTTP, against the same frozen champion detection runs on.

The session fixture in ``conftest.py`` is deliberately shared with the detection
suite, so ``/api/v1/explain`` and ``/api/v1/detect`` here are demonstrably the
same model, the same preprocessor, and the same window -- which is what lets the
central assertion below mean anything: the decomposition reconstructs the
decision the *scoring path* produced, not a decision this endpoint made up.

What is checked:

* the contributions add up to the model's own decision quantity;
* the response is bounded, ranked, and honest about what it dropped;
* it decomposes the decision function and never the calibrated probability;
* it is a read -- the same window explained twice gives the same answer, and no
  artifact under the deployment changes;
* nothing in it is a path, a coefficient, a pseudonym, or a split.
"""

from __future__ import annotations

import hashlib
from typing import Any

from password_attack_detector.api.config import APISettings
from password_attack_detector.ml.explain import RECONSTRUCTION_TOLERANCE
from tests.api.factories import brute_force_window, event, normal_window


def _explain(client: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Post a window to the attribution endpoint and return the document."""
    response = client.post("/api/v1/explain", json={"events": events})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------


def test_an_explanation_is_available_for_the_frozen_champion(client: Any) -> None:
    """This champion's family has an exact method, so an explanation exists."""
    body = _explain(client, brute_force_window(20))
    assert body["available"] is True
    assert body["unavailable_reason"] is None
    assert body["method"] in {
        "linear_logit_contribution",
        "tree_path_contribution",
        "single_feature_step_contribution",
    }
    assert body["model_family"]
    assert body["contributions"]


def test_the_contributions_reconstruct_the_model_s_own_decision(client: Any) -> None:
    """The residual is reported, and it is inside the declared tolerance.

    This is the property that separates an attribution from a plausible-looking
    table of numbers: the parts add up to the quantity the model actually
    produced. The service refuses to publish one that does not.
    """
    body = _explain(client, brute_force_window(20))
    assert abs(body["reconstruction_residual"]) <= RECONSTRUCTION_TOLERANCE


def test_the_explained_anchor_is_the_one_detection_scores(client: Any) -> None:
    """Both endpoints answer about the same row of the same window."""
    events = brute_force_window(20)
    explained = _explain(client, events)
    detected = client.post("/api/v1/detect", json={"events": events}).json()
    assert explained["anchor_event_id"] == detected["anchor"]["anchor_event_id"]
    assert explained["anchor_event_time"] == detected["anchor"]["anchor_event_time"]


def test_the_decomposition_is_of_the_decision_function_not_the_probability(
    client: Any,
) -> None:
    """A calibrated probability is bounded in [0, 1]; a decision value is not.

    The two are different quantities, and conflating them is the single most
    likely way an explanation endpoint starts lying. The contributions sum to
    ``decision_value``; the probability, where the lineage has one, is reported
    only by ``/api/v1/detect`` and is never summed toward here.
    """
    events = brute_force_window(20)
    body = _explain(client, events)
    ml = client.post("/api/v1/detect", json={"events": events}).json()["anchor"]["ml"]
    assert body["score_kind"] == ml["score_kind"]
    total = body["baseline_value"] + sum(
        item["contribution"] for item in body["contributions"]
    )
    if body["omitted_contribution_count"] == 0:
        assert abs(total - body["decision_value"]) <= 1e-6
    if ml["probability"] is not None:
        # Reported by detection, and absent from the decomposition entirely.
        assert "probability" not in body


# ---------------------------------------------------------------------------
# Bounded, ranked, and honest about the bound
# ---------------------------------------------------------------------------


def test_the_reported_and_omitted_contributions_account_for_every_column(
    client: Any,
) -> None:
    """A truncated table that did not say so would read as the whole story."""
    body = _explain(client, brute_force_window(20))
    assert (
        len(body["contributions"]) + body["omitted_contribution_count"]
        == (body["transformed_feature_count"])
    )
    assert body["transformed_feature_count"] > 0


def test_the_contributions_are_ranked_by_magnitude(client: Any) -> None:
    """What is dropped is always the part that moved the decision least."""
    body = _explain(client, brute_force_window(20))
    magnitudes = [abs(item["contribution"]) for item in body["contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_the_response_is_bounded_however_many_columns_there_are(client: Any) -> None:
    """A per-row export of the fitted function's shape is not an explanation."""
    from password_attack_detector.api.schemas import MAX_REPORTED_CONTRIBUTIONS

    body = _explain(client, brute_force_window(20))
    assert len(body["contributions"]) <= MAX_REPORTED_CONTRIBUTIONS


def test_each_column_is_credited_at_most_once(client: Any) -> None:
    """A repeated column would double-count in any reading of the table."""
    body = _explain(client, brute_force_window(20))
    names = [item["transformed_feature"] for item in body["contributions"]]
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# It is a read
# ---------------------------------------------------------------------------


def test_the_same_window_explains_identically(client: Any) -> None:
    """Determinism: nothing here samples, permutes, or reads a clock."""
    events = brute_force_window(20)
    assert _explain(client, events) == _explain(client, events)


def test_explaining_changes_nothing_under_the_deployment(
    client: Any, served: APISettings
) -> None:
    """Every artifact byte is identical before and after. Attribution writes none."""
    root = served.artifact_root
    assert root is not None

    def snapshot() -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    _explain(client, brute_force_window(20))
    _explain(client, normal_window(12))
    assert snapshot() == before


def test_two_different_windows_explain_differently(client: Any) -> None:
    """The decomposition is of *this* row, not a constant the model carries."""
    burst = _explain(client, brute_force_window(30))
    quiet = _explain(client, normal_window(12))
    assert burst["decision_value"] != quiet["decision_value"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_window_that_selects_many_anchors_is_refused(client: Any) -> None:
    """One explanation answers one row; 'all' is not a selection this can answer."""
    response = client.post(
        "/api/v1/explain",
        json={"events": brute_force_window(6), "anchor_selection": "all"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "API011"


def test_a_credential_field_is_refused_before_anything_is_explained(
    client: Any,
) -> None:
    """The same refusal the detection endpoints make, on the same schema."""
    events = brute_force_window(4)
    events[0]["password"] = "irrelevant"
    response = client.post("/api/v1/explain", json={"events": events})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "API013"


def test_a_misordered_window_is_refused(client: Any) -> None:
    """Attribution inherits every window rule; it restates none of them."""
    events = [event("late", offset_seconds=90.0), event("early", offset_seconds=0.0)]
    response = client.post("/api/v1/explain", json={"events": events})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "API004"


# ---------------------------------------------------------------------------
# Disclosure
# ---------------------------------------------------------------------------


def test_an_explanation_carries_no_feature_value(client: Any) -> None:
    """Phase 5 gates value disclosure behind review; the wire surface has no gate.

    A transformed value can be a country code, so the response schema has no
    field for one at all rather than a field that defaults to off.
    """
    body = _explain(client, brute_force_window(20))
    for item in body["contributions"]:
        assert set(item) == {"transformed_feature", "contribution"}


def test_an_explanation_names_no_location_identity_or_split(client: Any) -> None:
    """The sweep every serving document gets, applied to the widest one."""
    events = brute_force_window(20)
    text = client.post("/api/v1/explain", json={"events": events}).text
    for identifier in ("user_id", "source_id", "device_id", "session_id"):
        assert events[0][identifier] not in text
    for forbidden in ("/home/", "/tmp/", ".parquet", ".yaml", "Traceback"):
        assert forbidden not in text
    for split in ("train", "validation", "holdout", '"test"'):
        assert split not in text
    for term in ("password", "secret", "token", "credential", "coefficient"):
        assert term not in text.lower()


def test_the_explanation_reports_no_threshold(client: Any) -> None:
    """The operating point belongs to the decision, not to the decomposition.

    Reporting it here would invite exactly the comparison the decomposition
    cannot support: the contributions sum to a logit, and the threshold sits on
    a calibrated probability.
    """
    body = _explain(client, brute_force_window(20))
    for forbidden in ("decision_threshold", "threshold", "flagged", "probability"):
        assert forbidden not in body
