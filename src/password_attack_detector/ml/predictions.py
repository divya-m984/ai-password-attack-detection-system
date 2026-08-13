"""Batch inference under a frozen champion, and the rows it emits.

**Prediction is not evaluation.**  Everything in this module produces model
output for rows whose outcome nobody here has looked at.  There is no parameter,
no keyword, and no code path by which a label reaches any of it: the inference
input is assembled by :func:`~password_attack_detector.ml.dataset.\
load_inference_dataset`, whose signature has no label argument at all, and the
rows that come out carry a score, a decision, and a join key.  Whether the
decision was *right* is a question a later milestone asks, once, against the
frozen artifacts published here.

That distinction is what makes scoring the test split safe.  A test prediction
changes nothing about the model, the calibrator, or the operating point --
every one of them was fixed by the champion lock before this module ran, and
this module refuses to run without one.  Changing a test label cannot change a
byte of what is published, because no test label was opened.

**Everything is verified before anything is scored.**  A champion lock names
seventeen things, and :meth:`FrozenChampion.load` checks each of them against
the artifact it names rather than trusting the lock's own account of itself.
The order below is the order they run in, and each step only assumes what the
previous ones established:

===  ===================================  ==========================================
1    the lock file parses at this          a lock written under a contract this build
     contract version                      does not implement
2    the lock recomputes its own digest    a hand-edited lock
3    the scope directory names the lock's  a lock moved into another scope's
     own scope key                         directory
4    a champion_freeze receipt in the      a lock nobody recorded freezing
     ledger names this lock
5    the validation_selection it came      a lock pointing at a selection that is not
     from is on record and agrees          the one the ledger holds
6    the selected training run is on       a champion whose run left the history
     record and its published receipt
     matches
7    the run's model identity matches      a lock naming a different model than the
                                           run it cites
8    the model manifest's bytes digest     a manifest replaced after the freeze
     to the recorded fingerprint
9    the model artifact verifies in full   every structural and integrity check
     (Milestone 4)                         Milestone 4 performs
10   serializer and inference-adapter      a reader that cannot read this writer
     contract
11   the preprocessor fingerprint          a matrix built by different rules
12   the calibrator fingerprint, where     probabilities from a different calibrator
     the lineage has one
13   the binary threshold fingerprint      an operating point swapped after freezing
14   the feature catalog fingerprint       a model scored against a different feature
                                           catalog
15   the reviewed allowlist fingerprint    a feature contract nobody reviewed
16   the ordered eligible-feature          the right columns in a different order
     fingerprint
17   the ML config fingerprint and the     a runtime whose estimator internals may
     declared dependency contract          have moved since the arrays were extracted
===  ===================================  ==========================================

There is no ``--force``, no ``--ignore-lock``, and no way to hand this module a
model path or a model identifier in place of a lock.  Each refusal below is a
state in which the published predictions would not be attributable to anything
in particular, and an override would be a way to publish them anyway.

**Category triage is downstream of the binary decision.**  The category head was
fitted on known-malicious training rows only, so it has never been shown a
benign row.  Asking it about a row the binary champion did not flag produces a
number and not a finding, and publishing that number beside the ones that mean
something would make the two indistinguishable.  So the category artifact
contains **only the rows the binary head flagged**, and two states that look
alike are kept permanently apart:

* **not applicable** -- the binary head did not route this row to triage.  The
  row is absent from the category artifact entirely.  Nothing was asked, so
  nothing was answered.
* ``unknown`` -- the binary head *did* route this row to triage, the category
  head was asked, and its best class score fell below the frozen abstention
  floor.  That is a measured abstention on a row that reached the head.

Collapsing the first into the second would inflate the abstention rate with rows
the head never saw, and would let a benign-looking dataset read as a category
head that constantly refuses to commit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, model_validator

from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    ModelNotReadyError,
)
from password_attack_detector.ml.calibration import CalibrationState, canonical_json
from password_attack_detector.ml.catalog import MODEL_CATALOG, ModelCatalog
from password_attack_detector.ml.champion import (
    CHAMPION_DIR,
    CHAMPION_LOCK_FILE,
    ChampionLock,
)
from password_attack_detector.ml.config import MLConfig
from password_attack_detector.ml.dataset import InferenceDataset, InferenceFrame
from password_attack_detector.ml.dependencies import (
    dependency_contract_fingerprint,
    installed_version,
    sklearn_compatible,
)
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    CalibrationMethod,
    MLSplit,
    MLTask,
    ScoreKind,
    SelectionStatus,
    is_probability,
)
from password_attack_detector.ml.experiments import (
    CALIBRATION_DIR,
    MODEL_DIR,
    RUNS_DIR,
    THRESHOLD_DIR,
    TRAINING_RUN_FILE,
)
from password_attack_detector.ml.imbalance import BINARY_CLASS_ORDER
from password_attack_detector.ml.inference import InferenceModel, ModelCompatibility
from password_attack_detector.ml.ledger import ExperimentLedger, TrainingRunRecord
from password_attack_detector.ml.thresholds import (
    AnomalyThresholdSelection,
    CategoryAbstentionSelection,
    ThresholdSelection,
    assign_category,
    flagged_anomalous,
    flagged_malicious,
)

__all__ = [
    "ANOMALY_PREDICTION_COLUMNS",
    "BINARY_PREDICTION_COLUMNS",
    "CATEGORY_PREDICTION_COLUMNS",
    "PREDICTION_SCHEMA_VERSION",
    "PROHIBITED_PREDICTION_COLUMNS",
    "AnomalyScore",
    "BinaryPrediction",
    "CategoryPrediction",
    "ExperimentalAnomalyRun",
    "FrozenCategoryModel",
    "FrozenChampion",
    "category_applicable_anchors",
    "category_scores_payload",
    "predict_anomaly",
    "predict_binary",
    "predict_category",
    "verify_inference_feature_contract",
]

#: The prediction contract's own version.  Separate from the model contract,
#: the threshold contract, and the freeze contract: what a prediction row
#: carries can change without any of those changing, and the reverse.
PREDICTION_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Columns a binary prediction table carries, in the order it carries them.
BINARY_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "anchor_event_id",
    "anchor_event_time",
    "malicious_decision_score",
    "malicious_probability",
    "decision_threshold",
    "flagged_malicious",
)

#: Columns a category prediction table carries, in the order it carries them.
CATEGORY_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "anchor_event_id",
    "anchor_event_time",
    "predicted_scenario",
    "category_scores_json",
    "max_category_score",
    "min_category_score",
)

#: Columns an experimental anomaly table carries, in the order it carries them.
#:
#: No probability column, and not because none is written -- because none
#: exists.  Thresholding an unsupervised magnitude does not turn it into a
#: likelihood, and a nullable column named for one would be filled in by
#: somebody eventually.
ANOMALY_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "anchor_event_id",
    "anchor_event_time",
    "anomaly_score",
    "anomaly_threshold",
    "flagged_anomalous",
)

#: Columns no prediction table may carry.
#:
#: The join key is deliberately absent from this list: a prediction row exists
#: to be joined to an outcome later, and ``anchor_event_id`` is the whole
#: mechanism by which that happens.  Everything below is context the prediction
#: does not need and must not republish -- entity pseudonyms, campaign identity,
#: coordinates, credentials, the raw feature vector, and every form of ground
#: truth.
PROHIBITED_PREDICTION_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        # ground truth, in every spelling the project uses
        "label",
        "labels",
        "target",
        "malicious",
        "attack_class",
        "attack_label",
        "is_attack",
        "scenario",
        "scenario_variant",
        "y_true",
        "supervised_training_eligible",
        # split and campaign membership
        "split",
        "exclusion_reason",
        "campaign",
        "campaign_id",
        "campaign_stage",
        # entity scope and pseudonyms
        "user_id",
        "source_id",
        "device_id",
        "session_id",
        "scope_value",
        "user_scope",
        "source_scope",
        # coordinates
        "latitude",
        "longitude",
        "coordinates",
        # credentials
        "password",
        "secret",
        "token",
        "credential",
        "api_key",
        "private_key",
        # Phase 4 output and the fused verdict, which belongs to a later
        # milestone and must not appear in an artifact this one publishes.
        "risk_score",
        "fused_flagged",
    }
)


def _require(condition: bool, message: str) -> None:
    """Raise a sanitized refusal unless *condition* holds."""
    if not condition:
        raise ModelNotReadyError(message)


def _finite(value: float, what: str) -> float:
    """Return *value* when it is finite, else refuse."""
    if not math.isfinite(float(value)):
        raise ValueError(f"{what} must be finite")
    return float(value)


# ---------------------------------------------------------------------------
# Row schemas
# ---------------------------------------------------------------------------


class BinaryPrediction(BaseModel):
    """One row's malicious decision under the frozen operating point.

    Carries both the score the threshold was applied to and the threshold
    itself, so the decision can be recomputed from the row rather than taken on
    trust.  It carries *no* lineage fingerprint: those are identical for every
    row in a publication and live on the manifest, where they can be checked
    once instead of a hundred thousand times.

    ``malicious_probability`` exists only when a verified calibrator produced
    it.  An uncalibrated score is never relabelled a probability, and the
    absence is a null rather than a copy of the decision score.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_event_id: str
    anchor_event_time: datetime
    #: What the threshold was compared against: the raw decision score, or the
    #: calibrated probability.  Named on the row so a reader never has to infer
    #: it from which column happens to be populated.
    score_kind: ScoreKind
    malicious_decision_score: float
    malicious_probability: float | None
    decision_threshold: float
    flagged_malicious: bool

    @model_validator(mode="after")
    def check_row(self) -> Self:
        """The stored decision must be the one the stored numbers imply."""
        _finite(self.malicious_decision_score, "malicious_decision_score")
        _finite(self.decision_threshold, "decision_threshold")
        if self.score_kind not in {
            ScoreKind.DECISION_SCORE,
            ScoreKind.CALIBRATED_PROBABILITY,
        }:
            raise ValueError(
                f"a binary prediction is decided on a decision score or a "
                f"calibrated probability, not {str(self.score_kind)!r}"
            )
        calibrated = is_probability(self.score_kind)
        if calibrated and self.malicious_probability is None:
            raise ValueError(
                "a row decided on a calibrated probability must carry the "
                "probability it was decided on"
            )
        if not calibrated and self.malicious_probability is not None:
            raise ValueError(
                "a row decided on a raw decision score carries no probability; "
                "an uncalibrated score is not a probability under another name"
            )
        if self.malicious_probability is not None:
            value = _finite(self.malicious_probability, "malicious_probability")
            if not 0.0 <= value <= 1.0:
                raise ValueError("malicious_probability lies outside [0, 1]")
        if self.tzinfo_missing:
            raise ValueError("anchor_event_time must be timezone-aware")
        decided = self.decided_score
        if self.flagged_malicious != flagged_malicious(
            decided, threshold=self.decision_threshold
        ):
            raise ValueError(
                "flagged_malicious contradicts the frozen predicate applied to "
                "this row's own score and threshold"
            )
        return self

    @property
    def tzinfo_missing(self) -> bool:
        """Return whether the anchor time carries no timezone."""
        return (
            self.anchor_event_time.tzinfo is None
            or self.anchor_event_time.utcoffset() is None
        )

    @property
    def decided_score(self) -> float:
        """Return the score the frozen threshold was applied to."""
        if is_probability(self.score_kind):
            assert self.malicious_probability is not None  # checked above
            return self.malicious_probability
        return self.malicious_decision_score


class CategoryPrediction(BaseModel):
    """One triage result for a row the binary champion flagged.

    **Every row of this schema is category-applicable by construction.** A row
    exists here only because the binary head flagged its anchor, so the table's
    membership *is* the applicability record and there is no per-row flag to
    disagree with it.

    ``predicted_scenario`` is a known class name or
    :data:`~password_attack_detector.ml.enums.UNKNOWN_CATEGORY`.  Abstention is
    an outcome, not a failure: a row whose best class score falls below the
    frozen threshold has no category the head is willing to claim, and saying so
    is more useful than the nearest guess.  It never means "the binary head said
    benign" -- that row is not in this table at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_event_id: str
    anchor_event_time: datetime
    predicted_scenario: str
    #: Canonical JSON: every declared class, in the frozen class order, mapped
    #: to a finite score.  A map rather than one column per class, because the
    #: class space is a property of the frozen head and a fixed column set would
    #: make a differently-classed head unpublishable.
    category_scores_json: str
    max_category_score: float
    min_category_score: float

    @model_validator(mode="after")
    def check_row(self) -> Self:
        """The assignment must be the one the stored scores and floor imply."""
        _finite(self.max_category_score, "max_category_score")
        _finite(self.min_category_score, "min_category_score")
        scores = self.class_scores()
        if not scores:
            raise ValueError("a category prediction carries at least one class score")
        class_order = tuple(scores)
        assigned, best = assign_category(
            [scores[name] for name in class_order],
            class_order,
            min_category_score=self.min_category_score,
        )
        if best != self.max_category_score:
            raise ValueError(
                "max_category_score is not the largest score this row records"
            )
        if assigned != self.predicted_scenario:
            raise ValueError(
                "predicted_scenario contradicts the frozen abstention rule "
                "applied to this row's own scores and floor"
            )
        return self

    def class_scores(self) -> dict[str, float]:
        """Return the decoded class-score map, or raise.

        Bounded and strict: the payload must be a JSON object of finite numbers
        whose keys are in the deterministic order they were written in.  A
        prediction table is untrusted data during validation, and a class map is
        the one free-form field on the row.
        """
        if len(self.category_scores_json) > _MAX_CATEGORY_PAYLOAD_BYTES:
            raise ValueError("the category score payload exceeds its size ceiling")
        try:
            payload = json.loads(self.category_scores_json)
        except json.JSONDecodeError:
            raise ValueError("category_scores_json is not valid JSON") from None
        if not isinstance(payload, dict):
            raise ValueError("category_scores_json is not a JSON object")
        if len(payload) > _MAX_CATEGORY_CLASSES:
            raise ValueError("the category score map declares too many classes")
        scores: dict[str, float] = {}
        for name, value in payload.items():
            if not isinstance(name, str) or not name:
                raise ValueError("a category score map key must be a class name")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"the score for class {name!r} is not a number")
            scores[name] = _finite(float(value), f"the score for class {name!r}")
        if list(scores) != sorted(scores):
            raise ValueError(
                "the category score map is not in the deterministic class order"
            )
        if UNKNOWN_CATEGORY in scores:
            raise ValueError(
                f"{UNKNOWN_CATEGORY!r} is the abstention outcome, never a scored class"
            )
        return scores


class AnomalyScore(BaseModel):
    """One experimental anomaly magnitude, kept apart from the champion.

    Never a probability, before or after thresholding.  ``experimental`` and
    ``influences_champion_selection`` are pinned rather than configurable, so a
    row of this shape cannot be mistaken for supervised output however it is
    later joined.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_event_id: str
    anchor_event_time: datetime
    anomaly_score: float
    anomaly_threshold: float | None
    flagged_anomalous: bool | None
    experimental: bool = True
    influences_champion_selection: bool = False

    @model_validator(mode="after")
    def check_row(self) -> Self:
        """A flag exists exactly when a frozen threshold does, and agrees with it."""
        _finite(self.anomaly_score, "anomaly_score")
        if not self.experimental:
            raise ValueError("an anomaly row is permanently experimental")
        if self.influences_champion_selection:
            raise ValueError(
                "the anomaly probe never influences champion selection; a "
                "measurement that can change what it measures is not one"
            )
        if (self.anomaly_threshold is None) != (self.flagged_anomalous is None):
            raise ValueError(
                "an anomaly flag and the threshold that produced it are recorded "
                "together, or neither is"
            )
        if self.anomaly_threshold is None:
            return self
        threshold = _finite(self.anomaly_threshold, "anomaly_threshold")
        if self.flagged_anomalous != flagged_anomalous(
            self.anomaly_score, threshold=threshold
        ):
            raise ValueError(
                "flagged_anomalous contradicts the frozen predicate applied to "
                "this row's own score and threshold"
            )
        return self


#: Ceilings applied to the one free-form field on a prediction row.  A category
#: map is small by construction -- the class space is the scenario vocabulary --
#: so anything larger is a malformed or hostile artifact rather than a big one.
_MAX_CATEGORY_PAYLOAD_BYTES: Final[int] = 8192
_MAX_CATEGORY_CLASSES: Final[int] = 64


def category_scores_payload(scores: Sequence[float], class_order: Sequence[str]) -> str:
    """Return the canonical JSON class-score map for one row.

    Every declared class is present, none that was not declared is, and the key
    order is the frozen deterministic one.  Values are stored **exactly**: the
    largest of them is compared against the frozen abstention floor, and a
    rounded score would move whichever row sits on that boundary.  Full-precision
    floats round-trip exactly through canonical JSON, so the payload a row
    carries is still the payload a re-run produces.
    """
    if len(scores) != len(class_order):
        raise ValueError("class scores and the class order disagree in width")
    payload = {
        name: float(value) for name, value in zip(class_order, scores, strict=True)
    }
    return canonical_json(payload)


# ---------------------------------------------------------------------------
# The frozen champion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrozenCategoryModel:
    """The verified category head a lock bound alongside its binary champion."""

    model: InferenceModel
    abstention: CategoryAbstentionSelection
    class_order: tuple[str, ...]
    run_id: str

    @property
    def min_category_score(self) -> float:
        """Return the frozen abstention floor, exactly as Milestone 5 stored it.

        Not rounded: the floor is a comparison boundary, and moving it by a
        rounding step moves whichever row sits on it from a known class to
        ``unknown``.
        """
        return self.abstention.min_category_score


@dataclass(frozen=True, slots=True)
class FrozenChampion:
    """Everything a batch inference is permitted to run, verified end to end.

    Constructed only by :meth:`load`, and only from a champion lock.  There is
    no constructor argument for a model directory or a model identifier: a
    prediction attributable to "whichever model was in this folder" is not
    attributable to anything.
    """

    lock: ChampionLock
    binary: InferenceModel
    threshold: ThresholdSelection
    calibrator: CalibrationState | None
    category: FrozenCategoryModel | None
    scope_key: str
    freeze_record_id: str

    @property
    def score_kind(self) -> ScoreKind:
        """Return the score kind the frozen threshold was selected against."""
        return self.threshold.score_kind

    @property
    def decision_threshold(self) -> float:
        """Return the frozen operating point."""
        assert self.threshold.selected_threshold is not None  # checked at load
        return self.threshold.selected_threshold

    @property
    def compatibility(self) -> ModelCompatibility:
        """Return the feature contract this champion must be scored under."""
        return ModelCompatibility(
            eligible_feature_list_fingerprint=(
                self.lock.eligible_feature_list_fingerprint
            ),
            preprocessor_fingerprint=self.lock.preprocessor_fingerprint,
            feature_catalog_fingerprint=self.lock.feature_catalog_fingerprint,
            allowlist_fingerprint=self.lock.allowlist_fingerprint,
        )

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        ledger: ExperimentLedger,
        scope_key: str | None = None,
        catalog: ModelCatalog = MODEL_CATALOG,
        config: MLConfig | None = None,
    ) -> FrozenChampion:
        """Verify the frozen champion under *root* and return it, or refuse.

        Args:
            root: the artifact root holding ``champion/``, ``runs/``, and the
                ledger.
            ledger: the append-only experiment ledger, consulted for the freeze
                receipt, the selection, and the selected run.
            scope_key: which frozen scope to load.  Required when more than one
                champion is frozen under *root*, because picking whichever came
                first would make the prediction depend on directory order.
            catalog: the reviewed model catalog.
            config: the ML configuration this runtime is operating under.  When
                supplied, its declared dependency ranges are checked against the
                lock as well.  Omitting it skips that one check and nothing
                else, which is honest -- the lock is still checked against the
                run that produced it.

        Raises:
            ArtifactNotFoundError: no lock is frozen under *root*.
            ModelNotReadyError: any step of the verification chain failed.
        """
        lock, resolved_scope = _read_lock(Path(root), scope_key)

        # Step 4. A lock nobody recorded freezing is a file, not a decision.
        freezes = [
            record
            for record in ledger.champion_freezes()
            if record.scope_key == resolved_scope
        ]
        _require(
            bool(freezes),
            "no champion-freeze record in the ledger names this scope; a lock "
            "the history does not record is not a frozen champion",
        )
        receipts = [
            record
            for record in freezes
            if record.champion_lock_fingerprint == lock.lock_fingerprint
        ]
        _require(
            bool(receipts),
            "the frozen lock and the ledger's freeze receipt disagree about "
            "which champion was frozen",
        )
        freeze = receipts[0]

        # Step 5. The selection the freeze came from, on record and agreeing.
        selection = ledger.read_selection(lock.validation_selection_id)
        _require(
            selection.record_fingerprint == lock.validation_selection_fingerprint,
            "the lock and the recorded validation selection disagree; the lock "
            "names a selection whose content has changed",
        )
        _require(
            selection.record_id == freeze.validation_selection_id,
            "the freeze receipt and the lock name different validation selections",
        )

        # Step 6-7. The selected run, on record, on disk, and naming this model.
        run, directory = _read_run(Path(root), lock.training_run_id, ledger=ledger)
        _require(
            run.model_id == lock.model_id,
            "the published run names a different model than the lock froze",
        )
        _require(
            run.identity.model_content_fingerprint == lock.model_content_fingerprint,
            "the published run and the lock disagree about the champion's content",
        )
        spec = catalog.for_family(lock.model_family)
        _require(
            not spec.reference_baseline,
            "the reference baseline is never a prediction champion; it is the "
            "comparator every candidate is measured against",
        )
        _require(
            not (spec.experimental or spec.anomaly_only),
            "an experimental or anomaly-only family is never a supervised champion",
        )

        # Step 8. The manifest's bytes, not the manifest's claims.
        _require(
            _file_digest(directory / MODEL_DIR / "model_manifest.json")
            == lock.model_manifest_fingerprint,
            "the champion's model manifest is not the one that was frozen",
        )

        # Steps 9-11 and 14-16. The Milestone 4 loader performs the whole
        # artifact verification and checks the feature contract it is handed.
        binary = _load_model(
            directory / MODEL_DIR,
            compatibility=ModelCompatibility(
                eligible_feature_list_fingerprint=(
                    lock.eligible_feature_list_fingerprint
                ),
                preprocessor_fingerprint=lock.preprocessor_fingerprint,
                feature_catalog_fingerprint=lock.feature_catalog_fingerprint,
                allowlist_fingerprint=lock.allowlist_fingerprint,
            ),
            require_champion_eligible=True,
        )
        _require(
            binary.task is MLTask.BINARY_MALICIOUS,
            "the frozen champion is not a binary model",
        )
        _require(
            binary.model_id == lock.model_id,
            "the loaded artifact derives a different model identifier than the "
            "lock froze",
        )
        _require(
            binary.document.model_content_fingerprint == lock.model_content_fingerprint,
            "the loaded artifact's content is not the content that was frozen",
        )
        # Step 10, restated against the lock rather than against the build: the
        # loader already refused an adapter it cannot run, and this refuses one
        # the lock did not name.
        _require(
            binary.document.serializer_id == lock.serializer_id
            and binary.document.serializer_version == lock.serializer_version
            and binary.document.inference_adapter_id == lock.inference_adapter_id,
            "the artifact's serializer or inference adapter is not the one the "
            "lock froze",
        )
        _require(
            tuple(binary.fitted.class_order) == BINARY_CLASS_ORDER,
            "the frozen champion's class order is not the binary class order; a "
            "score column read by position would be the wrong column",
        )

        # Steps 12-13. The calibrator and the operating point.
        calibrator = _load_calibrator(directory, lock=lock)
        threshold = _load_threshold(directory, lock=lock, calibrator=calibrator)

        # Step 17. The configuration the selection was carried out under, and
        # the reviewed dependency ranges the arrays were extracted under.
        _require(
            selection.ml_config_fingerprint == lock.ml_config_fingerprint,
            "the lock and its selection were carried out under different ML "
            "configurations",
        )
        _require(
            run.identity.dependency_contract_fingerprint
            == lock.dependency_contract_fingerprint,
            "the lock and the run it froze declare different dependency "
            "contracts; the ranges a later reader checks against would be ranges "
            "nobody reviewed for this model",
        )
        if config is not None:
            _require(
                dependency_contract_fingerprint(config.dependency_requirements)
                == lock.dependency_contract_fingerprint,
                "this configuration declares a different dependency contract "
                "than the champion was produced under",
            )
        if spec.requires_sklearn:
            _require(
                sklearn_compatible(installed_version("scikit-learn")),
                "the installed scikit-learn lies outside the reviewed range this "
                "champion was produced under",
            )

        category = _load_category_head(Path(root), lock=lock, ledger=ledger)
        return cls(
            lock=lock,
            binary=binary,
            threshold=threshold,
            calibrator=calibrator,
            category=category,
            scope_key=resolved_scope,
            freeze_record_id=freeze.record_id,
        )


def _read_lock(root: Path, scope_key: str | None) -> tuple[ChampionLock, str]:
    """Return the frozen lock under *root*, and the scope it occupies."""
    champion_root = root / CHAMPION_DIR
    if not champion_root.is_dir():
        raise ArtifactNotFoundError(
            "no champion has been frozen under this artifact root; run "
            "'ml freeze-champion' first, and if selection found no champion, "
            "that is the finding rather than a missing file"
        )
    scopes = sorted(
        directory.name
        for directory in champion_root.iterdir()
        if directory.is_dir() and (directory / CHAMPION_LOCK_FILE).is_file()
    )
    if not scopes:
        raise ArtifactNotFoundError("no champion lock is present under this root")
    if scope_key is None:
        _require(
            len(scopes) == 1,
            f"{len(scopes)} frozen champions are present; name one with a scope "
            f"key. Predicting under whichever came first would make the model "
            f"depend on directory enumeration order",
        )
        scope_key = scopes[0]
    elif scope_key not in scopes:
        raise ArtifactNotFoundError("no champion is frozen under that scope key")

    path = champion_root / scope_key / CHAMPION_LOCK_FILE
    # Steps 1-2. The sealed reader checks the contract version before it
    # validates, and the seal validator refuses a payload whose content and
    # digest disagree.
    lock = ChampionLock.from_json(path.read_text(encoding="utf-8"))
    # Step 3.
    _require(
        lock.scope_key == scope_key,
        "the lock's scope key does not match the directory it was found in",
    )
    return (lock, scope_key)


def _read_run(
    root: Path, run_id: str, *, ledger: ExperimentLedger
) -> tuple[TrainingRunRecord, Path]:
    """Return the published run *run_id* and its directory, or refuse."""
    directory = root / RUNS_DIR / run_id
    receipt = directory / TRAINING_RUN_FILE
    if not receipt.is_file():
        raise ArtifactNotFoundError(
            "the run the lock names has no published directory under this root"
        )
    stored = TrainingRunRecord.from_json(receipt.read_text(encoding="utf-8"))
    indexed = ledger.read(run_id)
    _require(
        stored.to_json() == indexed.to_json(),
        "a published run and its ledger record disagree; prediction refuses a "
        "champion whose own history contradicts it",
    )
    return (stored, directory)


def _load_model(
    directory: Path,
    *,
    compatibility: ModelCompatibility,
    require_champion_eligible: bool,
) -> InferenceModel:
    """Load and fully verify one published model directory."""
    return InferenceModel.load(
        directory,
        compatibility=compatibility,
        require_champion_eligible=require_champion_eligible,
    )


def _load_calibrator(directory: Path, *, lock: ChampionLock) -> CalibrationState | None:
    """Return the champion's calibrator, or ``None`` when it has none."""
    path = directory / CALIBRATION_DIR / "calibration_state.json"
    if lock.calibration_state_fingerprint is None:
        _require(
            lock.calibration_method is CalibrationMethod.NONE,
            "the lock names a calibration method without naming a calibrator",
        )
        _require(
            not path.is_file(),
            "an uncalibrated champion's run publishes a calibrator; the lock and "
            "the artifact disagree about whether a probability exists",
        )
        return None
    _require(
        path.is_file(),
        "the lock names a calibrator the published run does not carry",
    )
    state = CalibrationState.from_json(path.read_text(encoding="utf-8"))
    _require(
        state.calibration_state_fingerprint == lock.calibration_state_fingerprint,
        "the published calibrator is not the one the lock froze",
    )
    _require(
        state.method is lock.calibration_method,
        "the published calibrator was fitted by a different method than the "
        "lock records",
    )
    _require(
        state.model_content_fingerprint == lock.model_content_fingerprint,
        "the published calibrator was fitted against a different model",
    )
    return state


def _load_threshold(
    directory: Path, *, lock: ChampionLock, calibrator: CalibrationState | None
) -> ThresholdSelection:
    """Return the champion's frozen operating point, or refuse."""
    path = directory / THRESHOLD_DIR / "binary_threshold.json"
    _require(
        path.is_file(),
        "the champion's run publishes no binary operating point",
    )
    selection = ThresholdSelection.from_json(path.read_text(encoding="utf-8"))
    _require(
        selection.selection_fingerprint == lock.binary_threshold_fingerprint,
        "the published operating point is not the one the lock froze",
    )
    _require(
        selection.status is SelectionStatus.SELECTED
        and selection.selected_threshold is not None,
        "the frozen operating point records no selected threshold; there is "
        "nothing to apply",
    )
    _require(
        selection.model_content_fingerprint == lock.model_content_fingerprint,
        "the frozen operating point was selected for a different model",
    )
    # The score kind and the calibrator travel together, in both directions.
    # Silently switching between a raw score and a calibrated probability is the
    # one substitution that would change every published decision without
    # changing anything a reader could see.
    _require(
        is_probability(selection.score_kind) == (calibrator is not None),
        "the frozen threshold's score kind and the champion's calibration "
        "lineage disagree about whether a probability exists",
    )
    if calibrator is not None:
        _require(
            selection.calibration_state_fingerprint
            == calibrator.calibration_state_fingerprint,
            "the frozen threshold was selected against a different calibrator",
        )
    return selection


def _load_category_head(
    root: Path, *, lock: ChampionLock, ledger: ExperimentLedger
) -> FrozenCategoryModel | None:
    """Return the verified frozen category head, or ``None`` when none was bound.

    Absence is represented rather than filled in.  A binary champion existing is
    no reason to publish a category model nobody selected, and an all-``unknown``
    stand-in head would be a model output nothing produced.
    """
    head = lock.category_head
    if head is None:
        return None
    run, directory = _read_run(root, head.training_run_id, ledger=ledger)
    _require(
        run.task is MLTask.ATTACK_CATEGORY,
        "the frozen category head names a run that is not a category run",
    )
    _require(
        run.model_id == head.model_id,
        "the frozen category head names a different model than its run published",
    )
    selection = ledger.read_selection(head.validation_selection_id)
    _require(
        selection.task is MLTask.ATTACK_CATEGORY,
        "the category head cites a selection for a different task",
    )
    _require(
        selection.selected_run_id == head.training_run_id,
        "the category selection and the frozen head name different runs",
    )
    model = _load_model(
        directory / MODEL_DIR,
        compatibility=ModelCompatibility(
            eligible_feature_list_fingerprint=lock.eligible_feature_list_fingerprint,
            preprocessor_fingerprint=head.preprocessor_fingerprint,
            feature_catalog_fingerprint=lock.feature_catalog_fingerprint,
            allowlist_fingerprint=lock.allowlist_fingerprint,
        ),
        require_champion_eligible=False,
    )
    _require(
        model.document.model_content_fingerprint == head.model_content_fingerprint,
        "the published category model is not the one the lock froze",
    )
    path = directory / THRESHOLD_DIR / "category_abstention.json"
    _require(
        path.is_file(),
        "the frozen category head's run publishes no abstention point",
    )
    abstention = CategoryAbstentionSelection.from_json(path.read_text(encoding="utf-8"))
    _require(
        abstention.selection_fingerprint == head.category_abstention_fingerprint,
        "the published abstention point is not the one the lock froze",
    )
    _require(
        abstention.class_order == head.class_order,
        "the frozen class order and the published abstention point disagree",
    )
    _require(
        tuple(model.fitted.class_order) == head.class_order,
        "the category model's fitted class order is not the frozen one; a score "
        "column read by position would be the wrong class",
    )
    return FrozenCategoryModel(
        model=model,
        abstention=abstention,
        class_order=head.class_order,
        run_id=head.training_run_id,
    )


def _file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes, or refuse when it is absent."""
    if not path.is_file():
        raise ArtifactNotFoundError(
            "an artifact the champion lock names is not present under this root"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inference_feature_contract(
    lock: ChampionLock,
    *,
    feature_manifest: Mapping[str, Any],
    catalog_fingerprint: str,
    allowlist_fingerprint: str,
    eligible_feature_list_fingerprint: str,
    required_feature_schema_version: str,
    compatible_catalog_fingerprints: Sequence[str],
) -> None:
    """Raise unless the rows about to be scored come from the frozen contract.

    Six things have to agree before a frozen model may be pointed at a feature
    table: the Phase 3 manifest that describes the table, the executable feature
    catalog this build computes, the reviewed allowlist, the resolved eligible
    feature list, the configured schema version, and the lock.  A model scored
    against a table built under a different feature contract would produce
    numbers that look exactly like predictions.

    Raises:
        ModelNotReadyError: naming which provenance fields disagreed, and no
            more than that.
    """
    problems: list[str] = []
    if feature_manifest.get("feature_catalog_fingerprint") != catalog_fingerprint:
        problems.append("manifest feature_catalog_fingerprint")
    if feature_manifest.get("feature_schema_version") != (
        required_feature_schema_version
    ):
        problems.append("manifest feature_schema_version")
    if catalog_fingerprint not in set(compatible_catalog_fingerprints):
        problems.append("allowlist compatible_feature_catalog_fingerprints")
    if catalog_fingerprint != lock.feature_catalog_fingerprint:
        problems.append("champion lock feature_catalog_fingerprint")
    if allowlist_fingerprint != lock.allowlist_fingerprint:
        problems.append("champion lock allowlist_fingerprint")
    if eligible_feature_list_fingerprint != lock.eligible_feature_list_fingerprint:
        problems.append("champion lock eligible_feature_list_fingerprint")
    if problems:
        raise ModelNotReadyError(
            f"the feature contract these rows were built under is not the one "
            f"the champion was frozen against; disagreeing provenance field(s): "
            f"{sorted(problems)}"
        )


# ---------------------------------------------------------------------------
# The experimental anomaly probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExperimentalAnomalyRun:
    """A verified experimental anomaly run, deliberately outside the lock.

    Loaded from a run identifier a caller names explicitly.  It is never read
    out of ``champion.lock`` and never attached to one: M-030 is a
    generalisation probe, its output is not comparable with a supervised score,
    and a lock that acquired one retroactively would make the champion's
    identity depend on an experiment that never influenced it.
    """

    run_id: str
    model: InferenceModel
    threshold: AnomalyThresholdSelection | None

    @classmethod
    def load(
        cls,
        root: Path,
        run_id: str,
        *,
        ledger: ExperimentLedger,
        catalog: ModelCatalog = MODEL_CATALOG,
    ) -> ExperimentalAnomalyRun:
        """Verify the named anomaly run and return it, or refuse.

        Raises:
            ArtifactNotFoundError: the run has no published directory.
            ModelNotReadyError: the run is not an experimental anomaly run, or
                its artifacts do not verify.
        """
        run, directory = _read_run(Path(root), run_id, ledger=ledger)
        _require(
            run.task is MLTask.ANOMALY,
            "the named run is not an anomaly run; the experimental probe is "
            "published from its own lineage and never from a supervised one",
        )
        spec = catalog.for_family(run.model_family)
        _require(
            spec.experimental or spec.anomaly_only,
            "the named run's family is not experimental; an anomaly artifact is "
            "published as an experimental probe or not at all",
        )
        _require(
            not run.champion_eligible,
            "the named run claims champion eligibility, which an anomaly probe "
            "never has",
        )
        model = _load_model(
            directory / MODEL_DIR,
            compatibility=ModelCompatibility(
                eligible_feature_list_fingerprint=(
                    run.identity.eligible_feature_list_fingerprint
                ),
                preprocessor_fingerprint=run.identity.preprocessor_fingerprint,
                feature_catalog_fingerprint=(run.identity.feature_catalog_fingerprint),
                allowlist_fingerprint=run.identity.allowlist_fingerprint,
            ),
            require_champion_eligible=False,
        )
        _require(
            model.task is MLTask.ANOMALY,
            "the published artifact is not an anomaly model",
        )
        path = directory / THRESHOLD_DIR / "anomaly_threshold.json"
        threshold: AnomalyThresholdSelection | None = None
        if path.is_file():
            threshold = AnomalyThresholdSelection.from_json(
                path.read_text(encoding="utf-8")
            )
            _require(
                threshold.selection_fingerprint
                == run.identity.anomaly_threshold_fingerprint,
                "the published anomaly threshold is not the one its run records",
            )
            _require(
                not threshold.influences_champion_selection,
                "an anomaly threshold that influences champion selection is not "
                "a probe",
            )
            if threshold.status is not SelectionStatus.SELECTED:
                threshold = None
        return cls(run_id=run_id, model=model, threshold=threshold)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _malicious_column(model: InferenceModel) -> int:
    """Return the index of the malicious class in a fitted binary class order.

    By **name**, never by position.  A model whose classes were ordered
    differently would otherwise have its benign score read as its malicious one,
    and every number downstream would be exactly inverted.
    """
    order = tuple(model.fitted.class_order)
    if BINARY_CLASS_ORDER[1] not in order:
        raise ModelNotReadyError(
            "the frozen champion declares no malicious class; there is no column "
            "to read a malicious score from"
        )
    return order.index(BINARY_CLASS_ORDER[1])


def _require_scope(dataset: InferenceDataset) -> None:
    """Raise unless the dataset names a split predictions may be published for."""
    if dataset.scope is MLSplit.EXCLUDED:
        raise ModelNotReadyError(
            "excluded rows are excluded from every stage of this layer, "
            "prediction included"
        )
    if dataset.frame.row_count == 0:
        raise ModelNotReadyError("there are no rows in scope to score")


def predict_binary(
    champion: FrozenChampion, dataset: InferenceDataset
) -> tuple[BinaryPrediction, ...]:
    """Score *dataset* under the frozen champion and apply its operating point.

    The Milestone 8 order, and every step of it is frozen state rather than a
    choice made here: the reviewed raw feature order, the frozen preprocessor,
    the project-owned inference adapter, the raw decision score preserved as
    published, the frozen calibrator applied only when the lineage has one, and
    the frozen threshold applied to whichever score it was selected against.
    """
    _require_scope(dataset)
    index = _malicious_column(champion.binary)
    scored = champion.binary.transform_and_score(dataset.frame)
    # Stored exactly, never rounded. A frozen threshold is an *observed score*
    # kept at full precision for the reason Milestone 5 states: rounding moves
    # the boundary of a step function. Rounding the score moves the row instead,
    # and the row it moves is the one sitting on the threshold -- so a candidate
    # measured at 95% detection would publish a different set of flags than the
    # selection that promoted it.
    raw = tuple(float(row[index]) for row in scored)

    probabilities: tuple[float, ...] | None = None
    if champion.calibrator is not None:
        probabilities = tuple(champion.calibrator.transform(raw))
    calibrated = is_probability(champion.score_kind)
    if calibrated and probabilities is None:
        raise ModelNotReadyError(
            "the frozen threshold expects a calibrated probability and the "
            "champion has no calibrator to produce one"
        )
    if not calibrated and probabilities is not None:
        # A calibrator exists and the threshold was frozen on the raw score.
        # Publishing the probability anyway would put a number on every row that
        # nothing in the decision used, and somebody would eventually compare it
        # with the threshold beside it.
        probabilities = None

    threshold = champion.decision_threshold
    rows: list[BinaryPrediction] = []
    for position, anchor in enumerate(dataset.frame.anchors):
        probability = None if probabilities is None else probabilities[position]
        decided = raw[position] if probability is None else probability
        rows.append(
            BinaryPrediction(
                anchor_event_id=anchor.anchor_event_id,
                anchor_event_time=anchor.anchor_event_time,
                score_kind=champion.score_kind,
                malicious_decision_score=raw[position],
                malicious_probability=probability,
                decision_threshold=threshold,
                flagged_malicious=flagged_malicious(decided, threshold=threshold),
            )
        )
    return tuple(rows)


def category_applicable_anchors(
    binary: Sequence[BinaryPrediction],
) -> tuple[str, ...]:
    """Return the anchors the category head is applicable to, in canonical order.

    Exactly the rows the frozen binary champion flagged. The category head was
    fitted on **known-malicious training rows only**, so it has never been shown
    a benign row and has no opinion worth recording about one: asking it about a
    row the binary head did not route to triage produces a number, not a
    finding.
    """
    return tuple(row.anchor_event_id for row in binary if row.flagged_malicious)


def predict_category(
    champion: FrozenChampion,
    dataset: InferenceDataset,
    *,
    binary: Sequence[BinaryPrediction],
) -> tuple[CategoryPrediction, ...]:
    """Assign a known category to each **binary-positive** row, or abstain.

    Category triage is downstream of the binary decision, and the artifact says
    so by containing only the rows the binary champion flagged. A row the binary
    head called benign is *not applicable* to triage, and its absence from this
    table is what records that: it is a different state from ``unknown``, which
    means the head was asked and its best class score fell below the frozen
    abstention floor.

    Args:
        champion: the verified frozen champion.
        dataset: the label-free inference input, canonically ordered.
        binary: this publication's binary predictions, aligned anchor for anchor
            with *dataset*. Required rather than optional: a category artifact
            built without knowing which rows were flagged would be a triage
            result for rows nobody routed to triage.

    Raises:
        ModelNotReadyError: when no category head was frozen, or when the binary
            predictions do not describe the rows in *dataset*. There is no
            fabricated all-``unknown`` head: a category artifact that nothing
            fitted would be indistinguishable from a head that abstained.
    """
    _require_scope(dataset)
    head = champion.category
    if head is None:
        raise ModelNotReadyError(
            "no category head was frozen alongside this champion; there is "
            "nothing to score, and an all-unknown stand-in would be a model "
            "output nothing produced"
        )
    anchors = dataset.frame.anchors
    if len(binary) != len(anchors) or any(
        row.anchor_event_id != anchor.anchor_event_id
        for row, anchor in zip(binary, anchors, strict=True)
    ):
        raise ModelNotReadyError(
            "the binary predictions and the inference input describe different "
            "rows; category applicability cannot be resolved against a "
            "different population"
        )

    applicable = tuple(
        position for position, row in enumerate(binary) if row.flagged_malicious
    )
    if not applicable:
        # Zero applicable rows is an honest outcome, and it is published as an
        # empty table rather than as a table of abstentions: nothing was routed
        # to triage, so nothing abstained.
        return ()

    frame = InferenceFrame(
        split=dataset.frame.split,
        feature_names=dataset.frame.feature_names,
        anchors=tuple(anchors[position] for position in applicable),
        feature_matrix=tuple(
            dataset.frame.feature_matrix[position] for position in applicable
        ),
    )
    scored = head.model.transform_and_score(frame)
    floor = head.min_category_score
    rows: list[CategoryPrediction] = []
    for anchor, values in zip(frame.anchors, scored, strict=True):
        if len(values) != len(head.class_order):
            raise ModelNotReadyError(
                "the category head emitted a score row of the wrong width for "
                "its frozen class order"
            )
        exact = tuple(float(value) for value in values)
        assigned, best = assign_category(
            exact, head.class_order, min_category_score=floor
        )
        rows.append(
            CategoryPrediction(
                anchor_event_id=anchor.anchor_event_id,
                anchor_event_time=anchor.anchor_event_time,
                predicted_scenario=assigned,
                category_scores_json=category_scores_payload(exact, head.class_order),
                max_category_score=best,
                min_category_score=floor,
            )
        )
    return tuple(rows)


def predict_anomaly(
    probe: ExperimentalAnomalyRun, dataset: InferenceDataset
) -> tuple[AnomalyScore, ...]:
    """Score *dataset* with the experimental probe, emitting magnitudes only.

    No probability is produced, and none can be: the row schema has no field
    for one.  The flag, where a frozen threshold exists, is the inverted
    anomaly predicate and is never combined with the supervised decision.
    """
    _require_scope(dataset)
    scored = probe.model.transform_and_score(dataset.frame)
    threshold = None if probe.threshold is None else _selected(probe.threshold)
    rows: list[AnomalyScore] = []
    for anchor, values in zip(dataset.frame.anchors, scored, strict=True):
        score = float(values[0])
        rows.append(
            AnomalyScore(
                anchor_event_id=anchor.anchor_event_id,
                anchor_event_time=anchor.anchor_event_time,
                anomaly_score=score,
                anomaly_threshold=threshold,
                flagged_anomalous=(
                    None
                    if threshold is None
                    else flagged_anomalous(score, threshold=threshold)
                ),
            )
        )
    return tuple(rows)


def _selected(threshold: AnomalyThresholdSelection) -> float:
    """Return a selected anomaly threshold's value, or refuse."""
    if threshold.threshold is None:
        raise ModelNotReadyError("the anomaly threshold records no selected value")
    return threshold.threshold


def _assert_no_prohibited_prediction_column() -> None:
    """Fail at import if a prediction row declares a column it must not carry."""
    schemas: tuple[tuple[type[BaseModel], tuple[str, ...]], ...] = (
        (BinaryPrediction, BINARY_PREDICTION_COLUMNS),
        (CategoryPrediction, CATEGORY_PREDICTION_COLUMNS),
        (AnomalyScore, ANOMALY_PREDICTION_COLUMNS),
    )
    for model, columns in schemas:
        offending = sorted(set(model.model_fields) & PROHIBITED_PREDICTION_COLUMNS)
        if offending:
            raise ValueError(
                f"{model.__name__} declares prohibited prediction column(s) {offending}"
            )
        missing = sorted(set(columns) - set(model.model_fields))
        if missing:
            raise ValueError(
                f"{model.__name__} does not declare its published column(s) {missing}"
            )


_assert_no_prohibited_prediction_column()


def _assert_prediction_contract_versions_are_semantic() -> None:
    """Fail at import if the contract version is not a semantic version."""
    parts = PREDICTION_SCHEMA_VERSION.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError("PREDICTION_SCHEMA_VERSION must be a semantic version")


_assert_prediction_contract_versions_are_semantic()
