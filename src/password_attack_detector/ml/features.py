"""The reviewed, versioned ML feature allowlist -- eligibility is opt-in.

A Phase 3 catalog feature is **inert** to this layer until a human writes it
into ``configs/ml/features-allowlist-v1.yaml``.  That inversion is deliberate.
The obvious alternative -- trusting ``FeatureSpec.intended_use`` -- does not
work, and not for a subtle reason: 198 of the catalog's 201 specs carry the
identical default string, and the field is absent from the catalog's
``_FINGERPRINT_FIELDS``, so changing it invalidates nothing.  A screen that
every feature passes by default, and that no artifact records, cannot be the
authority for admitting a feature into a trained model.  It stays as a
*necessary* condition here; the authority is the reviewed file.

The consequence worth stating plainly: adding a feature to the Phase 3 catalog
does **not** add it to any model.  It makes
:data:`~password_attack_detector.ml.eligibility.CHECK_NAMES`'s
``NO_UNREVIEWED_CATALOG_FEATURE`` check fail by name, which is the point.  The
new feature is either admitted with a written rationale or listed under
``pending_review`` with the decision deferred explicitly.  Neither path is
silent.

Everything this module produces is fingerprinted over *semantic content*:
allowlist identity and version, the required feature-schema version, and the
ordered per-feature contract.  Paths, file bytes, modification times, and the
directory a file was read from contribute nothing, so the same reviewed
allowlist in two checkouts yields the same digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from password_attack_detector.exceptions import ConfigurationError, MLConfigurationError
from password_attack_detector.features.catalog import (
    PROHIBITED_FEATURE_COLUMNS,
    FeatureCatalog,
    FeatureGroup,
    FeatureSpec,
    LeakageClass,
)
from password_attack_detector.features.config import FEATURE_SCHEMA_VERSION
from password_attack_detector.ml.enums import FeatureDecisionPoint

__all__ = [
    "ALLOWLIST_SCHEMA_VERSION",
    "ML_OUTPUT_COLUMNS",
    "ML_PERMITTED_INTENDED_USE",
    "RESERVED_MATRIX_COLUMNS",
    "EligibleFeatureList",
    "FeatureAdmission",
    "FeatureAllowlist",
    "decision_point_for",
    "emit_allowlist_document",
    "load_feature_allowlist",
    "resolve_eligible_features",
    "unreviewed_catalog_features",
]

#: Version of the allowlist *file format*, distinct from the version of any
#: particular reviewed allowlist.  A format change is a code change; an
#: allowlist change is a review.
ALLOWLIST_SCHEMA_VERSION: Final = "1.0.0"

#: The ``intended_use`` values that permit a feature to reach a model.
#:
#: A necessary screen, never a sufficient one -- see the module docstring.  It
#: is kept because it is the catalog's own statement of purpose, and a feature
#: whose declared use is "join key; never a model input" should be refused
#: twice rather than once.
ML_PERMITTED_INTENDED_USE: Final[frozenset[str]] = frozenset(
    {"supervised classification and anomaly detection"}
)

#: Column names this layer *emits*.
#:
#: Listed here so they can be refused as **inputs**.  Each remains a legitimate
#: field of a Phase 5 prediction table, evaluation report, or model artifact --
#: that is what they are *for*.  What is forbidden is the direction of travel: a
#: model input named after a model output is how a pipeline learns to predict
#: its own previous answer.  Being on this list is never a reason to rename an
#: output field.
#:
#: Kept in step with ``features.catalog.PROHIBITED_FEATURE_COLUMNS``, which
#: carries the same names so the Phase 3 catalog builder rejects them at source.
ML_OUTPUT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "malicious_probability",
        "malicious_decision_score",
        "flagged_malicious",
        "predicted_scenario",
        "category_scores_json",
        "anomaly_score",
        "fused_flagged",
        "decision_threshold",
        "min_category_score",
    }
)

#: Join, label, split, and campaign columns.  Present in the assembled dataset,
#: absent from the design matrix.
RESERVED_MATRIX_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "anchor_event_id",
        "anchor_event_time",
        "feature_schema_version",
        "attack_class",
        "malicious",
        "supervised_training_eligible",
        "split",
        "exclusion_reason",
        "campaign_id",
    }
)

_NAME = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", max_length=128)]
_SHA256 = Annotated[
    str, StringConstraints(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)
]


def decision_point_for(spec: FeatureSpec) -> FeatureDecisionPoint:
    """Return the decision point *spec* is actually available at.

    Derived from the catalog rather than read from the allowlist, so the
    allowlist's recorded value can be *checked* instead of believed.  A feature
    that needs a fitted baseline is available only where that baseline exists,
    and no review comment changes that.
    """
    if spec.requires_baseline:
        return FeatureDecisionPoint.REQUIRES_FITTED_BASELINE
    return FeatureDecisionPoint.POST_EVENT


class FeatureAdmission(BaseModel):
    """One reviewed decision to admit one feature into the ML design matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: _NAME
    decision_point: FeatureDecisionPoint
    #: The phase or version in which this feature was admitted.  Recorded so a
    #: later reviewer can tell a founding admission from a subsequent one.
    admitted_in: str = Field(min_length=1, max_length=64)
    #: Why this feature may inform a detection decision.  Required: an entry
    #: nobody could justify in one line is an entry nobody reviewed.
    rationale: str = Field(min_length=8, max_length=512)
    leakage_class: LeakageClass
    feature_group: FeatureGroup
    #: Named experiments this feature is restricted to.  Empty means "any
    #: experiment this allowlist governs", which is the ordinary case.
    experiments: tuple[str, ...] = ()

    @field_validator("experiments")
    @classmethod
    def check_experiments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Experiment restrictions are distinct, non-empty names."""
        if len(set(value)) != len(value):
            raise ValueError("experiments repeats a name")
        if any(not item.strip() for item in value):
            raise ValueError("experiments must not contain a blank name")
        return value

    @model_validator(mode="after")
    def check_not_reserved(self) -> Self:
        """An admission may never name a label, split, campaign, or output column."""
        if self.name in PROHIBITED_FEATURE_COLUMNS:
            raise ValueError(f"{self.name!r} is a prohibited feature column")
        if self.name in ML_OUTPUT_COLUMNS:
            raise ValueError(f"{self.name!r} is a machine-learning output column")
        if self.name in RESERVED_MATRIX_COLUMNS:
            raise ValueError(f"{self.name!r} is a join, label, or split column")
        if self.leakage_class is LeakageClass.KEY:
            raise ValueError(
                f"{self.name!r} is a key column and is never a model input"
            )
        if self.feature_group is FeatureGroup.KEY:
            raise ValueError(f"{self.name!r} is in the key group and is never an input")
        return self

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the allowlist digest.

        ``rationale`` and ``admitted_in`` are excluded on purpose, mirroring the
        Phase 3 catalog's exclusion of ``description``: correcting the wording of
        a review note must not invalidate every model that recorded this digest.
        The *contract* -- which features, under which classification, available
        when -- is what the fingerprint pins.
        """
        return {
            "name": self.name,
            "decision_point": str(self.decision_point),
            "leakage_class": str(self.leakage_class),
            "feature_group": str(self.feature_group),
            "experiments": sorted(self.experiments),
        }


class FeatureAllowlist(BaseModel):
    """A reviewed, versioned, separately fingerprinted set of admitted features.

    Two of these ship: the champion contract and a prior-only ablation.  They
    carry different ``allowlist_id`` values and therefore different
    fingerprints, which is what stops the ablation quietly becoming the
    champion's feature contract -- a champion lock records the fingerprint it
    was frozen under, and a disagreeing one is rejected rather than merged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowlist_schema_version: Literal["1.0.0"] = ALLOWLIST_SCHEMA_VERSION
    #: Identity of this reviewed contract, e.g. ``champion`` or
    #: ``prior_only_ablation``.  Part of the fingerprint, so two allowlists that
    #: happened to admit identical features still digest differently.
    allowlist_id: _NAME
    allowlist_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    required_feature_schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    #: Catalog fingerprints a reviewer has checked this allowlist against.  A
    #: feature *set* is stable across the shipped feature configurations, but
    #: their catalogs digest differently because windows differ, so this is a
    #: tuple rather than a single value.  Empty is refused: an allowlist that
    #: pins no catalog records no review.
    compatible_feature_catalog_fingerprints: tuple[_SHA256, ...]
    #: The leakage classes this allowlist takes responsibility for, or ``None``
    #: for every eligible class.
    #:
    #: This is what makes a *scoped* allowlist honest.  The prior-only ablation
    #: governs ``prior_only`` alone; the fourteen baseline-derived features it
    #: omits are out of its scope, not unreviewed by it.  Without this field the
    #: audit's ``NO_UNREVIEWED_CATALOG_FEATURE`` check could not tell a
    #: deliberate experimental narrowing from a feature nobody looked at, and
    #: would have to report the ablation as incomplete forever.
    governed_leakage_classes: tuple[str, ...] | None = None
    entries: tuple[FeatureAdmission, ...]
    #: Catalog features that are eligible by every mechanical condition but
    #: whose admission has been deliberately deferred.  Listing a name here is a
    #: decision; leaving it out of both lists is an oversight, and the audit
    #: tells them apart.
    pending_review: tuple[_NAME, ...] = ()

    @field_validator("compatible_feature_catalog_fingerprints")
    @classmethod
    def check_fingerprints(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """At least one catalog must be pinned, and none twice."""
        if not value:
            raise ValueError(
                "compatible_feature_catalog_fingerprints must pin at least one "
                "reviewed catalog"
            )
        if len(set(value)) != len(value):
            raise ValueError("compatible_feature_catalog_fingerprints repeats a digest")
        return value

    @field_validator("governed_leakage_classes")
    @classmethod
    def check_governed(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """A declared scope names distinct, eligible, non-key leakage classes."""
        if value is None:
            return value
        if not value:
            raise ValueError(
                "governed_leakage_classes must be omitted for 'every class' "
                "rather than set to an empty list"
            )
        if len(set(value)) != len(value):
            raise ValueError("governed_leakage_classes repeats a class")
        eligible = {str(item) for item in LeakageClass if item is not LeakageClass.KEY}
        unknown = sorted(set(value) - eligible)
        if unknown:
            raise ValueError(
                f"governed_leakage_classes names ineligible class(es) {unknown}; "
                f"eligible: {sorted(eligible)}"
            )
        return value

    @field_validator("entries")
    @classmethod
    def check_entries(
        cls, value: tuple[FeatureAdmission, ...]
    ) -> tuple[FeatureAdmission, ...]:
        """Entries are non-empty and name each feature exactly once."""
        if not value:
            raise ValueError("entries must admit at least one feature")
        names = [entry.name for entry in value]
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"entries admits {duplicates} more than once")
        return value

    @model_validator(mode="after")
    def check_pending_disjoint(self) -> Self:
        """A feature is admitted or deferred, never recorded as both."""
        admitted = {entry.name for entry in self.entries}
        overlap = sorted(admitted & set(self.pending_review))
        if overlap:
            raise ValueError(f"{overlap} appear in both entries and pending_review")
        if len(set(self.pending_review)) != len(self.pending_review):
            raise ValueError("pending_review repeats a feature name")
        if self.governed_leakage_classes is not None:
            scope = set(self.governed_leakage_classes)
            outside = sorted(
                {
                    entry.name
                    for entry in self.entries
                    if str(entry.leakage_class) not in scope
                }
            )
            if outside:
                raise ValueError(
                    f"{outside} are admitted but fall outside the declared "
                    f"governed_leakage_classes {sorted(scope)}"
                )
        return self

    def governs(self, leakage_class: LeakageClass) -> bool:
        """Return whether this allowlist takes responsibility for *leakage_class*."""
        if self.governed_leakage_classes is None:
            return leakage_class is not LeakageClass.KEY
        return str(leakage_class) in set(self.governed_leakage_classes)

    @property
    def names(self) -> frozenset[str]:
        """Return every admitted feature name."""
        return frozenset(entry.name for entry in self.entries)

    def entry_for(self, name: str) -> FeatureAdmission | None:
        """Return the admission for *name*, or ``None`` when it is not admitted."""
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def fingerprint(self) -> str:
        """Return a SHA-256 digest of this allowlist's semantic contract.

        Entries are sorted by name before hashing, so the order they were typed
        in does not change the digest -- only which features are admitted, under
        which classification, at which decision point.
        """
        payload = {
            "allowlist_schema_version": self.allowlist_schema_version,
            "allowlist_id": self.allowlist_id,
            "allowlist_version": self.allowlist_version,
            "required_feature_schema_version": self.required_feature_schema_version,
            "governed_leakage_classes": (
                None
                if self.governed_leakage_classes is None
                else sorted(self.governed_leakage_classes)
            ),
            "entries": [
                entry.fingerprint_data()
                for entry in sorted(self.entries, key=lambda item: item.name)
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


class EligibleFeatureList(BaseModel):
    """The ordered feature set one run may fit on, plus its identity.

    Order is the Phase 3 catalog's order restricted to the admitted features --
    not the order they appear in the YAML, and not alphabetical.  Every artifact
    in the layer writes columns in this order, so a design matrix, a serialized
    model's ``feature_order``, and a prediction table cannot disagree about
    which column is which.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowlist_id: str
    allowlist_version: str
    allowlist_fingerprint: _SHA256
    feature_names: tuple[str, ...]
    decision_points: tuple[FeatureDecisionPoint, ...]
    leakage_classes: tuple[LeakageClass, ...]
    feature_groups: tuple[FeatureGroup, ...]
    #: Admitted features the *configuration* narrowed away -- an admitted
    #: baseline-derived feature under a prior-only configuration, say.  Recorded
    #: rather than dropped silently, so a shrunken matrix is explicable.
    excluded_by_configuration: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_shapes(self) -> Self:
        """Every parallel tuple describes the same features, in the same order."""
        if not self.feature_names:
            raise ValueError("no feature survived eligibility resolution")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names repeats a feature")
        widths = {
            len(self.feature_names),
            len(self.decision_points),
            len(self.leakage_classes),
            len(self.feature_groups),
        }
        if len(widths) != 1:
            raise ValueError("feature metadata tuples have mismatched lengths")
        return self

    def __len__(self) -> int:
        return len(self.feature_names)

    def fingerprint(self) -> str:
        """Return the digest that gives a fitted model its feature identity.

        Includes the *resolved* order, so narrowing a run by configuration
        yields a different identity from the full contract even though both
        resolve from the same reviewed allowlist.
        """
        payload = {
            "allowlist_id": self.allowlist_id,
            "allowlist_version": self.allowlist_version,
            "allowlist_fingerprint": self.allowlist_fingerprint,
            "features": [
                {
                    "name": name,
                    "decision_point": str(point),
                    "leakage_class": str(leakage),
                    "feature_group": str(group),
                }
                for name, point, leakage, group in zip(
                    self.feature_names,
                    self.decision_points,
                    self.leakage_classes,
                    self.feature_groups,
                    strict=True,
                )
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_feature_allowlist(path: Path) -> FeatureAllowlist:
    """Load and validate a reviewed feature allowlist from YAML.

    Uses ``yaml.safe_load``, which constructs plain scalars, lists, and mappings
    only: no Python object or import path can be materialised from a reviewed
    file.

    Raises:
        ConfigurationError: if the file is missing, unreadable, not a mapping,
            or fails validation.  The message carries the failure, never the
            absolute path it was read from.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read the feature allowlist: {type(exc).__name__}"
        ) from None

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        raise ConfigurationError("Feature allowlist is not valid YAML") from None

    if not isinstance(data, dict):
        raise ConfigurationError("Feature allowlist must be a YAML mapping")

    try:
        return FeatureAllowlist(**data)
    except Exception as exc:
        raise ConfigurationError(f"Invalid feature allowlist: {exc}") from None


def _screen_catalog(
    catalog: FeatureCatalog,
    *,
    include_leakage_classes: Sequence[str],
    include_feature_groups: Sequence[str] | None,
) -> tuple[tuple[FeatureSpec, ...], tuple[FeatureSpec, ...]]:
    """Return the specs passing the mechanical screens, and those passing all but config.

    The first tuple is what a reviewer must have an opinion about: features that
    are mechanically usable *and* enabled by this configuration.  The second is
    the configuration-independent set, used by the audit so narrowing a run
    cannot make an unreviewed feature disappear from the report.
    """
    mechanical: list[FeatureSpec] = []
    for spec in catalog.specs:
        if spec.leakage_class is LeakageClass.KEY or spec.group is FeatureGroup.KEY:
            continue
        if spec.deprecated:
            continue
        if spec.intended_use not in ML_PERMITTED_INTENDED_USE:
            continue
        if spec.name in PROHIBITED_FEATURE_COLUMNS or spec.name in ML_OUTPUT_COLUMNS:
            continue
        if spec.name in RESERVED_MATRIX_COLUMNS:
            continue
        mechanical.append(spec)

    classes = set(include_leakage_classes)
    groups = None if include_feature_groups is None else set(include_feature_groups)
    configured = tuple(
        spec
        for spec in mechanical
        if str(spec.leakage_class) in classes
        and (groups is None or str(spec.group) in groups)
    )
    return configured, tuple(mechanical)


def unreviewed_catalog_features(
    catalog: FeatureCatalog, allowlist: FeatureAllowlist
) -> tuple[str, ...]:
    """Return catalog features within this allowlist's scope that nobody ruled on.

    Deliberately **configuration-independent**.  Narrowing a *run* to prior-only
    features must not make a newly added baseline feature vanish from the review
    queue -- otherwise an unreviewed feature could be hidden by choosing a
    narrower configuration, which is precisely the escape hatch this check
    exists to close.  What does bound the queue is the allowlist's own declared
    ``governed_leakage_classes``, which is a reviewed statement rather than a
    run-time choice.
    """
    _, mechanical = _screen_catalog(
        catalog,
        include_leakage_classes=[str(item) for item in LeakageClass],
        include_feature_groups=None,
    )
    decided = allowlist.names | set(allowlist.pending_review)
    return tuple(
        sorted(
            spec.name
            for spec in mechanical
            if spec.name not in decided and allowlist.governs(spec.leakage_class)
        )
    )


def resolve_eligible_features(
    catalog: FeatureCatalog,
    allowlist: FeatureAllowlist,
    *,
    include_leakage_classes: Sequence[str],
    include_feature_groups: Sequence[str] | None = None,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> EligibleFeatureList:
    """Resolve the reviewed allowlist against an executable catalog.

    Two kinds of outcome, kept distinct on purpose:

    * A **disagreement** raises.  An admitted feature the catalog no longer
      declares, or declares under a different leakage class, group, or decision
      point, means the review and the code describe different systems.  Silently
      preferring one of them is how a leakage classification stops being true.
    * A **narrowing** is recorded.  An admitted feature the configuration
      excludes -- a baseline-derived feature under a prior-only run -- is listed
      in ``excluded_by_configuration`` and left out of the matrix.

    Raises:
        MLConfigurationError: on any disagreement between allowlist and catalog,
            or on a schema-version mismatch.
    """
    if allowlist.required_feature_schema_version != feature_schema_version:
        raise MLConfigurationError(
            f"Allowlist {allowlist.allowlist_id!r} requires feature schema "
            f"{allowlist.required_feature_schema_version}, but the catalog "
            f"declares {feature_schema_version}"
        )

    problems: list[str] = []
    for entry in allowlist.entries:
        if not catalog.has(entry.name):
            problems.append(f"{entry.name}: not declared by the feature catalog")
            continue
        spec = catalog.get(entry.name)
        if spec.deprecated:
            problems.append(f"{entry.name}: deprecated in the feature catalog")
        if spec.intended_use not in ML_PERMITTED_INTENDED_USE:
            problems.append(f"{entry.name}: intended_use does not permit modelling")
        if spec.leakage_class is not entry.leakage_class:
            problems.append(
                f"{entry.name}: leakage class {str(entry.leakage_class)!r} in the "
                f"allowlist, {str(spec.leakage_class)!r} in the catalog"
            )
        if spec.group is not entry.feature_group:
            problems.append(
                f"{entry.name}: feature group {str(entry.feature_group)!r} in the "
                f"allowlist, {str(spec.group)!r} in the catalog"
            )
        actual_point = decision_point_for(spec)
        if actual_point is not entry.decision_point:
            problems.append(
                f"{entry.name}: decision point {str(entry.decision_point)!r} in the "
                f"allowlist, {str(actual_point)!r} in the catalog"
            )

    if problems:
        raise MLConfigurationError(
            f"Feature allowlist {allowlist.allowlist_id!r} disagrees with the "
            f"feature catalog: {'; '.join(sorted(problems))}"
        )

    classes = set(include_leakage_classes)
    groups = None if include_feature_groups is None else set(include_feature_groups)

    admitted = allowlist.names
    names: list[str] = []
    points: list[FeatureDecisionPoint] = []
    leakages: list[LeakageClass] = []
    feature_groups: list[FeatureGroup] = []
    excluded: list[str] = []

    # Catalog order, not allowlist order: the catalog is the authority on column
    # ordering for every artifact this project writes.
    for spec in catalog.specs:
        if spec.name not in admitted:
            continue
        if str(spec.leakage_class) not in classes:
            excluded.append(spec.name)
            continue
        if groups is not None and str(spec.group) not in groups:
            excluded.append(spec.name)
            continue
        names.append(spec.name)
        points.append(decision_point_for(spec))
        leakages.append(spec.leakage_class)
        feature_groups.append(spec.group)

    if not names:
        raise MLConfigurationError(
            f"No admitted feature survived this configuration: allowlist "
            f"{allowlist.allowlist_id!r} admits {len(admitted)} feature(s), and "
            f"the configured leakage classes and groups exclude every one"
        )

    return EligibleFeatureList(
        allowlist_id=allowlist.allowlist_id,
        allowlist_version=allowlist.allowlist_version,
        allowlist_fingerprint=allowlist.fingerprint(),
        feature_names=tuple(names),
        decision_points=tuple(points),
        leakage_classes=tuple(leakages),
        feature_groups=tuple(feature_groups),
        excluded_by_configuration=tuple(sorted(excluded)),
    )


def emit_allowlist_document(
    catalog: FeatureCatalog,
    *,
    allowlist_id: str = "champion",
    allowlist_version: str = "1.0.0",
    compatible_feature_catalog_fingerprints: Sequence[str],
    admitted_in: str,
    leakage_classes: Sequence[str] | None = None,
    rationales: Mapping[str, str] | None = None,
    notes: Sequence[str] = (),
) -> str:
    """Render a reviewed allowlist as YAML.

    Without *rationales* this is emphatically a **draft**: the justification it
    writes for each feature is generated from that feature's own classification,
    which is exactly the kind of self-referential reasoning a review exists to
    catch.  The drafting command says so in the file it writes.  The committed
    allowlists supply real rationales and their own header *notes*.
    """
    selected = [
        spec
        for spec in catalog.specs
        if spec.leakage_class is not LeakageClass.KEY
        and spec.group is not FeatureGroup.KEY
        and not spec.deprecated
        and spec.intended_use in ML_PERMITTED_INTENDED_USE
        and (leakage_classes is None or str(spec.leakage_class) in set(leakage_classes))
    ]
    default_notes = (
        "Reviewed machine-learning feature allowlist.",
        "",
        "Generated as a draft, then reviewed and committed by a person. A",
        "feature reaches a model because it appears here, not because it",
        "appears in the Phase 3 catalog.",
    )
    lines = [f"# {note}".rstrip() for note in (notes or default_notes)]
    lines += [
        "",
        f'allowlist_schema_version: "{ALLOWLIST_SCHEMA_VERSION}"',
        f"allowlist_id: {allowlist_id}",
        f'allowlist_version: "{allowlist_version}"',
        f'required_feature_schema_version: "{FEATURE_SCHEMA_VERSION}"',
        "",
        "compatible_feature_catalog_fingerprints:",
    ]
    lines.extend(f"  - {digest}" for digest in compatible_feature_catalog_fingerprints)
    if leakage_classes is not None:
        lines.extend(["", "governed_leakage_classes:"])
        lines.extend(f"  - {item}" for item in leakage_classes)
    lines.extend(["", "entries:"])
    for spec in selected:
        rationale = (rationales or {}).get(spec.name) or _default_rationale(spec)
        lines.extend(
            [
                f"  - name: {spec.name}",
                f"    decision_point: {decision_point_for(spec)}",
                f'    admitted_in: "{admitted_in}"',
                f"    leakage_class: {spec.leakage_class}",
                f"    feature_group: {spec.group}",
                f"    rationale: {json.dumps(rationale, ensure_ascii=True)}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _default_rationale(spec: FeatureSpec) -> str:
    """Return a placeholder rationale for a drafted admission."""
    window = f" over {spec.window}" if spec.window else ""
    return (
        f"{str(spec.group).replace('_', ' ')} signal{window}, classified "
        f"{spec.leakage_class} by the Phase 3 catalog. Review before admission."
    )
