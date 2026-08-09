"""The ML eligibility and leakage audit -- fifteen named checks.

Shaped after the Phase 3 :class:`~password_attack_detector.features.leakage.LeakageAuditor`
and inheriting its central rule: **a skipped check is not a pass.**  A check
whose input was not supplied reports :attr:`~password_attack_detector.ml.enums.AuditCheckStatus.SKIPPED`
and the overall audit fails.  The temptation this closes is a real one -- an
audit that quietly reports fourteen passes and one omission reads, at a glance,
like fifteen passes, and the one that was omitted is always the expensive one.

Every check here is evaluable in Milestone 2.  Nothing in this module reports a
future capability as verified: there is no gate check, no threshold-provenance
check over thresholds that do not exist yet, and no calibration check.  Those
arrive with the code that makes them meaningful.  What *is* checked now is the
whole of what the assembled dataset can be asked: which columns exist, where
they came from, who reviewed them, how the rows are ordered, how the splits
divide, and whether the validation partition is campaign-disjoint and supported.

**Output is aggregate only.**  Messages carry counts and declared column names.
No anchor identifier, no campaign identifier, no entity pseudonym, no feature
value, no absolute path, and no secret reaches a message, a JSON report, or the
Markdown rendering -- and a test sweeps all three for exactly those shapes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from password_attack_detector.features.catalog import (
    PROHIBITED_FEATURE_COLUMNS,
    FeatureCatalog,
    LeakageClass,
)
from password_attack_detector.ml.config import MLConfig
from password_attack_detector.ml.dataset import CAMPAIGN_COLUMNS, MLDataset
from password_attack_detector.ml.enums import (
    FIT_ELIGIBLE_SPLITS,
    AuditCheckStatus,
    AuditStatus,
    MLSplit,
    ValidationPartition,
    ValidationPartitionStatus,
)
from password_attack_detector.ml.features import (
    ML_OUTPUT_COLUMNS,
    RESERVED_MATRIX_COLUMNS,
    EligibleFeatureList,
    FeatureAllowlist,
    decision_point_for,
    unreviewed_catalog_features,
)
from password_attack_detector.ml.ordering import is_canonical
from password_attack_detector.ml.partition import ValidationPartitionResult

__all__ = [
    "CHECK_NAMES",
    "ML_AUDIT_JSON_FILE",
    "ML_AUDIT_MD_FILE",
    "MLAuditCheck",
    "MLEligibilityAuditResult",
    "MLEligibilityAuditor",
    "ml_audit_result_to_markdown",
]

#: Published report names, matching the ``<topic>.{json,md}`` convention the
#: detection layer's quality and evaluation reports already follow.
ML_AUDIT_JSON_FILE: Final = "ml_eligibility_audit.json"
ML_AUDIT_MD_FILE: Final = "ml_eligibility_audit.md"

#: Every check this auditor performs, in report order.
#:
#: The tuple is the contract: a test asserts the result carries exactly these
#: names, so a check cannot be dropped by deleting its implementation.
CHECK_NAMES: Final[tuple[str, ...]] = (
    "NO_PROHIBITED_COLUMNS",
    "NO_KEY_CLASS_COLUMNS",
    "EVERY_COLUMN_TRACES_TO_CATALOG",
    "ALLOWLIST_COVERS_EVERY_MATRIX_COLUMN",
    "NO_UNREVIEWED_CATALOG_FEATURE",
    "NO_LABEL_OR_SPLIT_IN_MATRIX",
    "NO_CAMPAIGN_IDENTIFIER_IN_MATRIX",
    "ROWS_ARE_CANONICALLY_ORDERED",
    "SPLIT_SETS_DISJOINT",
    "HOLDOUT_AND_EXCLUDED_ABSENT_FROM_FIT",
    "SCHEMA_AND_CATALOG_FINGERPRINT_MATCH",
    "JOIN_KEY_INTEGRITY",
    "VALIDATION_HALVES_CAMPAIGN_DISJOINT",
    "VALIDATION_SUPPORT_SUFFICIENT",
    "NO_TEST_OR_HOLDOUT_SELECTION_SOURCE",
)


@dataclass(frozen=True, slots=True)
class MLAuditCheck:
    """The outcome of one named check."""

    name: str
    status: AuditCheckStatus
    message: str

    @property
    def passed(self) -> bool:
        """Return whether this check passed.  ``SKIPPED`` is not passed."""
        return self.status is AuditCheckStatus.PASS


@dataclass(frozen=True)
class MLEligibilityAuditResult:
    """Structured outcome of a full ML eligibility audit.

    Counts, column names, and stable check names only.  Safe to embed in a
    report or print to a terminal.
    """

    status: AuditStatus
    checks: tuple[MLAuditCheck, ...]
    failures: tuple[str, ...]
    allowlist_id: str
    allowlist_version: str
    allowlist_fingerprint: str
    eligible_feature_list_fingerprint: str
    checked_feature_count: int
    checked_row_count: int
    split_row_counts: Mapping[str, int]
    validation_partition: Mapping[str, Any] | None

    @property
    def passed(self) -> bool:
        """Return whether every check passed."""
        return self.status is AuditStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-serialisable mapping."""
        return {
            "status": str(self.status),
            "checks": [
                {
                    "name": check.name,
                    "status": str(check.status),
                    "passed": check.passed,
                    "message": check.message,
                }
                for check in self.checks
            ],
            "failures": list(self.failures),
            "allowlist_id": self.allowlist_id,
            "allowlist_version": self.allowlist_version,
            "allowlist_fingerprint": self.allowlist_fingerprint,
            "eligible_feature_list_fingerprint": (
                self.eligible_feature_list_fingerprint
            ),
            "checked_feature_count": self.checked_feature_count,
            "checked_row_count": self.checked_row_count,
            "split_row_counts": dict(sorted(self.split_row_counts.items())),
            "validation_partition": (
                None
                if self.validation_partition is None
                else dict(sorted(self.validation_partition.items()))
            ),
        }


class MLEligibilityAuditor:
    """Runs the fifteen eligibility checks over an assembled ML dataset.

    Constructed with everything a check might need.  ``feature_manifest`` and
    ``partition`` are optional *arguments*, never optional *checks*: omitting
    one produces a ``SKIPPED`` status and a failing audit, which is how the
    caller finds out it forgot an input rather than receiving a clean bill of
    health for something nobody looked at.
    """

    def __init__(
        self,
        *,
        catalog: FeatureCatalog,
        allowlist: FeatureAllowlist,
        eligible: EligibleFeatureList,
        config: MLConfig,
        feature_manifest: Mapping[str, Any] | None = None,
        partition: ValidationPartitionResult | None = None,
    ) -> None:
        self._catalog = catalog
        self._allowlist = allowlist
        self._eligible = eligible
        self._config = config
        self._manifest = feature_manifest
        self._partition = partition

    def audit(self, dataset: MLDataset) -> MLEligibilityAuditResult:
        """Run every check and return the aggregate result."""
        checks: list[MLAuditCheck] = [
            self._no_prohibited_columns(dataset),
            self._no_key_class_columns(dataset),
            self._every_column_traces_to_catalog(dataset),
            self._allowlist_covers_every_matrix_column(dataset),
            self._no_unreviewed_catalog_feature(),
            self._no_label_or_split_in_matrix(dataset),
            self._no_campaign_identifier_in_matrix(dataset),
            self._rows_are_canonically_ordered(dataset),
            self._split_sets_disjoint(dataset),
            self._holdout_and_excluded_absent_from_fit(dataset),
            self._schema_and_catalog_fingerprint_match(dataset),
            self._join_key_integrity(dataset),
            self._validation_halves_campaign_disjoint(dataset),
            self._validation_support_sufficient(),
            self._no_test_or_holdout_selection_source(dataset),
        ]

        ordered = tuple(
            next(check for check in checks if check.name == name)
            for name in CHECK_NAMES
        )
        failures = tuple(check.name for check in ordered if not check.passed)

        return MLEligibilityAuditResult(
            status=AuditStatus.FAIL if failures else AuditStatus.PASS,
            checks=ordered,
            failures=failures,
            allowlist_id=self._allowlist.allowlist_id,
            allowlist_version=self._allowlist.allowlist_version,
            allowlist_fingerprint=self._allowlist.fingerprint(),
            eligible_feature_list_fingerprint=(
                dataset.eligible_feature_list_fingerprint
            ),
            checked_feature_count=len(dataset.feature_names),
            checked_row_count=dataset.joined_row_count,
            split_row_counts={
                str(split): dataset.for_split(split).row_count for split in MLSplit
            },
            validation_partition=(
                None if self._partition is None else self._partition.to_dict()
            ),
        )

    # -- checks -------------------------------------------------------------

    def _no_prohibited_columns(self, dataset: MLDataset) -> MLAuditCheck:
        """Ground truth, campaign metadata, and model output names in the matrix."""
        forbidden = PROHIBITED_FEATURE_COLUMNS | ML_OUTPUT_COLUMNS
        offending = sorted(set(dataset.feature_names) & forbidden)
        return _verdict(
            "NO_PROHIBITED_COLUMNS",
            not offending,
            f"{len(dataset.feature_names)} matrix column(s) checked against "
            f"{len(forbidden)} prohibited name(s)",
            f"prohibited column(s) in the design matrix: {offending}",
        )

    def _no_key_class_columns(self, dataset: MLDataset) -> MLAuditCheck:
        """Join and provenance keys are never model inputs."""
        offending = sorted(
            name
            for name in dataset.feature_names
            if self._catalog.has(name)
            and self._catalog.get(name).leakage_class is LeakageClass.KEY
        )
        return _verdict(
            "NO_KEY_CLASS_COLUMNS",
            not offending,
            "no column is classified 'key' by the feature catalog",
            f"key-class column(s) in the design matrix: {offending}",
        )

    def _every_column_traces_to_catalog(self, dataset: MLDataset) -> MLAuditCheck:
        """A matrix column with no catalog spec has no declared semantics."""
        offending = sorted(
            name for name in dataset.feature_names if not self._catalog.has(name)
        )
        return _verdict(
            "EVERY_COLUMN_TRACES_TO_CATALOG",
            not offending,
            f"all {len(dataset.feature_names)} matrix column(s) are declared by "
            f"the Phase 3 feature catalog",
            f"undeclared matrix column(s): {offending}",
        )

    def _allowlist_covers_every_matrix_column(self, dataset: MLDataset) -> MLAuditCheck:
        """Every column traces to a reviewed admission, with matching metadata."""
        problems: list[str] = []
        for name in dataset.feature_names:
            entry = self._allowlist.entry_for(name)
            if entry is None:
                problems.append(name)
                continue
            if not self._catalog.has(name):
                continue
            spec = self._catalog.get(name)
            if (
                spec.leakage_class is not entry.leakage_class
                or spec.group is not entry.feature_group
                or decision_point_for(spec) is not entry.decision_point
            ):
                problems.append(name)
        return _verdict(
            "ALLOWLIST_COVERS_EVERY_MATRIX_COLUMN",
            not problems,
            f"all {len(dataset.feature_names)} matrix column(s) trace to a "
            f"reviewed admission in allowlist {self._allowlist.allowlist_id!r} "
            f"v{self._allowlist.allowlist_version}",
            f"column(s) with no matching reviewed admission: {sorted(problems)}",
        )

    def _no_unreviewed_catalog_feature(self) -> MLAuditCheck:
        """A newly added catalog feature must be admitted or deferred, explicitly."""
        unreviewed = unreviewed_catalog_features(self._catalog, self._allowlist)
        return _verdict(
            "NO_UNREVIEWED_CATALOG_FEATURE",
            not unreviewed,
            f"every catalog feature within this allowlist's scope is admitted "
            f"or listed under pending_review "
            f"({len(self._allowlist.entries)} admitted, "
            f"{len(self._allowlist.pending_review)} deferred)",
            f"catalog feature(s) nobody has ruled on: {list(unreviewed)}. Admit "
            f"them with a rationale or list them under pending_review; a "
            f"feature never enters a model by being added to the catalog",
        )

    def _no_label_or_split_in_matrix(self, dataset: MLDataset) -> MLAuditCheck:
        """The label and split tables' columns stay out of the matrix.

        Checked against ``RESERVED_MATRIX_COLUMNS`` rather than by importing the
        Phase 3 table constants directly.  This module is not a permitted label
        reader, and reaching into ``features.serialization`` -- even only for a
        tuple of column *names* -- would put it on the wrong side of the
        import-graph boundary for no benefit.  ``ml.dataset``, which is
        permitted, asserts at import that the reserved set still covers both
        published tables, so the indirection cannot silently go stale.
        """
        offending = sorted(set(dataset.feature_names) & RESERVED_MATRIX_COLUMNS)
        return _verdict(
            "NO_LABEL_OR_SPLIT_IN_MATRIX",
            not offending,
            "no label, split, identifier, or provenance column reached the "
            "design matrix",
            f"label or split column(s) in the design matrix: {offending}",
        )

    def _no_campaign_identifier_in_matrix(self, dataset: MLDataset) -> MLAuditCheck:
        """Campaign identifiers group rows; they are never features."""
        offending = sorted(set(dataset.feature_names) & CAMPAIGN_COLUMNS)
        return _verdict(
            "NO_CAMPAIGN_IDENTIFIER_IN_MATRIX",
            not offending,
            "campaign metadata is retained beside the matrix and reaches the "
            "partitioner only",
            f"campaign column(s) in the design matrix: {offending}",
        )

    def _rows_are_canonically_ordered(self, dataset: MLDataset) -> MLAuditCheck:
        """Every split is sorted by anchor event time, then anchor identifier."""
        unordered = sorted(
            str(split)
            for split in MLSplit
            if not is_canonical(dataset.for_split(split).anchors)
        )
        return _verdict(
            "ROWS_ARE_CANONICALLY_ORDERED",
            not unordered,
            f"all {len(MLSplit)} split(s) are in canonical "
            f"(anchor_event_time, anchor_event_id) order",
            f"split(s) not in canonical order: {unordered}",
        )

    def _split_sets_disjoint(self, dataset: MLDataset) -> MLAuditCheck:
        """No anchor appears under two split labels."""
        overlaps: list[str] = []
        seen: dict[str, MLSplit] = {}
        for split in MLSplit:
            for anchor in dataset.for_split(split).anchors:
                previous = seen.get(anchor.anchor_event_id)
                if previous is not None and previous is not split:
                    overlaps.append(f"{previous}/{split}")
                seen[anchor.anchor_event_id] = split
        return _verdict(
            "SPLIT_SETS_DISJOINT",
            not overlaps,
            "train, validation, test, holdout, and excluded row sets are "
            "pairwise disjoint",
            f"{len(overlaps)} anchor(s) appear in more than one split: "
            f"{sorted(set(overlaps))}",
        )

    def _holdout_and_excluded_absent_from_fit(self, dataset: MLDataset) -> MLAuditCheck:
        """Holdout and excluded rows never reach a fittable split."""
        reserved = {
            anchor.anchor_event_id
            for split in (MLSplit.NOVEL_ANOMALY_HOLDOUT, MLSplit.EXCLUDED)
            for anchor in dataset.for_split(split).anchors
        }
        leaked = sum(
            1
            for split in FIT_ELIGIBLE_SPLITS
            for anchor in dataset.for_split(split).anchors
            if anchor.anchor_event_id in reserved
        )
        ineligible = sum(
            1
            for split in FIT_ELIGIBLE_SPLITS
            for anchor in dataset.for_split(split).anchors
            if not anchor.supervised_training_eligible
        )
        return _verdict(
            "HOLDOUT_AND_EXCLUDED_ABSENT_FROM_FIT",
            leaked == 0 and ineligible == 0,
            f"{len(reserved)} holdout or excluded row(s) are absent from the "
            f"fittable split(s), and every fittable row is marked supervised "
            f"training eligible",
            f"{leaked} reserved row(s) and {ineligible} "
            f"training-ineligible row(s) reached a fittable split",
        )

    def _schema_and_catalog_fingerprint_match(self, dataset: MLDataset) -> MLAuditCheck:
        """The catalog, the manifest, and the allowlist describe one contract."""
        if self._manifest is None:
            return MLAuditCheck(
                "SCHEMA_AND_CATALOG_FINGERPRINT_MATCH",
                AuditCheckStatus.SKIPPED,
                "no feature manifest was supplied, so the catalog and schema "
                "fingerprints could not be compared; a skipped check is not a "
                "passed check",
            )

        catalog_fingerprint = self._catalog.fingerprint()
        problems: list[str] = []
        manifest_catalog = self._manifest.get("feature_catalog_fingerprint")
        if manifest_catalog != catalog_fingerprint:
            problems.append("manifest feature_catalog_fingerprint")
        manifest_schema = self._manifest.get("feature_schema_version")
        if manifest_schema != self._config.required_feature_schema_version:
            problems.append("manifest feature_schema_version")
        if manifest_schema != self._allowlist.required_feature_schema_version:
            problems.append("allowlist required_feature_schema_version")
        if (
            catalog_fingerprint
            not in self._allowlist.compatible_feature_catalog_fingerprints
        ):
            problems.append("allowlist compatible_feature_catalog_fingerprints")
        if (
            dataset.feature_catalog_fingerprint is not None
            and dataset.feature_catalog_fingerprint != catalog_fingerprint
        ):
            problems.append("dataset feature_catalog_fingerprint")

        return _verdict(
            "SCHEMA_AND_CATALOG_FINGERPRINT_MATCH",
            not problems,
            "the feature manifest, the executable catalog, the reviewed "
            "allowlist, and the ML configuration agree on the feature schema "
            "version and catalog fingerprint",
            f"disagreeing provenance field(s): {sorted(problems)}",
        )

    def _join_key_integrity(self, dataset: MLDataset) -> MLAuditCheck:
        """Feature, label, and split rows accounted for exactly once each."""
        per_split = sum(dataset.for_split(split).row_count for split in MLSplit)
        distinct = len(
            {
                anchor.anchor_event_id
                for split in MLSplit
                for anchor in dataset.for_split(split).anchors
            }
        )
        consistent = per_split == dataset.joined_row_count == distinct
        return _verdict(
            "JOIN_KEY_INTEGRITY",
            consistent,
            f"{dataset.joined_row_count} anchor(s) joined one-to-one to labels "
            f"and split assignments, and account for every split row exactly "
            f"once",
            f"row accounting disagrees: {dataset.joined_row_count} joined, "
            f"{per_split} across splits, {distinct} distinct anchor(s)",
        )

    def _validation_halves_campaign_disjoint(self, dataset: MLDataset) -> MLAuditCheck:
        """No campaign has rows in both validation halves."""
        if self._partition is None:
            return MLAuditCheck(
                "VALIDATION_HALVES_CAMPAIGN_DISJOINT",
                AuditCheckStatus.SKIPPED,
                "no validation partition was supplied, so campaign "
                "disjointness could not be checked; campaign-grouped "
                "partitioning requires the Phase 2 label table, and a skipped "
                "check is not a passed check",
            )
        anchors = dataset.for_split(MLSplit.VALIDATION).anchors
        disjoint = self._partition.campaigns_are_disjoint(anchors)
        return _verdict(
            "VALIDATION_HALVES_CAMPAIGN_DISJOINT",
            disjoint,
            f"no campaign appears in both halves "
            f"({self._partition.partition_a_campaign_count} in validation-A, "
            f"{self._partition.partition_b_campaign_count} in validation-B)",
            "at least one campaign has rows in both validation halves; the "
            "boundary must fall between whole campaign groups",
        )

    def _validation_support_sufficient(self) -> MLAuditCheck:
        """Each half independently carries enough rows to measure anything."""
        if self._partition is None:
            return MLAuditCheck(
                "VALIDATION_SUPPORT_SUFFICIENT",
                AuditCheckStatus.SKIPPED,
                "no validation partition was supplied, so per-half support "
                "could not be checked; a skipped check is not a passed check",
            )
        return _verdict(
            "VALIDATION_SUPPORT_SUFFICIENT",
            self._partition.status is ValidationPartitionStatus.PARTITIONED,
            f"both halves meet every support floor "
            f"({self._partition.partition_a_row_count} and "
            f"{self._partition.partition_b_row_count} row(s))",
            f"insufficient validation support; failing requirement(s): "
            f"{list(self._partition.failing_requirements)}",
        )

    def _no_test_or_holdout_selection_source(self, dataset: MLDataset) -> MLAuditCheck:
        """Nothing fitted or selected may name the test or holdout split."""
        problems: list[str] = []

        # The enum has no TEST and no NOVEL_ANOMALY_HOLDOUT member, so no
        # configuration can name one as a source. Asserted rather than assumed:
        # this check exists to notice if a later edit adds one.
        named = {str(item) for item in ValidationPartition}
        if named & {str(MLSplit.TEST), str(MLSplit.NOVEL_ANOMALY_HOLDOUT)}:
            problems.append("ValidationPartition names an evaluation split")

        for label, source in (
            ("calibration", self._config.calibration.source_partition),
            ("thresholds", self._config.thresholds.source_partition),
            ("fusion", self._config.fusion.selection_partition),
        ):
            if str(source) not in named:
                problems.append(f"{label}.source_partition")

        if self._partition is not None:
            reserved = {
                anchor.anchor_event_id
                for split in (MLSplit.TEST, MLSplit.NOVEL_ANOMALY_HOLDOUT)
                for anchor in dataset.for_split(split).anchors
            }
            crossed = len(reserved & set(self._partition.assignment))
            if crossed:
                problems.append(f"{crossed} evaluation row(s) in a validation half")

        return _verdict(
            "NO_TEST_OR_HOLDOUT_SELECTION_SOURCE",
            not problems,
            "every fitted and selected quantity draws from a validation "
            "partition; the test and novel-holdout splits are not nameable as "
            "a source",
            f"selection provenance problem(s): {sorted(problems)}",
        )


def _verdict(
    name: str, passed: bool, pass_message: str, fail_message: str
) -> MLAuditCheck:
    """Return a pass or fail check, choosing the matching message."""
    return MLAuditCheck(
        name,
        AuditCheckStatus.PASS if passed else AuditCheckStatus.FAIL,
        pass_message if passed else fail_message,
    )


def ml_audit_result_to_markdown(result: MLEligibilityAuditResult) -> str:
    """Render an eligibility audit as Markdown, using aggregates only."""
    lines = [
        "# ML Eligibility Audit",
        "",
        f"- Status: **{result.status}**",
        f"- Allowlist: `{result.allowlist_id}` "
        f"v{result.allowlist_version} (`{result.allowlist_fingerprint[:16]}`)",
        f"- Eligible feature list: `{result.eligible_feature_list_fingerprint[:16]}`",
        f"- Features checked: {result.checked_feature_count}",
        f"- Rows checked: {result.checked_row_count}",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|--------|-------|-------|",
    ]
    for check in result.checks:
        verdict = "pass" if check.passed else str(check.status).upper()
        lines.append(f"| `{check.name}` | {verdict} | {check.message} |")

    lines.extend(["", "## Split sizes", "", "| Split | Rows |", "|--------|-------|"])
    for split, count in sorted(result.split_row_counts.items()):
        lines.append(f"| `{split}` | {count} |")

    if result.validation_partition is not None:
        lines.extend(
            [
                "",
                "## Validation partition",
                "",
                "| Property | Value |",
                "|--------|-------|",
            ]
        )
        for key, value in sorted(result.validation_partition.items()):
            rendered = (
                ", ".join(str(item) for item in value)
                if isinstance(value, list)
                else value
            )
            lines.append(f"| `{key}` | {rendered if rendered != '' else 'none'} |")

    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "- A **skipped** check is not a passed check. The overall status is "
            "`pass` only when every check passed; an unsupplied input makes the "
            "audit fail rather than shrink.",
            "- Passing says the feature contract, the row order, the split "
            "sets, and the validation partition are sound. It says nothing "
            "about detection effectiveness, and no model has been fitted.",
            "- Feature eligibility is opt-in. A feature reaches the design "
            "matrix because a reviewed allowlist admits it, not because the "
            "Phase 3 catalog declares it.",
            "",
        ]
    )
    return "\n".join(lines)
