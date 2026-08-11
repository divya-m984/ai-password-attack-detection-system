# Model contract

What the machine-learning layer may read, what it must return, and what it
guarantees. This document covers the data contract established in Phase 5
Milestone 2, the preprocessing and weighting of Milestone 3, the model adapters
and artifacts of Milestone 4, and the calibration and threshold selection of
Milestone 5.

**Nothing has been trained, and no result is claimed.** These milestones ship
library contracts: a model can be fitted, published, verified, calibrated, and
given an operating point, and every one of those is exercised by tests on
hand-specified fixtures. There is no training command, no experiment ledger, no
champion, no fusion, and no evaluation. No figure anywhere in this repository
describes detection performance, because no model has been evaluated.

---

## 1. What the layer reads

Phase 5 consumes the **point-in-time feature snapshots** Phase 3 publishes. Each
row describes one anchor event using only information available when that event
completed:

| Leakage class | Meaning | Count |
|---|---|---|
| `prior_only` | Computed strictly from events in `[t - window, t)` | 169 |
| `current_event_context` | The anchor event's own recorded fields | 15 |
| `baseline_derived` | Derived from a baseline fitted on an approved reference interval | 14 |
| `key` | Join and provenance columns | 3 |

`current_event_context` is legitimate because detection runs *after* an attempt
completes: the outcome, method, and client of the event being scored are
observable at decision time. The three `key` columns — `anchor_event_id`,
`anchor_event_time`, `feature_schema_version` — are never model inputs.

## 2. Labels, splits, and campaigns are isolated to one module

`password_attack_detector.ml.dataset` is **the only module in the ML layer
permitted to read ground truth, split assignments, or campaign metadata.**

The complete project-wide allowlist is exactly two modules:

```
password_attack_detector.detection.evaluation
password_attack_detector.ml.dataset
```

This is enforced, not documented: a test parses the syntax tree of every module
under `src/password_attack_detector/` and fails if any module outside that pair
imports a label-bearing module or symbol — **and** fails if either named module
is absent from the allowlist. Both directions matter. The first stops a third
reader being admitted quietly; the second stops the allowlist being emptied to
make a failure go away. Extending the pair is a reviewed edit to that test.

The boundary is narrow on purpose. One module that reads labels, joins them, and
returns a structure with them in named fields can be reviewed in an afternoon.
A layer where the preprocessor reads "just the split column" and the partitioner
reads "just the campaign id" cannot, and the second reader is always the one that
seems harmless.

Consuming `ml.dataset`'s *output* — `SplitDataset`, `AnchorMetadata`,
`MLDataset` — is not reading a label. That is the whole point of concentrating
the read: downstream modules receive typed structures instead of re-deriving the
join.

### Campaign metadata must be supplied explicitly

`campaign_id` is deliberately absent from every published Phase 3 table. The
splitter reads it internally for group isolation but never publishes it beside a
model input. Campaign-grouped validation partitioning therefore requires the
Phase 2 label table to be passed explicitly, and **refuses to run without it**.
There is no ungrouped fallback.

## 3. Feature inclusion is explicit and opt-in

A Phase 3 catalog feature is **inert** to this layer until a human writes it into
a reviewed allowlist.

The obvious alternative — trusting `FeatureSpec.intended_use` — does not work,
and not for a subtle reason: 198 of the catalog's 201 specs carry the identical
default string, and the field is absent from the catalog's fingerprint fields, so
changing it invalidates nothing. A screen that every feature passes by default,
and that no artifact records, cannot be the authority for admitting a feature
into a trained model. It remains a *necessary* condition; the authority is the
reviewed file.

A feature is eligible only when **all** of the following hold:

1. it exists in the executable Phase 3 feature catalog;
2. its leakage class is not `key`, and its group is not `key`;
3. it is not deprecated;
4. its `intended_use` permits modelling;
5. its leakage class is in the configured allowlist (`preprocessing.include_leakage_classes`);
6. its feature group is in the configured allowlist (`preprocessing.include_feature_groups`);
7. it is explicitly listed in the reviewed allowlist;
8. its recorded decision point matches the one the catalog implies;
9. its name appears in none of the prohibited label, split, campaign,
   post-decision, or model-output lists.

### Adding a catalog feature does not add it to any model

This is the property the whole design exists to provide. A new Phase 3 feature
makes the `NO_UNREVIEWED_CATALOG_FEATURE` audit check **fail by name**. It is
then either admitted with a written rationale or listed under `pending_review`
with the decision deferred explicitly. Neither path is silent, and neither
happens by default.

The review queue is deliberately **configuration-independent**. Narrowing a run
to prior-only features must not make a newly added baseline feature vanish from
the queue — otherwise an unreviewed feature could be hidden by choosing a
narrower configuration, which is exactly the escape hatch this check closes.
What *does* bound the queue is the allowlist's own declared
`governed_leakage_classes`, which is a reviewed statement rather than a run-time
choice.

### Two allowlists ship

| File | `allowlist_id` | Entries | Governs |
|---|---|---|---|
| `configs/ml/features-allowlist-v1.yaml` | `champion` | 198 | every eligible leakage class |
| `configs/ml/features-allowlist-prior-only-v1.yaml` | `prior_only_ablation` | 169 | `prior_only` only |

Both are checked entry-by-entry against the executable catalog when loaded. A
leakage class, feature group, or decision point recorded in the file that
disagrees with the catalog is **refused**, not reconciled — the two would be
describing different systems, and silently preferring one of them is how a
leakage classification stops being true.

Two kinds of outcome are kept distinct:

- A **disagreement** raises. Removed features, changed classifications, and
  schema-version mismatches all fail loudly.
- A **narrowing** is recorded. An admitted feature the configuration excludes is
  listed in `excluded_by_configuration` and left out of the matrix, so a shrunken
  matrix is always explicable.

### Prior-only ablation semantics

The ablation is a named experiment, not a replacement for the champion contract.
It admits only features computed strictly from events before the anchor, so a run
against it answers "how much of the signal is in the history alone" without the
anchor event's own fields or a fitted baseline.

It carries a different `allowlist_id` and therefore a different fingerprint. When
champion freezing arrives in Milestone 7, the lock will record the exact
`eligible_feature_list_fingerprint` it was frozen under, and a lock whose
fingerprint disagrees with the model manifest is rejected. **The ablation can
never silently become the champion's feature contract.**

### Fingerprints

Two digests, both over semantic content only:

- **Allowlist fingerprint** — schema version, allowlist identity and version,
  required feature-schema version, governed scope, and the ordered per-feature
  contract (name, decision point, leakage class, feature group, experiment
  restrictions), sorted by name before hashing.
- **Eligible-feature-list fingerprint** — the allowlist fingerprint plus the
  *resolved* order after configuration narrowing. A narrowed run is therefore a
  different model identity, not the same one smaller.

Neither digest depends on a file path, file bytes, modification time, or the
directory a file was read from. `rationale` and `admitted_in` are excluded,
mirroring the Phase 3 catalog's exclusion of `description`: correcting the wording
of a review note must not invalidate every model that recorded the digest.

## 4. Feature keys and identifiers stay outside X

The design matrix carries the reviewed eligible features and nothing else.

Anchor identifiers, anchor times, split assignments, campaign identifiers,
ground-truth labels, and supervised eligibility all leave `ml.dataset` too — as
*separate, named fields* on `AnchorMetadata` and `SplitDataset`. They are needed:
the partitioner groups by campaign, the auditor checks split disjointness, the
trainer selects rows by split. What must never happen is one of them becoming a
**column**.

Three layers enforce this:

1. the allowlist refuses to *admit* a reserved, prohibited, or output name;
2. `ml.dataset` refuses to *build a matrix* from one, so a hand-constructed
   feature list cannot bypass the file;
3. the audit checks the matrix against every prohibited name anyway.

`features.catalog.PROHIBITED_FEATURE_COLUMNS` now carries the Phase 5 output
vocabulary as well, so a *feature* column with one of those names cannot come
into existence rather than being detected after it has:

```
malicious_probability     malicious_decision_score   flagged_malicious
predicted_scenario        category_scores_json       anomaly_score
fused_flagged             decision_threshold         min_category_score
```

Reading a model's own output back in as a feature is how a pipeline learns to
predict its previous answer, and it is invisible in the metrics because the
feature genuinely is predictive.

### The prohibition is on inputs, not on names

`PROHIBITED_FEATURE_COLUMNS` is an **input boundary**. It governs what may be
accepted *into* a Phase 3 feature snapshot or a Phase 5 design matrix. It says
nothing about what a name may be used for anywhere else.

Every name above remains a legitimate field of a Phase 5 **prediction table,
evaluation report, model artifact, or champion lock** — that is what those names
are *for*. `decision_threshold` belongs on a prediction row; `malicious_probability`
is the calibrated output the whole layer exists to produce. Being on this list is
never a reason to rename an output field, and the approved Phase 5 output names
are not renamed.

The prohibition is on the **direction of travel**:

| Direction | Verdict | Enforced by |
|---|---|---|
| output name → feature snapshot / design matrix | **rejected** | catalog builder, `FeatureValidator`, `ml.dataset`, the allowlist, the audit |
| output name → prediction / evaluation / model artifact | **permitted** | `PROHIBITED_METADATA_FIELDS` deliberately excludes them |

Both directions are asserted by tests: a feature snapshot carrying
`decision_threshold` is rejected at every enforcement point, and a typed Phase 5
output schema declaring all nine names validates cleanly and passes the ML
layer's own field guard.

## 5. Canonical row ordering

Every row that enters a fit, transform, or prediction is sorted by:

```
(anchor_event_time, str(anchor_event_id))
```

The identical key `features/engine.py` sorts events by and `detection/engine.py`
sorts snapshots by. Reusing it rather than inventing a third ordering is the
point: three orderings would agree on most inputs and diverge on the ones that
matter.

The order is applied once, in `ml.dataset`, immediately after assembly, and
re-asserted by `ml.ordering.assert_canonical` at every downstream entry point.
Re-asserting rather than trusting the loader means a stage that reorders rows
internally is caught by the next one.

Duplicate anchors and invalid timestamps raise **before** any sorting happens.
After a sort, duplicates sit next to each other and look like a legitimate tie;
the sort would succeed and the duplicate would ride into a design matrix as two
rows describing one event.

Three things the ordering refuses to use: Python's `hash` (randomised per
process), locale-sensitive comparison (`str` comparison is by code point, and no
collation is consulted), and the wall clock (the only timestamps are the anchors'
own recorded event times).

### The guarantee is the trainer's, not the estimators'

Several scikit-learn estimators are sensitive to row order — ties broken during a
tree split, floating-point accumulation in a summation, the sequence a solver
visits samples in. **Nothing here makes them order-invariant.** What the canonical
sort does is remove the *input's* order from the equation: rows arrive from
Parquet in whatever order they were written and are sorted once. Reproducibility
then follows from the canonical order plus a fixed seed.

## 6. Validation-A / validation-B campaign separation

Validation is used for two different jobs, and doing both on the same rows is a
leak: a calibrator fitted on rows that also chose the operating point makes the
operating point look better calibrated than it is.

| Partition | Reserved for |
|---|---|
| **validation-A** | calibration fitting; the permitted anomaly-threshold source |
| **validation-B** | binary threshold, category abstention threshold, model selection, fusion-strategy selection |

The procedure:

1. take the validation rows, canonically ordered;
2. form groups — **every non-null `campaign_id` is one indivisible group**,
   whatever mix of benign and malicious rows it holds; only rows with **no
   campaign association** become deterministic singletons, keyed by anchor
   identifier;
3. sort groups by `(minimum anchor_event_time, minimum anchor_event_id)`;
4. accumulate whole groups into validation-A until the configured target fraction
   is **first** reached;
5. check support requirements independently for each half.

Grouping happens at step 2 — **before** support is tallied and before the
boundary is placed — so nothing downstream can divide a campaign to satisfy a
target fraction or a support floor.

**The boundary always falls between groups.** A campaign is a coordinated burst by
one operator; its early and late events are the same behaviour twice. A row
midpoint cut would put some in the half that fits the calibrator and the rest in
the half that chooses the threshold, and the threshold would be chosen partly by
memorising a campaign the calibrator had already seen.

**No row-level label overrides campaign membership.** A campaign containing both
benign and malicious events stays whole. Splitting it because some of its rows
are benign would put the same coordinated activity on both sides of the boundary
— the exact leak this partition exists to prevent — and the disjointness check
inspects every non-null `campaign_id`, not only the malicious rows, so a straddle
carried by a campaign's *benign* members is caught too.

### Phase 5 is deliberately stricter than Phase 3 here

The Phase 3 splitter applies `normal_grouping: singleton`, which keys on the
recorded scenario and therefore treats a benign row as ungrouped **even when it
carries a campaign identifier**. That is defensible for the train/validation/test
split, whose benign class would otherwise be discarded wholesale by the exclusion
policy.

It is not good enough for this partition. Validation-A fits the calibrator and
validation-B chooses the operating point, so a campaign with rows on both sides
means the operating point was chosen partly by memorising activity the calibrator
had already fitted on. **Phase 5 keeps every campaign whole and accepts the cost.
The Phase 5 invariant is not weakened to match Phase 3.**

### Campaign metadata is the authority — not identifier spelling

Whether a recorded `campaign_id` denotes a campaign is decided in `ml.dataset`,
at the reader boundary, before any grouping happens. Downstream, `campaign_id` is
either a real campaign or `None`, and the partitioner applies one unconditional
rule to it and reads no label at all.

The **supplied campaign metadata is the authoritative source of campaign
membership.** Resolution consults it first, and only then considers anything
else:

| Recorded identifier | Declared by the metadata? | Resolves to |
|---|---|---|
| any spelling, including `normal-123` | yes | that campaign — one indivisible group |
| matches the generator's documented placeholder contract `normal-<digits>` | no | no campaign association (deterministic singleton rows) |
| anything else | no | **failure**, code `MLD001` — a dangling campaign reference |
| null, empty, whitespace-only | — | no campaign association |

A campaign is *declared* when the metadata records campaign structure for it —
`campaign_stage`, which the Phase 2 generator writes on every campaign it runs
and on nothing else — **or** when the identifier is simply not the documented
placeholder shape. The asymmetry is deliberate: a recorded stage is a positive
declaration and outranks the placeholder shape, while the absence of a stage
declares nothing either way and can never demote an identifier on its own. An
ordinary campaign table that carries no stages therefore behaves exactly as it
always did — everything in it is a campaign.

**Identifier shape is a recogniser of last resort, never an authority.** Matching
`normal-<digits>` is a *necessary* condition for resolving to "no campaign",
never a sufficient one. A real campaign named `normal-123` that the metadata
declares stays a campaign, and its rows stay together; an integration test
publishes the same dataset twice under both namings and asserts the two
partitions are identical.

**An undeclared, non-placeholder identifier stops the assembly.** It is never
quietly turned into singleton rows: doing so would scatter whatever it names
across both validation halves, and the resulting partition report would look
perfectly healthy — the exact failure this layer exists to make impossible.

None of this reads a label. Whether a row is benign or malicious has no bearing
on which campaign it belongs to, so a genuine campaign containing benign rows
stays whole — a test asserts exactly that.

#### Why the placeholder is recognised at all

`GroundTruthLabel.campaign_id` is a required field, so the Phase 2 generator has
to write something on every row — including ordinary background traffic that
belongs to no campaign. It writes `normal-<seed>`: one value shared by every
benign event in the dataset, naming no coordinated activity.

Treating it as a campaign would fuse the entire benign class into a single
indivisible group. Because that group is most of the validation split, it would
land wholly in one half and leave the other with **zero benign rows**, making
every partition on generator output report insufficient support.

### There is no fallback

Two halves that are each too small to measure anything are reported as
`INSUFFICIENT_VALIDATION_SUPPORT`, naming the failing requirements as stable
codes (`validation_a:min_partition_positive_rows`, and so on). The boundary index
is never adjusted after it lands, and the rows are never halved anyway. A
partition that looks usable and is not would be worse than no partition.

Support is checked **independently for each half**. A combined check would pass a
partition where one half carries everything, which is the failure mode most worth
catching.

The partition fingerprint covers the grouping policy, the support policy, and the
ordered group assignment. The support policy is included because two runs that
partitioned the same rows the same way under different support floors reached
different *conclusions* about whether those halves were usable.

Test and novel-holdout rows never enter either half. The
`ValidationPartition` enum has **no** `TEST` member and no holdout member — the
absence of the member is the enforcement.

## 7. Null versus zero semantics

Unchanged from Phase 3, and load-bearing:

- **null** means the quantity is **undefined** for this row — no attempts were
  made in the window, no baseline was fitted;
- **zero** means it was **observed to be zero** — attempts were made and none
  failed.

Milestone 2 imputes nothing; nulls are carried through assembly unchanged.
Milestone 3's preprocessing imputes, and ships a `<name>__missing` indicator
beside every nullable column so the distinction survives — see §13.

`NaN` and infinity are **refused**, not carried. `NaN != NaN` breaks the exact
equality comparisons this layer's determinism guarantees rest on, and a null
already has a precise meaning that `NaN` would blur.

## 8. Known-category class derivation

The secondary head's class space is derived from `ScenarioType` at runtime:

```
sorted(set(ScenarioType) - {normal, novel_anomaly_holdout})
```

Seven classes. Derived rather than written down, so adding a scenario to the
generator extends the class space, and renaming one fails a test instead of
scoring quietly against a class that no longer exists.

`normal` is excluded because the head is fitted on malicious rows only.
`novel_anomaly_holdout` is excluded because its whole purpose is to measure
behaviour on attacks no class was fitted for.

A row whose class is outside the space — a benign row, a holdout row, or a
malicious row carrying an unrecognised class — carries `None`, never the nearest
known class. `UNKNOWN_CATEGORY` exists as a first-class *prediction* outcome even
though it is never a *fitted* class: forcing an unrecognised row into the nearest
known category would report a confident answer the model does not have, and would
quietly convert novel behaviour into a class somebody already wrote a rule for.

## 9. Novel holdout isolation

Novel-anomaly-holdout rows are assembled and split-scoped like any other, and
then go nowhere near a fit. `FIT_ELIGIBLE_SPLITS` has exactly one member —
`train` — and the audit's `HOLDOUT_AND_EXCLUDED_ABSENT_FROM_FIT` check verifies
that no holdout or excluded anchor appears in it and that every fittable row is
marked supervised-training-eligible.

Holdout rows never enter a validation partition, never contribute to a threshold
grid, and are reported separately from supervised metrics when evaluation arrives.

## 10. The eligibility audit

`MLEligibilityAuditor` runs fifteen named checks:

| Check | What it verifies |
|---|---|
| `NO_PROHIBITED_COLUMNS` | no ground truth, campaign, or model-output name in the matrix |
| `NO_KEY_CLASS_COLUMNS` | no column classified `key` by the catalog |
| `EVERY_COLUMN_TRACES_TO_CATALOG` | every column has declared semantics |
| `ALLOWLIST_COVERS_EVERY_MATRIX_COLUMN` | every column traces to a reviewed admission with matching metadata |
| `NO_UNREVIEWED_CATALOG_FEATURE` | no catalog feature in scope is undecided |
| `NO_LABEL_OR_SPLIT_IN_MATRIX` | no label, split, identifier, or provenance column |
| `NO_CAMPAIGN_IDENTIFIER_IN_MATRIX` | campaign metadata reached the partitioner only |
| `ROWS_ARE_CANONICALLY_ORDERED` | every split is in canonical order |
| `SPLIT_SETS_DISJOINT` | no anchor under two split labels |
| `HOLDOUT_AND_EXCLUDED_ABSENT_FROM_FIT` | reserved rows are absent from fittable splits |
| `SCHEMA_AND_CATALOG_FINGERPRINT_MATCH` | manifest, catalog, allowlist, and config agree |
| `JOIN_KEY_INTEGRITY` | rows joined, rows across splits, and distinct anchors all agree |
| `VALIDATION_HALVES_CAMPAIGN_DISJOINT` | no campaign in both halves |
| `VALIDATION_SUPPORT_SUFFICIENT` | each half independently meets its floors |
| `NO_TEST_OR_HOLDOUT_SELECTION_SOURCE` | nothing selected draws on an evaluation split |

### A skipped check is not a pass

This is the rule the audit inherits from the Phase 3 leakage auditor, and it is
the reason `ml audit-features` **requires** `--campaign-labels` and
`--feature-manifest` rather than treating them as optional. A check whose input
was not supplied reports `SKIPPED`, and the overall status is `FAIL`.

Fourteen passes and one omission reads, at a glance, like fifteen passes — and
the omitted one is always the expensive one. Reporting an unevaluated check as
passed would make an audit look like evidence for something nobody measured.

There are exactly two aggregate statuses, `pass` and `fail`. There is no
`warning`: a leakage finding is not a matter of degree, and a third status would
invite a run to proceed on one.

### Every check here is evaluable now

Nothing in this module reports a future capability as verified. There is no gate
check, no threshold-provenance check over thresholds that do not exist yet, and
no calibration check. Those arrive with the code that makes them meaningful.

### Output is aggregate only

Messages carry counts and declared column names. No anchor identifier, campaign
identifier, entity pseudonym, feature value, absolute path, or secret reaches a
message, the JSON report, or the Markdown rendering — and a test sweeps all three
plus the console output for exactly those shapes.

Reports are published as `reports/ml_eligibility_audit.json` and
`reports/ml_eligibility_audit.md`.

## 11. Running the audit

```bash
uv run password-attack-detector ml audit-features \
  --features         data/processed/feature_snapshots.parquet \
  --labels           data/processed/feature_labels.parquet \
  --splits           data/processed/feature_splits.parquet \
  --campaign-labels  data/interim/labels.parquet \
  --feature-manifest data/processed/feature_manifest.json \
  --allowlist        configs/ml/features-allowlist-v1.yaml \
  --config           configs/ml/model-development.yaml
```

Exits zero only when every check passed.

To draft a new allowlist after a catalog change:

```bash
uv run password-attack-detector ml catalog --emit-allowlist configs/ml/draft.yaml
```

The result is a **starting point for review**, never a finished contract: the
rationale it writes for each feature is derived from that feature's own
classification, which is precisely the reasoning a review exists to challenge.
Read it, replace the rationales, decide what to defer, and commit it.

## 12. Preprocessing (Milestone 3)

`ml.preprocessing` turns the reviewed raw feature matrix into a numeric design
matrix. It reads no Parquet, no label, no split assignment, and no campaign
identifier: rows arrive as a typed frame exposing exactly `feature_names`,
`feature_matrix`, `anchors`, and `split`, and an anchor exposes only its
identifier and event time — enough to assert canonical order, and nothing more.
The label-reader allowlist stays the two modules §2 fixed it at.

### Fitting is train-only, and the split decides

`fit_preprocessor` refuses any frame whose split is not `train`. The check is
against `FIT_ELIGIBLE_SPLITS`, the same one-member set that governs model
fitting, so widening one widens both visibly.

`transform` accepts every split — encoding validation, test, and holdout rows is
the normal case. What it never does is write back: `FittedPreprocessor` is a
frozen pydantic model, so there is no `partial_fit`, no attribute a transform
could update, and no historical state a re-fit could overwrite. Fitting again
produces a *new* object.

### Null versus zero, carried through

Phase 3's doctrine (§7) survives the transform:

| Raw value | Value channel | `<name>__missing` |
|---|---|---|
| observed number | that number | `0.0` |
| observed zero | `0.0` | `0.0` |
| null (nullable feature) | train statistic | `1.0` |
| null (non-nullable feature) | **rejected** | — |

The indicator is not optional and cannot be configured off: imputing a null
without flagging it hands an estimator a fabricated observation and deletes the
distinction in the same step.

The imputation statistic is the **train-only median** by default (`zero` is the
other declared policy). The even-length rule is pinned in project code — the two
middle order statistics, averaged — rather than delegated, so a library changing
its tie convention cannot move every imputed cell in the repository. Sorting is
by value, so the constant does not depend on row order.

A feature that is **null throughout training** has an undefined median. The
behaviour is defined rather than left to chance: the value channel is filled
with `0.0`, `all_null_in_train` records why, and the indicator — constant `1.0`
across every training row — carries the whole story. Failing the run instead
would let one feature that happens to be undefined over the training window
block two hundred that are fine.

`NaN` and infinity are refused on entry and cannot appear on exit.

### Booleans

`False` encodes to `0.0` and `True` to `1.0`. A nullable boolean also carries a
`<name>__missing` indicator, and the **pair** is the encoding: `(0.0, 0.0)` is an
observed false and `(0.0, 1.0)` is a missing one. Filling the value channel with
a third number — `-1`, say — was rejected: it invents an observation that never
happened and orders it below false, which is meaningless for a boolean and
actively wrong for anything treating the column as continuous.

Booleans are never standardized. A 0/1 channel stays on 0/1.

### Categorical encoding

Vocabularies are learned from canonical training rows only and **sorted by code
point**, never by encounter order, so file order cannot reorder the columns.

Three synthetic buckets, and they mean different things:

| Bucket | Meaning |
|---|---|
| `__missing` | the value was null (nullable features only) |
| `__unknown` | training never saw this value |
| `__other` | training saw this value and it was too rare to keep |

Exactly one column is hot in every block, including for a null and for an unseen
value. A block of all zeros would say "none of the above" without saying which
none.

The two frozen rules:

- **A category rare in training stays `__other`** however common it later
  becomes.
- **A category unseen in training stays `__unknown`**, never `__other`.
  Collapsing them would tell a model that an unheard-of country resembles the
  rare ones.

Neither can enlarge the vocabulary: an unseen value at transform time maps to
`__unknown` and changes no serialized state.

**Reserved-token collisions are refused, not resolved.** Every bucket label
begins with `__` (enforced by config validation), and an observed training
category beginning with `__` fails the fit. Either resolution — renaming the
value or merging it into the bucket — would silently make one column mean two
things.

**Rare bucketing is a reviewed list, not a rule that switches itself on.**
`PreprocessingConfig.rare_category_features` names the features it applies to;
`current_country_code` is the reviewed member, because a long tail of countries
each seen a handful of times is a memorisation surface rather than a signal. A
threshold that armed itself when a vocabulary happened to grow would change a
model's feature set because the *data* changed. Within a named feature, a
training count strictly below `min_category_frequency` is bucketed; a count
equal to it is kept. The `__other` column is emitted only when at least one
training category was actually bucketed.

A retained vocabulary above `max_category_cardinality` fails the fit: a
categorical wide enough to identify a row is not a categorical.

### Standardization

Governed by `standardize_numeric_for_linear_models`. When enabled, location and
scale are fitted on training rows only, using the mean and the **population**
standard deviation.

A constant column gets `scale = 1.0`, recorded with `zero_variance: true`.
Centring already sends it to zero, so any other divisor is either a division by
zero or an arbitrary inflation of a column carrying no variation. The test is
applied at the precision the scale is *stored* at, so a deviation of `1e-12`
counts as zero rather than being written and then divided by.

Only numeric **value channels** are scaled. Missingness indicators and one-hot
columns are already on `{0, 1}`; standardizing them would turn "this value was
absent" into a number whose meaning depends on how often it was absent in
training.

One frozen preprocessor serves every model family. The configuration field is
named for linear models because that is *why* the policy exists, not because the
matrix is family-specific: standardization is strictly monotone per column, so
tree families are unaffected by it. No family-specific preprocessing exists, and
none is planned before the adapters land.

### The output contract

Output columns are emitted in **raw feature order**, and within each feature:

- numeric: `<name>`, then `<name>__missing` if nullable;
- boolean: `<name>`, then `<name>__missing` if nullable;
- categorical: one column per retained category in vocabulary order, then
  `<name>=__other` if emitted, `<name>=__missing` if nullable, then
  `<name>=__unknown`.

`=` cannot occur in a Phase 3 feature name, so `a=b` can only have come from
feature `a`. Uniqueness across the whole output list is asserted anyway — a
separator argument is not a proof.

The transformed matrix is `float64` for a common estimator interface. The
original semantics stay explicit in state: which features were numeric, which
were boolean, which categories existed, and what each column means.

### Serialized state

`FittedPreprocessor` round-trips through canonical JSON — sorted keys, ASCII, no
pickle and no joblib — and carries:

preprocessing schema version · raw feature order · eligible-feature-list
fingerprint · preprocessing-config fingerprint · numeric imputation statistics ·
missing-indicator declarations · categorical vocabularies · reserved-bucket
policy · rare-category thresholds · boolean encoding policy · scaling state ·
transformed output order · transformed feature count · fitted train row count ·
its own fingerprint.

Fitted statistics are quantized to nine decimals **when they are fitted**, not
only when they are written, so the number a transform multiplies by is the number
the JSON carries: `from_json(to_json())` is byte-identical and produces
numerically identical output.

Loading is strict. An unknown field, a missing field, malformed JSON, or a schema
version this build does not implement all fail. Loading loosely would drop what
it did not understand and then recompute a fingerprint over the remainder.

Identity is semantic: no path, no wall-clock timestamp, and no machine detail
ever enters the state, so the same rows fitted twice, a year apart, in two
directories produce the same fingerprint.

### Privacy boundaries

Fitted state is aggregate and semantic. It carries no anchor identifier, no
campaign identifier, no pseudonym, no coordinate, no credential, and no absolute
path — swept by test.

Category vocabularies get special treatment, because a vocabulary is the one part
of the state built from **observed values** rather than from counts. Two rules:

- only features the catalog declares `non_sensitive` may have a vocabulary
  serialized at all;
- an individual value shaped like a pseudonym (`usr_`, `src_`, `dev_`, `app_`,
  `ses_`), a path, or a `lat,lon` pair **fails the fit**.

The audit fails rather than publishing. Eligibility asks "may a model use this";
this asks "may an artifact publish these values", and they are different
questions — a feature can pass every eligibility check and still be the wrong
thing to write a vocabulary of.

## 13. Class weights (Milestone 3)

`ml.imbalance` computes class weights and does nothing else. It imports no label
type, opens no file, and receives a plain sequence of class values from whichever
component already read the labels.

**No resampling, ever.** `ImbalanceConfig.resampling` admits one value.
Oversampling duplicates rows a split boundary already placed, undersampling
discards evidence, and synthetic minority generation invents authentication
events that never happened and then measures a detector against them.

Three policies:

| Policy | Weight |
|---|---|
| `none` | `1.0` for every class; counts still recorded |
| `balanced` | `n_samples / (n_classes * count(class))` |
| `fixed` | the reviewed weights from configuration |

The balanced formula is implemented in project code rather than delegated to
`sklearn.utils.class_weight`, so the number in a manifest can be recomputed by
hand from the counts printed beside it. A test asserts the module imports no
`sklearn` name.

What fails rather than defaulting:

- a class with **zero training rows** under `balanced` — that is a split or label
  problem to fix, not a weight to invent;
- a class value outside the declared class order;
- a fixed mapping naming a class the task does not declare, or omitting one it
  does;
- a zero or negative weight, at configuration validation — zero silences a class
  and a negative weight inverts it.

`ClassWeightState` is frozen, JSON-round-trippable, and fingerprinted over task,
class order, policy, training counts, weights, and the configuration fingerprint.
Weights are quantized on computation, so a reload cannot drift. Class order is
declared, never inferred from which classes a particular split happened to
contain — an order that moved with the data would leave every recorded
fingerprint pointing at the wrong arrangement.

Row order does not affect the result: values are counted, not scanned.

## 14. Leakage is proven behaviourally

The strongest Milestone 3 acceptance test is not structural. A structural check
("no split column is read") can be satisfied by code that leaks anyway.

The behavioural test fits preprocessing and class weights on training rows, then
transforms validation, test, and holdout fixtures built to move anything that
reads them — different numeric distributions, different missingness, different
category frequencies, categories training never saw, a numeric scale a million
times larger, flipped booleans — and asserts the serialized state is **byte
identical** afterwards. Imputation values, vocabularies, rare buckets, scale
statistics, output order, and the fingerprint are each asserted individually, so
a failure names the part that leaked.

Two converses keep it honest:

- a **fresh fit** on the same training rows after all that perturbation reaches
  the same state, which is what proves the statistics are a function of the
  training rows alone rather than of whatever the process happened to see;
- **perturbing training** *does* move the fingerprint — five separate edits, one
  per semantic — because a state that never changed would pass the first test
  trivially.

Row-order determinism is proven the same way: shuffled source rows are
canonicalized by Milestone 2 and reach byte-identical fitted state, while the
low-level fit and transform **reject** non-canonical rows rather than sorting
them. Dataset assembly may sort; a fitted-stage API that sorted silently would
hide the fact that somebody handed the trainer rows it had not canonicalized.

## 15. Model adapters and artifacts (Milestone 4)

### Two halves, deliberately separated

Every family has a **fit** path that may use scikit-learn, and a **score** path
that may not. Scoring reconstructs the output from the stored artifact alone, in
project code, with no estimator present.

That is what makes the round-trip parity test evidence rather than tautology:
the two paths do not share an implementation, so agreement between them means
something. It is also what lets a published model outlive the release it was
fitted under — the artifact is numbers and metadata, and reading it needs no
estimator at all.

Adapters read no Parquet, no label file, no split file, no campaign metadata,
and no feature manifest. A `TrainingBatch` arrives already assembled by
Milestone 2, already transformed by Milestone 3, and already canonically
ordered — and `fit` *asserts* that ordering rather than trusting it. The
preprocessor is carried through frozen; nothing can mutate it, because
`FittedPreprocessor` has no mutable field.

### Why pickle and joblib are not authoritative

Unpickling executes whatever the payload asks for. That makes a model file a
code file — and a model file is exactly the artifact most likely to be copied
between machines by somebody who did not produce it. A pickle is also opaque to
review, unstable across library versions, and impossible to diff.

So the authoritative artifact is canonical JSON beside a deterministic array
archive, and loading it constructs project types only. Neither `pickle` nor
`joblib` is imported anywhere in the ML layer, asserted by an AST test over
every module.

### The published directory

| File | Contents |
|---|---|
| `model.json` | contract, hyperparameters, orders, array manifest, score semantics |
| `arrays.npz` | every fitted number, deterministically encoded |
| `preprocessor.json` | the Milestone 3 state the matrix was built by |
| `model_manifest.json` | integrity and provenance, **written last** |

A closed set, not a minimum. An unexpected file fails verification: a loader
that tolerates extra files is a loader somebody can put something in.

There is no `calibrator.json`, and its absence is *stated* rather than left to
be inferred — both the document and the manifest record
`calibration_status: not_fitted`, and a document claiming otherwise is refused.

### Deterministic archive semantics

`numpy.savez` is not used for publication. It writes a ZIP whose members carry
the wall-clock time they were written, in whatever order the mapping iterated,
with whatever compression and platform markers the interpreter defaulted to. Two
runs producing the same model would produce different files.

Every varying field is pinned instead:

| Property | Value |
|---|---|
| member order | sorted by array name |
| member timestamp | 1980-01-01 00:00:00, the ZIP epoch |
| compression | deflate, fixed level |
| platform marker | `create_system = 0` |
| permissions | `0o644` |
| `.npy` format version | 1.0 |
| dtype | declared per array, little-endian |
| memory order | C-contiguous |

Arrays are normalised before writing: object, void, and string dtypes are
refused outright; NaN and infinity are refused; declared sizes and member counts
are bounded. Nothing about the machine reaches the bytes — no path, no host
name, no user name, no temporary directory, no modification time.

Tests write the same semantic arrays in two directories, at wall-clock times a
measurable interval apart, from mappings with different insertion order, and
assert the bytes and the SHA-256 are identical.

**Deterministic bytes are a property, not the identity.** A future compression
change would alter every byte while changing no model.

### Model identity

`model_content_fingerprint` is a SHA-256 over the canonical content: the array
values, the feature and class orders, the hyperparameters, and the upstream
fingerprints. `model_id` is a UUIDv5 over that fingerprint together with the
task and family.

Identity never reads the directory name, the model alias, the output path, a
file timestamp, an archive timestamp, or the moment of publication. The same
model written twice in two places is the same model — asserted by test, as is
the converse: one changed coefficient, or one reordered column, changes it.

`published_at` is recorded because the Phase 2 manifest convention has one, and
it is excluded from every fingerprint and from the identity.

### Round-trip parity

Every publishable family is fitted, published, reloaded, and re-scored, and the
two score matrices are compared.

| Family | Bound | Observed |
|---|---|---|
| M-000 prior baseline | exact | exact |
| M-001 threshold baseline | exact | exact |
| M-010 logistic regression | `1e-12` | `1.1e-16` |
| M-020 random forest | **exact** | exact |
| M-021 histogram boosting | `1e-12` | exact (not published) |
| M-030 isolation forest | `1e-9` | `1.1e-16` |

The bounds differ because the arithmetic does. A tree traversal is comparisons
and a mean of stored rows, so anything short of bit-equality would mean the two
implementations disagree about a *split* rather than about a rounding step —
exact is the only defensible target. A logistic score adds an exponential, and
the isolation forest reimplements a formula involving a logarithm and Euler's
constant; for those, one float64 rounding step is what is defensible, and the
observed error is a single ULP.

**A tolerance was never widened to make a test pass.** The forest's exact target
caught a real defect: scikit-learn's tree predictor casts its input to float32
before comparing against a float64 threshold, and a fitted threshold is a
midpoint computed in that same float32 space. A float64 comparison agrees on
almost every row and disagrees on any row whose value ties the threshold at
float32 — which cost `1.5e-2` on the first matrix containing a standardised
column. The cast is reproduced (`TREE_INPUT_DTYPE`), and a looser tolerance
would have accepted the bug.

Parity is tested on ordinary values, imputed values, `__unknown` and `__other`
channels, rows placed exactly on real split thresholds, repeated scoring,
reordered rows, and models saved and loaded in different directories.

### The verification chain

`ml verify-manifest` and `InferenceModel.load` share the same fail-closed order,
cheapest and most sceptical first. Each step assumes only what the previous ones
established:

1. directory exists — 2. exactly the declared files — 3. no symbolic links —
4. size ceilings — 5. manifest parses under a supported schema — 6. every
recorded SHA-256 matches — 7. `model.json` parses — 8. manifest and document
agree on identity, family, task, orders, and every upstream fingerprint —
9. the archive holds exactly the declared arrays with the declared dtypes,
shapes, and value digests — 10. the model identity recomputes.

Loading continues: 11. serializer id and version supported — 12. inference
adapter id supported — 13. family present in the closed registry — 14. catalog
entry agrees — 15. runtime scikit-learn inside the recorded bounded range —
16. feature-contract fingerprints match — 17. preprocessor fingerprint matches —
18. transformed order matches — 19. class order valid.

Only then is an adapter constructed. A test proves the order holds by making
every adapter constructor raise and asserting a tampered artifact still fails
with a verification error.

### Security restrictions

A model directory is untrusted data. The loader refuses path-traversal and
absolute member names, symbolic links, unexpected files, duplicate logical
entries, malformed JSON, unknown JSON fields, object and void arrays, members
requiring `allow_pickle`, unsupported dtypes, non-finite arrays, declared shapes
and counts above the configured ceilings, checksum mismatches, fingerprint
mismatches, serializer mismatches, and dependency incompatibility.

Family dispatch is a dictionary lookup against a hand-written registry. There is
no `importlib`, no `__import__`, no `eval`, no `exec`, and no attribute path from
artifact content to code — asserted by walking the syntax tree rather than
grepping, so the prohibitions named in a docstring are not mistaken for uses.
`np.load` is never called with `allow_pickle=True`; the archive reader validates
names, counts, and sizes before any buffer is interpreted.

Verification executes nothing. A test plants an `__init__.py` and a
`sitecustomize.py` in a model directory and asserts neither runs.

### The registry and the catalog

`MODEL_IMPLEMENTATIONS` is a closed, hand-written mapping, checked against the
Milestone 1 catalog at import in both directions: a declared family with no
implementation and an implementation with no catalog entry are both failures, as
are disagreements about the catalog model id, the serializer id, the inference
adapter id, or the supported tasks.

| Model | Family | Publishable | Champion-eligible | Serializer |
|---|---|---|---|---|
| M-000 | prior baseline | yes | **no** (reference baseline) | `json_prior_v1` v1 |
| M-001 | single-feature threshold | yes | yes | `json_threshold_v1` v1 |
| M-010 | logistic regression | yes | yes | `json_linear_v1` v1 |
| M-020 | random forest | yes | yes | `json_tree_ensemble_v1` v1 |
| M-021 | histogram boosting | **no** | **no** (private-interface gate) | `json_histogram_ensemble_v1` v1 |
| M-030 | isolation forest | yes | **no** (anomaly-only) | `json_isolation_forest_v1` v1 |

The champion-eligible set is therefore exactly **{M-001, M-010, M-020}**, and
each absence has its own reason rather than a shared one.

Champion *eligibility* is not selection. Nothing in this milestone selects
anything, and every manifest records `champion_status: not_selected`.

### M-000: the reference baseline, never a candidate

M-000 is `champion_eligible = false`, permanently, with
`eligibility_status: reference_baseline` and a typed `reference_baseline` flag.
It is **not** experimental: it is fully implemented, fully publishable, fully
reportable, and its serializer and inference adapter are as complete as any
other family's. The only thing it may never be is the winner.

The reason is architectural, not a judgement about its quality:

- a candidate qualifies by **beating M-000** on the configured validation gate,
  and a model cannot meaningfully beat itself;
- `NO_ELIGIBLE_CHAMPION` has to stay reachable when M-001, M-010, and M-020 all
  fail their gates — and it would not be, if the comparator were itself a
  candidate;
- a baseline inside its own contest becomes an automatic fallback champion,
  which is the quietest possible way for a selection process to always succeed.

The invariant is stated in three places and required to agree in all three: the
catalog entry, the adapter class, and every fitted model. `ModelSpec`'s
validator refuses `champion_eligible` on a reference baseline and ties the flag
to the status in both directions, and `assert_registry_matches_catalog` compares
the adapter's declaration against the reviewed entry — so neither can be edited
alone. `InferenceModel.load(require_champion_eligible=True)` refuses an M-000
artifact while loading it perfectly well without that requirement.

Selection logic itself belongs to a later milestone; nothing here implements it.

### M-021: the private-attribute gate

Histogram gradient boosting fits, and reproduces its estimator exactly. It is
still not champion-eligible, and the reason is not accuracy.

Serialising it requires reading `_predictors` and `_baseline_prediction`. Both
are private. Neither appears in the scikit-learn API reference, neither carries a
deprecation policy, and the structured dtype of `predictor.nodes` is an internal
layout a patch release may rearrange. Every other family is serialised from
documented attributes, which is what lets the reviewed version range be a review
gate rather than a hope.

So the dependency is made explicit rather than hidden: `PRIVATE_ATTRIBUTES`
names exactly what is read and why, `REQUIRED_NODE_FIELDS` names the structured
fields and their dtype kinds, and `probe_compatibility()` checks all of it
against the installed release and returns a structured verdict.

**The probe passes on scikit-learn 1.9.0, and the family is still not
promoted.** The gate is whether a serializer contract may rest on an
undocumented interface, and the answer does not change when the interface
happens to be present. The adapter is registered as unpublishable, so its
private-state dependency never reaches a stored file, and it can be deleted
without touching anything else.

### M-030: experimental, unsupervised, never a probability

The isolation forest receives no target — `fit` is handed a design matrix and
nothing else, and the anomaly batch carries no target column at all. Choosing
which rows are benign happens upstream, where labels are legitimately readable.

It cannot become champion: the flag is false on every model it produces, the
catalog records `anomaly_only`, and `MLTask.ANOMALY` is absent from
`SUPERVISED_TASKS`. Its output is an `anomaly_score` under scikit-learn's
convention — negative, lower meaning more anomalous. Its flag threshold is
selected separately, from a benign source and never from the holdout it is
measured against; see §17.

Its scoring path reimplements the published expected-path-length formula because
scikit-learn's `_average_path_length` is private, which is exactly why parity is
asserted numerically rather than assumed.

### Probability terminology

Nothing in this milestone produces a probability.

| Task | Score kind |
|---|---|
| binary malicious | `decision_score` |
| attack category | `class_score` |
| anomaly | `anomaly_score` |

That holds for every family, including the baselines: M-000 and M-001 emit the
same uncalibrated vocabulary as the learned families. No artifact may advertise
`calibrated_probability`, `malicious_probability`, or `attack_probability`, and
every published manifest and document records `calibration_status: not_fitted`.

Two independent guards enforce it. `ScoreSemantics` requires a fitted
calibration method before a calibrated kind is constructible at all, and refuses
prose using "probability", "likelihood", or "confidence" for an uncalibrated
kind. Separately, the model document and the manifest both require
`calibration_status: not_fitted` at this contract version, so an artifact cannot
claim a calibrated score kind while recording that no calibrator was fitted —
tested in both directions.

**Reproducing `predict_proba` is not inheriting its vocabulary.** The parity
tests compare project-owned output against scikit-learn's `predict_proba`,
because that is the estimator output being reproduced and the comparison is the
only way to prove the reconstruction is right. A logistic sigmoid and a forest
vote both land in `[0, 1]` and look exactly like probabilities. Calling one a
probability before a calibrator has been fitted *and* its calibration error
measured is the easiest way for this layer to mislead somebody, so the internal
comparison is explicitly walled off from the external contract, and a test
asserts the same model that passes the parity comparison still declares
`decision_score`.

### Publication is staged

`write_model_directory` builds in a temporary sibling, validates there,
round-trips the archive there, writes the manifest last, and moves the finished
directory into place. An existing destination is backed up only when overwriting
was explicitly permitted, and is restored completely on any failure. A sibling
rather than the system temporary area, because `rename` is atomic only within a
filesystem.

A partial directory is therefore impossible: a failed manifest build leaves no
destination at all, and a failed overwrite leaves the previous model byte for
byte. This is the publication *primitive*; the orchestration that decides when
to call it belongs to the training milestone.

## 16. Calibration (Milestone 5)

### What each split is for

| split | what it may be used for |
|---|---|
| **train** | fit the model, the preprocessor, the class weights, and the anomaly probe's benign quantile |
| **validation-A** | **fit calibration** — and produce an in-sample *fit diagnostic* |
| **validation-B** | **assess the frozen calibrator's quality**; select operating thresholds; later, select the model and the fusion strategy |
| **test** | final locked evaluation only, once everything above is frozen |
| **novel-anomaly holdout** | a separate generalisation probe, and nothing else |

**Validation-A calibration metrics are in-sample diagnostics and are not
evidence for champion eligibility.** A calibrator asked to describe the rows it
was fitted on will describe them well whether or not it generalises. That number
is worth having — it exposes a fit that collapsed to a constant or landed
somewhere absurd — and it is not a measurement of calibration quality.
Authoritative calibration quality comes from applying the **frozen** calibrator
to validation-B, which it has never seen.

Nothing in Milestone 5 reads test or holdout, and no calibration report can name
either: `ValidationPartition` has no member for them.

### Validation-A fits it, and nothing else may

A calibrator is fitted on **validation-A** — the half of the validation split
Milestone 2 set aside for it. Validation-B chooses the operating point. Test and
the novel-anomaly holdout are read once, after everything is frozen, by a
Milestone 6 evaluation with its own record.

Every entry point in `ml/calibration.py` and `ml/thresholds.py` takes a typed
`ScoreSampleSource` and checks it. Provenance is never inferred from a filename,
a directory, or a column that happens to be present: those describe where bytes
were stored, not what the rows are. **There is no flag, keyword, environment
variable, or configuration key that widens a source**, and a test asserts the
refusal message offers none.

A source object *can* name the test split, deliberately. Refusing to construct
one would move the firewall into the type system, where no test could
demonstrate that the selectors themselves refuse — and the selectors refusing is
the property that matters.

### What the module may see

Scores, binary labels, anchors for the ordering assertion, and fingerprints. It
opens no Parquet file, reads no label table, no split table, and no campaign
table. It never refits the model whose scores it consumes and never touches the
fitted preprocessor. The label-reader allowlist is unchanged and remains exactly
two modules — `detection.evaluation` and `ml.dataset`.

### The method is configured, never selected

`CalibrationConfig.method` names **one** method. It is not a candidate list, and
no comparison chooses among candidates.

Selecting a calibration method on validation-A would score every candidate on
the rows that fitted it, and the winner of that comparison is whichever method
overfits hardest. Splitting validation-A again to referee it would leave neither
half large enough to fit or to judge. Using validation-B would spend the
operating point's rows on a measurement, and using test is not an option at all.
So the method is a reviewed decision: written down, fingerprinted, and visible.

### Platt

A two-parameter logistic calibrator, `sigmoid(a · score + b)`, fitted by
Newton-Raphson with a deterministic start, a deterministic step-halving rule,
and a fixed convergence test. `CalibratedClassifierCV` is not used.

Targets are **smoothed** as in Platt (1999): a positive row is fitted against
`(N⁺ + 1)/(N⁺ + 2)` and a negative one against `1/(N⁻ + 2)`. Without the
smoothing the maximum-likelihood estimate does not exist whenever the score
separates the two classes perfectly — `a` diverges — and perfect separation is
entirely possible on generated data. The price is that no calibrated probability
reaches exactly zero or one, which is the correct behaviour for a fit over
finitely many rows.

A fit that exhausts its iteration budget, produces a non-finite parameter, or
stalls reports `convergence_failed` and returns **no state**. That is a
different answer from `insufficient_calibration_support`, and the two are
reported separately because they have different remedies.

Inference needs only the two stored scalars. The sigmoid is evaluated
branch-wise, because `exp(710)` overflows float64 and the naive form returns
`NaN` for exactly the strongly separated rows a detector cares most about.

### Isotonic

A weighted pool-adjacent-violators fit, written in project code. The
authoritative state reduces to the two attributes scikit-learn documents on a
fitted `IsotonicRegression`:

| stored field | scikit-learn attribute |
|---|---|
| `x_thresholds` | `X_thresholds_` |
| `y_thresholds` | `y_thresholds_` |

No private attribute is read, stored, or depended on. Duplicate scores are
collapsed into one breakpoint whose target is the mean of theirs and whose
weight is their count, so forty rows at one score weigh forty and their
presentation order does not matter. Interior breakpoints equal to both
neighbours are dropped, which is what makes the stored arrays *equal* to
scikit-learn's rather than merely equivalent.

Outside the fitted domain the output is **clamped**, not extrapolated —
matching `out_of_bounds="clip"`. Extrapolating a monotone step function past its
last breakpoint would invent a relationship nothing was fitted on.

Below `min_isotonic_distinct_scores` distinct scores the fit reports
insufficient support rather than producing a lookup table for a handful of
values.

**Why the fit is project-owned.** No module outside `ml/models` imports
scikit-learn, for the reason §15 gives: an estimator object is not an artifact.
The consequence here is a benefit rather than a cost — a parity test compares
two genuinely independent implementations instead of one implementation with
itself. `X_thresholds_` must match **exactly**, since both sides carry observed
scores unrounded; `y_thresholds_` are fitted values, quantized to the nine
decimals every fitted number in this project is stored at, and agree to `5e-10`.

### `x_thresholds` and selected thresholds are stored unrounded

Fitted *parameters* are quantized. Breakpoints and thresholds are not, because
they are observed scores: rounding one moves the boundary of a step function,
which changes which rows it flags. The number stored is the number measured, and
the number measured is the number a later prediction will apply. Floats
round-trip exactly through JSON, so this costs nothing in reproducibility.

### The probability vocabulary transition

Milestone 4 emits `decision_score`, `class_score`, and `anomaly_score`, and none
of them is a probability. `calibrated_probability` becomes available in exactly
one place — `apply_calibration` — and it is a construction rather than a rename.
A calibrated batch carries, and is validated to carry:

- the source model's identifier and content fingerprint;
- the calibration-state fingerprint;
- the calibration method;
- the source score kind;
- `output_score_kind = calibrated_probability`.

Four claims are refused outright:

| refused | why |
|---|---|
| a state with `method: none` | there is no fitted state for "no calibrator" |
| a calibrated field with no calibrator named | a claim nobody can check |
| calibrating an `anomaly_score` | an unsupervised magnitude is not a supervised probability |
| calibrating an already-calibrated score | that measures the calibrator, not the model |

An ordinary uncalibrated Milestone 4 score is never renamed into a probability
field. `ScoreSemantics` continues to refuse prose using "probability",
"likelihood", or "confidence" for an uncalibrated kind.

### Two evaluations, and only one of them is evidence

The same measurement code produces both reports. What differs is provenance,
and every report says which it is rather than leaving a reader to infer it from
the partition named beside it.

| | `diagnose_calibration_fit` | `evaluate_calibration_quality` |
|---|---|---|
| measured on | validation-A | validation-B |
| `evaluation_kind` | `in_sample_fit_diagnostic` | `out_of_sample_validation` |
| calibrator saw these rows | yes, they fitted it | no |
| `admissible_as_champion_evidence` | **always false** | true when measured on adequate support *and* within the configured ECE ceiling |
| what it is for | spotting a pathological fit | authoritative calibration quality |

Both record `fit_source_partition: validation_a` — where the calibrator was
*fitted* — alongside `source_partition`, where it was *measured*. The state's
field is deliberately named `fit_source_partition` so the two can never be
confused for each other.

Three independent things stop an in-sample diagnostic being substituted for an
out-of-sample measurement:

1. `require_out_of_sample_evidence(report, stage=...)` refuses the wrong kind
   outright — the guard a later champion gate calls instead of reading two
   fields and hoping;
2. `evaluation_kind` and `source_partition` are tied to each other by a
   validator, in both directions;
3. `admissible_as_champion_evidence` is pinned false for every in-sample
   report, whatever else the record claims.

And because each report carries the digest of its own content, editing a stored
payload to relabel it fails the seal check before it reaches any of the three.

The calibrator is **applied** on validation-B, never refitted. It arrives
frozen, `CalibrationState` has no mutable field, and a test asserts its bytes
and its digest are identical before and after the evaluation.

**A real limitation, stated rather than worked around.** Validation-B both
measures calibration and selects the operating threshold. A third partition
would separate them, but the validation split is already halved on campaign
boundaries and a third slice would leave none of them able to measure anything.
Borrowing rows from test is not an option. So the reuse is documented here
rather than hidden.

The measurement primitive itself (`brier_score`) takes numbers and labels and
knows nothing about splits. That is what will let a Milestone 6 test evaluation
reuse it rather than growing a second implementation that drifts — and it is
safe to expose precisely because it decides nothing about where its inputs came
from.

### Calibration metrics

Computed over probabilities this module produces itself from the frozen decision
score — so the numbers measured are provably that calibrator's.

**Brier score** — the mean squared difference between the predicted probability
and the outcome, `mean((p − y)²)`.

**Reliability bins** — equal-width over `[0, 1]`, with edges at `i/n` fixed by
`reliability_bin_count`. Bin `i` covers `[i/n, (i+1)/n)`; the last bin is closed
on the right so a probability of exactly `1.0` lands somewhere. Quantile bins are
deliberately not used: they would move with the score distribution, so two runs
over different data could not be compared bin for bin.

**Expected calibration error** — the support-weighted mean gap between the mean
predicted probability and the observed positive rate, over **populated bins
only**. An ECE that counted an empty bin as a perfect one would improve every
time the data got sparser.

**Support statuses** are not decoration:

| status | meaning |
|---|---|
| `measured` | the quantity is defined and rests on adequate support |
| `insufficient_support` | defined and computed, over too few rows to be evidence |
| `unavailable` | not defined at all — an empty denominator, an empty bin |

An empty bin is present in the output and says it is empty; its rates are
`null`, never `0.0`. A report below `min_calibration_rows`, or missing a class
entirely, carries `insufficient_support` and **no verdict at all** — the
comparison against the configured error ceiling is `null`, because a comparison
against an unmeasured quantity is not a verdict. A small ECE on thin support is
not a pass, and the contract makes it impossible to read as one.

There is no plotting dependency, and matplotlib is not a dependency of this
project.

### The raw-score Brier comparator

A report may carry `raw_score_brier`: the same Brier score computed on the
**uncalibrated** model output, so a reader can see whether calibration helped.
It is available only under a condition, and the condition is mathematical rather
than stylistic.

A Brier score is the mean squared error of a forecast on `[0, 1]`. A decision
score is an ordered magnitude and carries no such promise — the anomaly head's
declared range is `[-1, 0]`, and a binary head's need not be narrower. So:

| the model's declared score contract | comparator |
|---|---|
| bounded to `[0, 1]` | computed, and labelled `raw_score_kind: decision_score` |
| not bounded to `[0, 1]` | `null`, reason `raw_score_not_unit_bounded` |
| not supplied | `null`, reason `raw_score_contract_not_supplied` |
| bounded, but an observation falls outside | `null`, reason `raw_score_outside_declared_bounds` |
| already calibrated | `null`, reason `raw_score_is_already_calibrated` |

**An arbitrary decision score is never clipped into `[0, 1]` to manufacture a
comparison.** A clipped comparator would be the Brier score of a forecast the
model never made, and it would flatter or damn calibration according to how far
outside the interval the raw scores happened to fall. When the model's own
contract and its observations disagree about its range, that is a fault worth
seeing, so the comparator is withdrawn rather than repaired.

**Computing a Brier score against a field does not rename it.** The field stays
`decision_score`, the report records that kind beside the number, and a report
claiming `raw_score_kind: calibrated_probability` is refused. The calibrated
output remains the only thing in this project carrying
`score_kind: calibrated_probability`.

`improves_on_raw_score` is `null` whenever no valid comparator exists — the
common case. A later champion gate should therefore require the configured ECE
and Brier limits on the **validation-B** report, and apply any
improvement-over-raw-score condition **only** where a comparator actually
exists.

---

## 17. Threshold selection (Milestone 5)

### Three thresholds, three predicates

| threshold | read from | predicate |
|---|---|---|
| binary decision | validation-B | `score >= threshold` |
| category abstention | validation-B | `max(class_score) >= min_category_score` |
| anomaly flag | train **or** validation-A | `anomaly_score <= threshold` |

The binary predicate is `>=`, pinned once and stored in every selection. The
difference between `>=` and `>` is only ever one row wide, and it is exactly the
row sitting on the threshold — so the choice is written down rather than left to
whichever comparison somebody typed at a call site.

The anomaly predicate is deliberately inverted. An anomaly score follows the
scikit-learn convention where **lower is more anomalous**, so a flag is `<=`.
The two predicates are separately named constants; two opposite comparisons
sharing one name is how a detector ends up flagging the calmest traffic it can
find.

### Nothing here fits anything

The model is frozen, the preprocessor is frozen, and the calibrator — if there
is one — is frozen. This module counts rows above candidate thresholds and picks
one.

### Candidates are observed scores

A candidate is a value some row actually produced, taken in ascending order, and
capped at `search_grid_size` by an evenly spaced subset that always includes both
extremes. Two consequences:

- **every candidate flags at least one row**, so a "flag nothing" threshold is
  never returned. A detector that never fires is not an operating point, and
  admitting one would make `no_feasible_threshold` unreachable under any
  false-positive ceiling;
- a fixed arithmetic grid is not used, because a grid point between two observed
  scores produces the same confusion matrix as the observed score above it while
  reporting a threshold no row ever justified.

When the search is capped, the selection records `candidates_truncated: true`
alongside `candidate_count` and `distinct_score_count`. A silent cap would let a
report read as though every operating point had been considered.

### Objectives

| objective | maximised / minimised | constraint |
|---|---|---|
| `max_recall_at_max_fpr` | maximise detection rate | false-positive rate ≤ ceiling |
| `max_f1` | maximise F1 | none |
| `min_fpr_at_min_recall` | minimise false-positive rate | detection rate ≥ floor |

Feasibility and objective are decided together, in one place: a constraint
evaluated separately from the quantity it constrains is a constraint somebody
eventually forgets to apply. Candidates are visited in ascending threshold
order, and `tie_break` decides which of several tied candidates survives —
`lowest_threshold` by default, because a tie means the lower threshold detects
at least as much at no extra cost. Every comparison is made on the same
quantized number the selection goes on to report, so a tie is reproducible.

### Support-aware statuses

| status | meaning | `data_selected` |
|---|---|---|
| `selected` | a threshold was chosen from validation data | `true` |
| `insufficient_validation_support` | the partition could not resolve the constraint | `false` |
| `no_feasible_threshold` | support was adequate; nothing satisfied the constraint | `false` |

**Neither negative is answered with a best-available threshold.** A
best-available threshold under a constraint nobody met is a threshold that
quietly violates it.

The distinction between the two is worked through in the false-positive case.
With `N` benign rows the smallest observable non-zero false-positive rate is
`1/N`. If the configured ceiling sits below that, the only way to satisfy it is
to flag no benign row at all — so the ceiling has not been *held*, it has merely
not been *tested*. That is missing evidence:
`insufficient_validation_support`, requirement `false_positive_rate_resolution`.
If the support is ample and every candidate still exceeds the ceiling, that is a
measured negative: `no_feasible_threshold`, requirement
`max_false_positive_rate`. Different facts, different statuses, different
remedies.

Every rate is reported beside its numerator and its denominator, so a reader can
recompute it rather than trust it — and can see that a 0% false-positive rate
over eleven benign rows is not the same claim as one over eleven thousand.

### Curve artifacts

A selection whose support gate passed carries one curve point per candidate:
threshold, TP, FP, TN, FN, precision, recall, false-positive rate, and F1.
Points are unique by semantic threshold and ascending. A rate whose denominator
is empty is `null`, never `0.0`. There is no plotting, no event identifier, and
never one row per event.

No curve is emitted when the support gate fails. Per-threshold rates over
support the layer has just declared unusable are exactly the numbers somebody
would go on to plot.

No test curve is produced by anything in Milestone 5, and no selection schema
declares a field a test number could be written into.

### Category abstention

Chosen on **known-malicious validation-B rows only**. Three things are refused
rather than filtered:

- benign rows — the threshold trades coverage against precision *among attacks*;
- a row whose true category is outside the declared class order — a novel
  pattern has no known category to be right or wrong about, and admitting one
  would let the holdout influence a threshold that exists to test it;
- an unsorted class order — every per-class count is reported positionally.

The predicted class is the argmax, with ties going to the **earliest class in
the declared order**. An arbitrary tie-break would make the reported precision
depend on which class a library happened to enumerate first.

The objective is `max_coverage_at_min_precision`: take the widest coverage whose
known-category precision still clears `min_known_category_precision`
(equivalently, whose category error stays under `1 − that`). Coverage falls as
the threshold rises, but precision does not rise monotonically with it, so every
candidate is evaluated rather than the search stopping at the first feasible one.

At inference a row with `max(class_score) < min_category_score` is emitted as
`unknown`. Equality goes the other way: `max(class_score) >= min_category_score`
means the known class may be emitted.

**The conservative fallback.** When validation-B cannot support the choice — too
few known-malicious rows, or a class with too little support for its precision
to mean anything — or when no candidate clears the floor, the selection returns
the reviewed constant `category.min_category_score` from configuration and
records:

```
data_selected: false
```

It is a usable threshold and a permanently visible admission that no measurement
chose it. A fallback is never described as selected from validation, no coverage
or precision is published alongside it, and flipping the boolean on a stored
record fails the digest check.

### Anomaly threshold

Two configured provenance methods, both benign-only:

| method | reads | target |
|---|---|---|
| `train_benign_quantile` | benign TRAIN scores | `1 − quantile` flagged |
| `validation_a_benign_fpr` | benign validation-A scores | `target_benign_flag_rate` |

Both resolve to the same rule — the largest observed score whose flagged share
stays at or under the target — so the two differ in *where they read*, which is
the only thing that should distinguish them. Ties are never split: a threshold
that flagged some of the rows carrying one score and not others would not be a
threshold.

`AnomalyThresholdMethod` has no member naming test or the novel-anomaly holdout.
The holdout is what this probe is measured against, so a threshold tuned on it
would be measuring itself. The estimator receives no supervised target, the
sample is validated to carry no malicious row, and the output stays an
`anomaly_score` — thresholding a magnitude does not turn it into a probability.
`influences_champion_selection` is pinned false in the record as it is in the
configuration, and the anomaly signal is not part of any fusion.

---

## 18. Three identities, kept separate

| identity | what it is | fingerprint over |
|---|---|---|
| **model** | the fitted estimator's semantics | numbers, orders, hyperparameters, upstream digests |
| **calibrator** | model + validation-A calibration | method, parameters, support, and the model it calibrates |
| **threshold** | model/calibrator + validation-B operating point | objective, threshold, counts, and the chain above |

Selecting an operating point **does not change a model's identity**. No
Milestone 5 code path rewrites a published `model.json`, `arrays.npz`,
`preprocessor.json`, or `model_manifest.json`; the Milestone 4 artifact contract
is untouched and still records `calibration_status: not_fitted`. Milestone 6's
`champion.lock` is what binds the three together, and it can only do that if
they are three.

### The provenance chain

Four links, each checked independently wherever a calibrator meets data:

- **model** — a calibrator fitted for one model cannot be applied to another's
  scores;
- **preprocessor** — nor to a matrix built by different rules;
- **validation partition** — validation-A and validation-B come from one
  campaign-disjoint partitioning and therefore carry the same *parent* digest,
  so a calibrator from one dataset's validation-A cannot be combined with
  another dataset's validation-B;
- **configuration** — a calibrator cannot be reported under a configuration it
  was not fitted under.

Any of the four disagreeing means two runs have been spliced together, and the
splice is refused rather than reported.

### Sealed records

`CalibrationState`, `CalibrationReport`, `ThresholdSelection`,
`CategoryAbstentionSelection`, and `AnomalyThresholdSelection` each carry the
SHA-256 digest of their own semantic content as a **field**, recomputed on every
construction including every deserialization. A record whose content and digest
disagree is refused rather than repaired, so an edited payload is detected
without the reader having to be handed the expected value separately.

All five are frozen, reject unknown fields, check their schema version before
validating anything else, reject non-finite numbers, and serialise to canonical
JSON — sorted keys, ASCII, no incidental whitespace — that round-trips
byte-identically. None carries a path, a timestamp, a host, an event identifier,
a campaign identifier, or a pseudonym; the field names of every published schema
are swept against the prohibited-metadata list at import.

`CALIBRATION_SCHEMA_VERSION` and `THRESHOLD_SCHEMA_VERSION` are both `1.0.0` and
are independent of the model contract's `ML_SCHEMA_VERSION`: a model, its
calibrator, and its operating point change for different reasons.

### The leakage argument, behaviourally

Every Milestone 5 entry point is a pure function of train, validation-A, and
validation-B. There is no parameter on any of them through which a test or
holdout row could arrive except the typed provenance each one checks, and those
checks are exercised directly. Beyond that:

- changing validation-A changes the calibrator, its fit diagnostic, and — since
  the calibrator moved — its validation-B quality report, while leaving the
  category and anomaly thresholds byte-identical;
- changing validation-B changes the quality report and the binary and category
  thresholds, and leaves the calibrator and its fit diagnostic
  **byte-identical** — the sharper of the two directions, since a calibrator
  that shifted when the measurement rows changed would mean validation-B had
  reached the fit;
- changing validation-A *after* a calibrator is fitted changes neither it nor
  its validation-B report: a frozen calibrator is a value, not a view onto the
  rows that produced it;
- changing the benign training scores changes the anomaly threshold and nothing
  else;
- test and holdout rows change nothing at all, because every entry point refuses
  them — demonstrated by offering them to each in turn and comparing the
  calibrator's and the quality report's bytes before and after.

---

## 19. Known limitations

**No training orchestration exists yet.** Milestones 4 and 5 ship model
adapters, artifacts, loading, calibration, and threshold selection. There is no
training command, no experiment ledger, no champion selection, no prediction
publication, no fusion, no evaluation, no explainability, and no drift
detection. No figure in this repository describes model performance, because no
model has been evaluated.

**There is no `ml train`.** Fitting, calibrating, and threshold selection are
library contracts exercised by tests. The model commands are `ml catalog`,
`ml audit-features`, and `ml verify-manifest`; none of them calibrates, selects a
threshold, or reads the test split.

**Calibration on synthetic validation data is not real-world calibration.** A
fitted calibrator here is calibrated against the frozen synthetic validation-A
distribution under the declared protocol, and that is an internal property.
Concretely, none of the following is claimed:

- that these calibrated probabilities are reliable on real authentication
  traffic;
- that a low ECE or Brier score is evidence of production calibration;
- that reliability bins measured on generated traffic generalise to a real
  authentication system.

The generator produces the behaviour somebody wrote into it, so its score
distribution is more separable and less varied than real traffic. A calibration
map fitted to that distribution describes that distribution.

**An in-sample fit diagnostic is not calibration quality.** The validation-A
report exists to expose a pathological fit, and it is marked
`admissible_as_champion_evidence: false` so nothing can offer it as evidence of
calibration. Authoritative quality is the validation-B measurement — which is
itself measured on the same rows that select the operating threshold, since a
two-way split has no third partition to spare. See §16.

**A raw-score Brier comparison is often unavailable, and that is correct.**
Most decision scores carry no promise of living on `[0, 1]`, so there is nothing
mathematically valid to compare against. No score is ever clipped to
manufacture one.

**A threshold is only as resolvable as its denominator.** A false-positive
ceiling below `1/N` on `N` benign rows cannot be held by any threshold that
fires; the layer reports that rather than returning a threshold under it. The
same applies to a benign flag-rate target for the anomaly probe.

**Champion eligibility is a property, not a decision.** Three families are
eligible — M-001, M-010, M-020 — and none has been selected, compared, or
promoted; every manifest records `champion_status: not_selected`. M-000 is
eligible for nothing by design, being the reference the others are measured
against.

**Parity is measured on this release.** The bounds in §15 hold for scikit-learn
1.9.0 with `n_jobs=1`. Bit-for-bit reproduction across machines also needs
`OMP_NUM_THREADS=1`; the tests that depend on it set it rather than mutating any
user-wide configuration.

**A gated family may disappear.** If a future scikit-learn moves the histogram
boosting node layout, M-021's compatibility probe fails and the family is
dropped. Nothing depends on it.

**Imputation is a fabrication, honestly labelled.** A median fills a cell nobody
observed. The indicator says so, and a model may learn from the indicator — but
no statistic recovers the value that was never measured, and a feature null
across most of the training window contributes far less than its column count
suggests.

**Category coverage is training-bounded.** A deployment seeing genuinely new
countries or client types routes them all to one `__unknown` column. That is the
honest encoding, not a good one: the model has no way to distinguish among them,
and that is a reason to refit rather than a property to rely on.

**No real authentication traffic is generated.** This system is defensive and
offline. It never stores plaintext passwords, never cracks credentials, never
automates authentication attempts against any service, and never serves a model
over a network.

**Synthetic data has a ceiling.** Everything measurable here describes generated
traffic with known ground truth. A synthetic generator produces the attack
patterns somebody wrote into it, which means:

- attack behaviour is more separable than real traffic, because it was
  parameterised rather than observed;
- benign behaviour is less varied than real traffic, so a false-positive rate
  measured here is a lower bound at best;
- the class balance is a configuration choice, not a measurement;
- the novel-anomaly holdout probes generalisation to *one designed* unseen
  pattern, not to genuinely novel attacker behaviour.

Passing the eligibility audit says the feature contract, the row order, the split
sets, and the validation partition are sound. **It says nothing about detection
effectiveness.**

**The audit's scope is what it can see.** It checks the assembled dataset against
the declared contract. It cannot detect a feature that was computed incorrectly
but classified correctly — that is what the Phase 3 leakage auditor's behavioural
checks are for — and it cannot detect a leak in data that never reached the split
table.
