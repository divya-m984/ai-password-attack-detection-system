# Model contract

What the machine-learning layer may read, what it must return, and what it
guarantees. This document describes the data contract established in Phase 5
Milestone 2.

**No model exists yet.** Milestone 2 assembles data and audits it. There is no
preprocessing, no fitting, no calibration, no threshold selection, no inference,
and no fusion. Nothing in this repository has been trained, and no figure
anywhere in it describes detection performance.

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

## 15. Known limitations

**No model training exists yet.** Milestone 2 ships the data contract and
Milestone 3 the preparation layer. There is no fitting, no calibration, no
threshold selection, no model serialization, no experiment tracking, no champion
selection, no inference, no fusion, no evaluation, no explainability, and no
drift detection. No figure in this repository describes model performance,
because no model has produced one.

**Preprocessing has no CLI.** It is a library contract exercised by tests. There
is no `ml train`, and preprocessing state is not published as an artifact yet;
`preprocessor.json` arrives with model serialization in Milestone 4, and the
state is shaped to be included there unchanged.

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
