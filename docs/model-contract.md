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
Milestone 3's preprocessing will impute, and will ship a `<name>__missing`
indicator beside every imputed column so the distinction survives.

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

## 12. Known limitations

**No model training exists yet.** Milestone 2 ships the data contract. There is
no preprocessing, no imbalance handling, no fitting, no calibration, no threshold
selection, no model serialization, no experiment tracking, no champion selection,
no inference, no fusion, no evaluation, no explainability, and no drift
detection. No figure in this repository describes model performance, because no
model has produced one.

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
