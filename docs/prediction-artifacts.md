# Batch prediction and prediction artifacts

Milestone 8 scores rows under a frozen champion and publishes what it said.

It is the first milestone permitted to touch the TEST split, and it is permitted
to precisely because it cannot learn anything from doing so: the model, the
preprocessing state, the calibrator, and the operating point were all fixed by
`champion.lock` before this milestone ran, and nothing here fits, tunes, selects,
or re-derives any of them.

**Prediction is not evaluation.** Everything below produces model output for rows
whose outcome nobody here has looked at. Whether that output was *right* is a
different question, asked once, later, by a milestone that is allowed to open the
test labels. This document is careful about that distinction throughout, because
the two are easy to conflate and expensive to conflate.

---

## 1. The label firewall

Three independent mechanisms, and each is asserted by tests rather than
documented as an intention.

**The signature.** `ml.dataset.load_inference_dataset` takes a feature snapshot
path and a split path. There is no third parameter. A caller cannot supply a
label table by mistake, and a reviewer does not have to check whether one was.
The same holds for `assemble_inference_dataset`, for `predict_binary`, for
`predict_category`, and for `predict_anomaly`.

**The command.** `ml predict` has no `--labels` and no `--campaign-labels`
option. `ml validate` and `ml profile` take a published directory and nothing
else.

**The behaviour.** The integration suite rewrites the Phase 3 label table with
every outcome inverted, re-runs the same prediction, and asserts the published
bytes are identical. If a single byte moved, something read them.

Split membership *is* read, because scoring "the test rows" requires knowing
which rows those are. A split assignment is not an outcome: it was fixed by
Phase 3's chronological splitter long before any model existed.

---

## 2. What may be published, and for which split

| Scope | Role | Requires |
|---|---|---|
| `train` | `supervised_prediction` | a verified `champion.lock` |
| `validation` | `supervised_prediction` | a verified `champion.lock` |
| `test` | `supervised_prediction` | a verified `champion.lock` |
| `novel_anomaly_holdout` | `generalisation_probe` | a verified `champion.lock` |
| `excluded` | — | refused at every layer |

The role is a property of the publication, not of the split enum, and it is part
of prediction identity. The novel-anomaly holdout is a **generalisation probe**
wherever it appears: it exists to be scored by a model that has never seen
anything like it, and merging its rows into a supervised artifact would produce a
table that answers neither question. Publishing it changes nothing about a TEST
publication, the champion, the thresholds, or the category abstention point.

A scope with no rows is refused. Nothing to score is a finding, not an empty
publication.

---

## 3. Champion-locked inference

`FrozenChampion.load` verifies seventeen things against the artifacts they name,
in an order where each step only assumes what the previous ones established:

| # | Checked | Prevents |
|---|---|---|
| 1 | the lock parses at this contract version | a lock from a contract this build does not implement |
| 2 | the lock recomputes its own digest | a hand-edited lock |
| 3 | the scope directory names the lock's own scope key | a lock moved into another scope |
| 4 | a `champion_freeze` receipt names this lock | a lock nobody recorded freezing |
| 5 | the `validation_selection` is on record and agrees | a lock citing a changed selection |
| 6 | the selected run is on record and its receipt matches | a champion whose run left the history |
| 7 | the run's model identity matches the lock | a lock naming a different model than the run it cites |
| 8 | the model manifest's **bytes** digest to the recorded fingerprint | a manifest replaced after the freeze |
| 9 | the model artifact passes the Milestone 4 verifier in full | every structural and integrity failure that verifier catches |
| 10 | serializer and inference-adapter identity | a reader that cannot read this writer |
| 11 | the preprocessor fingerprint | a matrix built by different rules |
| 12 | the calibrator fingerprint, where the lineage has one | probabilities from a different calibrator |
| 13 | the binary threshold fingerprint | an operating point swapped after freezing |
| 14 | the feature catalog fingerprint | a model scored against a different feature catalog |
| 15 | the reviewed allowlist fingerprint | a feature contract nobody reviewed |
| 16 | the ordered eligible-feature fingerprint | the right columns in a different order |
| 17 | the ML config fingerprint and the declared dependency contract | a runtime whose estimator internals may have moved |

`ml predict` additionally checks the Phase 3 **feature manifest** against the
executable catalog, the reviewed allowlist, the configured schema version, and
the lock, so a table built under a different feature contract is refused rather
than scored.

There is no `--force`, no `--ignore-lock`, and no `--model-id`. Every refusal is
a state in which the published predictions would not be attributable to anything
in particular, and an override would be a way to publish them anyway.

M-000 is never a prediction champion — the lock refuses the reference baseline
outright, and the catalog check refuses it again. M-021 remains unpublishable and
M-030 remains outside supervised champion identity.

---

## 4. Binary predictions

For each row, in this order and no other:

1. take the exact reviewed raw feature order;
2. transform with the **frozen** Milestone 3 preprocessor from the champion's own
   lineage;
3. score with the project-owned Milestone 4 inference adapter — no estimator is
   reconstructed and nothing is unpickled;
4. preserve the raw `decision_score` as published;
5. apply the frozen calibrator **only** when one exists in the lineage;
6. apply the frozen Milestone 5 threshold to whichever score it was selected
   against;
7. emit the decision.

The predicate is the frozen one, and there is exactly one implementation of it
(`thresholds.flagged_malicious`), used both to *choose* an operating point and to
*apply* one:

```
flagged_malicious == (score >= decision_threshold)
```

**No silent substitution.** If the threshold was selected over
`calibrated_probability`, the probability is what it is compared against. If the
lock legitimately freezes a raw-score threshold, the raw score is compared and no
`malicious_probability` is fabricated. The row schema enforces both directions: a
calibrated row must carry its probability, and an uncalibrated row must not carry
one at all. `malicious_probability` exists only where a verified fitted
calibrator produced it.

Every stored row can be checked without reference to anything else, because it
carries both the score it was decided on and the threshold it was decided
against. Validation recomputes the decision from those two numbers for every row
and refuses a contradiction rather than repairing it.

---

## 5. Category triage

Published only when `champion.lock` bound a frozen category head. The head's own
selection, run, model, preprocessor, class order, and abstention fingerprint are
all verified before it is used.

### Triage is downstream of the binary decision

The category head was fitted on **known-malicious training rows only**. It has
never been shown a benign row, so asking it about one produces a number rather
than a finding. Category triage therefore runs *after* the binary decision and
only over the rows it flagged, and `category_predictions.parquet` contains
**exactly the binary-positive anchors** — the table's membership is the
applicability record.

Two states that look alike are kept permanently apart:

| State | Meaning | How it is represented |
|---|---|---|
| **not applicable** | the binary champion did not route this row to triage | the row is **absent** from the category artifact |
| `unknown` | the binary champion *did* route it, the head was asked, and its best class score fell below the frozen floor | a row with `predicted_scenario = "unknown"` |

`predicted_scenario = "unknown"` never means "the binary detector said benign".
Collapsing the first state into the second would inflate the abstention rate with
rows the head never saw, and a mostly-benign dataset would read as a triage model
that constantly refuses to commit.

For the rows that *are* applicable:

```
predicted_scenario = argmax(class_score)  when max(class_score) >= min_category_score
                   = "unknown"            otherwise
```

Ties go to the **earliest class in the declared order**, under a strict `>`
comparison — the same `thresholds.best_class` Milestone 5 used to measure the
abstention point, so a precision measured during selection describes the rows a
prediction actually produces.

`category_scores_json` is canonical JSON: every declared class, none that was not
declared, in the frozen deterministic order, with finite quantized values. It is
the one free-form field on a prediction row, so it is bounded in size and class
count and parsed strictly on read.

If no head was frozen, **no category artifact is published**. An all-`unknown`
stand-in would be a model output nothing produced, and it would be
indistinguishable from a head that abstained on every row.

If a head was frozen and **no row was flagged**, the artifact is published and
empty. Zero applicable rows is an honest outcome and is reported as such: every
category rate becomes `unavailable` rather than zero, because a head that was
never asked is not a head that abstained.

The category output never changes the binary decision. `predict_category` reads
the binary predictions to decide applicability and returns new rows; it writes
nothing back.

---

## 6. The experimental anomaly artifact

Published only when a caller explicitly names an experimental anomaly run
(`--anomaly-run`). It is **never** read out of `champion.lock` and never attached
to one: M-030 is a generalisation probe, its output is not comparable with a
supervised score, and a lock that acquired one retroactively would make the
champion's identity depend on an experiment that never influenced it.

- the run must be an anomaly run from an experimental family that is not
  champion-eligible;
- its own preprocessor, model artifact, and anomaly threshold are verified;
- it emits `anomaly_score` and **never a probability** — there is no probability
  column on the schema at all, because a nullable one would be filled in
  eventually;
- the flag, where a frozen threshold exists, is the inverted predicate
  `anomaly_score <= threshold`;
- `experimental` is pinned true and `influences_champion_selection` is pinned
  false, on every row and in the aggregate report.

It cannot change `flagged_malicious`, `predicted_scenario`, champion selection, or
any threshold. **No `FusionDecision` is published** — fusion belongs to a later
milestone, and no part of it exists here.

---

## 7. Row identity and canonical order

A prediction row carries the minimum technical identity a later evaluation needs
to join it to an outcome:

- `anchor_event_id`
- `anchor_event_time`

and nothing else identity-bearing. Prediction tables carry **no** user or source
pseudonym, campaign identifier, raw IP, credential, raw feature vector,
coordinate, target label, or attack ground truth. `PROHIBITED_PREDICTION_COLUMNS`
names each category and an import-time guard refuses a schema that declares one.

Rows are emitted in the project's single canonical order:

```
(anchor_event_time, str(anchor_event_id))
```

Physical source-file order reaches neither the output order nor the semantic
identity — the whole table is canonicalised before it is scoped. Duplicate
anchors are refused.

**The aggregate artifacts render no identifier.** The manifest, the validation
result, the quality report in both renderings, the CLI output, and every error
message carry counts, declared names, and stable codes. The join keys stay in the
Parquet that the next milestone will join on.

---

## 8. Prediction identity

`prediction_id` is `uuid5` over a canonical payload binding:

- the prediction and manifest schema versions;
- the scope and its role;
- the whole frozen lineage — lock fingerprint, freeze receipt, selection, run,
  model, model manifest, preprocessor, calibrator, threshold, score kind and
  value, the category head's fingerprints and class order where present, the
  anomaly run's lineage where present, the serializer and adapter identity, and
  the dependency contract;
- the feature catalog, allowlist, and eligible-feature fingerprints;
- the inference-input fingerprint and the split-membership fingerprint;
- the exact semantic content of every published row;
- the row counts.

It binds **nothing observational**: no output directory, no checkout path, no
hostname, no username, no publication time, no filesystem timestamp, and no
temporary-directory name.

The consequences are the point:

| Change | Effect on identity |
|---|---|
| a predicted value, a decision, or a threshold | **changes** |
| a row entering or leaving the scope | **changes** |
| a semantic feature value in an inference row | **changes** (via the input fingerprint) |
| a different champion, calibrator, or operating point | **changes** |
| the output directory, the machine, or the hour | unchanged |
| the physical order of the source Parquet rows | unchanged |
| **a TEST label** | unchanged — none was read |

The per-file digests and byte sizes are deliberately *outside* identity. They are
integrity evidence about a particular writing of the rows, so re-publishing
identical predictions through a writer whose page layout changed keeps the same
identifier and produces a different manifest digest.

---

## 9. The artifacts

A publication lives at `<root>/predictions/<prediction_id>/` and contains exactly:

| File | Format | Present |
|---|---|---|
| `binary_predictions.parquet` | Parquet | always |
| `category_predictions.parquet` | Parquet | when a frozen head was published |
| `anomaly_scores.parquet` | Parquet | when an experimental probe was named |
| `prediction_validation.json` | JSON | always |
| `ml_quality.json` | JSON | always |
| `ml_quality.md` | Markdown | always |
| `prediction_manifest.json` | JSON | always, **written last** |

Anything else in the directory fails validation.

### Deterministic Parquet

Column names, order, Arrow types, and nullability are **pinned**, not inferred.
Inferring them would make the file's shape depend on the data: an all-`null`
probability column would come out as `null` type rather than nullable `float64`,
and a reader would then have to guess whether the calibrator was absent or the
inference broke.

Writer settings are fixed — format version `2.6`, `snappy`, statistics off,
dictionary encoding off, a pinned page and row-group size, microsecond UTC
timestamps. Nothing observational is written. The same rows written twice, in two
directories, at two times, produce **byte-identical files**, and a test asserts it
directly rather than arguing it from the settings.

The semantic content fingerprint, not the file, is authoritative.

---

## 10. The prediction manifest

`PredictionManifest` is sealed: its digest is a field, recomputed on every
construction and every deserialization, so an edited manifest is refused rather
than believed. It binds the prediction identity, the content fingerprint, the
scope and role, the whole frozen lineage, the inference-input and
split-membership fingerprints, the row counts, and every declared file with its
SHA-256 and byte size — plus the digests of the validation result and the quality
report.

It carries **no metric of any kind**, no label fingerprint, and no list of
identifiers. Import-time guards refuse a field named for any of them, because the
way this artifact would stop being label-free is one plausible-sounding field at a
time.

It is written **last**, so its presence means everything it covers is already
there.

---

## 11. Validation

`ml validate` checks a published *prediction* directory. It is not `ml
verify-manifest`, which checks a published *model* directory; the two take
different inputs and neither accepts the other's.

Twenty-eight checks, each with a stable `M0xx` code:

| Code | Check |
|---|---|
| M001 | the manifest is present and parses at this contract version |
| M002 | the manifest recomputes its own seal |
| M003 | the prediction identity is the one the manifest's content derives |
| M004 | exactly the declared files are present, and nothing else |
| M005 | no symbolic link, no directory, no name outside the permitted set |
| M006 | every declared digest matches the bytes on disk |
| M007 | every declared byte size matches the bytes on disk |
| M008 | the declared row counts match the tables |
| M009 | the content fingerprint recomputes from the published rows |
| M010 | the tables carry the declared Arrow schema and column order |
| M011 | the binary table is readable and non-empty |
| M012 | anchors are unique |
| M013 | rows are in canonical order |
| M014 | every stored number is finite |
| M015 | calibrated probabilities lie in `[0, 1]` |
| M016 | a probability exists exactly where a calibrator produced one |
| M017 | every stored decision recomputes from its own score and threshold |
| M018 | one frozen operating point across the table |
| M019 | class maps are complete, finite, and in class order |
| M020 | every category assignment recomputes from its own scores and floor |
| M021 | the category table covers exactly the binary-positive rows |
| M022 | anomaly rows carry a magnitude, are experimental, and influence nothing |
| M023 | every anomaly flag recomputes from its own score and threshold |
| M024 | no prohibited column appears in any published table |
| M025 | the scope and its declared role agree |
| M026 | the bound aggregate reports are the ones declared |
| M027 | every row applies the operating point the lineage names |
| M028 | every category row records the frozen class order and abstention floor |

**A skipped mandatory check is not a pass.** A failure early in the chain leaves
the checks that depended on it recorded as `skipped`, and the aggregate is `pass`
only when every mandatory check ran and passed. The two check sets — staged and
published — are declared as constants, and a result is refused unless it reports
exactly its stage's set, so a check that quietly stopped being emitted fails the
result rather than improving it.

A prediction publication is treated as **untrusted data**: nothing is executed,
no Parquet metadata is interpreted, no pickle is deserialized, no module is
imported from artifact content, JSON payloads are size-bounded, and the declared
row count is capped before anything is materialised.

Validation **repairs nothing**. A row contradicting its own frozen threshold is
invalid, and it stays invalid.

---

## 12. Quality and profile

`ml profile` emits `MLQualityReport` as deterministic JSON and Markdown, rebuilt
from the publication alone — the rows, the manifest, and a fresh validation pass.
That is what makes it checkable rather than merely informative.

**It profiles a valid publication, and nothing else.** Validation runs first, in
full, and the order is fixed:

1. resolve and parse the publication safely;
2. run the complete published check set;
3. **if validation fails** — exit non-zero, print the sanitized failing codes and
   a pointer to `ml validate`, and write **no JSON and no Markdown**. Any report
   from an earlier successful run is left exactly as it was;
4. only after it passes — derive the distribution aggregates, build the report,
   and render it.

A tampered artifact is `ml validate`'s subject. Profiling a publication whose
manifest, checksums, lineage, or rows do not verify would produce a document
indistinguishable from a description of a sound one, and would make profiling a
way past verification.

What it reports:

- **binary** — rows scored, flagged and unflagged counts and rates, score kind,
  decision threshold, decision-score range, mean, and quantiles, whether a
  calibrated probability is available and its distribution when it is, and the
  count of rows carrying no probability;
- **category** — the two populations reported separately: rows scored, rows the
  binary head did not flag (`not_applicable_count`), and rows routed to triage
  (`applicable_row_count`). **Every category rate uses the applicable population
  as its denominator**, never the whole table: known-class assignments per
  declared class, abstentions, the abstention rate among applicable rows, the
  frozen floor, and the best-score distribution;
- **anomaly** — rows scored, the magnitude range and distribution, the frozen
  threshold and flag count where one exists, and its permanent experimental
  status;
- **validation** — the aggregate status, the failing codes, and the number of
  checks run.

**Unavailable is not zero.** An uncalibrated champion has no probability
distribution, and the report says `unavailable` rather than `0.0` — a mean of zero
would describe a model that was certain every row was benign.

**No outcome-dependent figure appears.** There is no accuracy, no precision or
recall against truth, no F1, no false-positive rate against truth, no PR-AUC, no
Brier score, and no calibration error. Each requires labels this milestone never
opens. An import-time guard refuses a field named for one, and the test suite
sweeps the schemas, the JSON, the Markdown, and the terminal output for every
spelling.

**Structural validity is not predictive quality.** A publication can pass every
check in this report while the model behind it is useless. The checks establish
that the rows are internally consistent, correctly typed, canonically ordered, and
attributable to the champion that produced them. Whether flagging those particular
rows was a good idea is a different question, not a weaker version of this one.

---

## 13. Publication

Staged, re-read, validated, promoted, manifest last:

1. verify the frozen champion — before anything is scored;
2. build the canonical inference input;
3. compute predictions in memory, where every row validates itself against the
   frozen predicate as it is constructed;
4. write the row artifacts into a temporary **sibling** directory — a sibling
   because `rename` is atomic only within a filesystem;
5. **re-read every artifact from disk** and revalidate it, which is what proves
   the file is the rows rather than something that merely looked like them in
   memory;
6. run the staged validation and write its result;
7. derive the aggregate profile from those re-read rows and write both renderings;
8. write the manifest **last**;
9. validate the complete staged publication exactly as `ml validate` would
   validate it after promotion;
10. flush the files and the directory, promote atomically, remove the staging
    directory on every exit path.

On any failure: no partial destination, no promoted manifest, no staging left
behind, and every previously published artifact untouched.

### Idempotency and conflict

A prediction identifier is derived from the predictions themselves, so:

- **identical semantic publication** — idempotent. The existing artifact is
  confirmed byte for byte and left exactly as it was; nothing is rewritten and no
  second directory appears.
- **different content** — a different identity, and therefore a different
  directory. Nothing is ever overwritten.
- **the same identity with different bytes** — a hard conflict, refused. A
  published prediction is evidence.
- **a corrupt or incomplete destination** — a fault. The publisher fails closed
  rather than "completing" it in place, because a publication finished by a later
  run is a publication whose two halves came from two runs. Corruption is not
  idempotency.

There is no overwrite flag, because the only thing it could do is destroy the
earlier evidence.

---

## 14. The ledger

**No prediction record is appended.** The immutable experiment ledger owns
`training_run`, `validation_selection`, `champion_freeze`, and the reserved
`test_evaluation`. Prediction publication identity belongs to the
`PredictionManifest`.

Milestone 8 writes no `test_evaluation` record and mutates no existing record.
The frozen champion and every published run are read and never written.

---

## 15. Determinism

Asserted, not argued:

- the same frozen workflow run **in two directories** produces the same canonical
  rows, the same content fingerprint, the same `prediction_id`, the same manifest
  bytes, the same validation result, and byte-identical JSON and Markdown reports;
- the same workflow run with the **source rows physically shuffled** produces the
  same publication — and is recognised as the *same* publication, so it writes
  nothing;
- changing **one semantic feature value** in an inference row moves the
  inference-input fingerprint and the prediction identity;
- changing **TEST labels only** changes nothing, because none was read.

---

## 16. Commands

```bash
# score a split under the frozen champion and publish the result
uv run password-attack-detector ml predict \
    --features processed/feature_snapshots.parquet \
    --splits processed/feature_splits.parquet \
    --feature-manifest processed/feature_manifest.json \
    --allowlist configs/ml/features-allowlist-v1.yaml \
    --config configs/ml/model-development.yaml \
    --split test \
    --output-root artifacts/ml

# check a published prediction artifact, without opening a label
uv run password-attack-detector ml validate --output-root artifacts/ml

# report the aggregate shape of that output
uv run password-attack-detector ml profile --output-root artifacts/ml \
    --reports-dir reports
```

`ml predict` exits non-zero when the lock does not verify, the feature contract
disagrees, the scope is empty, or the published artifact fails its own
validation. `ml validate` exits non-zero on any invalid or tampered artifact,
naming the stable code that caught it.

No command prints an anchor identifier, a row, a feature value, a pseudonym, a
coefficient, an absolute path, or an outcome metric.

---

## 17. Known limitations

**The predictions are on synthetic authentication traffic.** The score
distributions in a quality report reflect the generator's assumptions — a
population somebody wrote — and not a production one. A flagged rate measured
here says how often this model fires on generated data, and nothing about how
often it would fire on real traffic.

**No test metric exists.** This milestone deliberately produces none, and no
figure it publishes can be turned into one without the labels it never read.

**A validated publication is not a good model.** See §12.

**Category triage is only as good as the binary head that routes to it.** A row
the binary champion missed never reaches triage, and the category artifact will
never mention it. The abstention rate describes the rows that were routed, not
the rows that should have been.

**The category head is optional and often absent.** On a small dataset the
category selection frequently finds no eligible head, and the correct outcome is
no category artifact rather than a fabricated one.

**The anomaly artifact is experimental and comparative only.** Its magnitudes are
not probabilities, are not comparable with the supervised score, and influence
nothing.

**Fusion does not exist yet.** No rule-versus-model comparison, no
`FusionDecision`, and no hybrid verdict is produced or published anywhere in this
milestone.
