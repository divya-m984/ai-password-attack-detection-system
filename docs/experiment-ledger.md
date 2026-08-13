# Experiment ledger and training runs

How a model gets trained, what gets written down about it, and what can never
be changed afterwards. This document covers Phase 5 Milestone 6.

**Training ranks nothing.** Milestone 6 trains every configured candidate,
publishes an immutable run for each, and records what happened. It promotes
nothing, writes no `champion.lock`, and reads no test split. Choosing a champion
from those runs — on validation-B alone — is Milestone 7's, and is documented in
`docs/champion-selection.md`. No figure anywhere in this repository describes
detection performance on unseen data.

---

## 1. Three tracks, kept apart

| track | fitted on | operating point | artifacts |
|---|---|---|---|
| **binary supervised** | TRAIN, supervised-eligible rows | threshold on validation-B | model, calibrator, both calibration reports, binary threshold |
| **category supervised** | TRAIN, known-malicious rows only | abstention on validation-B | model, abstention threshold |
| **anomaly experimental** | TRAIN, benign rows only, no target | benign quantile or validation-A benign rate | model, anomaly threshold |

The separation is enforced, not merely intended. A category head never inherits
the binary threshold, an anomaly run never carries a calibrator or a calibrated
probability, and a binary run never invents an abstention artifact — each is a
validator on the training-run record, and each is tested in both directions.

**Only applicable artifacts are written.** An anomaly run has no `calibration/`
directory at all, rather than an empty one: a reader finding an empty artifact
has to guess whether it is absent or broken.

### The category population

Malicious TRAIN rows carrying a category the Phase 2 scenario contract declares,
and nothing else. Benign rows are excluded because the head answers *which
attack*, not *whether*. Novel-anomaly rows are excluded because they carry no
known category to be right about — and they are in a different split entirely.

The class space is derived from `KNOWN_CATEGORY_CLASSES`, which comes from
`ScenarioType`. It is never taken from a Phase 4 rule name: the rule engine and
the category head answer different questions, and borrowing one vocabulary for
the other would make the two look comparable when they are not. Classes with no
representation in the training population are dropped from the fitted class
order, and a population with fewer than two represented classes reports
`insufficient_support` rather than fitting a one-class classifier.

### The anomaly probe

Fitted on benign TRAIN rows with **no target column**. Which rows are benign is
decided in the orchestration, where labels are legitimately readable; the
estimator receives a design matrix and nothing else.

Its threshold comes from one of two configured provenances, both benign-only:
`train_benign_quantile` (benign TRAIN scores) or `validation_a_benign_fpr`
(benign validation-A scores). Neither can name validation-B, the test split, or
the novel-anomaly holdout — `AnomalyThresholdMethod` has no member for any of
them, and the holdout in particular is what the probe is *measured against*, so
a threshold tuned on it would be measuring itself.

Its output stays an `anomaly_score`, it is permanently `experimental`, and
`influences_champion_selection` is pinned false in the configuration and again
in the published selection.

---

## 2. Exact split usage

| split | read by Milestone 6 for |
|---|---|
| **TRAIN** | preprocessing statistics, class weights, every model fit, the anomaly benign quantile |
| **validation-A** | fitting the calibrator; the in-sample fit diagnostic; optionally the anomaly benign flag rate |
| **validation-B** | out-of-sample calibration quality; the binary threshold; the category abstention threshold |
| **TEST** | nothing |
| **novel-anomaly holdout** | nothing |

### Preprocessing is fitted per track, not per run

A fitted preprocessor is not a formatting step. It learns imputation constants,
category vocabularies, rare-value buckets, and scaling statistics, so its
fitting population is part of the learned pipeline — and each track may only
learn from the rows its own task permits.

| track | preprocessing fitted on |
|---|---|
| binary | supervised-eligible TRAIN rows |
| category | known-malicious TRAIN rows only |
| anomaly | benign TRAIN rows only |

An anomaly probe whose medians came from malicious rows has read, through its
own encoder, the labels it is supposed to be blind to. A category head whose
country vocabulary came from benign traffic has been shaped by rows it will
never be asked about. Sharing one state would do both quietly, and every
fingerprint downstream would still look correct.

The reviewed **raw** feature allowlist is shared; the fitted state is not. Two
families fitted for the same task reuse one state and one fingerprint, because
their preprocessing configuration is identical.

### The frozen splits cannot reach a training run

There is no parameter on any entry point through which a test or holdout row
could arrive: row selection happens inside the orchestration, from a dataset
that already knows which split each row belongs to.

One consequence is worth stating explicitly, because it is where a leak would
otherwise arrive through the back door. The dataset's own
`training_data_fingerprint`, `label_fingerprint`, and `split_fingerprint` cover
**every** row the dataset holds — test and holdout included. A run identity built
from them would change whenever somebody added a test row, which is the firewall
leaking through the *identifier* rather than through the fit.

So a training run carries **three role-scoped digests** instead, each over
exactly the rows its own track may read:

| field | binds |
|---|---|
| `readable_training_data_fingerprint` | the feature values of every readable row |
| `readable_label_fingerprint` | supervised eligibility, the binary target, and the known-category target |
| `readable_split_fingerprint` | which readable role each row is in, plus the parent validation-partition digest |

Three and not one. Collapsing them would tell a reader *that* the lineage moved
and never *what* moved it, and a changed feature value, a changed label, and a
row entering or leaving a readable role are three different findings with three
different investigations behind them.

The scope is per track, because the three tracks read genuinely different rows:

| track | readable roles |
|---|---|
| binary | TRAIN, validation-A, validation-B |
| category | known-malicious TRAIN, known-malicious validation-B |
| anomaly | benign TRAIN — plus benign validation-A when the configured threshold provenance reads it |

Row identity is used **inside** each digest and published by none of them.
Pairing a label to the row that carries it is what makes the label digest move
when two rows swap labels; publishing the pairing would put anchor identifiers
in a record, so only the digest leaves.

The three unscoped dataset fingerprints are left unset in a run record rather
than published as non-semantic decoration.

The behavioural tests perturb one thing at a time and assert exactly what moves:

| perturbation | moves |
|---|---|
| a TRAIN feature value | `readable_training_data_fingerprint`, model, run identity |
| a TRAIN label | `readable_label_fingerprint`, run identity |
| a validation-A label | label lineage, the calibrator, run identity |
| a validation-B label | label lineage, the quality report, the threshold, run identity |
| a readable row's split assignment | `readable_split_fingerprint`, run identity |
| a TEST feature, label, or split assignment | **nothing** |
| a novel-holdout feature, label, or split assignment | **nothing** |
| a malicious TRAIN row | binary track only — the anomaly probe is unmoved |
| a benign TRAIN row | anomaly and binary tracks — the category head is unmoved |

A companion test asserts the dataset's *own* whole-row fingerprints **do** move
when a test row changes. The firewall is scoping, not blindness, and that is
precisely why none of those digests may enter a run identity.

A further test replaces the frozen splits wholesale and asserts every
preprocessing statistic, every fitted model, every calibrator, every report,
every operating point, and every run identifier comes out byte-identical.

---

## 3. Candidates are declared, never searched

Candidates are enumerated from the reviewed configuration and the executable
catalog, ordered by task and then by catalog identifier. There is no grid, no
random sampling, no Bayesian optimiser, and no criterion evaluated on held-out
data to decide what to try next. A candidate exists because somebody wrote it
down.

Each carries a `candidate_fingerprint` over its task, its catalog entry, and its
**effective** hyperparameters — what would actually be fitted, rather than only
the overrides that happened to be written down. Two configurations differing in
a default they both accepted are the same candidate.

No wall clock takes part in candidate identity, or in any identity here.

### M-001 is configured, never discovered

The single-feature threshold baseline cuts on a transformed column named in
`single_feature_baseline_column`. Scanning every feature for the best split
would be a model-selection procedure run on training data and then reported as a
baseline, which flatters the baseline and understates whatever it is compared
against. Left unset, the M-001 candidate reports `unavailable` rather than
picking a column for itself.

---

## 4. Run statuses

A configured candidate that cannot be trained does **not** disappear. It
produces a run outcome and a ledger record carrying one of:

| status | meaning |
|---|---|
| `completed` | fitted, published, and carrying every artifact its task requires |
| `unavailable` | the family cannot be trained under this configuration at all |
| `failed_validation` | the model fitted and the published artifact failed verification |
| `insufficient_support` | a support floor was not met, so nothing downstream could mean anything |
| `calibration_unavailable` | configured calibration could not be fitted on validation-A |
| `threshold_unavailable` | no operating point could be selected on validation-B |
| `serializer_unavailable` | the family has no proven serializer, so a fit may be compared in process but never stored |

A candidate list that silently shrinks is a comparison nobody can audit: a
family missing from a later report must be distinguishable from a family that
was never configured.

**A candidate can fit a model and still be ineligible downstream.** A candidate
whose configured protocol requires calibration and cannot fit one records
`calibration_unavailable` with the requirement it failed. That is neither a
failure of the run nor silently a success.

### The reference baseline is not calibrated, by contract

M-000 emits the training class prior for every row. There is no score variation
for a calibrator to map, and both Milestone 5 methods require distinct scores by
contract.

Two ways out were available and both were refused: fabricating a calibrator for
a constant, and relaxing the minimum-support or distinct-score rules until Platt
accepted one. Each would have made the comparator's number mean *less*.

So calibration is declared **not applicable** for a reference-baseline family.
The run keeps `score_kind: decision_score`, chooses its operating point on that
raw bounded score under the Milestone 5 raw-score contract, records
`calibration_status: not_calibrated` with `calibration_method: none`, publishes
no calibration artifact at all, and **can reach `completed`**. The mandatory
comparator every candidate is measured against must be usable, and an absent
calibrator is not a defect in it.

This is an exemption for the comparator alone. An ordinary champion candidate
whose configured protocol requires calibration and cannot fit one still reports
`calibration_unavailable`, and M-000 remains `champion_eligible: false` however
complete its run is.

M-021 is registered, implementable, and permanently unpublishable — its
serializer would rest on undocumented estimator internals — so it reports
`serializer_unavailable` and no artifact is ever written for it.

---

## 5. Training-run identity

`run_id` is a UUIDv5 over the canonical rendering of the record's identity, so
the same semantics derive the same identifier on any machine, in any directory,
in any year. It **cannot be assigned**: `ExperimentRecordIdentity.derive`
refuses a caller-supplied `run_id`, because an identifier somebody could choose
would let two different runs claim to be the same run.

Identity binds the whole semantic lineage:

- task, model family, catalog model id, and the catalog and feature-schema
  versions;
- the candidate fingerprint (effective hyperparameters) and the seed;
- the ML configuration fingerprint and the model-catalog fingerprint;
- the feature-catalog fingerprint, the reviewed allowlist fingerprint, and the
  ordered eligible-feature fingerprint;
- the **task-specific** preprocessing fingerprint and the class-weight
  fingerprint;
- the three role-scoped lineage fingerprints (see §2) and the parent
  validation-partition fingerprint;
- the model content fingerprint;
- the calibrator, binary-threshold, category-abstention, and anomaly-threshold
  fingerprints, each where applicable;
- the serializer identifier and version, and a digest over the declared
  dependency ranges.

Identity excludes, and has nowhere to put: an output path, a checkout path, a
publication timestamp, a hostname, a username, or a temporary directory.

### No test metric, ever

A training-run record has no field a test metric could be written into — not an
empty one, not a null one, not one to be filled in later. A schema with
somewhere to put it is a schema somebody eventually fills in early. A test
evaluation is its own record type, written after a champion is frozen.

The record also has no `champion`, `is_champion`, `rank`, or `ranking` field.
`champion_eligible` is present and is a property of the *family*, copied from the
reviewed catalog; it says nothing about the run.

---

## 6. The ledger

Append-only, file-backed, and inspectable with `cat`.

```
artifacts/ml/ledger/
    ledger.json                      the ledger's own contract version
    training_run/<run_id>.json       one immutable record
```

Only the directory for a record type actually being written is created. Two
further types are written by Milestone 7 — `validation_selection/` for each
champion selection and `champion_freeze/` for each freeze receipt, both described
in `docs/champion-selection.md`. `test_evaluation/` has a reserved location and
no files, because nothing in this build evaluates on the test split.

**One record per file, not a JSONL stream.** Appending a line to a shared file
is a torn write waiting to happen: an interrupted process leaves a truncated
final line and every later reader has to decide what to do about it. A record
written to its own file with `O_EXCL` cannot be half-appended — either it exists
complete or it does not exist — and the exclusive creation *is* the collision
check, performed by the filesystem rather than by a read-then-write race. The
write is flushed and fsynced before the handle closes.

### Idempotency and conflict are different answers

| offered | outcome |
|---|---|
| an unused identifier | the record is written; `created: true` |
| a used identifier, identical bytes | nothing is written; `created: false`; success |
| a used identifier, different bytes | `LedgerConflictError` |

Nothing is overwritten and nothing is merged, because both would destroy the
earlier claim in order to record the later one. **There is no update API** — no
`update`, `upsert`, `replace`, `delete`, `amend`, or `patch`, public or private,
and a test sweeps the class surface to keep it that way.

A record that fails to parse, exceeds the size a record of this contract can
have, or does not recompute its own digest is refused rather than skipped: a
ledger that quietly skipped one would report a shorter history as a complete one.

---

## 7. Publication

A run directory is a claim that a model was fitted under stated conditions. A
half-written one is a claim nobody can check, so a run is built elsewhere and
moved into place only once it is complete.

```
artifacts/ml/runs/<run_id>/
    training_run.json                   the receipt, written last
    model/
        model.json  arrays.npz  preprocessor.json  model_manifest.json
    calibration/                        binary runs only
        calibration_state.json
        calibration_fit_diagnostic.json
        calibration_validation_report.json
    thresholds/                         only what the task has
        binary_threshold.json | category_abstention.json | anomaly_threshold.json
    ranking/                            binary runs with both classes present
        validation_b_ranking.json
```

**`ranking/` is not a second view of `thresholds/`.** The threshold curve comes
off a *bounded* candidate grid and exists to justify one operating point; the
ranking evidence is built from **every distinct validation-B score level** and
exists to measure discrimination. Keeping them in separate artifacts is what
stops a performance setting — `search_grid_size` — from reaching a champion
gate. The evidence is written whenever the binary validation half carried both
classes, including for a run that found no feasible threshold: a constant
reference baseline is exactly that case, and its discrimination is still exactly
measurable. Its contract is `docs/champion-selection.md` §6.

The order:

1. build the whole run in a temporary **sibling** directory — a sibling because
   `rename` is atomic only within a filesystem, and a cross-device move degrades
   into a copy that can be interrupted halfway;
2. write only the artifacts this run's task actually has;
3. verify the model artifact with the **Milestone 4 verifier** — the same checker
   a loader uses, not a lighter one written for publication;
4. re-read every typed artifact from the bytes on disk and confirm it parses and
   recomputes its own digest;
5. build and validate the training-run record;
6. write `training_run.json` last, so its presence means everything it covers is
   already there;
7. promote atomically;
8. append to the ledger.

On any failure the destination is untouched, the ledger is untouched, no partial
run is visible, and the staging directory is removed.

**A published run is never overwritten.** A second publication of the same
identifier is either the same run again — confirmed byte for byte and left
alone — or a contradiction, which is refused. There is no overwrite flag, because
the only thing it could do is destroy the earlier evidence.

### Why the ledger goes last

The two orderings fail differently. Ledger-first can leave the ledger asserting a
run that does not exist, and the only repair would be deleting an immutable
record. Run-first can leave a complete, valid, *unindexed* run — recoverable by
reading the run and appending the record it already contains, without rewriting
anything.

`ml experiments --reconcile` performs that recovery. It reads each published
run's own receipt and appends it; it never rebuilds a record, never rewrites one,
and never deletes one. A run whose stored record disagrees with the ledger raises
rather than being reconciled, because that is a contradiction and not a gap.

### Model identity is not rewritten

The model keeps the identity Milestone 4 gave it, the calibrator and the
thresholds keep the identities Milestone 5 gave them, and the training-run record
**binds** them by fingerprint. It does not restate them, so a later milestone
reads the artifact it wants rather than a copy that could disagree.

---

## 8. Reproducibility

Two runs of identical semantics, in two different directories, at two different
times, produce:

- the same preprocessing, class-weight, model-content, calibration, threshold,
  and abstention fingerprints;
- the same `model_id` and the same `run_id`;
- byte-identical `model.json`, `arrays.npz`, `preprocessor.json`,
  `model_manifest.json`, every calibration and threshold artifact, and every
  training-run receipt.

No publication timestamp is written anywhere. `published_at` is left null rather
than merely excluded from a digest, which is what makes identical runs *identical*
rather than only equivalent.

Shuffling the source Parquet row order changes nothing: canonical ordering is
applied once, on the way out of dataset assembly, and every later stage asserts
it rather than re-sorting.

Changing a genuine semantic input — a TRAIN feature value, the reviewed allowlist,
a hyperparameter, the seed — changes the run identity, and a test asserts it.

---

## 9. Privacy

Run and ledger artifacts are model state and aggregates. No anchor identifier,
event identifier, campaign identifier, entity pseudonym, IP address, credential,
coordinate, raw row, or absolute home path appears in any of them. Artifact
digests name paths relative to the run directory, and a record naming an
absolute path or a traversal component is refused.

The role-scoped lineage fingerprints identify the readable data semantically.
Row identity takes part in computing them and appears in none of them: what
leaves is a digest.

CLI output is run identifiers, statuses, tasks, model identifiers, and counts.
Sweeps run over `training_run.json`, ledger entries, and both commands' terminal
output.

---

## 10. Known limitations

**Training records what was run; it does not decide which run was best.** The
ledger listing deliberately shows no metric of any kind, because a listing that
ranked runs would be a champion selection under another name. Selection happens
in its own step, against predeclared gates, and writes its own immutable record —
see `docs/champion-selection.md`.

**No test evaluation.** Neither training nor selection reads the TEST split or
the novel-anomaly holdout. Milestone 8 *predicts* on them under a frozen
champion, which is not the same thing: it publishes what the model said without
opening a label, so no outcome-dependent number is computable from what it
writes. The `test_evaluation` record type is reserved and unwritten.

**Predictions are not ledger records.** Milestone 8 appends nothing here. A
prediction publication is identified by its own `PredictionManifest`, and the
ledger's four record types are unchanged — see `docs/prediction-artifacts.md`.

**A completed run is not a good model.** `completed` means every artifact the
task requires was published. It says nothing about detection effectiveness, and
nothing here measures any.

**Synthetic development data has a ceiling.** Everything trainable here is
generated traffic with known ground truth. Attack behaviour is more separable
than real traffic because it was parameterised rather than observed; benign
behaviour is less varied, so a false-positive rate measured on it is a lower
bound at best; the class balance is a configuration choice, not a measurement.

**Calibration on synthetic validation data is not real-world calibration.** See
`docs/model-contract.md` §16.

**Ranking evidence is exact, and exactness is not accuracy.** The published
curve carries every distinct score level, so the metric derived from it is not
an approximation of itself — but it was still measured on synthetic validation
traffic, and inherits every limitation above.

**M-001's column is a reviewed decision in both shipped configurations.** Both
name `user_failure_count__5m`: prior-only, post-event, admitted by the reviewed
allowlist, non-nullable, and the most direct expression of the pattern this
system exists to detect. No validation figure took part in the choice, and none
may. On the 160-event CI fixture the baseline's cut puts every validation-A row
on one side, so its calibration reports one distinct score — a real downstream
outcome rather than a missing setting, and the path where it completes is
covered by the unit suite.

**The CI-sized configuration is not a development run.** `configs/ml/model-testing.yaml`
shrinks every count so a contract test can run in seconds. Its policies match
development exactly — calibration reads validation-A, thresholds read
validation-B, missingness indicators are on, resampling is off — but its floors
are not appropriate for a real run, and the full 720-hour development workflow is
deliberately never executed in the ordinary test suite.
