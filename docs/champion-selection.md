# Champion selection and the champion freeze

How one trained run becomes the champion, what has to be true before it can, and
what is written down so the decision can be checked afterwards. This document
covers Phase 5 Milestone 7.

**No test data has been read.** Selection compares candidates on validation-B
alone. The TEST split and the novel-anomaly holdout are untouched by every
function, command, record, and report described here, and there is no flag,
option, or parameter through which either could be supplied. Nothing in this
repository yet describes detection performance on unseen data.

**Selection reads artifacts, not rows.** Every number compared here was
measured, frozen, and digest-sealed by Milestone 6. The selection layer opens no
Parquet table, imports no label reader, and refits nothing. The direct
label-reader allowlist is unchanged at exactly two modules —
`detection.evaluation` and `ml.dataset` — and Milestone 7 is in neither.

---

## 1. The candidate universe

Derived from the reviewed model catalog, never written down a second time:

```
champion_candidate_model_ids(task) =
    every catalog entry with champion_eligible = true
    that declares support for `task`
```

| task | candidates | why the others are out |
|---|---|---|
| `binary_malicious` | M-001, M-010, M-020 | M-000 is the reference baseline and is not champion-eligible; M-021's serializer is unproven; M-030 is anomaly-only |
| `attack_category` | M-010, M-020 | M-001 and M-000 do not support the category task |
| `anomaly` | *(none)* | the anomaly track is experimental and promotes nothing |

A second hard-coded list would be the thing that quietly disagreed with the
reviewed one, and the disagreement would surface only as a promotion nobody
expected. There is no such list.

---

## 2. Exactly which data decides

| partition | used for | used by selection |
|---|---|---|
| TRAIN | fitting models and preprocessing | no |
| validation-A | fitting calibrators | no — only its *out-of-sample* consequence is read |
| validation-B | operating points, calibration quality, **every gate** | **yes** |
| TEST | nothing yet | **no** |
| novel-anomaly holdout | nothing yet | **no** |

Calibration evidence is admitted only when it was measured out of sample. An
in-sample validation-A fit diagnostic is refused with the reason code
`in_sample_diagnostic_is_not_evidence`: it measures a calibrator on the rows
that fitted it, and a gate that accepted it would be satisfied by a calibrator
that had merely memorised.

---

## 3. The mandatory reference baseline

Every binary selection requires exactly **one** M-000 reference run from the same
experiment lineage. A candidate qualifies by beating it, and nothing beats
itself, so M-000 is never a candidate, never ranked, and never frozen.

### The readiness contract

M-000 reference evidence is usable when:

- its published run and model artifact verify;
- it carries **exact validation-B ranking evidence** (§6);
- that evidence declares the same metric, integration convention, and scoring
  stage every candidate is measured at;
- its lineage matches the selection scope — checked per candidate by the
  lineage gate.

It does **not** require champion eligibility, a calibrator, or a feasible
operating threshold. A prior-probability baseline scores every row alike: it
flags everything or nothing, so under any false-positive ceiling worth
configuring it has no feasible threshold and its run stops at
`threshold_unavailable`. That is the baseline behaving exactly as a baseline
should, and its discrimination is still exactly measurable — for a constant
scorer the metric reduces to the positive prevalence, computed with no operating
point involved anywhere.

None of that excuses a malformed comparator. Each refusal has its own reason
code:

| reason | meaning |
|---|---|
| `reference_baseline_run_missing` | no M-000 run for this task |
| `reference_baseline_ambiguous` | more than one, so "the baseline" has no referent |
| `reference_baseline_artifact_unverified` | its published model does not verify |
| `reference_ranking_evidence_unavailable` | it published no exact ranking evidence |
| `reference_metric_definition_mismatch` | its evidence declares another metric or scoring stage |

When the reference cannot be established, the gain gate is **inconclusive for
every candidate** and the selection fails closed. The gate is never waived, and
the baseline is never promoted in its own absence.

---

## 4. Gates: three outcomes, and the third is the point

Every gate is mandatory and answers `pass`, `fail`, or `inconclusive`. **An
inconclusive gate blocks promotion exactly as a failed one does.** A thin
validation half is not a clean bill of health, and a project that lets a
measurement nobody could make become a passed check will eventually promote a
model on the strength of eleven benign rows.

Every gate in its set is evaluated and recorded whatever the verdict — a gate
missing from a report reads as a gate that passed.

### Binary gates

| gate | asks |
|---|---|
| `validation_support` | does validation-B carry enough rows of each class to measure anything |
| `operating_threshold` | was a threshold *selected* on validation-B, rather than fallen back to |
| `false_positive_ceiling` | is the false-positive rate at or under the configured ceiling |
| `detection_rate_floor` | is the detection rate at or above the configured floor |
| `baseline_pr_auc_gain` | does exact PR-AUC exceed M-000's by the configured margin |
| `calibration_quality` | is the out-of-sample expected calibration error within the ceiling |
| `serializer_eligibility` | is the family champion-eligible and did its artifact verify |
| `lineage_compatibility` | was this candidate produced under the same experiment lineage as the comparator |

### Category gates

| gate | asks |
|---|---|
| `category_validation_support` | enough known-malicious validation-B rows for coverage to mean anything |
| `category_abstention_point` | was the abstention threshold selected, not a predeclared fallback |
| `category_precision_floor` | is known-category precision at or above the floor in force |
| `category_class_support` | does **every** known class carry enough rows, checked per class |
| `lineage_compatibility` | same experiment lineage |

The per-class check is deliberate. A head that is excellent at two categories and
has seen three rows of the third has not been evaluated on the third, and an
aggregate precision would hide that.

### The floor a gate is judged against

The precision gate applies the stricter of the floor the artifact was selected
under and the floor currently configured. A head selected under a looser
criterion is not promoted by the criterion it happened to be produced with.

---

## 5. Every rate arrives with its arithmetic

A gate publishes the numerator, the denominator, the rate, and a Wilson score
interval at the configured confidence.

**A rate over an empty denominator is `null`, never zero.** "No false positives
out of none" and "no false positives out of nine thousand" are different facts,
and only one of them is a false-positive rate.

**The resolution rule.** A false-positive ceiling of 1% cannot be *held* by a
sample of fifty benign rows: the coarsest non-zero rate such a sample can express
is 2%, so the only way to appear compliant is to flag nothing. The gate requires
`benign_rows × ceiling ≥ 1` before it will decide, and reports
`false_positive_rate_resolution` — inconclusive — when the sample cannot resolve
the constraint.

**Wilson intervals are published, not applied.** The configured ceiling is a
constraint on the rate, so the gate compares the rate and prints the interval
beside it. Requiring the interval's upper bound to clear the ceiling would be a
*stricter* criterion than the one written down, and inventing an acceptance
criterion during selection is exactly what a predeclared gate set exists to
prevent.

---

## 6. The discrimination metric: exact PR-AUC

Two things are kept apart that a single "curve" would quietly merge.

| | operating-point search | ranking evidence |
|---|---|---|
| purpose | choose one decision threshold | measure discrimination |
| built from | the bounded candidate grid (`search_grid_size`) | **every distinct score level** |
| may be sampled | yes, and says so (`candidates_truncated`) | never |
| belongs to | the Milestone 5/6 threshold contract | the champion gate |
| published at | `thresholds/binary_threshold.json` | `ranking/validation_b_ranking.json` |

**The metric is PR-AUC**, because that is what the reviewed gate configuration
names (`min_pr_auc_gain_over_baseline`). It is not average precision published
under PR-AUC's name; the name in the record and the arithmetic behind it are
pinned together by test.

**The declared integration rule** is step-wise and right-continuous, with no
interpolation of any kind:

```
levels sorted by score, descending, one level per distinct score
for level i:  TP_i = cumulative true positives at scores >= s_i
              FP_i = cumulative false positives at scores >= s_i
              precision_i = TP_i / (TP_i + FP_i)
              recall_i    = TP_i / P
PR-AUC = sum over i of  precision_i * (recall_i - recall_{i-1}),  recall_0 = 0
```

Trapezoidal integration is *not* used: linear interpolation between two points
of a precision-recall curve does not correspond to any achievable operating
point.

**Ties are one level.** Rows sharing a score enter the cumulative counts
together. Nothing consults an anchor, a row index, or arrival order — breaking a
tie by any of those would manufacture ranking performance *inside* rows the
model scored identically. Reordering tied rows, or shuffling the input, produces
byte-identical evidence.

**One scoring stage for every model.** The evidence is built from the frozen
pre-threshold model score (`decision_score`) — the quantity the model itself
produces, before any calibration map and before any threshold. The ranking
contract refuses evidence at any other stage, so a calibrated score for one
candidate and a raw score for the next can never reach one comparison.
Calibration quality is a separate gate; so is the operating point.

**For the constant M-000 baseline** all validation-B rows form one score group,
and the metric reduces to the positive prevalence — computed with no operating
threshold, and with no extra score levels invented to make a curve out of.

**No approximation may satisfy this gate.** If a candidate or the comparator has
no exact evidence, the gate is `inconclusive` (`ranking_evidence_unavailable`,
`reference_ranking_evidence_unavailable`) and the candidate is blocked. There is
no reason code by which a sampled or estimated number can pass, and the
threshold grid — whatever its size — cannot move the metric.

## 7. Ranking never substitutes for a gate

A candidate outside the gates is not ranked last. It is **not ranked**. Ordering
happens only among candidates that already cleared everything mandatory.

The objective and the tie-break chain are declared in configuration *before* the
evidence is seen, and they are part of the gate-configuration fingerprint:

```
objective:   max_detection_rate
tie-break:   min_false_positive_rate
           > max_baseline_pr_auc_gain
           > min_expected_calibration_error
           > catalog_model_id
```

The chain must end with `catalog_model_id` so the order is total: two candidates
that tie on every measurement are separated by a name rather than by whichever
the filesystem returned first. When that final criterion is reached, the record
says so — `tie_resolved_by=catalog_model_id` — because a selection decided by a
name is a weaker claim than one decided by a measurement.

The rationale is emitted as stable codes, not prose: `objective=…`,
`tie_break=…`, `rank=N model=…`.

---

## 8. Three outcomes, and only one is a winner

| status | meaning |
|---|---|
| `eligible` | one candidate cleared every mandatory gate and the ranking chose it |
| `no_eligible_champion` | every candidate carries at least one **measured** failure |
| `insufficient_validation_support` | some candidate's only blockers were inconclusive |

The precedence is deliberate. A candidate blocked only by inconclusive gates
might have qualified on data this run did not carry, so the question is open and
the honest answer is that it could not be resolved. Only when every candidate
carries a measured failure is the negative a finding. An empty candidate universe
is `no_eligible_champion`: nothing promotable was configured, which is a fact
rather than missing support.

A negative outcome is published exactly like a positive one. A history that
recorded only successes would be a history of successes rather than of what
happened.

---

## 9. The category head is a separate question

It is selected separately, from its own evidence, with its own gates and its own
record. It has no reference baseline to beat and inherits no binary operating
point, and **a binary champion existing is no reason to invent one**. When the
category selection finds nothing, the freeze binds no head and the absence is
represented rather than filled in.

---

## 10. The `validation_selection` record

One immutable record per selection, sealed by a digest that is recomputed on
every load. It carries:

- the task and the outcome;
- every candidate run identifier and model fingerprint — **the whole comparison,
  not only the winner**;
- each candidate's gate verdicts, observed values, requirements, and reason
  codes;
- the reference run identifier, which is never among the candidates;
- the ranking and the rationale, present only when a champion was found;
- the selected run, model, and content fingerprint, present exactly when the
  outcome is `eligible`;
- the configuration, catalog, gate-configuration, validation-partition, and
  role-scoped readable lineage fingerprints the comparison was carried out
  under;
- the discrimination metric, its integration convention, and the scoring stage
  it was measured at — named in the record so a later refactor cannot swap the
  arithmetic while keeping the name, and absent for the category head, which has
  no reference comparison to declare one for.

The identity is derived, not supplied. Two selections over the same candidates
under the same criteria derive the same identifier and are the same record; a
change to any acceptance criterion makes a different question with a different
answer and a different identifier.

There is no field in this record through which a test figure could arrive.

---

## 11. Publication and recovery

The Milestone 6 ordering, unchanged and for the same reason:

1. stage into a sibling directory;
2. write the reports, then the record **last** — its presence means everything it
   covers is already there;
3. verify the record reads back as itself;
4. rename the directory into place, atomically;
5. append the ledger entry.

A ledger asserting a selection that does not exist would need an immutable record
deleted to repair; an unindexed valid selection needs only to be read.
`reconcile_selections` and `reconcile_freezes` do exactly that — read the
published record and append it, never rebuild it, never rewrite anything.

Re-publishing an identical selection is idempotent. Publishing a *different*
selection under an existing identifier is refused: a published selection is
evidence and is never overwritten.

---

## 12. The champion freeze

A `champion.lock` is a promise about **what will be evaluated**. Everything a
later test evaluation may load — the model, the preprocessing state it was fitted
against, the calibrator, the operating point, and the contracts they were all
produced under — is named there by fingerprint, once, before anyone has seen a
test row. An evaluation whose subject could still change afterwards would not be
an evaluation of anything in particular.

**The lock carries no metric of any kind.** Not a validation metric and certainly
not a test one. It says which model was chosen; the selection record beside it
says why. Keeping the two apart is what stops the lock from becoming a place
where a number could later be revised. It also carries no path, no host, and no
timestamp: two freezes of the same selection in two directories produce identical
bytes.

**Freezing verifies again.** The selection already checked every gate; freezing
re-reads the artifacts, re-runs the Milestone 4 verifier, re-derives the candidate
universe from the ledger, and re-checks every fingerprint against the artifact it
names. Trusting the selection record would make the lock a copy of a claim rather
than a check of one.

### There is no force

Not a flag, not an environment variable, not a keyword argument. Freezing refuses
when:

- the selection outcome is not `eligible`;
- the chosen candidate is not among the selection's own results;
- the chosen candidate carries a blocking gate of either kind;
- the candidate is not champion-eligible for the task, or is a reference
  baseline, experimental, or anomaly-only family;
- the published model artifact does not verify;
- the operating point, a required calibrator, or the out-of-sample calibration
  report is missing;
- a fingerprint in the run disagrees with the selection record, or the operating
  point was selected for a different model;
- a lineage or dependency-contract fingerprint the lock must name is absent;
- the installed scikit-learn lies outside the reviewed range the model's arrays
  were extracted under;
- a candidate or the comparator named by the selection is no longer in the
  ledger.

Each of those is a state in which promoting would assert something nobody
established, and an override would be a way to assert it anyway.

### One champion per scope

The scope key digests the **experiment**: task, configuration, catalogs, gate
configuration, validation partition, role-scoped readable lineage, feature
catalog, and allowlist. Deliberately more than `task | split | labels`, because
those three collide across materially different experiments and a scope that
collides is a scope in which one champion silently replaces another.

Two selections that differ only in which candidates were available belong to the
same scope and cannot both be frozen. A second freeze is either the same freeze
again — confirmed byte for byte and left alone — or a contradiction, which is
refused. A materially different experiment is a different scope and gets its own
champion.

The `champion_freeze` ledger record is the immutable receipt: the scope key, the
lock's digest, the selection that justified it, the selected run and model, and
the category head's selection when one was bound.

---

## 13. Commands

```bash
# Select a champion from published runs, on validation-B only
uv run password-attack-detector ml select \
  --output-root artifacts/ml \
  --config configs/ml/model-development.yaml \
  --reports-dir reports

# Freeze the champion an eligible selection chose
uv run password-attack-detector ml freeze-champion \
  --output-root artifacts/ml \
  --config configs/ml/model-development.yaml
```

Neither command takes a feature, label, or split path. There is nowhere to hand
either of them a row, and no `--allow-test`, `--force-test`, or `--force`.

`ml select` exit codes:

| code | meaning |
|---|---|
| `0` | a champion was selected; `ml freeze-champion` may proceed |
| `2` | `no_eligible_champion` or `insufficient_validation_support` |
| `1` | the command could not run — unreadable artifacts, bad configuration |

Neither negative outcome is an error in the command, so neither is exit `1`. Both
are findings, and both are recorded.

`ml select` writes `reports/ml_gates.json` and `reports/ml_gates.md`: every
candidate, every gate, every observed value, and every reason code, with the
validation-only limit stated in the report itself.

---

## 14. Privacy

Selection output is model identifiers, gate verdicts, counts, rates, and stable
reason codes. No event identifier, pseudonym, campaign, row, coefficient,
threshold *value*, or absolute path appears in a record, a report, or on the
terminal — asserted by sweeps over all three.

---

## 15. Known limitations

**Nothing here measures generalisation.** Every figure in a selection record was
measured on validation-B, which took part in choosing the operating point, the
calibrator, and now the champion. Validation performance is not an estimate of
test performance and must never be quoted as one.

**Synthetic development data has a ceiling.** Attack behaviour is more separable
than real traffic because it was parameterised rather than observed; benign
behaviour is less varied, so a false-positive rate measured on it is a lower
bound at best; the class balance is a configuration choice, not a measurement.
Every gate verdict inherits all of that.

**PR-AUC is a summary, and a summary hides shape.** Two models with equal
PR-AUC can behave very differently at the operating point a deployment would
actually use, which is why the ceiling, floor, and calibration gates are
separate and mandatory rather than folded into one number.

**A passed gate is a passed gate, not a good model.** The gates express minimum
acceptability under the configured criteria. Clearing them means no mandatory
check was failed or unmeasurable; it does not mean the model is effective.

**The comparator is a prior-probability baseline.** Beating it establishes that a
candidate learned something beyond the class prior. It is a floor, not a
competitive benchmark, and no claim in this repository compares any model here to
a published system.

**Selection is scoped to one experiment.** Candidates from different
configurations, catalogs, feature contracts, or validation partitions are not
comparable, and the lineage gate blocks the attempt rather than ranking across
them.

**Freezing is not evaluating, and predicting is not either.** The lock names
what a later evaluation may run. Milestone 8 uses it to publish predictions —
including on the TEST split — without opening a single test label, so nothing it
writes revises, confirms, or contradicts any figure in a selection record. See
`docs/prediction-artifacts.md`.

**The CI-sized configuration is not a development run.**
`configs/ml/model-testing.yaml` shrinks every count so a contract test can run in
seconds. Its gate thresholds are loosened to match its loosened threshold search
and are not appropriate for a real run; the full 720-hour development workflow is
deliberately never executed in the ordinary test suite.
