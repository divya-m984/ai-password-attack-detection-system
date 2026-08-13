# Locked test evaluation and system comparison

Milestone 9 opens the TEST labels. Once.

Everything before it was arranged so that opening them changes nothing: the
model, the preprocessing state, the class weights, the calibrator, the operating
point, the category head, the Phase 4 rule configuration, and — where one
exists — the hybrid fusion strategy were all fixed before this milestone ran. A
test label cannot reach any of them, because by the time a label is read there
is nothing left that a label could move.

This document describes what that lock consists of, what is measured, and what
the resulting numbers do and do not mean.

---

## 1. Why a test split can be read at all

A held-out split stops being held out the moment a decision is made from it. Not
when it is used to fit a model — that is only the most obvious case — but when
*any* choice is made because of what it showed: a threshold nudged, a candidate
preferred, a feature dropped, a configuration re-run "because the first one
looked wrong".

Milestone 9 can read the TEST labels because it has no such choice available.

**Everything tunable is frozen, and the freeze is verified before a label is
opened.** `evaluate_test` verifies the whole lineage first: the champion lock,
the freeze receipt, the validation selection, the training run, the model
content fingerprint, the manifest bytes, the preprocessor, the calibrator, the
operating point, and the prediction publication that scored the rows. Any
disagreement raises `ModelNotReadyError` and the evaluation does not happen.
There is no force option and no partial mode.

**The evaluation is a pure function of frozen inputs.** Re-running it produces
byte-identical artifacts, and publishing an identical evaluation twice writes
nothing the second time. An operator who dislikes the result has no lever: the
only way to get a different number is to change something upstream, which
changes the lineage, which produces a different receipt with a different
identity and leaves the first one standing.

**Nothing feeds back.** No threshold is re-selected, no candidate is re-ranked,
no configuration is emitted. The receipt records what was measured and under
which frozen lineage, and that is the whole of its effect on the system.

---

## 2. The reader boundary

Exactly two modules in this layer may open a table containing ground truth:
`detection.evaluation` and `ml.dataset`. That allowlist is asserted in both
directions by tests — a module added to it fails a test, and a module removed
from it fails a different one.

**Milestone 9 did not widen it.**

`ml.test_evaluation` — the module that computes every test metric — opens
nothing. It receives `TestOutcome` values as arguments:

```python
@dataclass(frozen=True, slots=True)
class TestOutcome:
    anchor_event_id: str
    malicious: bool
    known_category: str | None
    split: MLSplit
```

The composition root (`ml evaluate`) reads the labels through `ml.dataset`,
which was already permitted, and hands over typed values. An import guard in
`ml.test_evaluation` fails at import time if that module ever grows a label
reader import.

This is deliberately a boundary of *capability*, not of intent. A module that
cannot open a label table cannot open one by accident, and a reviewer does not
have to read it to find out.

---

## 3. What is measured

### 3.1 Binary detection

Precision, recall, false-positive rate, F1, balanced accuracy, accuracy, and a
confusion matrix, each with Wilson score intervals on the rates. A rate whose
denominator is zero has a `value` of `None` — not `0.0`. "No benign rows, so no
false-positive rate" and "a false-positive rate of zero" are different findings
and are never rendered the same way.

**PR-AUC is exact.** Distinct score levels are collected, ties are grouped, the
levels are walked in descending order, and the area is integrated step-wise and
right-continuous. There is no threshold grid and no interpolation, so the number
does not depend on a resolution parameter nobody chose deliberately. The same
implementation (`ml.ranking.score_levels`) serves selection and evaluation, so
a champion's validation PR-AUC and its test PR-AUC are the same quantity.

A system with no continuous score gets no PR-AUC and carries a stated reason
instead — see §4.

### 3.2 Calibration

Reliability across ten equal-width probability buckets, plus expected and
maximum calibration error. Reported only where a verified calibrator produced
probabilities; a decision score is never reinterpreted as one.

### 3.3 Attack category — downstream triage

The category head runs **after** the binary head, on the rows the binary head
flagged. Its metrics respect that:

| Population | Meaning |
|---|---|
| applicable | binary-positive, and the category head was asked |
| not applicable | the binary head said benign — the head was never asked |
| unknown | the head was asked and abstained |

`not_applicable` and `unknown` are different outcomes and are counted
separately. Collapsing them would report "the binary head said benign" as "the
category head could not decide". Per-class precision, recall, and F1 are
computed over the **applicable** population, because a denominator that included
rows the head never saw would measure the binary head instead.

### 3.4 Novel-anomaly holdout — experimental

Reported on its own, marked experimental, and never folded into a supervised
figure. No rule and no supervised model was fitted for these anomalies, so a
combined number would describe neither population. The holdout evaluation
carries `champion_evidence: false`: it is not evidence about the champion and is
not permitted to become evidence about it.

---

## 4. The three-system comparison

Rule-only, ML-only, and hybrid, measured over **one identical population**.

`require_common_universe` enforces this structurally: every system's decisions
must cover exactly the same anchor set, and a mismatch raises rather than being
reconciled. A comparison where one arm quietly scored fewer rows is not a
comparison, and it is easy to produce by accident.

**A system that could not be measured is reported, not dropped.** The hybrid arm
is absent whenever no fusion strategy qualified on validation-B, and the
comparison carries the reason:

- `no_fusion_selection: none was established before TEST` — no validation
  evidence was supplied, so nothing was chosen.
- `fusion_<status>: no hybrid strategy qualified on validation-B, and none is
  substituted` — candidates competed and none cleared the gates.

Dropping the arm instead would make a comparison of two systems look like a
comparison of three that the third happened to lose.

**The rule arm has no discrimination metric, on purpose.** Phase 4 emits a
bounded 0–100 severity magnitude. It is an ordinal severity, not a ranking
score, and integrating a PR curve over it would manufacture a metric out of a
quantity that was never meant to rank. The comparison records:

```
ordinal_rule_risk_is_not_a_discrimination_score
```

A fused decision has none either, for a different reason: it is a boolean, and
the two systems behind it live on different scales
(`fused_boolean`).

**Alert-level comparison.** Beyond row-level metrics, decisions are run through
the Phase 4 alert lifecycle — the same inclusive grouping window, inclusive
cooldown, escalation bypass, and rate limiting — so the systems can be compared
in the units a SOC actually receives. `ml.alerts` reproduces Phase 4's temporal
semantics exactly and is tested against it. It refuses to carry rule-only fields
(`contributing_rule_ids`, `correlation_group`, `attack_category`,
`signal_strengths`) so an ML alert cannot borrow a rule's explanation.

**Nothing declares a winner.** There is no `winner` field — an import guard
fails if one is added — and no report names a best system. Which system to run
depends on alert budget and on what a missed detection costs, and neither
quantity is in this artifact.

---

## 5. Hybrid fusion, and why it is chosen before TEST

Three strategies compete:

| Strategy | Decision |
|---|---|
| `OR_GATE` | flagged if either system flagged |
| `AND_GATE` | flagged only if both flagged |
| `STACKED` | a fitted meta-learner over (ml_score, rule_flag) |

The set is closed: `select_fusion_strategy` raises if handed anything other than
exactly these three.

Selection happens on **validation-B**, by `establish_fusion_selection`, driven by
`prepare_fusion_selection` — the pre-TEST orchestration `ml evaluate` runs before
it is able to open a label. Every argument to both is TRAIN-side, validation-side,
or frozen lineage. There is no TEST parameter, and an import-time guard fails if
one is ever added — a structural check rather than a review note, because the way
this stops being true is one plausible-sounding keyword at a time.

Ties are broken by declared strategy order, not by iteration order.

**All three candidates are constructed, every run.** OR_GATE and AND_GATE are
pure functions of the validation-B evidence. STACKED is fitted here, from real
out-of-fold refits — it is not a library capability the command declines to use.
The candidate universe is recorded whole, so a strategy that was not selected is
visibly *considered and rejected* rather than absent.

### Out-of-fold meta-features

A stacker fitted on scores from a model that saw those rows during training is
fitted on evidence that was never out of sample, and it will prefer the model
component for the wrong reason. So `STACKED` is fitted from **out-of-fold** TRAIN
scores:

- folds are cut at **campaign** boundaries, because a campaign is a coordinated
  burst whose events share a cause, and splitting one would let the base model
  learn a campaign and then be scored on the rest of it;
- assignment is a pure function of campaign identifiers and canonical row order
  — no shuffling, no random state;
- **each fold refits the whole pipeline** — preprocessing, class weights, and
  the model — on that fold's legal population, because a preprocessor fitted on
  all of TRAIN leaks just as surely as a model does;
- only TRAIN rows take part. Validation, TEST, and the holdout are never in a
  fit population and are never scored here.

Each fold instantiates the **frozen champion's own family and hyperparameters**
as its base recipe — a stacker whose meta-feature came from some other model
would be a stacker over a base model nobody froze. The already-frozen champion
itself is never refitted, and its full-TRAIN score is never used as an
out-of-fold score.

`StackedFusionState` pins `meta_feature_source="out_of_fold_train"`.

### When STACKED is legitimately unavailable

A stacker that cannot be built carries a typed reason, `OR_GATE` and `AND_GATE`
still compete, and the unavailable candidate enters the comparison with every
rate honestly unavailable rather than zero. The legitimate causes are all
scientific or data-shaped:

| Reason | Meaning |
|---|---|
| `insufficient_campaign_groups` | fewer campaigns than folds; a fold would have to split one |
| `insufficient_fold_support` | a fold lacks the positive or benign rows to fit the base family |
| `rule_evidence_unavailable:` | some out-of-fold TRAIN anchor carries no frozen rule decision |
| `base_recipe_unavailable:` | the frozen champion's model is not a candidate this configuration declares |
| `stacked_fit_failed:` | the meta-learner itself could not be fitted |

**"The CLI does not wire out-of-fold fitting" is not on that list, and cannot
be.** A test asserts that no unavailable reason blames the orchestration.

### Freezing before the reader

The ordering is enforced by types rather than by the order of the lines.
`prepare_fusion_selection` returns a `FusionFreezeProof`, and the TEST
ground-truth reader takes one as an argument. Only that function can produce a
proof — it holds a module-private token, and constructing one by hand raises.
A proof must also name the whole declared candidate universe, so a strategy
dropped from the record is a strategy nobody can claim was considered.

So a TEST label is unreachable until every candidate has been built or typed
unavailable, the validation-B comparison is complete, and the selection outcome
and fingerprint are final. When the fusion stage fails, the command returns
before a single TEST label becomes a value.

---

## 6. Publication

Staged into a sibling directory, re-read, verified, promoted atomically, and
only then appended to the experiment ledger. The receipt is written **last**.

The ordering is deliberate. A ledger entry asserting an evaluation that does not
exist would need an immutable record deleted to repair; an unindexed but valid
evaluation needs only to be read, which `reconcile_evaluations` does.

Published under `evaluations/<record_id>/`:

| File | Contents |
|---|---|
| `test_evaluation.json` | the receipt — full frozen lineage, no observational fields |
| `ml_evaluation.json` / `.md` | the champion's own test metrics |
| `system_comparison.json` / `.md` | rule-only, ML-only, hybrid |
| `category_evaluation.json` / `.md` | downstream triage, when a head was frozen |
| `anomaly_holdout.json` / `.md` | the experimental holdout, when scored |

The receipt records fingerprints and counts. It records no per-row output, no
anchor identifier, no pseudonym, and no feature value — an evaluation is an
aggregate statement, and it names no subject.

`TestEvaluationRecord` is the ledger's fourth record type, alongside
`training_run`, `validation_selection`, and `champion_freeze`. Appends are
`O_EXCL` plus `fsync`; an identical append is idempotent and a conflicting one
raises `LedgerConflictError`.

---

## 7. The commands

```bash
# The one command that reads a TEST label.
uv run password-attack-detector ml evaluate \
    --features processed/feature_snapshots.parquet \
    --labels processed/feature_labels.parquet \
    --splits processed/feature_splits.parquet \
    --allowlist allowlist.yaml \
    --risk-assessments detection/risk_assessments.parquet \
    --output-root artifacts/ml \
    --reports-dir reports

# Report an evaluation that was already published. Reads no label.
uv run password-attack-detector ml compare --output-root artifacts/ml
```

`ml evaluate` exits `0` when the evaluation completed and `2` when it could not
be completed on the available support — a finding, not an error in the command,
which is why it is not exit `1`.

To include the hybrid arm, supply validation evidence so a strategy can be
chosen before TEST is touched:

```bash
    --validation-prediction <prediction-id> \
    --validation-risk-assessments detection/validation_risk_assessments.parquet
```

To evaluate the experimental novel-anomaly holdout, supply a holdout prediction
carrying anomaly scores:

```bash
    --holdout-prediction <prediction-id>
```

`ml compare` reads only the published JSON and Markdown. It recomputes nothing —
re-deriving a figure at report time would let the report and the receipt
disagree, and the receipt is the record.

---

## 8. What these numbers are not

**They describe synthetic traffic.** Every figure this milestone produces was
measured on generated authentication logs with known ground truth. The generator
produces attacks it was written to produce, and a detector evaluated against
them is being asked whether it finds the attacks somebody already decided to
simulate. Every report carries this caveat in its own text, not only here.

**They are not evidence of real-world detection effectiveness.** Real
authentication traffic has volumes, seasonality, application diversity, and
adversary adaptation that no generator here models. A recall figure measured
above says nothing about recall against an attacker who knows the rules.

**They are not a deployment recommendation.** No report names a best system, and
the comparison deliberately withholds one. Alert budget, analyst capacity, and
the cost asymmetry between a missed detection and a false positive are all
operational facts that live outside this artifact.

**A single evaluation is a point estimate.** The Wilson intervals are reported
precisely so that a rate measured on a few dozen positive rows is not read as a
precise quantity. A metric whose support was thin says so through its interval
and through `support_status`.
