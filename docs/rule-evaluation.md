# Rule Evaluation

How the detection layer is measured against synthetic ground truth, and why
that measurement is kept strictly outside the detection path.

## Synthetic, and labelled as such

**These metrics describe synthetic authentication traffic with known ground
truth. They measure how this rule set behaves on generated data and are not
evidence of real-world detection effectiveness.**

That sentence is a module constant rendered at the top of both report formats,
so no reader can receive a number without it. No performance figure appears in
this repository's documentation: every metric is produced by running the
tooling, never transcribed into prose.

## The separation

`evaluation.py` is the **only** module in the detection package permitted to
read labels, split assignments, or campaign metadata. `DetectionEngine`,
`RiskScorer`, and `AlertBuilder` accept no such argument and import nothing
from it — an import-graph test parses each module's syntax tree and asserts it.

At the CLI, `detection run` has no `--labels` and no `--splits` option. The
absence is the enforcement: there is no argument to carry ground truth into a
detection run. `detection evaluate` is the only workflow that reads either.

## No threshold optimization

There is no threshold-search code path anywhere. Nothing reads a metric and
writes a configuration value back, and a test asserts the evaluation module
contains no configuration constructor and no mutation helper. Every report
records `threshold_optimization_performed: false`, so the absence is stated
rather than assumed.

Detection thresholds are chosen from the train and validation splits by a
human. A threshold fitted to a test set turns that set's numbers into a
description of the fitting rather than of the detector.

## Split discipline

Metrics are reported per split — `train`, `validation`, `test` — and each split
sees only its own events.

The **novel-anomaly holdout** is reported entirely separately and never folded
into the supervised category metrics. Its purpose is to measure behaviour on
anomalies no rule was written for; treating it as an ordinary supervised class
would destroy that measurement. It is also absent from
`CATEGORY_TO_SCENARIO`, so no rule can claim it.

## Metrics

Every number is calculated from the joined artifacts. None is a literal.

**Event level, per split and overall**

- normal (benign) false-positive rate
- malicious-event detection rate
- per-scenario detection rate
- per-category precision, recall, and F1
- macro and support-weighted summaries
- primary-category confusion matrix
- insufficient-data rate

**Alert level**

- alerts per 1,000 authentication events
- alert precision — an alert counts as correct when the majority of the
  assessments in its category and window are malicious. Exact membership would
  require replaying grouping; the approximation is stated here rather than
  presented as exact.
- alert recall over detectable malicious anchors
- duplicate-reduction ratio (fired detections ÷ alerts)

**Campaign level** — requires `--campaign-labels`

- campaign detection coverage
- campaign-level false-negative count
- time to first detection per campaign, as a distribution

A campaign counts as detected **once**, however many of its events fired, so
repeated detections cannot inflate coverage.

## Category mapping

`CATEGORY_TO_SCENARIO` is declared explicitly rather than inferred from a name
match, so a renamed enum member fails a test instead of silently scoring
against the wrong class.

| Attack category | Synthetic scenario |
|--------|-------|
| `brute_force` | `brute_force` |
| `password_spraying` | `password_spraying` |
| `credential_stuffing` | `credential_stuffing` |
| `distributed_brute_force` | `distributed_brute_force` |
| `account_takeover_indicator` | `account_takeover_indicator` |
| `impossible_travel_indicator` | `impossible_travel` |
| `bot_activity` | `bot_activity` |
| `mfa_sequence_anomaly` | *(none — the generator has no MFA scenario)* |

`PAD-MFA-001` is therefore scored at the event level only, never as a class.

## Null semantics

**A metric over an empty denominator is `None`, never zero and never one.**
Zero would read as a measured absence and one as measured perfection; both are
claims the data does not support. In JSON the value is `null`; in Markdown it
renders as `unavailable`.

Campaign metrics without a campaign table report `available: false` rather
than being fabricated. The `campaign_id` column is deliberately absent from
`FEATURE_LABEL_COLUMNS` — the splitter reads it internally for group isolation
but it is never published beside a model input — so campaign metrics require
the Phase 2 label table as a separate, explicit input.

## Determinism

Input row order cannot affect any result: every join is by identifier into a
dictionary, and every reported collection is sorted before emission. The
confusion matrix's class order is the sorted claimable scenarios followed by
`none`, so a matrix rendered twice reads the same twice.

## Privacy

Evaluation outputs are aggregate only. No event identifier, campaign
identifier, pseudonym, coordinate, or absolute path appears in either report
format — a sweep over a fixture with deliberately UUID-shaped anchors asserts
it.
