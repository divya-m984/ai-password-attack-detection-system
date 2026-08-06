# Risk Scoring

How the rules that fired on one authentication event become a single bounded
number, and what that number does and does not mean.

## Not a probability

**`risk_score` is a bounded ordinal magnitude on a 0–100 scale. It is not a
probability, a likelihood, or a confidence.** Nothing in this system estimates
how likely an attack is; the score orders findings by how much configured
evidence accumulated behind them. The same applies to `signal_strength`, which
is the same kind of quantity on a `(0, 1]` scale.

A test enforces this: no field name, dataclass attribute, or public symbol in
the scoring module may contain "probability", "likelihood", or "confidence",
and the words appear in its prose only inside a sentence that denies them.

## The procedure

```
contribution_r = family_weight[family(r)] × signal_strength(r)     ∈ (0, 1]
c_g            = reduce(contribution_r for r in group g)           ∈ (0, 1]
combined       = 1 − Π over sorted g of (1 − c_g)                  ∈ [0, 1)
risk_score     = max(round(100 × combined, 4), min_fired_risk_score)
```

With **no rule fired**, `risk_score` is exactly `0.0` — a module constant, not
a configurable floor. That is what lets validation, alerting, and evaluation
all read a zero as "nothing fired" rather than "scored very low".

### Step 1 — weighted contribution

Each fired rule's signal strength is scaled by its family's configured weight.
The defaults rank account compromise highest (a successful authentication from
a novel context is the finding least tolerable to miss) and automation lowest
(legitimate service accounts share its shape exactly).

| Family | Default weight |
|--------|-------|
| `account_compromise` | 0.95 |
| `brute_force` | 0.90 |
| `spraying` | 0.85 |
| `stuffing` | 0.85 |
| `location` | 0.80 |
| `automation` | 0.70 |

### Step 2 — reduce within a correlation group

Rules in one correlation group describe **the same underlying behaviour**. The
default reducer is `max`, which is idempotent: three rules restating one
failure burst score exactly what the strongest of them scores alone. That is
the entire purpose of the groups.

| Correlation group | Rules |
|--------|-------|
| `credential_guessing_single_target` | PAD-BF-001, PAD-BF-002, PAD-DBF-001 |
| `source_fanout` | PAD-CS-001, PAD-PS-001 |
| `session_anomaly` | PAD-ATO-001, PAD-MFA-001 |
| `location_movement` | PAD-GEO-001 |
| `automation_timing` | PAD-BOT-001 |

A `bounded_sum` reducer is available as a configured alternative:
`1 − Π(1 − contribution)` within the group, which rewards breadth while staying
bounded.

### Step 3 — combine across groups

Noisy-OR over `sorted(group_keys)`. Sorting fixes the order of float
multiplication, so two runs agree bit for bit.

### Step 4 — map to severity

Three strictly ordered boundaries, all inclusive from below:

```
risk_score == 0.0            →  LOW   (only reachable with zero fired rules)
0.0 < score <  medium        →  LOW
medium   <= score < high     →  MEDIUM
high     <= score < critical →  HIGH
critical <= score            →  CRITICAL
```

Defaults are `medium: 40.0`, `high: 65.0`, `critical: 85.0`.

**`LOW` is an ordinary severity, not a suppressed one.** A fired detection
scoring below the `medium` boundary is a genuine `LOW` finding, and whether it
becomes an alert is decided by two configured gates, not by its band. See
[alert-lifecycle.md](alert-lifecycle.md).

## Proven properties

Each of these is pinned by a dedicated test.

| Property | Why it holds |
|--------|-------|
| `risk_score ∈ [0, 100]` | Noisy-OR over factors in `[0, 1]` yields `combined ∈ [0, 1)` |
| Zero rules fired ⟺ `risk_score == 0.0` | The zero branch is the only path to zero, and `min_fired_risk_score > 0` keeps every firing strictly above it — tested in both directions |
| Correlated rules cannot out-score the strongest of them | `max` within a group is idempotent |
| An unrelated signal can never lower risk | Adding a group multiplies by `(1 − c_g) ≤ 1` |
| Stronger evidence never lowers risk | `max` and noisy-OR are monotone non-decreasing in each argument |
| Order-invariant | Commutative reducers, plus sorted group iteration before the product |
| Deterministic | Pure float arithmetic, fixed rounding, no RNG, no wall clock |
| No `NaN` or infinity | Every input is bounded by validation; a non-finite value raises rather than propagating |

## Deterministic tie-breaking

The **primary attack category** is the category of the single strongest
contribution, ranked by `(−contribution, −severity, −signal_strength, rule_id)`.
The rule identifier is last and is always distinct within an anchor, so the
order is total: there is no case where two runs could disagree.

`fired_rule_ids` and `contributing_categories` are sorted before publication.

## Run identity

Every `RiskAssessment` records the **detection configuration fingerprint** of
the run that produced it. Two executions under different weights or thresholds
therefore stay distinguishable.

The fingerprint is deliberately *not* folded into `detection_id`. A detection's
identity is "rule R fired on anchor A", which should stay stable across
retunings; the *score* depends on every weight and threshold in the run, which
is why the fingerprint lives on the assessment instead.

## What the scorer cannot see

`RiskScorer` takes fired detections and nothing else. There is no parameter for
feature snapshots, labels, splits, campaigns, entity scope, canonical events, or
model output anywhere in its interface, and the module imports no reader for
any of them. A signature test and an import-graph test both enforce it.
