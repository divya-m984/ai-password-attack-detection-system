# Alert Lifecycle

How risk assessments become alerts, when repeats are suppressed, and why
suppression is safe to trust.

## The tension

An alert is a claim that something is worth an analyst's attention. Emitting
one per fired detection would make that claim thousands of times about the same
behaviour. So this layer groups and suppresses — and the whole design question
is: **how do you reduce duplicate noise without ever hiding materially new
risk?**

Three answers, each enforced rather than intended:

1. **Every suppressed detection is counted.** `AlertingStats` carries an
   accounting identity that a schema validator enforces, so an alert cannot
   vanish without leaving a tally behind.
2. **A higher severity bypasses cooldown** rather than being swallowed by it.
3. **The grouping key includes the attack category and correlation group**, so
   an unrelated finding is never suppressed by an open alert about something
   else.

## The two alert gates

An assessment becomes an alert when **all three** hold:

1. `fired_rule_count > 0`
2. `risk_score >= min_alert_risk_score` — the minimum alert score
3. `severity >= min_alert_severity` — default `low`, so this gate is open by
   default

Consequences, all configuration-driven:

- An assessment with `fired_rule_count == 0` scores `0.0` and is never offered
  to the builder.
- A fired detection **below** `min_alert_risk_score` creates no alert. It stays
  fully visible in `rule_detections`, `risk_assessments`, the quality report's
  `below_score_floor_count`, and evaluation — diagnostics, not an alert.
- **At or above `min_alert_risk_score`, `LOW` is a valid alert severity** and a
  `LOW` alert row is written. Nothing in the code rejects an alert for being
  `LOW`; a test asserts the string `Severity.LOW` appears nowhere in the alert
  module.
- Raising `min_alert_severity` to `medium` is how an operator chooses to keep
  `LOW` assessments as non-alert diagnostics. All four settings are supported
  and tested.
- The quality report records `low_alert_reachable`, so a configuration that
  makes the `LOW` band unreachable is visible rather than surprising.

## Grouping

**Key:** `(attack_category, correlation_group, scope_kind, scope_value)`.

Assessments are sorted by `(anchor_event_time, anchor_event_id)` and processed
in one indexed pass. An open alert absorbs a new assessment when
`t − last_seen <= grouping_window` (inclusive at the boundary); a larger gap
closes it and may open a new one.

- `first_seen` is the first contributing event's time and **never moves**.
- `last_seen` **only advances**, so the window an alert reports always contains
  every event that contributed to it.
- `contributing_rule_ids` is the sorted union across contributors.
- `contributing_event_count` is exact.

### Aggregate and peak risk

- **`aggregate_risk_score` is the arithmetic mean** of the risk scores of the
  qualifying assessments grouped into the alert.
- **`peak_risk_score` is their maximum.**

The pair says "how bad typically" and "how bad at worst". The mean is bounded
above by the maximum, which is the invariant `SecurityAlert` enforces.

## Grouping modes

| Mode | When | `scope_kind` | `scope_value` |
|--------|-------|-------|-------|
| `category_scoped` | No scope table, or a degraded group | `none` | `null` |
| `entity_scoped` | A scope table supplied a value for this group's dimension | `user` or `source` | pseudonym |

Every alert records the mode that produced it, and `AlertingStats` records the
run-level regime — so two runs under different regimes can never be mistaken
for one another.

### The optional entity-scope table

```
anchor_event_id  str (non-null, UNIQUE)
user_scope       str (nullable)
source_scope     str (nullable)
```

Keyed one-to-one by `anchor_event_id` — the same key every other Phase 4 table
uses, so no secondary join key exists anywhere.

Which dimension each correlation group uses is a **typed, declared, validated**
mapping:

| Correlation group | Dimension |
|--------|-------|
| `credential_guessing_single_target` | `user` |
| `session_anomaly` | `user` |
| `location_movement` | `user` |
| `source_fanout` | `source` |
| `automation_timing` | `source` |

`user_scope` and `source_scope` are consumed **independently**: a
`source_fanout` alert groups on the source while a `session_anomaly` alert
groups on the account, in the same run.

**Validation runs before any alert is constructed.** A duplicate anchor is a
hard failure, never a last-wins merge; the relationship with the assessment
anchors is checked symmetrically in both directions. Either failure aborts the
run with a non-zero exit and no partial artifacts.

When the required dimension is **missing** for an anchor, that group alone
degrades to category-scoped grouping and `scope_missing_count` increments. The
run continues — losing one grouping key is a smaller harm than losing the alert
set — unless `strict_scope` is configured, which makes it fatal.

### Scope values are sensitive operational metadata

A scope value reaches **exactly one field of one artifact**:
`security_alerts.parquet`'s `scope_value` column. It never appears in evidence,
a reason code, any report, any CLI summary, the manifest, a statistic, or a
validation message. `EntityScopeRecord.__repr__` and `EntityScopeTable.__repr__`
are both redacted, because a repr reaches logs and tracebacks.

`DetectionEngine` and `RiskScorer` accept no scope argument and import nothing
from the alert module, so the confinement is a type error rather than a
convention.

## Suppression and escalation

### Cooldown

After an alert closes, a further qualifying assessment on the same key within
`cooldown` of `last_seen` is suppressed and counted under
`SuppressionReason.COOLDOWN`. The boundary is **inclusive**: exactly one
cooldown after the previous alert's last contribution is still suppressed; one
microsecond later is not.

### Materiality bypass

Cooldown suppresses *repeats*. A finding that is **more severe**, or that
**peaks above anything the previous alert saw**, is not a repeat — suppressing
it would be the failure mode this whole module exists to avoid. Such an
assessment opens a new alert and increments `AlertingStats.escalated_count`.
The behaviour is configured by `escalation_bypasses_cooldown` (default `true`).

### In-place escalation

Within an open alert's grouping window:

- A **higher severity** raises `current_severity` and increments the alert's
  own `escalation_count`.
- A **greater risk score at unchanged severity** updates `peak_risk_score`
  without an escalation count — the alert got worse, but not by a band.
- A **lower severity never reduces** an alert.

### Rate limit

At most `max_alerts_per_group_per_window` alerts per key within
`alert_limit_window`.

The horizon is deliberately **longer than `grouping_window`**. Two alerts for
one key are always at least one grouping window apart — a closer assessment is
absorbed into the open alert rather than opening a new one — so a limit
measured over the grouping window could never bite. The limit exists to
backstop the materiality bypass, which can legitimately emit several alerts for
one key in quick succession.

## The accounting identity

```
qualifying_count == grouped_detection_count
                  + suppressed_by_cooldown_count
                  + suppressed_by_rate_limit_count
entity_scoped_count + category_scoped_count == alert_count
```

Enforced by a validator on `AlertingStats`, so a statistics record that does
not balance cannot be constructed. Assessments rejected by the two gates are
counted separately under `below_score_floor_count` and
`below_severity_floor_count`.

`AlertingStats` exposes **counts only**. No grouping key, no scope value, and
no anchor identifier reaches it — these statistics are written to reports that
must stay safe to share.

## Deterministic alert identifiers

```
alert_id = uuid5(NS_ALERT, canonical_json({
    alerting_version, attack_category, correlation_group,
    scope_kind, scope_value, first_seen
}))
```

Never `uuid4`, wall-clock time, a process identifier, `hash()`, machine
identity, or ambient random state — a test scans the module's executable source
for all of them.

The contributing rule set is deliberately **excluded**: an alert that later
absorbs one more rule is the same alert, and its identity should not change
underneath a reader who already recorded it.

## What Phase 4 does not implement

No persistent alert store, no analyst workflow, no acknowledgement or
assignment state, and no notification of any kind.
