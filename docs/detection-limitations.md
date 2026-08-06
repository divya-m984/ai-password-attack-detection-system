# Detection Limitations

What Phase 4 does not do, cannot do, and should not be read as doing.

## No machine learning

**No ML model is implemented in Phase 4.** Nothing here trains, loads, scores,
or serves a model. Every decision comes from a reviewed Python rule with
declared, typed thresholds. `risk_score` and `signal_strength` are bounded
ordinal magnitudes computed by arithmetic — not model outputs, and not
probabilities.

## No real authentication traffic

The system generates no authentication requests, contacts no identity provider,
and sends nothing to any external service. It reads feature snapshots computed
from recorded events. It stores no password, no hash, no credential list, and
no token; it performs no authentication and attempts no bypass.

## What the rules can and cannot see

Rules consume **Phase 3 point-in-time feature snapshots and nothing else**.
The detection engine does not query raw event history — there is no handle to
the canonical event table anywhere in the package. A rule that needed more
history would need a new Phase 3 feature.

This means detection quality is bounded by feature quality. A behaviour Phase 3
does not express is a behaviour no Phase 4 rule can detect.

## Indicators, not findings

`PAD-ATO-001` and `PAD-GEO-001` produce **indicators**, and their names,
categories, evidence templates, and catalog descriptions all say so.

- A successful authentication from unfamiliar contexts is what an account
  takeover looks like from the outside. It is also what a new laptop on a
  business trip looks like. Nothing in this system distinguishes the two.
- "Impossible travel" describes arithmetic, not the world. Coarse location is
  an approximation, and a VPN, a corporate egress point, or a mobile carrier
  gateway relocates an apparent origin by thousands of kilometres with nobody
  moving.

Confirming either requires investigation this system does not perform.

## Evidence is not proof

A fired rule reports that a configured condition matched against observed
behaviour. **Evidence is an indicator, not causal proof.** Templates that
assert proof, confirmation, guarantees, probability, or likelihood are rejected
at import, and the schema rejects an evidence message containing that language.

Every rule carries declared `limitations` in the catalog naming the benign
behaviour that reproduces its shape — a shared egress address, a misconfigured
retrying client, a directory-service outage, a failing authenticator app,
legitimate scheduled automation. Those are rendered by
`detection catalog` and in [rule-catalog.md](rule-catalog.md).

## Known false-positive sources

| Rule | Benign behaviour that reproduces it |
|--------|-------|
| PAD-BF-001 | Shared egress address concentrating unrelated failures; a client retrying a stale credential |
| PAD-BF-002 | A user who forgets a password, fails repeatedly, resets it, then signs in |
| PAD-PS-001 | Corporate egress serving many accounts during an outage; a directory misconfiguration |
| PAD-CS-001 | A NAT gateway or proxy producing broad fan-out with varied client characteristics |
| PAD-DBF-001 | A popular account behind carrier NAT appearing to be reached from many sources |
| PAD-ATO-001 | Travel, device replacement, or a new office producing several novelty flags at once |
| PAD-GEO-001 | VPN, corporate egress, or carrier gateway relocating an apparent origin |
| PAD-BOT-001 | Monitoring probes, service accounts, and scheduled integrations |
| PAD-MFA-001 | A failing authenticator app, a clock-drifted token, a newly enrolled device |

## Synthetic evaluation proves nothing about production

Every metric this project reports describes generated traffic with known ground
truth. **Synthetic evaluation does not demonstrate real-world effectiveness.**
Thresholds tuned on synthetic data reflect the generator's parameters, not any
real population's behaviour.

## Suppression trade-offs

Alert suppression reduces duplicate noise, and any suppression can in principle
hide something. The mitigations are stated in
[alert-lifecycle.md](alert-lifecycle.md): a materiality bypass for more severe
or higher-peaking findings, a grouping key that isolates unrelated categories,
and **complete aggregate accounting** — every suppressed detection is counted,
enforced by a validator on the statistics record.

What suppression does *not* provide is a guarantee that a suppressed event was
uninteresting. It guarantees only that it was counted.

## Correlation groups deliberately collapse breadth

Rules in one correlation group are reduced by `max`, so three distinct signals
in one group score what the strongest scores alone. That is intended — it stops
one behaviour being counted three times — but it does mean breadth *within* a
group carries no extra weight. Noisy-OR across groups still rewards breadth
across genuinely different behaviours, and a `bounded_sum` reducer is available
as a configured alternative.

## Entity scope is sensitive and optional

The entity-scope table carries pseudonymous identifiers. It is opt-in, consumed
**only during alert construction**, and confined to one column of one artifact.
It is never passed to the engine or the scorer, never enters a report, a
manifest, a CLI summary, or a validation message, and its path is git-ignored.
Real-data alert artifacts require protected storage.

Without it, alerts group on category, correlation group, and the configured
time window — which is coarser, and the mode is recorded on every alert so the
two regimes are never confused.

## A reconstructed quality report cannot report everything

`detection run` writes a **live-run** quality report: it observed every rule
evaluation and every suppression decision, so every counter is an integer,
including counters that are genuinely zero.

`detection profile` rebuilds a report from **published artifacts**. Those
artifacts record what fired, what was scored, and what alerted — not what did
*not* fire, which rules were disabled, or which suppression decisions were
taken. Those counters are reported as **unavailable** (`null` in JSON,
"Unavailable from published artifacts" in Markdown), and the report carries
warning code `Q001` explaining why.

**Unavailable is not zero.** Reporting zero would assert the event never
happened, which is a stronger claim than "this artifact set does not say". The
counters affected are listed on the report itself as `unavailable_metrics`:

`input_snapshot_count`, `total_rule_evaluation_count`, `not_fired_count`,
`disabled_rule_count`, `grouped_detection_count`,
`suppressed_by_cooldown_count`, `suppressed_by_rate_limit_count`,
`suppressed_total`, `escalated_count`, `below_score_floor_count`,
`below_severity_floor_count`, `scope_missing_count`.

Everything else — fired counts, per-rule and per-family triggers, category and
severity distributions, risk summaries, alert counts and durations,
insufficient-data totals — is derived exactly from the tables and matches the
live run. A rule with no rows in the detection table triggered zero times, and
that zero *is* measured.

## Scale and performance

Cost is `O(snapshots × enabled rules)` with bounded per-row work, verified by
structural operation counters rather than timings. No wall-clock benchmark is
asserted in CI, and no throughput figure is published here.

## Not implemented in Phase 4

- No persistent alert store, analyst workflow, acknowledgement state, or
  notification
- No streaming or online detection; the pipeline is batch
- No feedback loop, automatic threshold tuning, or adaptive thresholds
- No API service, dashboard, or database
- No response, blocking, or remediation action of any kind
