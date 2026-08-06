# Rule Contract

What a detection rule is allowed to see, what it must return, and what
guarantees it owes its callers.

## The detection-time boundary

A rule consumes **exactly one Phase 3 point-in-time feature snapshot** and
nothing else. Phase 3 is the boundary: every window, baseline, and sequence
value a rule needs was already computed there, over the half-open interval
`[t - window, t)`, with the anchor event and anything sharing its timestamp
excluded.

The detection engine **does not query raw event history**. There is no handle
to the canonical event table anywhere in the detection package, so a rule
cannot reconstruct a sequence, look backwards past its snapshot, or reach a
future row. A rule that needed more history would need a new Phase 3 feature,
not a new query.

**Labels and split assignments never enter the detection engine.** Neither
`DetectionEngine`, `RiskScorer`, nor `AlertBuilder` accepts an argument that
could carry them, and the engine rejects a snapshot table that merely
*contains* a ground-truth, split, campaign, scope, or model-output column
before any rule sees a row. Ground truth belongs to
`detection evaluate` and to nothing else.

## What a rule may read

| Leakage class | Permitted | Why |
|--------|-------|-------|
| `prior_only` | yes | Computed strictly from events before the anchor |
| `current_event_context` | yes | The anchor's own recorded fields; detection runs after the event completed |
| `baseline_derived` | yes | Derived from a baseline fitted on an approved reference interval |
| `key` | no | Identity and timing; the engine reads the anchor keys, rules never do |

Every feature a rule declares is resolved against the configured feature
catalog **once, at preparation time**. Resolution is where the boundary is
enforced: a template that names a column the catalog does not declare, a
column in the Phase 3 prohibition set, or a column whose leakage class a rule
may not read fails the run rather than the row.

## The two-phase contract

```python
class Rule(Protocol):
    spec: RuleSpec

    def prepare(self, config, feature_catalog) -> PreparedRule: ...


class PreparedRule(Protocol):
    preparation: RulePreparation

    def evaluate(self, row: Mapping[str, Any]) -> RuleEvaluationResult: ...
```

`prepare()` runs **once per run**. It resolves `{window}` templates into
concrete column names, checks every boundary above, and freezes the effective
thresholds. `evaluate()` then does dictionary reads and bounded arithmetic —
no catalog lookups, no configuration parsing, no name validation. That split is
what makes the cost `O(snapshots × enabled rules)` with bounded per-row work.

## Configuration is data, never logic

A rule's thresholds are declared as typed `RuleParameter` objects in the
catalog, each with a kind, a default, and bounds. YAML supplies *values for
declared parameters* and selects registered rules — nothing more. There is no
expression evaluation, no callable reference, no import path, and no `eval`,
`exec`, or dynamic import anywhere in the detection package; a test scans every
module to keep it that way. Configuration is read with `yaml.safe_load`, which
constructs only plain scalars, lists, and mappings.

A configuration key that looks like a credential is rejected outright.
Detection needs no secret, so accepting one could only ever be a mistake.

## What a rule must return

Every evaluation returns exactly one status:

| Status | Meaning |
|--------|-------|
| `fired` | The configured conditions matched; carries evidence, reason codes, and a strictly positive signal strength |
| `not_fired` | The conditions were evaluated and did not match |
| `insufficient_data` | The rule did not observe the history it declared, so it reached no verdict |
| `disabled` | The rule was switched off for this run |

The distinction between `not_fired` and `insufficient_data` is load-bearing.
A null feature means *unobserved*, not *clean*: reporting a quiet negative for
history a rule never saw would make a false-positive rate meaningless. The
minimum-history gate runs before any threshold comparison.

Status and payload are enforced rather than documented. A fired result must
carry evidence and at least one reason code; a non-fired result must carry
neither and must score exactly zero.

## Signal strength

```
sat(observed, threshold, k)     = clamp((observed/threshold − 1) / (k − 1))
sat_inv(observed, threshold, k) = clamp((threshold/max(observed, ε) − 1) / (k − 1))
raw             = Σ wᵢ·componentᵢ / Σ wᵢ
signal_strength = min_signal_strength + (1 − min_signal_strength) · raw
```

**Signal strength is not a probability.** It is a bounded ordinal magnitude in
`(0, 1]` describing how far a rule's observations exceeded its configured
thresholds. Nothing estimates how likely an attack is.

At exact threshold equality every saturating component contributes `0.0`, so
the strength is the configured floor — strictly positive, which is what
guarantees a fired rule never scores as though nothing fired. Every helper is
total: no division by zero, no `NaN`, no infinity.

## Evidence

Evidence messages come from **frozen templates declared in the catalog**,
rendered with sanitized numeric and enum values. No caller-supplied string ever
reaches a message, and the template grammar is restricted to a handful of bare
placeholders and validated at catalog build time.

**Evidence is an indicator, not causal proof.** Templates are rejected at
import if they contain proof-asserting or probability-asserting language, and
the schema rejects an evidence item whose message does. The permitted register
is *"matched the configured condition"*, *"contributed to this detection"*,
*"is consistent with"*, *"may indicate"*.

Evidence carries no identifier. A feature snapshot contains no username, user,
source, device, or session identifier, no IP address, and no coordinate — only
derived distances, velocities, and flags — so an identifier has no path into an
evidence payload. The schema additionally rejects any value shaped like a UUID
or a pseudonym.

## Stability guarantees

- **Rule identifiers and versions are frozen literals.** A test pins the
  current `(rule_id, rule_version)` set, so a change is a deliberate diff.
- **Registration is static.** `ALL_RULES` is a literal tuple; there is no
  discovery, plugin path, or configuration-supplied import. An implementation
  whose specification is not the catalog's own object is rejected.
- **Catalog order is sorted by rule identifier**, so evaluation order is a
  property of the data rather than of import order.
- **The catalog fingerprint excludes prose**, so fixing a typo in a description
  does not invalidate every artifact that recorded it.

## Where the manifest lives

The detection manifest is written into the detection **output directory**, not
`artifacts/`. The shared `verify_dataset` implementation requires artifact
paths relative to the manifest's own directory and rejects `..`, so a manifest
in `artifacts/` describing data in `data/processed/` cannot pass path
containment. Phase 3 makes the same choice for the same reason.
