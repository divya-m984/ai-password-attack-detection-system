# Rule Catalog

Generated from the rule registry. Do not edit by hand; regenerate with `password-attack-detector detection catalog --format markdown`.

- Rule catalog version: `1.0.0`
- Catalog fingerprint: `e56c7d5879cc2678e31217124b54fa6c9801f7a8315d80bc8dc4d9483fef5699`
- Registered rules: 9

Rules describe behaviour observed in point-in-time feature snapshots. A fired rule reports that a configured condition matched; it is not evidence that an attack occurred.

## Rules by family

| Family | Rules |
|--------|-------|
| `account_compromise` | 2 |
| `automation` | 1 |
| `brute_force` | 3 |
| `location` | 1 |
| `spraying` | 1 |
| `stuffing` | 1 |

## Correlation groups

Rules in one group describe the same underlying behaviour. Their contributions are reduced within the group before groups are combined, so restating one behaviour cannot inflate a risk score.

| Correlation group | Rules |
|--------|-------|
| `credential_guessing_single_target` | `PAD-BF-001`, `PAD-BF-002`, `PAD-DBF-001` |
| `source_fanout` | `PAD-CS-001`, `PAD-PS-001` |
| `session_anomaly` | `PAD-ATO-001`, `PAD-MFA-001` |
| `location_movement` | `PAD-GEO-001` |
| `automation_timing` | `PAD-BOT-001` |

## Rule index

| Rule | Version | Name | Severity |
|--------|-------|-------|-------|
| `PAD-ATO-001` | `1.0.0` | Account-takeover indicator | `critical` |
| `PAD-BF-001` | `1.0.0` | Concentrated brute-force indicator | `high` |
| `PAD-BF-002` | `1.0.0` | Successful authentication after failure burst | `high` |
| `PAD-BOT-001` | `1.0.0` | Bot-like authentication indicator | `medium` |
| `PAD-CS-001` | `1.0.0` | Credential-stuffing indicator | `high` |
| `PAD-DBF-001` | `1.0.0` | Distributed brute-force indicator | `critical` |
| `PAD-GEO-001` | `1.0.0` | Impossible-travel indicator | `high` |
| `PAD-MFA-001` | `1.0.0` | MFA sequence anomaly indicator | `medium` |
| `PAD-PS-001` | `1.0.0` | Password-spraying indicator | `high` |

## PAD-ATO-001 -- Account-takeover indicator

A successful authentication from a combination of contexts new to the account, supported by at least one behavioural deviation. This is an indicator of behaviour consistent with account takeover, not a finding that an account was taken over.

| Property | Value |
|--------|-------|
| Rule version | `1.0.0` |
| Family | `account_compromise` |
| Attack category | `account_takeover_indicator` |
| Default severity | `critical` |
| Correlation group | `session_anomaly` |
| Evaluation scope | `anchor_event` |
| Privacy class | `non_sensitive` |
| Deprecated | no |

### Required features

- `current_authentication_outcome`
- `user_in_baseline`
- `is_new_device_for_user`
- `is_new_source_for_user`
- `is_new_country_for_user`
- `is_new_application_for_user`
- `is_new_auth_method_for_user`

### Optional features

- `prior_failures_since_user_success`
- `login_hour_deviation`
- `response_time_zscore`
- `user_event_rate_ratio`
- `current_mfa_outcome`
- `distance_from_user_baseline_centroid_km`

### Thresholds

| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |
|--------|-------|-------|-------|-------|-------|-------|
| `min_novel_context_count` | `int` | `2` | 1.0 | 5.0 | count | How many of the five novelty indicators must be true. |
| `min_supporting_signals` | `int` | `1` | 1.0 | 6.0 | count | How many supporting behavioural deviations must be present. |
| `min_prior_failures` | `int` | `3` | 1.0 | - | count | Failures since this account last succeeded. |
| `min_login_hour_deviation` | `float` | `0.9` | 0.0 | 1.0 | ratio | How unusual the anchor hour must be for this account. |
| `min_abs_response_time_zscore` | `float` | `3.0` | 0.0 | - | zscore | Absolute standardised response-time deviation. |
| `min_event_rate_ratio` | `float` | `3.0` | 1.0 | - | ratio | Recent event rate relative to the account's baseline rate. |
| `min_baseline_distance_km` | `float` | `1000.0` | 0.0 | 20100.0 | km | Distance from the account's baseline location centroid. |

### Minimum history

Requires a non-null value for:

- `user_in_baseline`

Novelty is meaningless for an account absent from the fitted baseline: a cold account is not the same as a known account seeing a new device.

### Evidence

| Code | Feature | Comparator | Unit | Meaning |
|--------|-------|-------|-------|-------|
| `ATO_CURRENT_SUCCESS` | `current_authentication_outcome` | `eq` | - | The success precondition. |
| `ATO_NOVEL_CONTEXT_COUNT` | derived | `gte` | count | Count of novel contexts across device, source, country, application, and method. |
| `ATO_PRIOR_FAILURES` | `prior_failures_since_user_success` | `gte` | count | Supporting signal: preceding failure pressure. |
| `ATO_LOGIN_HOUR_DEVIATION` | `login_hour_deviation` | `gte` | ratio | Supporting signal: unusual hour for this account. |
| `ATO_RESPONSE_TIME_DEVIATION` | `response_time_zscore` | `gte` | zscore | Supporting signal: response-time deviation. |
| `ATO_EVENT_RATE_DEVIATION` | `user_event_rate_ratio` | `gte` | ratio | Supporting signal: event-rate deviation. |
| `ATO_MFA_DEVIATION` | `current_mfa_outcome` | `in` | - | Supporting signal: multi-factor deviation. |
| `ATO_BASELINE_DISTANCE` | `distance_from_user_baseline_centroid_km` | `gte` | km | Supporting signal: distance from the account's usual area. |

### Expected limitations

- Travel, a device replacement, or a new office reproduces several novelty indicators at once for a legitimate user.
- The rule reports an indicator. Confirming account takeover requires investigation this system does not perform.

## PAD-BF-001 -- Concentrated brute-force indicator

Repeated failed authentication concentrated on one account from one source, at a cadence and failure share consistent with automated credential guessing. The source-fan-out ceiling keeps a password-spraying source out of this rule.

| Property | Value |
|--------|-------|
| Rule version | `1.0.0` |
| Family | `brute_force` |
| Attack category | `brute_force` |
| Default severity | `high` |
| Correlation group | `credential_guessing_single_target` |
| Evaluation scope | `anchor_event` |
| Privacy class | `non_sensitive` |
| Deprecated | no |

### Required features

- `pair_failure_count__{window}`
- `pair_failure_rate__{window}`
- `user_failure_count__{window}`
- `prior_consecutive_user_failures`
- `source_unique_user_count__{cardinality_window}`

### Optional features

- `pair_mean_interarrival_seconds__{window}`
- `user_blocked_count__{window}`

### Thresholds

| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |
|--------|-------|-------|-------|-------|-------|-------|
| `window` | `window` | `5m` | - | - | duration | Window for the pair and user failure counts. |
| `cardinality_window` | `window` | `5m` | - | - | duration | Window for the source's distinct-user count; must be a window at which cardinality features are emitted. |
| `min_pair_failures` | `int` | `8` | 1.0 | - | count | Failures for this user-source pair. |
| `min_user_failures` | `int` | `8` | 1.0 | - | count | Failures for this user. |
| `min_pair_failure_rate` | `float` | `0.8` | 0.0 | 1.0 | ratio | Share of this pair's attempts that failed. |
| `min_consecutive_failures` | `int` | `5` | 1.0 | - | count | Length of the unbroken failure run immediately before the anchor. |
| `max_source_unique_users` | `int` | `3` | 1.0 | - | count | Ceiling on the source's distinct-user count; above this the behaviour is fan-out, not concentration. |
| `max_mean_interarrival_seconds` | `float` | `60.0` | 0.0 | 86400.0 | seconds | Cadence ceiling for the supporting timing component. |
| `blocked_support_threshold` | `int` | `2` | 1.0 | - | count | Blocked-authentication count that adds supporting evidence. This is the only use of blocked-account activity; there is no standalone blocked-account rule. |

### Minimum history

Requires a non-null value for:

- `pair_failure_rate__{window}`

The pair failure rate is null when the window holds no attempts for this user-source pair, which is unseen history rather than a clean negative.

### Evidence

| Code | Feature | Comparator | Unit | Meaning |
|--------|-------|-------|-------|-------|
| `BF_PAIR_FAILURE_COUNT` | `pair_failure_count__{window}` | `gte` | count | Failure volume concentrated on one user-source pair. |
| `BF_USER_FAILURE_COUNT` | `user_failure_count__{window}` | `gte` | count | Failure volume against the targeted account. |
| `BF_PAIR_FAILURE_RATE` | `pair_failure_rate__{window}` | `gte` | ratio | Share of the pair's attempts that failed. |
| `BF_CONSECUTIVE_USER_FAILURES` | `prior_consecutive_user_failures` | `gte` | count | Length of the immediately preceding failure run. |
| `BF_SOURCE_TARGET_CONCENTRATION` | `source_unique_user_count__{cardinality_window}` | `lte` | count | Concentration guard separating this rule from password spraying. |
| `BF_PAIR_INTERARRIVAL` | `pair_mean_interarrival_seconds__{window}` | `lte` | seconds | Supporting cadence evidence; contributes signal, never required. |
| `BF_BLOCKED_ACTIVITY` | `user_blocked_count__{window}` | `gte` | count | Supporting blocked-account evidence; contributes signal, never required. |

### Expected limitations

- A shared egress address used by many colleagues can concentrate unrelated failures onto one apparent source.
- A misconfigured client retrying a stale credential reproduces this shape without an attacker.

## PAD-BF-002 -- Successful authentication after failure burst

A successful authentication immediately following a sustained run of failures for the same account or user-source pair. Apart from the current outcome, every condition reads prior-history sequence features only. Shares a correlation group with the other brute-force rules so one failure burst is never counted twice.

| Property | Value |
|--------|-------|
| Rule version | `1.0.0` |
| Family | `brute_force` |
| Attack category | `brute_force` |
| Default severity | `high` |
| Correlation group | `credential_guessing_single_target` |
| Evaluation scope | `anchor_event` |
| Privacy class | `non_sensitive` |
| Deprecated | no |

### Required features

- `current_authentication_outcome`
- `prior_failures_since_pair_success`
- `prior_failures_since_user_success`
- `previous_user_outcome`
- `seconds_since_user_previous_failure`

### Thresholds

| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |
|--------|-------|-------|-------|-------|-------|-------|
| `min_pair_failures_since_success` | `int` | `6` | 1.0 | - | count | Failures for this user-source pair since the pair last succeeded. |
| `min_user_failures_since_success` | `int` | `8` | 1.0 | - | count | Failures for this account since it last succeeded. |
| `max_seconds_since_previous_failure` | `float` | `300.0` | 0.0 | 86400.0 | seconds | How recently the preceding failure must have occurred. |

### Minimum history

Requires a non-null value for:

- `previous_user_outcome`
- `seconds_since_user_previous_failure`

Both are null for an account with no prior event or no prior failure, which is unseen history rather than a clean negative.

### Evidence

| Code | Feature | Comparator | Unit | Meaning |
|--------|-------|-------|-------|-------|
| `BF2_CURRENT_SUCCESS` | `current_authentication_outcome` | `eq` | - | The success condition that separates this rule from PAD-BF-001. |
| `BF2_PAIR_FAILURE_BURST` | `prior_failures_since_pair_success` | `gte` | count | Failure burst for the user-source pair. |
| `BF2_USER_FAILURE_BURST` | `prior_failures_since_user_success` | `gte` | count | Failure burst for the account. |
| `BF2_PREVIOUS_OUTCOME` | `previous_user_outcome` | `in` | - | The immediately preceding outcome. |
| `BF2_FAILURE_RECENCY` | `seconds_since_user_previous_failure` | `lte` | seconds | Recency of the preceding failure. |

### Expected limitations

- A user who forgets a password, fails repeatedly, resets it, and then signs in successfully reproduces this shape exactly.
- The rule reports a sequence, not an assessment of who supplied the successful credential.

## PAD-BOT-001 -- Bot-like authentication indicator

Sustained authentication volume from one source at machine-like regular intervals with uniform client characteristics. The volume floor and the dispersion requirement together mean no short sequence and no single fast interval can fire this rule.

| Property | Value |
|--------|-------|
| Rule version | `1.0.0` |
| Family | `automation` |
| Attack category | `bot_activity` |
| Default severity | `medium` |
| Correlation group | `automation_timing` |
| Evaluation scope | `anchor_event` |
| Privacy class | `non_sensitive` |
| Deprecated | no |

### Required features

- `source_attempt_count__{dispersion_window}`
- `source_interarrival_coefficient_of_variation__{dispersion_window}`
- `source_mean_interarrival_seconds__{dispersion_window}`
- `source_unique_user_agent_count__{cardinality_window}`

### Optional features

- `source_unique_user_count__{cardinality_window}`

### Thresholds

| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |
|--------|-------|-------|-------|-------|-------|-------|
| `dispersion_window` | `window` | `15m` | - | - | duration | Window for the timing statistics; must be a window at which dispersion features are emitted. |
| `cardinality_window` | `window` | `1h` | - | - | duration | Window for the source's distinct client counts. |
| `min_attempts` | `int` | `20` | 3.0 | - | count | Attempt floor; below this the timing statistics are not meaningful. |
| `max_interarrival_cov` | `float` | `0.15` | 0.0 | 10.0 | ratio | Ceiling on the interarrival coefficient of variation; low values indicate machine-like regularity. |
| `max_mean_interarrival_seconds` | `float` | `120.0` | 0.0 | 86400.0 | seconds | Ceiling on the mean gap between attempts. |
| `max_unique_user_agents` | `int` | `2` | 1.0 | - | count | Ceiling on distinct user-agent families; automation tends to repeat one client string. |
| `min_unique_users` | `int` | `0` | 0.0 | - | count | Optional account fan-out floor; zero disables the component. |

### Minimum history

Requires a non-null value for:

- `source_interarrival_coefficient_of_variation__{dispersion_window}`

The coefficient of variation is null below two in-window observations or at a zero mean, which is too little timing history to judge regularity.

### Evidence

| Code | Feature | Comparator | Unit | Meaning |
|--------|-------|-------|-------|-------|
| `BOT_ATTEMPT_VOLUME` | `source_attempt_count__{dispersion_window}` | `gte` | count | Attempt volume floor. |
| `BOT_TIMING_REGULARITY` | `source_interarrival_coefficient_of_variation__{dispersion_window}` | `lte` | ratio | Timing regularity; the automation discriminator. |
| `BOT_MEAN_INTERARRIVAL` | `source_mean_interarrival_seconds__{dispersion_window}` | `lte` | seconds | Mean interarrival interval. |
| `BOT_CLIENT_UNIFORMITY` | `source_unique_user_agent_count__{cardinality_window}` | `lte` | count | Repeated client characteristics. |
| `BOT_SOURCE_FANOUT` | `source_unique_user_count__{cardinality_window}` | `gte` | count | Optional fan-out evidence; contributes signal, never required. |

### Expected limitations

- Legitimate automation -- monitoring probes, service accounts, and scheduled integrations -- authenticates on exactly this rhythm.
- Regular timing describes a client, not an intent.

## PAD-CS-001 -- Credential-stuffing indicator

Broad account fan-out from one source with a mix of failures and occasional successes, varied client characteristics, and an unfamiliar device or country for the account. No credential value, password-reuse data, or credential list is read anywhere.

| Property | Value |
|--------|-------|
| Rule version | `1.0.0` |
| Family | `stuffing` |
| Attack category | `credential_stuffing` |
| Default severity | `high` |
| Correlation group | `source_fanout` |
| Evaluation scope | `anchor_event` |
| Privacy class | `non_sensitive` |
| Deprecated | no |

### Required features

- `source_unique_user_count__{cardinality_window}`
- `source_success_count__{window}`
- `source_failure_count__{window}`
- `source_attempt_count__{window}`
- `source_unique_device_count__{cardinality_window}`
- `source_unique_user_agent_count__{cardinality_window}`
- `user_in_baseline`
- `is_new_device_for_user`
- `is_new_country_for_user`

### Thresholds

| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |
|--------|-------|-------|-------|-------|-------|-------|
| `window` | `window` | `1h` | - | - | duration | Window for the source's outcome counts. |
| `cardinality_window` | `window` | `1h` | - | - | duration | Window for the source's distinct counts. |
| `min_unique_users` | `int` | `10` | 2.0 | - | count | Distinct accounts touched. |
| `min_successes` | `int` | `1` | 1.0 | - | count | Successes from this source; the mixed-outcome signature. |
| `min_failures` | `int` | `10` | 1.0 | - | count | Failures from this source. |
| `min_unique_devices` | `int` | `3` | 1.0 | - | count | Distinct devices seen. |
| `min_unique_user_agents` | `int` | `3` | 1.0 | - | count | Distinct user-agent families. |
| `max_attempts_per_user` | `float` | `4.0` | 1.0 | - | ratio | Ceiling on attempts per targeted account. |

### Minimum history

Requires a non-null value for:

- `user_in_baseline`

The unfamiliar-context condition needs a fitted baseline for the account. An account absent from the baseline yields insufficient_data, never a clean negative.

### Evidence

| Code | Feature | Comparator | Unit | Meaning |
|--------|-------|-------|-------|-------|
| `CS_SOURCE_USER_FANOUT` | `source_unique_user_count__{cardinality_window}` | `gte` | count | Account fan-out from a single source. |
| `CS_MIXED_OUTCOMES` | `source_success_count__{window}` | `gte` | count | The mixed-outcome signature separating stuffing from spraying. |
| `CS_CLIENT_DIVERSITY` | `source_unique_device_count__{cardinality_window}` | `gte` | count | Device and user-agent diversity from one source. |
| `CS_ATTEMPTS_PER_USER` | `source_attempt_count__{window}` | `lte` | ratio | Derived attempts-per-account ratio. |
| `CS_UNFAMILIAR_CONTEXT` | `is_new_device_for_user` | `is_true` | - | Unfamiliar-context requirement; needs a fitted baseline. |

### Expected limitations

- A NAT gateway or corporate proxy produces broad fan-out and varied client characteristics from a single apparent source.
- The rule observes behaviour only; it has no visibility into whether any credential was reused from another service.

## PAD-DBF-001 -- Distributed brute-force indicator

One account failing repeatedly across many distinct sources, each contributing few attempts. The per-source volume ceiling is what separates this from a single high-volume source, and the source fan-out ceiling keeps a spraying source out.

| Property | Value |
|--------|-------|
| Rule version | `1.0.0` |
| Family | `brute_force` |
| Attack category | `distributed_brute_force` |
| Default severity | `critical` |
| Correlation group | `credential_guessing_single_target` |
| Evaluation scope | `anchor_event` |
| Privacy class | `non_sensitive` |
| Deprecated | no |

### Required features

- `user_unique_source_count__{cardinality_window}`
- `user_failure_count__{window}`
- `user_failure_rate__{window}`
- `pair_attempt_count__{window}`
- `source_unique_user_count__{cardinality_window}`

### Thresholds

| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |
|--------|-------|-------|-------|-------|-------|-------|
| `window` | `window` | `1h` | - | - | duration | Window for the account's failure counts. |
| `cardinality_window` | `window` | `1h` | - | - | duration | Window for the distinct counts. |
| `min_unique_sources` | `int` | `8` | 2.0 | - | count | Distinct sources targeting this account. |
| `min_user_failures` | `int` | `20` | 1.0 | - | count | Failures against this account. |
| `min_user_failure_rate` | `float` | `0.8` | 0.0 | 1.0 | ratio | Share of this account's attempts that failed. |
| `max_pair_attempts` | `int` | `5` | 1.0 | - | count | Ceiling on attempts from any one source; the distribution guard. |
| `max_source_unique_users` | `int` | `3` | 1.0 | - | count | Ceiling on the anchor source's distinct-user count, so a spraying source is not reported as distributed brute force. |

### Minimum history

Requires a non-null value for:

- `user_failure_rate__{window}`

The account failure rate is null when the window holds no attempts for this account.

### Evidence

| Code | Feature | Comparator | Unit | Meaning |
|--------|-------|-------|-------|-------|
| `DBF_USER_SOURCE_FANOUT` | `user_unique_source_count__{cardinality_window}` | `gte` | count | Source fan-out against one account. |
| `DBF_USER_FAILURE_COUNT` | `user_failure_count__{window}` | `gte` | count | Aggregate failure volume against the account. |
| `DBF_USER_FAILURE_RATE` | `user_failure_rate__{window}` | `gte` | ratio | Share of the account's attempts that failed. |
| `DBF_LOW_PER_SOURCE_VOLUME` | `pair_attempt_count__{window}` | `lte` | count | Per-source volume ceiling; the distributed discriminator. |
| `DBF_SOURCE_NOT_FANNED_OUT` | `source_unique_user_count__{cardinality_window}` | `lte` | count | Spraying guard. |

### Expected limitations

- A popular account behind a mobile carrier NAT can legitimately appear to be reached from many distinct sources.
- Coarse source identity means one attacker behind a proxy pool and many unrelated clients look alike.

## PAD-GEO-001 -- Impossible-travel indicator

A successful authentication whose distance from the account's previous located success implies a travel speed above a configured plausible maximum. Derived from coarse location data, which is an approximation; this is an indicator, not a finding of compromise. No coordinate is read or emitted -- the rule consumes derived distance, velocity, and status columns only.

| Property | Value |
|--------|-------|
| Rule version | `1.0.0` |
| Family | `location` |
| Attack category | `impossible_travel_indicator` |
| Default severity | `high` |
| Correlation group | `location_movement` |
| Evaluation scope | `anchor_event` |
| Privacy class | `non_sensitive` |
| Deprecated | no |

### Required features

- `current_authentication_outcome`
- `user_previous_success_geo__status`
- `implied_velocity__status`
- `distance_km_from_user_previous_success`
- `implied_velocity_kmh_from_previous_success`

### Optional features

- `country_changed_since_previous_success`
- `seconds_since_user_previous_success_with_location`

### Thresholds

| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |
|--------|-------|-------|-------|-------|-------|-------|
| `min_distance_km` | `float` | `500.0` | 1.0 | 20100.0 | km | Minimum meaningful distance; below this, coarse-location error dominates. |
| `min_velocity_kmh` | `float` | `900.0` | 1.0 | - | kmh | Implied travel speed above which the movement is implausible. |
| `distance_rounding_km` | `float` | `10.0` | 1.0 | - | km | Rounding applied to the distance reported in evidence, so no precise location is inferable. |
| `require_country_change` | `bool` | `False` | - | - | - | Whether a country change is also required. |
| `zero_elapsed_policy` | `string` | `fire` | - | - | - | How to treat a zero elapsed interval, where implied velocity is undefined rather than ordinary travel. |

### Minimum history

Requires a non-null value for:

- `user_previous_success_geo__status`

Any status other than 'ok' means the comparison could not be made: no prior located success, or a missing coordinate on either side. Each yields insufficient_data with the status as the reason.

### Evidence

| Code | Feature | Comparator | Unit | Meaning |
|--------|-------|-------|-------|-------|
| `GEO_CURRENT_SUCCESS` | `current_authentication_outcome` | `eq` | - | The success precondition. |
| `GEO_DISTANCE` | `distance_km_from_user_previous_success` | `gte` | km | Rounded great-circle distance from the previous located success. |
| `GEO_IMPLIED_VELOCITY` | `implied_velocity_kmh_from_previous_success` | `gte` | kmh | Capped implied velocity between the two located successes. |
| `GEO_ZERO_ELAPSED_INTERVAL` | `implied_velocity__status` | `eq` | - | Explicit zero-elapsed branch; never treated as ordinary travel. |
| `GEO_COUNTRY_CHANGE` | `country_changed_since_previous_success` | `is_true` | - | Optional country-change requirement. |

### Expected limitations

- Coarse location is approximate; a VPN, a corporate egress point, or a mobile carrier gateway relocates an apparent origin by thousands of kilometres with no travel at all.
- The rule reports an indicator derived from approximate data, not evidence that an account was used by two people.

## PAD-MFA-001 -- MFA sequence anomaly indicator

Elevated recent multi-factor failure or challenge activity for an account, combined with an abnormal multi-factor outcome on the anchor event. Both parts are required, and a minimum-history gate means a single ordinary challenge cannot fire the rule. Shares a correlation group with the account-takeover indicator so one compromise narrative is not counted twice.

| Property | Value |
|--------|-------|
| Rule version | `1.0.0` |
| Family | `account_compromise` |
| Attack category | `mfa_sequence_anomaly` |
| Default severity | `medium` |
| Correlation group | `session_anomaly` |
| Evaluation scope | `anchor_event` |
| Privacy class | `non_sensitive` |
| Deprecated | no |

### Required features

- `user_attempt_count__{window}`
- `user_mfa_failure_count__{window}`
- `user_challenge_count__{window}`
- `current_mfa_outcome`

### Thresholds

| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |
|--------|-------|-------|-------|-------|-------|-------|
| `window` | `window` | `15m` | - | - | duration | Window for the account's multi-factor counts. |
| `min_mfa_history_events` | `int` | `6` | 2.0 | - | count | Attempts the account must have in the window before the rule will reach a verdict. |
| `min_mfa_observations` | `int` | `3` | 1.0 | - | count | Combined challenge and multi-factor failure observations required before the rule will reach a verdict. |
| `min_mfa_failures` | `int` | `4` | 1.0 | - | count | Multi-factor failures for this account. |
| `min_challenges` | `int` | `4` | 1.0 | - | count | Challenged authentications for this account. |

### Minimum history

Below min_mfa_history_events attempts, or below min_mfa_observations combined challenge and multi-factor failure observations, the rule returns insufficient_data. There is not enough multi-factor history to call a sequence anomalous.

### Evidence

| Code | Feature | Comparator | Unit | Meaning |
|--------|-------|-------|-------|-------|
| `MFA_HISTORY_SUFFICIENT` | `user_attempt_count__{window}` | `gte` | count | Minimum-history gate. |
| `MFA_PRIOR_FAILURES` | `user_mfa_failure_count__{window}` | `gte` | count | Elevated prior multi-factor failure activity. |
| `MFA_PRIOR_CHALLENGES` | `user_challenge_count__{window}` | `gte` | count | Elevated prior challenge activity. |
| `MFA_CURRENT_OUTCOME` | `current_mfa_outcome` | `in` | - | Abnormal multi-factor outcome on the anchor; always required. |

### Expected limitations

- A user with a failing authenticator app, a clock-drifted token, or a newly enrolled device reproduces this sequence.
- The rule reports a sequence anomaly. It is not a finding that a multi-factor control was defeated.

## PAD-PS-001 -- Password-spraying indicator

One source attempting a small number of authentications against many distinct accounts, with a high failure share. The low attempts-per-account ceiling is what separates spraying from single-account brute force.

| Property | Value |
|--------|-------|
| Rule version | `1.0.0` |
| Family | `spraying` |
| Attack category | `password_spraying` |
| Default severity | `high` |
| Correlation group | `source_fanout` |
| Evaluation scope | `anchor_event` |
| Privacy class | `non_sensitive` |
| Deprecated | no |

### Required features

- `source_unique_user_count__{cardinality_window}`
- `source_failure_count__{window}`
- `source_failure_rate__{window}`
- `source_attempt_count__{window}`

### Optional features

- `source_mean_interarrival_seconds__{window}`

### Thresholds

| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |
|--------|-------|-------|-------|-------|-------|-------|
| `window` | `window` | `1h` | - | - | duration | Window for the source's counts and rate. |
| `cardinality_window` | `window` | `1h` | - | - | duration | Window for the source's distinct-user count. |
| `min_unique_users` | `int` | `15` | 2.0 | - | count | Distinct accounts the source touched. |
| `min_source_failures` | `int` | `15` | 1.0 | - | count | Failures from this source. |
| `min_source_failure_rate` | `float` | `0.7` | 0.0 | 1.0 | ratio | Share of this source's attempts that failed. |
| `max_attempts_per_user` | `float` | `3.0` | 1.0 | - | ratio | Ceiling on attempts per targeted account; the low-and-slow guard. |
| `min_mean_interarrival_seconds` | `float` | `0.0` | 0.0 | 86400.0 | seconds | Optional cadence floor; zero disables the component. |

### Minimum history

Requires a non-null value for:

- `source_failure_rate__{window}`

The source failure rate is null when the window holds no attempts from this source.

### Evidence

| Code | Feature | Comparator | Unit | Meaning |
|--------|-------|-------|-------|-------|
| `PS_SOURCE_USER_FANOUT` | `source_unique_user_count__{cardinality_window}` | `gte` | count | Account fan-out from a single source. |
| `PS_SOURCE_FAILURE_COUNT` | `source_failure_count__{window}` | `gte` | count | Failure volume from the source. |
| `PS_SOURCE_FAILURE_RATE` | `source_failure_rate__{window}` | `gte` | ratio | Share of the source's attempts that failed. |
| `PS_ATTEMPTS_PER_USER` | `source_attempt_count__{window}` | `lte` | ratio | Derived attempts-per-account ratio; the spraying discriminator. |
| `PS_SOURCE_CADENCE` | `source_mean_interarrival_seconds__{window}` | `gte` | seconds | Optional cadence evidence; contributes signal, never required. |

### Expected limitations

- A shared corporate egress address legitimately serves many accounts and can exceed the fan-out threshold during an outage.
- A directory-service misconfiguration can fail many accounts at once without any attacker.

## Known limitations

- Rules consume Phase 3 point-in-time feature snapshots. They never read ground-truth labels, split assignments, campaign metadata, or the canonical event table.
- Signal strength is a bounded ordinal magnitude, not a statistical probability.
- Evidence records what was observed and what it is consistent with. It is not causal proof.
- Synthetic data exercises these rules but does not demonstrate real-world detection effectiveness.
