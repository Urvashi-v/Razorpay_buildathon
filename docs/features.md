# Feature dictionary

**Generated from code. Do not edit by hand.**

Regenerate with:

```bash
rto-sentinel features docs
```

Every feature below is declared in `src/rto_sentinel/features/` as a
`FeatureSpec`, which carries its own source columns, lookback window, observation
point and availability. The pipeline refuses to emit any feature whose
availability is not `at_order_time`, and
`tests/leakage/test_feature_leakage.py` checks the declarations against the data
itself - a feature that claims to be outcome-independent and behaves otherwise
fails a test.

## How to read the observation point

| Value | Meaning | Knowable at checkout? |
|---|---|---|
| `order_payload` | Comes straight off the order being scored | Yes, trivially |
| `customer_record` | Fixed when the account was created | Yes |
| `prior_orders_placed` | Over orders **placed** earlier | Yes - the merchant saw them |
| `prior_orders_resolved` | Over orders that **came back** earlier | Only outcomes that arrived |
| `population_resolved` | Aggregated across all customers' resolved orders | Yes, same caveat |
| `post_order` | Not knowable at scoring time | **No** - refused by the pipeline |

The distinction between `prior_orders_placed` and `prior_orders_resolved` is the
one that matters. "Orders placed in the last 30 days" is knowable instantly.
"Orders returned in the last 30 days" is knowable only for orders that had
actually come back - an order placed on day 40 that returns on day 47 is invisible
to an order placed on day 42. Confusing the two is the most common leak in this
class of problem, so the specs record which clock each window runs on and a
validator refuses an outcome window keyed on placement time.

**Feature version:** `1.0.0`  
**Feature-set fingerprint:** `798aef57ad3cefe94cfeeb9555287fefda16e0c502f236f7e226590923531dfb`  
**Total features:** 54 across 6 families

## `customer_history` — 13 features

| Feature | Type | Observation point | Lookback | Available | Description |
|---|---|---|---|---|---|
| `cust_account_age_days` | float | `customer_record` | — | yes | Days between the customer's signup and this order. |
| `cust_prior_order_count` | int | `prior_orders_placed` | all history by placed | yes | Orders this customer placed before this one. |
| `cust_prior_cod_count` | int | `prior_orders_placed` | all history by placed | yes | Prior orders paid cash on delivery. |
| `cust_prepaid_share` | float | `prior_orders_placed` | all history by placed | yes | Share of prior orders that were prepaid. |
| `cust_prior_resolved_count` | int | `prior_orders_resolved` | all history by resolved | yes | Prior orders whose delivery outcome was known by this instant. |
| `cust_prior_rto_count` | int | `prior_orders_resolved` | all history by resolved | yes | Prior orders that had already returned by this instant. |
| `cust_prior_rto_rate` | float | `prior_orders_resolved` | all history by resolved | yes | Raw share of resolved prior orders that returned. |
| `cust_prior_rto_rate_smoothed` | float | `prior_orders_resolved` | all history by resolved | yes | Return rate shrunk toward the population base rate with a Beta(0.4, 1.6) prior. |
| `cust_days_since_last_order` | float | `prior_orders_placed` | all history by placed | yes | Days since this customer last placed an order. |
| `cust_prior_value_mean` | float | `prior_orders_placed` | all history by placed | yes | Mean value of this customer's prior orders. |
| `cust_value_vs_prior_mean` | float | `prior_orders_placed` | all history by placed | yes | This order's value divided by the customer's prior mean. |
| `cust_mean_resolution_days` | float | `prior_orders_resolved` | all history by resolved | yes | Mean days from order to outcome across resolved prior orders. |
| `cust_is_new` | bool | `prior_orders_placed` | all history by placed | yes | True when this is the customer's first order. |

<details><summary>Source columns and risk notes</summary>

**`cust_account_age_days`**

- Source columns: `signup_at`, `ordered_at`
- Risk: Low risk. Account age is a legitimate tenure signal and is known at checkout. Correlated with order count, so the two are not independent evidence.

**`cust_prior_order_count`**

- Source columns: `customer_hash`, `ordered_at`
- Risk: Placed-clock, so knowable instantly. Zero for a first-time customer - a fact, not a missing value.

**`cust_prior_cod_count`**

- Source columns: `customer_hash`, `ordered_at`, `is_cod`
- Risk: Payment method is chosen at checkout, so this is placed-clock.

**`cust_prepaid_share`**

- Source columns: `customer_hash`, `ordered_at`, `is_cod`
- Expected null share: ~30%
- Risk: A customer who usually prepays and suddenly picks COD is a documented pattern. NaN for first-time customers.

**`cust_prior_resolved_count`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`
- Risk: The denominator behind every rate below. Emitted so the model can tell 'zero returns from one order' apart from 'zero returns from twenty' - those are very different pieces of evidence.

**`cust_prior_rto_count`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: increasing
- Risk: RESOLVED-clock. An order placed earlier that has not come back yet contributes nothing - that is the whole point of the as-of join.

**`cust_prior_rto_rate`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: increasing
- Expected null share: ~40%
- Risk: Strongest honest signal in the problem. NaN when nothing has resolved yet - never 0.0, which would claim a clean record the merchant has no basis for.

**`cust_prior_rto_rate_smoothed`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: increasing
- Risk: One return out of one order is not a 100% returner. Shrinkage stops the model treating a single unlucky delivery as a verdict on a person - which matters because that verdict costs them friction.

**`cust_days_since_last_order`**

- Source columns: `customer_hash`, `ordered_at`
- Expected null share: ~30%
- Risk: Placed-clock: the merchant knows when someone last ordered even if that order has not been delivered yet. NaN for first-timers.

**`cust_prior_value_mean`**

- Source columns: `customer_hash`, `ordered_at`, `order_value_inr`
- Expected null share: ~30%
- Risk: Order value is known at checkout, so placed-clock is correct.

**`cust_value_vs_prior_mean`**

- Source columns: `customer_hash`, `ordered_at`, `order_value_inr`
- Expected null share: ~30%
- Risk: An order far above someone's usual basket is a documented risk pattern. Ratio rather than z-score: most customers have too few orders for a standard deviation to mean anything.

**`cust_mean_resolution_days`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`, `maturity_days`
- Expected null share: ~40%
- Risk: SUBTLE. Resolution time is outcome-correlated - returns take longer than deliveries - so this is a partial proxy for past outcomes. That is legitimate for PRIOR orders whose outcome is genuinely known, and would be a severe leak for the current one. Resolved-clock, strictly.

**`cust_is_new`**

- Source columns: `customer_hash`, `ordered_at`
- Risk: The cold-start cohort. Reported separately in every evaluation, because a model that simply learns 'new equals risky' has found a population, not a behaviour.

</details>

## `temporal` — 9 features

| Feature | Type | Observation point | Lookback | Available | Description |
|---|---|---|---|---|---|
| `temporal_orders_last_7d` | int | `prior_orders_placed` | 7d by placed | yes | Orders this customer placed in the 7 days before this one. |
| `temporal_orders_last_30d` | int | `prior_orders_placed` | 30d by placed | yes | Orders this customer placed in the 30 days before this one. |
| `temporal_orders_last_90d` | int | `prior_orders_placed` | 90d by placed | yes | Orders this customer placed in the 90 days before this one. |
| `temporal_rto_count_last_30d` | int | `prior_orders_resolved` | 30d by resolved | yes | Orders that returned in the 30 days before this one was placed. |
| `temporal_rto_rate_last_30d` | float | `prior_orders_resolved` | 30d by resolved | yes | Share of orders resolving in the last 30 days that returned. |
| `temporal_rto_count_last_90d` | int | `prior_orders_resolved` | 90d by resolved | yes | Orders that returned in the 90 days before this one was placed. |
| `temporal_rto_rate_last_90d` | float | `prior_orders_resolved` | 90d by resolved | yes | Share of orders resolving in the last 90 days that returned. |
| `temporal_days_since_last_rto` | float | `prior_orders_resolved` | all history by resolved | yes | Days since this customer's most recent known return. |
| `temporal_order_burst` | float | `prior_orders_placed` | 90d by placed | yes | Orders in the last 7 days relative to the customer's 90-day daily rate. |

<details><summary>Source columns and risk notes</summary>

**`temporal_orders_last_7d`**

- Source columns: `customer_hash`, `ordered_at`
- Risk: Placed-clock, so knowable instantly. Zero is a fact here, not a missing value: the customer genuinely placed no orders.

**`temporal_orders_last_30d`**

- Source columns: `customer_hash`, `ordered_at`
- Risk: Placed-clock, so knowable instantly. Zero is a fact here, not a missing value: the customer genuinely placed no orders.

**`temporal_orders_last_90d`**

- Source columns: `customer_hash`, `ordered_at`
- Risk: Placed-clock, so knowable instantly. Zero is a fact here, not a missing value: the customer genuinely placed no orders.

**`temporal_rto_count_last_30d`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: increasing
- Risk: RESOLVED-clock. An order placed inside the window that has not come back yet contributes nothing, because on this date nobody knows how it ended.

**`temporal_rto_rate_last_30d`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: increasing
- Expected null share: ~45%
- Risk: Mostly NaN, and correctly so - most customers have nothing resolving in any given window. A recency-weighted rate is fairer than an all-time one: a bad month two years ago should decay.

**`temporal_rto_count_last_90d`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: increasing
- Risk: RESOLVED-clock. An order placed inside the window that has not come back yet contributes nothing, because on this date nobody knows how it ended.

**`temporal_rto_rate_last_90d`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: increasing
- Expected null share: ~45%
- Risk: Mostly NaN, and correctly so - most customers have nothing resolving in any given window. A recency-weighted rate is fairer than an all-time one: a bad month two years ago should decay.

**`temporal_days_since_last_rto`**

- Source columns: `customer_hash`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: decreasing
- Expected null share: ~78%
- Risk: Recency of the last return, which decays naturally. NaN for the large majority who have never had one - and NaN is right, because 'never' is not a very large number of days, it is a different state.

**`temporal_order_burst`**

- Source columns: `customer_hash`, `ordered_at`
- Expected null share: ~36%
- Risk: HIGHEST FAIRNESS RISK IN THIS FAMILY. A legitimate seasonal shopper and an impulsive one look identical over a short window. Kept only if the ablation study shows it pays for itself in rupees, not in AUC.

</details>

## `order_shape` — 9 features

| Feature | Type | Observation point | Lookback | Available | Description |
|---|---|---|---|---|---|
| `order_value_inr` | float | `order_payload` | — | yes | Net order value in rupees, after discount. |
| `order_log_value` | float | `order_payload` | — | yes | Natural log of (1 + order value). Compresses a long right tail. |
| `order_is_cod` | bool | `order_payload` | — | yes | True when the order is cash on delivery. |
| `order_discount_depth` | float | `order_payload` | — | yes | Discount as a fraction of gross order value. |
| `order_discount_inr` | float | `order_payload` | — | yes | Absolute discount in rupees. |
| `order_item_count` | int | `order_payload` | — | yes | Total units in the basket. |
| `order_value_per_item` | float | `order_payload` | — | yes | Net order value divided by unit count. |
| `order_category` | category | `order_payload` | — | yes | Product category of the order. |
| `order_cart_edited` | bool | `order_payload` | — | yes | True when the basket was modified before checkout. |

<details><summary>Source columns and risk notes</summary>

**`order_value_inr`**

- Source columns: `order_value_inr`
- Risk: Low leakage risk. Reported by order-value quartile in the fairness audit, because friction on a small order costs a customer proportionally more than on a large one.

**`order_log_value`**

- Source columns: `order_value_inr`
- Risk: Low risk. A monotone transform of order value; both are kept because trees split on raw scale while the log is easier to read in SHAP.

**`order_is_cod`**

- Source columns: `is_cod`
- Risk: The single strongest split in the problem - 26% against under 2%. Not a leak: payment method is chosen at checkout, before scoring.

**`order_discount_depth`**

- Source columns: `discount_depth`
- Monotonic constraint: increasing
- Risk: Predictive, and partly the merchant's own doing. Surfaced as a merchant insight in the evaluation, not only as a customer-level penalty. See the module docstring.

**`order_discount_inr`**

- Source columns: `discount_inr`
- Risk: Low risk. Kept alongside depth because a large absolute discount on an expensive item is a different situation from a deep one on a cheap item.

**`order_item_count`**

- Source columns: `item_count`
- Risk: Low risk. Multi-item baskets fail delivery differently from single ones.

**`order_value_per_item`**

- Source columns: `order_value_inr`, `item_count`
- Risk: Low risk. Separates 'one expensive thing' from 'many cheap things', which behave differently at the doorstep.

**`order_category`**

- Source columns: `category`
- Risk: Fashion returns more than electronics, which is a property of the goods rather than of the customer. Low fairness risk, and useful for merchant reporting.

**`order_cart_edited`**

- Source columns: `cart_edited`
- Risk: Weak signal. Included because hesitation before purchase is a plausible intent proxy; expected to earn little and be a candidate for removal in the ablation study.

</details>

## `address_quality` — 9 features

| Feature | Type | Observation point | Lookback | Available | Description |
|---|---|---|---|---|---|
| `addr_token_count` | int | `order_payload` | — | yes | Whitespace- and comma-separated tokens in the address line. |
| `addr_has_house_number` | bool | `order_payload` | — | yes | True when a house or building number is present. |
| `addr_has_floor_number` | bool | `order_payload` | — | yes | True when a floor or flat is present. |
| `addr_has_landmark` | bool | `order_payload` | — | yes | True when a landmark reference is present. |
| `addr_pincode_city_consistent` | bool | `order_payload` | — | yes | True when the typed city matches the city the pincode belongs to. |
| `addr_allcaps_ratio` | float | `order_payload` | — | yes | Share of alphabetic tokens typed entirely in capitals. |
| `addr_gibberish_ratio` | float | `order_payload` | — | yes | Share of tokens that are keyboard mash or autocomplete garbage. |
| `addr_completeness_score` | float | `order_payload` | — | yes | Weighted share of present address components: house number 0.40, pincode/city consistency 0.30, landmark 0.20, floor 0.10. |
| `addr_is_new_for_customer` | bool | `prior_orders_placed` | all history by placed | yes | True when this customer has not ordered to this address before. |

<details><summary>Source columns and risk notes</summary>

**`addr_token_count`**

- Source columns: `addr_token_count`
- Risk: A very short address is usually an incomplete one. Measures length, not quality of language.

**`addr_has_house_number`**

- Source columns: `addr_has_house_number`
- Risk: The component whose absence most reliably fails a delivery. Fair: a courier genuinely cannot find an unnumbered building.

**`addr_has_floor_number`**

- Source columns: `addr_has_floor_number`
- Risk: Weaker than a house number and irrelevant for standalone houses, so it carries the smallest weight in the composite score.

**`addr_has_landmark`**

- Source columns: `addr_has_landmark`
- Risk: How deliveries actually get completed across much of India. Its presence is a positive signal, and its absence is weak evidence rather than a red flag.

**`addr_pincode_city_consistent`**

- Source columns: `addr_pincode_city_consistent`
- Risk: One of the most common structural defects in Indian address data, and causally linked to delivery failure - a parcel addressed to the wrong city struggles to arrive. Checkable against a lookup table, so it is a fact rather than an inference about the person.

**`addr_allcaps_ratio`**

- Source columns: `addr_allcaps_ratio`
- Risk: WATCH THIS ONE. All-caps entry is common on older keyboards and among less confident typists, so it risks proxying for something other than deliverability. Kept because it is cheap to audit, and it is the first candidate for removal if the fairness audit trips.

**`addr_gibberish_ratio`**

- Source columns: `addr_gibberish_ratio`
- Risk: Matches a fixed list of consonant runs, not a language model. Cannot flag an unfamiliar-looking real word, which is the failure mode a learned gibberish detector would have.

**`addr_completeness_score`**

- Source columns: `addr_has_house_number`, `addr_pincode_city_consistent`, `addr_has_landmark`, `addr_has_floor_number`
- Monotonic constraint: decreasing
- Risk: A hand-set composite, not a learned one. Weights are a judgement, stated here so they can be argued with; the components are also emitted individually so the model is not forced to accept them.

**`addr_is_new_for_customer`**

- Source columns: `customer_hash`, `address_fingerprint`, `ordered_at`
- Risk: Placed-clock: the merchant knows an address was used before regardless of how those orders ended. A first delivery to a new address is genuinely riskier - but so is every order from a new customer, so this must not simply re-encode the cold-start cohort. Reported alongside it in the evaluation.

</details>

## `session_intent` — 9 features

| Feature | Type | Observation point | Lookback | Available | Description |
|---|---|---|---|---|---|
| `session_product_page_seconds` | float | `order_payload` | — | yes | Seconds spent on the product page before adding to cart. |
| `session_sessions_before_purchase` | int | `order_payload` | — | yes | Distinct sessions before this order was placed. |
| `session_time_to_checkout_seconds` | float | `order_payload` | — | yes | Seconds from cart creation to order placement. |
| `session_device_class` | category | `order_payload` | — | yes | Device the order was placed from. |
| `session_hour_of_day` | int | `order_payload` | — | yes | Hour the order was placed, 0-23, UTC. |
| `session_day_of_week` | int | `order_payload` | — | yes | Day of week, Monday=0. |
| `session_is_late_night` | bool | `order_payload` | — | yes | True when the order was placed between 23:00 and 04:59. |
| `session_is_sale_day` | bool | `order_payload` | — | yes | True when the order fell on a promotional day. |
| `session_cod_after_prepaid_failure` | bool | `order_payload` | — | yes | True when COD was chosen after a prepaid attempt failed. |

<details><summary>Source columns and risk notes</summary>

**`session_product_page_seconds`**

- Source columns: `product_page_seconds`
- Risk: Weak. A considered purchase and a distracted browser look the same. Low fairness risk - it describes a session, not a person.

**`session_sessions_before_purchase`**

- Source columns: `sessions_before_purchase`
- Risk: Weak. More sessions usually means more deliberation, but it also means a slower connection or a shared device.

**`session_time_to_checkout_seconds`**

- Source columns: `time_to_checkout_seconds`
- Risk: The clearest impulse proxy in this family. Still weak on its own.

**`session_device_class`**

- Source columns: `device_class`
- Risk: WATCH THIS ONE. Device class correlates with income in India, so it is a plausible proxy for something this project has no business predicting on. Kept for now because it is a documented delivery-context signal, and flagged as a removal candidate if the fairness audit trips.

**`session_hour_of_day`**

- Source columns: `hour_of_day`
- Risk: Low fairness risk. Encoded as an integer rather than cyclically so it reads directly in a reason code - see the module docstring.

**`session_day_of_week`**

- Source columns: `day_of_week`
- Risk: Low risk. Weekend orders deliver differently from weekday ones.

**`session_is_late_night`**

- Source columns: `is_late_night`
- Risk: The documented low-intent impulse window. A blunt encoding of hour_of_day, kept because it is what the reason code will say.

**`session_is_sale_day`**

- Source columns: `is_sale_day`
- Risk: A merchant-controlled variable, like discount depth. Surfaced as a merchant insight, not only as a customer-level signal.

**`session_cod_after_prepaid_failure`**

- Source columns: `cod_after_prepaid_failure`
- Risk: Known at checkout - the failure happened seconds earlier in the same session, so this is payload, not history. Rare, which limits how much the model can learn from it.

</details>

## `geography_route` — 5 features

| Feature | Type | Observation point | Lookback | Available | Description |
|---|---|---|---|---|---|
| `geo_pincode_tier` | category | `order_payload` | — | yes | Pincode tier: tier_1, tier_2 or tier_3. |
| `geo_pincode_rto_rate_smoothed` | float | `population_resolved` | all history by resolved | yes | Return rate for this pincode over orders resolved before this one, shrunk toward the population rate with strength 50. NaN below 30 resolved orders. |
| `geo_pincode_resolved_count` | int | `population_resolved` | all history by resolved | yes | Orders in this pincode that had resolved before this one. |
| `geo_courier_partner` | category | `order_payload` | — | yes | Courier assigned to this order. |
| `geo_courier_rto_rate_smoothed` | float | `population_resolved` | all history by resolved | yes | Return rate for this courier over orders resolved before this one, shrunk with strength 50. |

<details><summary>Source columns and risk notes</summary>

**`geo_pincode_tier`**

- Source columns: `pincode_tier`
- Risk: A coarse, three-level geography signal - deliberately coarser than a pincode. It is also the primary axis of the fairness audit, so its effect is measured rather than assumed.

**`geo_pincode_rto_rate_smoothed`**

- Source columns: `pincode`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: increasing
- Expected null share: ~92%
- Risk: THE HIGHEST-RISK FEATURE IN THE PROJECT. A raw pincode rate is an income and region proxy. Three guards: Bayesian shrinkage, a minimum support threshold below which it is NaN, and as-of computation so it cannot read the future. Never permitted as a top-3 SHAP feature without written justification in REPORT.md.

**`geo_pincode_resolved_count`**

- Source columns: `pincode`, `ordered_at`, `resolved_at`
- Risk: The evidence count behind the rate. Emitted so the model can distinguish a well-observed pincode from a thin one, rather than treating a shrunk estimate as equally reliable everywhere.

**`geo_courier_partner`**

- Source columns: `courier_partner`
- Risk: Low fairness risk - this is a merchant logistics choice, not a customer attribute. Assigned at order time in this dataset; a production system where the courier is picked after scoring would have to drop it.

**`geo_courier_rto_rate_smoothed`**

- Source columns: `courier_partner`, `ordered_at`, `resolved_at`, `is_rto`
- Monotonic constraint: increasing
- Expected null share: ~5%
- Risk: Lane performance is a property of the carrier, not the customer, so the fairness concern is much weaker here than for pincode. Still as-of computed: a global courier rate would leak test-window outcomes into training.

</details>

## What is deliberately absent

Four groups of features are refused outright, with the reasons recorded in
`config/features.yaml`:

- **Name-derived features of any kind.** Religion, caste and region inference
  from names is a live harm in Indian systems. Names are hashed for identity
  only, never featurised.
- **Raw pincode as a categorical.** With enough trees this becomes a redlining
  machine. Only smoothed, shrunk aggregates with a minimum support threshold.
- **Gender, age, or anything inferable from them.** No lift worth the exposure.
  Note that `cust_account_age_days` is account *tenure*, not customer age; it is
  listed as an explicit, justified exception in the config.
- **Cross-merchant behaviour.** A consent question this project cannot resolve
  responsibly.

Matching is by whole token, not substring. Substring matching was tried first and
was unusable - the pattern `age` matched `cust_account_age_days` and
`session_product_page_seconds` - and a check that cries wolf gets switched off,
which is worse than no check at all.
