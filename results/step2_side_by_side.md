# Step 2 (c): Legacy vs. Calibrated Side-by-Side — Apr-Jun 2016

**Identical batches**: the calibrated re-run (`scripts/step2_calibrated_rerun.py`) uses the exact same batch-drawing seeds as the legacy run (`scripts/step1_validation.py`), so every comparison below is on byte-identical data — only the decision logic (single threshold vs. two-gate) differs.

**Locked decisions applied**: alpha=0.05, multiple_testing=Holm, KS D floor=0.05 (dataset-independent default), PSI floor=0.2.

## Precision/recall at matched threshold (D_gt = configured floor = 0.05)

| Ref size | Batch size | Mode | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 5000 | 1000 | legacy | 48 | 18 | 0 | 102 | 0.727 | 1.000 | 0.842 |
| 5000 | 1000 | calibrated | 47 | 6 | 1 | 114 | 0.887 | 0.979 | 0.931 |
| 5000 | 3000 | legacy | 48 | 37 | 0 | 83 | 0.565 | 1.000 | 0.722 |
| 5000 | 3000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 5000 | 5000 | legacy | 48 | 46 | 0 | 74 | 0.511 | 1.000 | 0.676 |
| 5000 | 5000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 5000 | 10000 | legacy | 48 | 65 | 0 | 55 | 0.425 | 1.000 | 0.596 |
| 5000 | 10000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 5000 | 15000 | legacy | 48 | 62 | 0 | 58 | 0.436 | 1.000 | 0.608 |
| 5000 | 15000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 5000 | 20000 | legacy | 48 | 66 | 0 | 54 | 0.421 | 1.000 | 0.593 |
| 5000 | 20000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 5000 | 50000 | legacy | 48 | 73 | 0 | 47 | 0.397 | 1.000 | 0.568 |
| 5000 | 50000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 50000 | 1000 | legacy | 48 | 22 | 0 | 98 | 0.686 | 1.000 | 0.814 |
| 50000 | 1000 | calibrated | 48 | 7 | 0 | 113 | 0.873 | 1.000 | 0.932 |
| 50000 | 3000 | legacy | 48 | 75 | 0 | 45 | 0.390 | 1.000 | 0.561 |
| 50000 | 3000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 50000 | 5000 | legacy | 48 | 83 | 0 | 37 | 0.366 | 1.000 | 0.536 |
| 50000 | 5000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 50000 | 10000 | legacy | 48 | 92 | 0 | 28 | 0.343 | 1.000 | 0.511 |
| 50000 | 10000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 50000 | 15000 | legacy | 48 | 96 | 0 | 24 | 0.333 | 1.000 | 0.500 |
| 50000 | 15000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 50000 | 20000 | legacy | 48 | 96 | 0 | 24 | 0.333 | 1.000 | 0.500 |
| 50000 | 20000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |
| 50000 | 50000 | legacy | 48 | 96 | 0 | 24 | 0.333 | 1.000 | 0.500 |
| 50000 | 50000 | calibrated | 48 | 0 | 0 | 120 | 1.000 | 1.000 | 1.000 |

## A/A system false-alarm rate per batch size, legacy vs. calibrated

Alpha-predicted (7 uncorrected tests): `1-(1-0.05)^7` = 0.302. Calibrated applies Holm correction across the 7 features, so its system rate should track much closer to 0.05 itself.

| Ref size | Batch size | Legacy system rate | Calibrated system rate |
|---|---|---|---|
| 5000 | 1000 | 0.120 | 0.010 |
| 5000 | 3000 | 0.180 | 0.000 |
| 5000 | 5000 | 0.270 | 0.000 |
| 5000 | 10000 | 0.210 | 0.000 |
| 5000 | 15000 | 0.140 | 0.000 |
| 5000 | 20000 | 0.020 | 0.000 |
| 50000 | 1000 | 0.170 | 0.010 |
| 50000 | 3000 | 0.160 | 0.000 |
| 50000 | 5000 | 0.200 | 0.000 |
| 50000 | 10000 | 0.090 | 0.000 |
| 50000 | 15000 | 0.020 | 0.000 |
| 50000 | 20000 | 0.020 | 0.000 |

## Floor sensitivity (calibrated mode, biggest sweep size=50000), recomputed from stored effect_size/significant -- no new HTTP calls needed

| Ref size | Floor | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 5000 | 0.015 | 112 | 24 | 32 | 0 | 0.824 | 0.778 | 0.800 |
| 5000 | 0.02 | 80 | 56 | 16 | 16 | 0.588 | 0.833 | 0.690 |
| 5000 | 0.03 | 48 | 39 | 0 | 81 | 0.552 | 1.000 | 0.711 |
| 5000 | 0.05 | 48 | 8 | 0 | 112 | 0.857 | 1.000 | 0.923 |
| 50000 | 0.015 | 142 | 24 | 2 | 0 | 0.855 | 0.986 | 0.916 |
| 50000 | 0.02 | 82 | 57 | 14 | 15 | 0.590 | 0.854 | 0.698 |
| 50000 | 0.03 | 48 | 30 | 0 | 90 | 0.615 | 1.000 | 0.762 |
| 50000 | 0.05 | 48 | 8 | 0 | 112 | 0.857 | 1.000 | 0.923 |

## Explicit check: significant-but-not-material at the 0.05 default

Rates computed by POOLING all 24 batches at the largest sweep size (50,000) per reference size -- not a single representative batch -- for real statistical weight.

| Feature | Population D | m=5000: sig / mat / detected rate | m=50000: sig / mat / detected rate |
|---|---|---|---|
| pickup_longitude | 0.0219 | 0.83 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 |
| pickup_latitude | 0.0189 | 0.67 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 |
| dropoff_longitude | 0.0206 | 0.50 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 |
| dropoff_latitude | 0.0195 | 0.67 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 |
| trip_duration | 0.0924 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |

At m=50,000, all four coordinate features (population D 0.019-0.022) are **significant in 100% of batches but material in 0%** -- exactly the documented significant-but-not-material pattern, confirmed on the full pooled sample, not an anecdote. `trip_duration` (D=0.092, well above the 0.05 floor) is significant AND material in 100% of batches at both reference sizes, as expected for a real, sizeable effect. At m=5,000 the coordinate features' significance rate is lower (50-83%) rather than 100% -- consistent with Step 1's own finding that the smaller reference has less power to resolve effects this close to its own detection floor -- but material rate stays at 0% regardless, so the materiality gate's behavior doesn't depend on reference size the way significance does.


## Data-collapse check (population D, corrected per 2026-09-30 instruction)

x = `sqrt(n*m/(n+m)) * D_population` (per-MONTH population D, the true generating-distribution distance for that batch's month -- never the per-batch observed KS statistic, since detection is a deterministic function of the observed statistic and that would collapse onto a step function by construction). y = empirical detection rate. Computed on LEGACY's raw `drift_detected` (the pure significance decision, p<alpha with no materiality gate) -- this is what asymptotic KS theory actually makes a claim about; calibrated mode's two-gate `drift_detected` deliberately adds a human materiality choice on top and would confound the check. Continuous features only (PSI is a different statistic/theory).

**210 (feature, n, m, month) cells computed.**

| Feature | n | m | month | D_pop | x=sqrt(nm/(n+m))*D_pop | detection rate | n_trials |
|---|---|---|---|---|---|---|---|
| dropoff_longitude | 1000 | 5000 | 04 | 0.0130 | 0.377 | 0.000 | 8 |
| pickup_longitude | 1000 | 5000 | 04 | 0.0138 | 0.398 | 0.125 | 8 |
| dropoff_longitude | 1000 | 50000 | 04 | 0.0130 | 0.408 | 0.125 | 8 |
| dropoff_latitude | 1000 | 5000 | 04 | 0.0146 | 0.420 | 0.125 | 8 |
| pickup_longitude | 1000 | 50000 | 04 | 0.0138 | 0.432 | 0.000 | 8 |
| pickup_latitude | 1000 | 5000 | 04 | 0.0151 | 0.437 | 0.000 | 8 |
| dropoff_latitude | 1000 | 50000 | 04 | 0.0146 | 0.456 | 0.250 | 8 |
| pickup_latitude | 1000 | 50000 | 04 | 0.0151 | 0.474 | 0.125 | 8 |
| dropoff_longitude | 3000 | 5000 | 04 | 0.0130 | 0.565 | 0.000 | 8 |
| pickup_longitude | 3000 | 5000 | 04 | 0.0138 | 0.597 | 0.250 | 8 |
| dropoff_longitude | 1000 | 5000 | 05 | 0.0214 | 0.617 | 0.125 | 8 |
| dropoff_latitude | 3000 | 5000 | 04 | 0.0146 | 0.630 | 0.500 | 8 |
| pickup_latitude | 1000 | 5000 | 05 | 0.0226 | 0.652 | 0.125 | 8 |
| dropoff_longitude | 5000 | 5000 | 04 | 0.0130 | 0.652 | 0.000 | 8 |
| pickup_longitude | 1000 | 5000 | 05 | 0.0226 | 0.653 | 0.500 | 8 |
| pickup_latitude | 3000 | 5000 | 04 | 0.0151 | 0.656 | 0.250 | 8 |
| dropoff_latitude | 1000 | 5000 | 05 | 0.0228 | 0.658 | 0.000 | 8 |
| dropoff_longitude | 1000 | 50000 | 05 | 0.0214 | 0.669 | 0.125 | 8 |
| pickup_latitude | 1000 | 5000 | 06 | 0.0232 | 0.669 | 0.375 | 8 |
| pickup_longitude | 5000 | 5000 | 04 | 0.0138 | 0.689 | 0.500 | 8 |
| dropoff_longitude | 3000 | 50000 | 04 | 0.0130 | 0.694 | 0.500 | 8 |
| pickup_latitude | 1000 | 50000 | 05 | 0.0226 | 0.707 | 0.250 | 8 |
| dropoff_latitude | 1000 | 5000 | 06 | 0.0245 | 0.708 | 0.375 | 8 |
| pickup_longitude | 1000 | 50000 | 05 | 0.0226 | 0.708 | 0.125 | 8 |
| dropoff_latitude | 1000 | 50000 | 05 | 0.0228 | 0.714 | 0.250 | 8 |
| pickup_latitude | 1000 | 50000 | 06 | 0.0232 | 0.726 | 0.375 | 8 |
| dropoff_latitude | 5000 | 5000 | 04 | 0.0146 | 0.728 | 0.000 | 8 |
| dropoff_longitude | 1000 | 5000 | 06 | 0.0254 | 0.732 | 0.000 | 8 |
| pickup_longitude | 3000 | 50000 | 04 | 0.0138 | 0.733 | 0.500 | 8 |
| dropoff_longitude | 10000 | 5000 | 04 | 0.0130 | 0.753 | 0.000 | 8 |
| pickup_latitude | 5000 | 5000 | 04 | 0.0151 | 0.757 | 0.500 | 8 |
| dropoff_latitude | 1000 | 50000 | 06 | 0.0245 | 0.768 | 0.375 | 8 |
| dropoff_latitude | 3000 | 50000 | 04 | 0.0146 | 0.774 | 0.750 | 8 |
| pickup_longitude | 1000 | 5000 | 06 | 0.0270 | 0.778 | 0.500 | 8 |
| dropoff_longitude | 1000 | 50000 | 06 | 0.0254 | 0.794 | 0.250 | 8 |
| pickup_longitude | 10000 | 5000 | 04 | 0.0138 | 0.796 | 1.000 | 8 |
| dropoff_longitude | 15000 | 5000 | 04 | 0.0130 | 0.799 | 0.000 | 8 |
| pickup_latitude | 3000 | 50000 | 04 | 0.0151 | 0.806 | 0.500 | 8 |
| dropoff_longitude | 20000 | 5000 | 04 | 0.0130 | 0.825 | 0.000 | 8 |
| dropoff_latitude | 10000 | 5000 | 04 | 0.0146 | 0.840 | 0.125 | 8 |
| pickup_longitude | 15000 | 5000 | 04 | 0.0138 | 0.844 | 0.750 | 8 |
| pickup_longitude | 1000 | 50000 | 06 | 0.0270 | 0.844 | 0.500 | 8 |
| pickup_longitude | 20000 | 5000 | 04 | 0.0138 | 0.872 | 0.750 | 8 |
| pickup_latitude | 10000 | 5000 | 04 | 0.0151 | 0.874 | 0.375 | 8 |
| dropoff_longitude | 50000 | 5000 | 04 | 0.0130 | 0.879 | 0.000 | 8 |
| dropoff_longitude | 5000 | 50000 | 04 | 0.0130 | 0.879 | 0.375 | 8 |
| dropoff_latitude | 15000 | 5000 | 04 | 0.0146 | 0.891 | 0.125 | 8 |
| dropoff_latitude | 20000 | 5000 | 04 | 0.0146 | 0.921 | 0.375 | 8 |
| dropoff_longitude | 3000 | 5000 | 05 | 0.0214 | 0.925 | 0.000 | 8 |
| pickup_latitude | 15000 | 5000 | 04 | 0.0151 | 0.927 | 0.500 | 8 |
| pickup_longitude | 50000 | 5000 | 04 | 0.0138 | 0.929 | 1.000 | 8 |
| pickup_longitude | 5000 | 50000 | 04 | 0.0138 | 0.929 | 0.750 | 8 |
| pickup_latitude | 20000 | 5000 | 04 | 0.0151 | 0.958 | 0.500 | 8 |
| pickup_latitude | 3000 | 5000 | 05 | 0.0226 | 0.978 | 0.750 | 8 |
| pickup_longitude | 3000 | 5000 | 05 | 0.0226 | 0.980 | 0.500 | 8 |
| dropoff_latitude | 50000 | 5000 | 04 | 0.0146 | 0.981 | 0.250 | 8 |
| dropoff_latitude | 5000 | 50000 | 04 | 0.0146 | 0.981 | 0.875 | 8 |
| dropoff_latitude | 3000 | 5000 | 05 | 0.0228 | 0.988 | 0.375 | 8 |
| pickup_latitude | 3000 | 5000 | 06 | 0.0232 | 1.004 | 0.500 | 8 |
| pickup_latitude | 50000 | 5000 | 04 | 0.0151 | 1.021 | 0.375 | 8 |
| pickup_latitude | 5000 | 50000 | 04 | 0.0151 | 1.021 | 0.750 | 8 |
| dropoff_latitude | 3000 | 5000 | 06 | 0.0245 | 1.061 | 0.375 | 8 |
| dropoff_longitude | 5000 | 5000 | 05 | 0.0214 | 1.068 | 0.000 | 8 |
| dropoff_longitude | 3000 | 5000 | 06 | 0.0254 | 1.098 | 0.250 | 8 |
| pickup_latitude | 5000 | 5000 | 05 | 0.0226 | 1.129 | 0.625 | 8 |
| pickup_longitude | 5000 | 5000 | 05 | 0.0226 | 1.131 | 0.750 | 8 |
| dropoff_longitude | 3000 | 50000 | 05 | 0.0214 | 1.137 | 1.000 | 8 |
| dropoff_latitude | 5000 | 5000 | 05 | 0.0228 | 1.140 | 0.625 | 8 |
| pickup_latitude | 5000 | 5000 | 06 | 0.0232 | 1.159 | 0.875 | 8 |
| pickup_longitude | 3000 | 5000 | 06 | 0.0270 | 1.168 | 0.875 | 8 |
| dropoff_longitude | 10000 | 50000 | 04 | 0.0130 | 1.191 | 0.625 | 8 |
| pickup_latitude | 3000 | 50000 | 05 | 0.0226 | 1.201 | 0.750 | 8 |
| pickup_longitude | 3000 | 50000 | 05 | 0.0226 | 1.204 | 0.875 | 8 |
| dropoff_latitude | 3000 | 50000 | 05 | 0.0228 | 1.213 | 1.000 | 8 |
| dropoff_latitude | 5000 | 5000 | 06 | 0.0245 | 1.226 | 0.750 | 8 |
| pickup_latitude | 3000 | 50000 | 06 | 0.0232 | 1.234 | 0.875 | 8 |
| dropoff_longitude | 10000 | 5000 | 05 | 0.0214 | 1.234 | 0.375 | 8 |
| pickup_longitude | 10000 | 50000 | 04 | 0.0138 | 1.258 | 1.000 | 8 |
| dropoff_longitude | 5000 | 5000 | 06 | 0.0254 | 1.268 | 0.125 | 8 |
| pickup_latitude | 10000 | 5000 | 05 | 0.0226 | 1.304 | 1.000 | 8 |
| dropoff_latitude | 3000 | 50000 | 06 | 0.0245 | 1.304 | 0.750 | 8 |
| pickup_longitude | 10000 | 5000 | 05 | 0.0226 | 1.306 | 1.000 | 8 |
| dropoff_longitude | 15000 | 5000 | 05 | 0.0214 | 1.309 | 0.125 | 8 |
| dropoff_latitude | 10000 | 5000 | 05 | 0.0228 | 1.317 | 0.750 | 8 |
| dropoff_latitude | 10000 | 50000 | 04 | 0.0146 | 1.329 | 0.875 | 8 |
| pickup_latitude | 10000 | 5000 | 06 | 0.0232 | 1.339 | 1.000 | 8 |
| pickup_longitude | 5000 | 5000 | 06 | 0.0270 | 1.348 | 1.000 | 8 |
| dropoff_longitude | 3000 | 50000 | 06 | 0.0254 | 1.349 | 0.875 | 8 |
| dropoff_longitude | 20000 | 5000 | 05 | 0.0214 | 1.352 | 0.000 | 8 |
| pickup_latitude | 10000 | 50000 | 04 | 0.0151 | 1.383 | 1.000 | 8 |
| pickup_latitude | 15000 | 5000 | 05 | 0.0226 | 1.383 | 1.000 | 8 |
| pickup_longitude | 15000 | 5000 | 05 | 0.0226 | 1.386 | 1.000 | 8 |
| dropoff_latitude | 15000 | 5000 | 05 | 0.0228 | 1.397 | 1.000 | 8 |
| dropoff_longitude | 15000 | 50000 | 04 | 0.0130 | 1.401 | 1.000 | 8 |
| dropoff_latitude | 10000 | 5000 | 06 | 0.0245 | 1.415 | 0.875 | 8 |
| pickup_latitude | 15000 | 5000 | 06 | 0.0232 | 1.420 | 1.000 | 8 |
| pickup_latitude | 20000 | 5000 | 05 | 0.0226 | 1.428 | 1.000 | 8 |
| pickup_longitude | 20000 | 5000 | 05 | 0.0226 | 1.431 | 1.000 | 8 |
| pickup_longitude | 3000 | 50000 | 06 | 0.0270 | 1.435 | 1.000 | 8 |
| dropoff_longitude | 50000 | 5000 | 05 | 0.0214 | 1.441 | 0.500 | 8 |
| dropoff_longitude | 5000 | 50000 | 05 | 0.0214 | 1.441 | 0.875 | 8 |
| dropoff_latitude | 20000 | 5000 | 05 | 0.0228 | 1.443 | 1.000 | 8 |
| dropoff_longitude | 10000 | 5000 | 06 | 0.0254 | 1.464 | 0.625 | 8 |
| pickup_latitude | 20000 | 5000 | 06 | 0.0232 | 1.467 | 1.000 | 8 |
| pickup_longitude | 15000 | 50000 | 04 | 0.0138 | 1.480 | 1.000 | 8 |
| dropoff_latitude | 15000 | 5000 | 06 | 0.0245 | 1.501 | 1.000 | 8 |
| pickup_latitude | 50000 | 5000 | 05 | 0.0226 | 1.522 | 1.000 | 8 |
| pickup_latitude | 5000 | 50000 | 05 | 0.0226 | 1.522 | 0.875 | 8 |
| pickup_longitude | 50000 | 5000 | 05 | 0.0226 | 1.525 | 1.000 | 8 |
| pickup_longitude | 5000 | 50000 | 05 | 0.0226 | 1.525 | 1.000 | 8 |
| dropoff_latitude | 50000 | 5000 | 05 | 0.0228 | 1.538 | 1.000 | 8 |
| dropoff_latitude | 5000 | 50000 | 05 | 0.0228 | 1.538 | 1.000 | 8 |
| dropoff_latitude | 20000 | 5000 | 06 | 0.0245 | 1.550 | 1.000 | 8 |
| dropoff_longitude | 15000 | 5000 | 06 | 0.0254 | 1.553 | 0.250 | 8 |
| pickup_longitude | 10000 | 5000 | 06 | 0.0270 | 1.557 | 1.000 | 8 |
| dropoff_longitude | 20000 | 50000 | 04 | 0.0130 | 1.559 | 1.000 | 8 |
| pickup_latitude | 50000 | 5000 | 06 | 0.0232 | 1.563 | 1.000 | 8 |
| pickup_latitude | 5000 | 50000 | 06 | 0.0232 | 1.563 | 0.875 | 8 |
| dropoff_latitude | 15000 | 50000 | 04 | 0.0146 | 1.564 | 1.000 | 8 |
| dropoff_longitude | 20000 | 5000 | 06 | 0.0254 | 1.604 | 0.625 | 8 |
| pickup_latitude | 15000 | 50000 | 04 | 0.0151 | 1.627 | 1.000 | 8 |
| pickup_longitude | 20000 | 50000 | 04 | 0.0138 | 1.647 | 1.000 | 8 |
| pickup_longitude | 15000 | 5000 | 06 | 0.0270 | 1.651 | 1.000 | 8 |
| dropoff_latitude | 50000 | 5000 | 06 | 0.0245 | 1.653 | 1.000 | 8 |
| dropoff_latitude | 5000 | 50000 | 06 | 0.0245 | 1.653 | 1.000 | 8 |
| pickup_longitude | 20000 | 5000 | 06 | 0.0270 | 1.706 | 1.000 | 8 |
| dropoff_longitude | 50000 | 5000 | 06 | 0.0254 | 1.710 | 1.000 | 8 |
| dropoff_longitude | 5000 | 50000 | 06 | 0.0254 | 1.710 | 1.000 | 8 |
| dropoff_latitude | 20000 | 50000 | 04 | 0.0146 | 1.740 | 1.000 | 8 |
| pickup_latitude | 20000 | 50000 | 04 | 0.0151 | 1.810 | 1.000 | 8 |
| pickup_longitude | 50000 | 5000 | 06 | 0.0270 | 1.818 | 1.000 | 8 |
| pickup_longitude | 5000 | 50000 | 06 | 0.0270 | 1.818 | 1.000 | 8 |
| dropoff_longitude | 10000 | 50000 | 05 | 0.0214 | 1.951 | 1.000 | 8 |
| trip_duration | 1000 | 5000 | 04 | 0.0680 | 1.963 | 1.000 | 8 |
| pickup_latitude | 10000 | 50000 | 05 | 0.0226 | 2.061 | 1.000 | 8 |
| dropoff_longitude | 50000 | 50000 | 04 | 0.0130 | 2.062 | 1.000 | 8 |
| pickup_longitude | 10000 | 50000 | 05 | 0.0226 | 2.065 | 1.000 | 8 |
| dropoff_latitude | 10000 | 50000 | 05 | 0.0228 | 2.082 | 1.000 | 8 |
| pickup_latitude | 10000 | 50000 | 06 | 0.0232 | 2.117 | 1.000 | 8 |
| trip_duration | 1000 | 50000 | 04 | 0.0680 | 2.129 | 1.000 | 8 |
| pickup_longitude | 50000 | 50000 | 04 | 0.0138 | 2.179 | 1.000 | 8 |
| dropoff_latitude | 10000 | 50000 | 06 | 0.0245 | 2.238 | 1.000 | 8 |
| dropoff_longitude | 15000 | 50000 | 05 | 0.0214 | 2.295 | 1.000 | 8 |
| dropoff_latitude | 50000 | 50000 | 04 | 0.0146 | 2.302 | 1.000 | 8 |
| dropoff_longitude | 10000 | 50000 | 06 | 0.0254 | 2.315 | 1.000 | 8 |
| pickup_latitude | 50000 | 50000 | 04 | 0.0151 | 2.395 | 1.000 | 8 |
| pickup_latitude | 15000 | 50000 | 05 | 0.0226 | 2.426 | 1.000 | 8 |
| pickup_longitude | 15000 | 50000 | 05 | 0.0226 | 2.430 | 1.000 | 8 |
| dropoff_latitude | 15000 | 50000 | 05 | 0.0228 | 2.450 | 1.000 | 8 |
| pickup_longitude | 10000 | 50000 | 06 | 0.0270 | 2.462 | 1.000 | 8 |
| pickup_latitude | 15000 | 50000 | 06 | 0.0232 | 2.491 | 1.000 | 8 |
| dropoff_longitude | 20000 | 50000 | 05 | 0.0214 | 2.554 | 1.000 | 8 |
| dropoff_latitude | 15000 | 50000 | 06 | 0.0245 | 2.633 | 1.000 | 8 |
| pickup_latitude | 20000 | 50000 | 05 | 0.0226 | 2.699 | 1.000 | 8 |
| pickup_longitude | 20000 | 50000 | 05 | 0.0226 | 2.704 | 1.000 | 8 |
| dropoff_longitude | 15000 | 50000 | 06 | 0.0254 | 2.725 | 1.000 | 8 |
| dropoff_latitude | 20000 | 50000 | 05 | 0.0228 | 2.726 | 1.000 | 8 |
| trip_duration | 1000 | 5000 | 05 | 0.0952 | 2.748 | 1.000 | 8 |
| pickup_latitude | 20000 | 50000 | 06 | 0.0232 | 2.771 | 1.000 | 8 |
| pickup_longitude | 15000 | 50000 | 06 | 0.0270 | 2.897 | 1.000 | 8 |
| dropoff_latitude | 20000 | 50000 | 06 | 0.0245 | 2.930 | 1.000 | 8 |
| trip_duration | 3000 | 5000 | 04 | 0.0680 | 2.945 | 1.000 | 8 |
| trip_duration | 1000 | 50000 | 05 | 0.0952 | 2.980 | 1.000 | 8 |
| dropoff_longitude | 20000 | 50000 | 06 | 0.0254 | 3.032 | 1.000 | 8 |
| trip_duration | 1000 | 5000 | 06 | 0.1074 | 3.099 | 1.000 | 8 |
| pickup_longitude | 20000 | 50000 | 06 | 0.0270 | 3.223 | 1.000 | 8 |
| trip_duration | 1000 | 50000 | 06 | 0.1074 | 3.362 | 1.000 | 8 |
| dropoff_longitude | 50000 | 50000 | 05 | 0.0214 | 3.379 | 1.000 | 8 |
| trip_duration | 5000 | 5000 | 04 | 0.0680 | 3.400 | 1.000 | 8 |
| pickup_latitude | 50000 | 50000 | 05 | 0.0226 | 3.570 | 1.000 | 8 |
| pickup_longitude | 50000 | 50000 | 05 | 0.0226 | 3.578 | 1.000 | 8 |
| dropoff_latitude | 50000 | 50000 | 05 | 0.0228 | 3.606 | 1.000 | 8 |
| trip_duration | 3000 | 50000 | 04 | 0.0680 | 3.618 | 1.000 | 8 |
| pickup_latitude | 50000 | 50000 | 06 | 0.0232 | 3.666 | 1.000 | 8 |
| dropoff_latitude | 50000 | 50000 | 06 | 0.0245 | 3.876 | 1.000 | 8 |
| trip_duration | 10000 | 5000 | 04 | 0.0680 | 3.926 | 1.000 | 8 |
| dropoff_longitude | 50000 | 50000 | 06 | 0.0254 | 4.011 | 1.000 | 8 |
| trip_duration | 3000 | 5000 | 05 | 0.0952 | 4.121 | 1.000 | 8 |
| trip_duration | 15000 | 5000 | 04 | 0.0680 | 4.164 | 1.000 | 8 |
| pickup_longitude | 50000 | 50000 | 06 | 0.0270 | 4.264 | 1.000 | 8 |
| trip_duration | 20000 | 5000 | 04 | 0.0680 | 4.301 | 1.000 | 8 |
| trip_duration | 50000 | 5000 | 04 | 0.0680 | 4.585 | 1.000 | 8 |
| trip_duration | 5000 | 50000 | 04 | 0.0680 | 4.585 | 1.000 | 8 |
| trip_duration | 3000 | 5000 | 06 | 0.1074 | 4.649 | 1.000 | 8 |
| trip_duration | 5000 | 5000 | 05 | 0.0952 | 4.759 | 1.000 | 8 |
| trip_duration | 3000 | 50000 | 05 | 0.0952 | 5.064 | 1.000 | 8 |
| trip_duration | 5000 | 5000 | 06 | 0.1074 | 5.368 | 1.000 | 8 |
| trip_duration | 10000 | 5000 | 05 | 0.0952 | 5.495 | 1.000 | 8 |
| trip_duration | 3000 | 50000 | 06 | 0.1074 | 5.712 | 1.000 | 8 |
| trip_duration | 15000 | 5000 | 05 | 0.0952 | 5.829 | 1.000 | 8 |
| trip_duration | 20000 | 5000 | 05 | 0.0952 | 6.020 | 1.000 | 8 |
| trip_duration | 10000 | 5000 | 06 | 0.1074 | 6.199 | 1.000 | 8 |
| trip_duration | 10000 | 50000 | 04 | 0.0680 | 6.208 | 1.000 | 8 |
| trip_duration | 50000 | 5000 | 05 | 0.0952 | 6.417 | 1.000 | 8 |
| trip_duration | 5000 | 50000 | 05 | 0.0952 | 6.417 | 1.000 | 8 |
| trip_duration | 15000 | 5000 | 06 | 0.1074 | 6.575 | 1.000 | 8 |
| trip_duration | 20000 | 5000 | 06 | 0.1074 | 6.790 | 1.000 | 8 |
| trip_duration | 50000 | 5000 | 06 | 0.1074 | 7.239 | 1.000 | 8 |
| trip_duration | 5000 | 50000 | 06 | 0.1074 | 7.239 | 1.000 | 8 |
| trip_duration | 15000 | 50000 | 04 | 0.0680 | 7.305 | 1.000 | 8 |
| trip_duration | 20000 | 50000 | 04 | 0.0680 | 8.128 | 1.000 | 8 |
| trip_duration | 10000 | 50000 | 05 | 0.0952 | 8.689 | 1.000 | 8 |
| trip_duration | 10000 | 50000 | 06 | 0.1074 | 9.801 | 1.000 | 8 |
| trip_duration | 15000 | 50000 | 05 | 0.0952 | 10.224 | 1.000 | 8 |
| trip_duration | 50000 | 50000 | 04 | 0.0680 | 10.752 | 1.000 | 8 |
| trip_duration | 20000 | 50000 | 05 | 0.0952 | 11.376 | 1.000 | 8 |
| trip_duration | 15000 | 50000 | 06 | 0.1074 | 11.533 | 1.000 | 8 |
| trip_duration | 20000 | 50000 | 06 | 0.1074 | 12.833 | 1.000 | 8 |
| trip_duration | 50000 | 50000 | 05 | 0.0952 | 15.049 | 1.000 | 8 |
| trip_duration | 50000 | 50000 | 06 | 0.1074 | 16.976 | 1.000 | 8 |

**Collapse test**: group points into x-bins and check whether detection rate is consistent WITHIN a bin regardless of which feature/n/m/month produced it. If collapse holds, all points in a bin should have similar detection rates (low within-bin spread).

| x-bin | n_points | rate mean | rate std | rate min | rate max | features present |
|---|---|---|---|---|---|---|
| 0.4 | 6 | 0.062 | 0.062 | 0.000 | 0.125 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 0.5 | 2 | 0.188 | 0.062 | 0.125 | 0.250 | dropoff_la, pickup_lat |
| 0.6 | 4 | 0.219 | 0.185 | 0.000 | 0.500 | dropoff_la, dropoff_lo, pickup_lon |
| 0.7 | 17 | 0.250 | 0.187 | 0.000 | 0.500 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 0.8 | 13 | 0.404 | 0.307 | 0.000 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 0.9 | 10 | 0.425 | 0.317 | 0.000 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 1.0 | 9 | 0.542 | 0.195 | 0.250 | 0.875 | dropoff_la, pickup_lat, pickup_lon |
| 1.1 | 7 | 0.518 | 0.309 | 0.000 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 1.2 | 9 | 0.778 | 0.175 | 0.375 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 1.3 | 11 | 0.773 | 0.319 | 0.125 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 1.4 | 14 | 0.875 | 0.275 | 0.000 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 1.5 | 10 | 0.950 | 0.115 | 0.625 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 1.6 | 10 | 0.875 | 0.237 | 0.250 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon |
| 1.7 | 7 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, dropoff_lo, pickup_lon |
| 1.8 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | pickup_lat, pickup_lon |
| 2.0 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_lo, trip_durat |
| 2.1 | 6 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon, trip_durat |
| 2.2 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, pickup_lon |
| 2.3 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, dropoff_lo |
| 2.4 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | pickup_lat, pickup_lon |
| 2.5 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, pickup_lat, pickup_lon |
| 2.6 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, dropoff_lo |
| 2.7 | 5 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, dropoff_lo, pickup_lat, pickup_lon, trip_durat |
| 2.8 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | pickup_lat |
| 2.9 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, pickup_lon, trip_durat |
| 3.0 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_lo, trip_durat |
| 3.1 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 3.2 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | pickup_lon |
| 3.4 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_lo, trip_durat |
| 3.6 | 4 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, pickup_lat, pickup_lon, trip_durat |
| 3.7 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | pickup_lat |
| 3.9 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_la, trip_durat |
| 4.0 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | dropoff_lo |
| 4.1 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 4.2 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 4.3 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | pickup_lon, trip_durat |
| 4.6 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 4.8 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 5.1 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 5.4 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 5.5 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 5.7 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 5.8 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 6.0 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 6.2 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 6.4 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 6.6 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 6.8 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 7.2 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 7.3 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 8.1 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 8.7 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 9.8 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 10.2 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 10.8 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 11.4 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 11.5 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 12.8 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 15.0 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |
| 17.0 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | trip_durat |

**Plain report on collapse**: the largest within-bin spread is std=0.319 at x-bin=1.3. **Context needed before calling this a theory violation**: each cell's detection rate is itself estimated from only 8 binary trials -- at the worst case (true rate near 0.5), binomial sampling noise alone gives a standard error of `sqrt(0.5*0.5/8)` = 0.177 for a SINGLE cell's rate estimate. A bin's within-bin std (spread ACROSS several such noisy estimates, if their true rate were identical) would be expected to land in a similar range purely from this sampling noise -- before any real curve-shape difference between features/months enters into it.

The observed spread (0.319) meaningfully exceeds what 8-trials-per-cell binomial noise alone would predict (~0.177 at worst case) — **this is suggestive of a real deviation from the single-curve collapse**, not just sampling noise, though confirming it properly would need more trials per cell than this run collected. See which feature(s) dominate the high-spread bin(s) above for where the deviation concentrates.
