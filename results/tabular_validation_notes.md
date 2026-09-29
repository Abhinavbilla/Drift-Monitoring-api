# Tabular Validation Notes (Step 1) — Apr-Jun 2016

## Compact summary

**Scope**: Apr-Jun 2016 production only (see `results/citi_bike_provenance_forensics.md`). Old README "Apr-Dec" numbers remain superseded, not quoted here.

**Ground truth fix**: population truth is now computed once from the FULL baseline CSV (1,577,611 rows) vs full production data, independent of reference/batch size. The old reference-vs-production computation is kept only as a diagnostic (`reference_detectable_effect`), never as ground truth. See 'Ground truth fix, verified' below for the dropoff_longitude check.

**Real batches only, pooled Apr-Jun, decision unit=(batch,feature), D_gt=0.02 / PSI>=0.2, at the largest batch size tested (50,000), WITH and WITHOUT `month`:**

| Reference size | month? | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 5000 | with | 84 | 37 | 12 | 35 | 0.694 | 0.875 | 0.774 |
| 5000 | without | 60 | 37 | 12 | 35 | 0.619 | 0.833 | 0.710 |
| 50000 | with | 96 | 48 | 0 | 24 | 0.667 | 1.000 | 0.800 |
| 50000 | without | 72 | 48 | 0 | 24 | 0.600 | 1.000 | 0.750 |

**A/A per-feature and system false-alarm rates vs. alpha-predicted (`1-(1-0.05)^7`=0.302), both reference sizes, all A/A sizes:**

| Reference size | Batch size | System | pickup_longitude | pickup_latitude | dropoff_longitude | dropoff_latitude | trip_duration | gender_id | month |
|---|---|---|---|---|---|---|---|---|---|
| 5000 | 1000 | 0.120 | 0.100 | 0.010 | 0.040 | 0.000 | 0.010 | 0.000 | 0.000 |
| 5000 | 3000 | 0.180 | 0.160 | 0.000 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5000 | 5000 | 0.270 | 0.240 | 0.000 | 0.030 | 0.010 | 0.020 | 0.000 | 0.000 |
| 5000 | 10000 | 0.210 | 0.130 | 0.000 | 0.090 | 0.010 | 0.010 | 0.000 | 0.000 |
| 5000 | 15000 | 0.140 | 0.100 | 0.000 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5000 | 20000 | 0.020 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 50000 | 1000 | 0.170 | 0.050 | 0.050 | 0.050 | 0.010 | 0.030 | 0.000 | 0.000 |
| 50000 | 3000 | 0.160 | 0.040 | 0.040 | 0.040 | 0.010 | 0.030 | 0.000 | 0.000 |
| 50000 | 5000 | 0.200 | 0.020 | 0.060 | 0.120 | 0.010 | 0.030 | 0.000 | 0.000 |
| 50000 | 10000 | 0.090 | 0.010 | 0.020 | 0.040 | 0.010 | 0.010 | 0.000 | 0.000 |
| 50000 | 15000 | 0.020 | 0.000 | 0.000 | 0.010 | 0.010 | 0.000 | 0.000 | 0.000 |
| 50000 | 20000 | 0.020 | 0.000 | 0.000 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 |

**Smallest synthetic severity reliably detected (100% of trials, every continuous feature) at each reference size:**

| Reference size | Smallest reliable severity (σ) |
|---|---|
| 5000 | 0.05 |
| 50000 | 0.05 |

**Original README discrepancies (all about the now-superseded old numbers):**

- *Headline recall 0.939 vs. naive per-feature-table sum 81/90=0.900*: **unreproducible** — the original `/fit` call's exact reference rows and the script that produced that headline run no longer exist (`split_citi_bike.py` was missing, see `docs/recon.md` §4/§9); there is no way to recompute the exact old number.
- *Sweep row 10,000: recall 0.812 at precision 1.0 implies F1=0.896, but README says 0.886 (implying precision≈0.975)*: **unreproducible**, same reason — the exact old sweep run's data is gone.
- *Headline batch size (0.939) exceeds every sweep row including 50k (0.917)*: **explained**, not a contradiction — confirmed by reading the code (`docs/recon.md` §4) that the headline used a fixed `PRODUCTION_BATCH_SIZE=25000`, a size the separate 8-trial sweep never tested. Different experiments, not a discrepancy in one experiment — though the specific 0.939 value itself remains unreproducible for the reasons above.

---

## Ground truth fix, verified

**dropoff_longitude**, which flipped labels between reference sizes in the previous (flawed) version of this analysis, now has a single, fixed population D — because population truth no longer uses either sampled reference at all:

- Pooled Apr-Jun population D for `dropoff_longitude` (full baseline vs full production, independent of reference size): **0.0206** — one number, used for both m=5,000 and m=50,000 analyses.

**Full population truth (pooled Apr-Jun), raw statistics:**

| Feature | Population D or PSI |
|---|---|
| pickup_longitude | 0.0219 |
| pickup_latitude | 0.0189 |
| dropoff_longitude | 0.0206 |
| dropoff_latitude | 0.0195 |
| trip_duration | 0.0924 |
| gender_id | 0.0432 |
| month | 16.2660 |

**Per-month population truth, raw statistics (all vs. the SAME full baseline CSV):**

| Feature | Apr (D/PSI) | May (D/PSI) | Jun (D/PSI) |
|---|---|---|---|
| pickup_longitude | 0.0138 | 0.0226 | 0.0270 |
| pickup_latitude | 0.0151 | 0.0226 | 0.0232 |
| dropoff_longitude | 0.0130 | 0.0214 | 0.0254 |
| dropoff_latitude | 0.0146 | 0.0228 | 0.0245 |
| trip_duration | 0.0680 | 0.0952 | 0.1074 |
| gender_id | 0.0337 | 0.0546 | 0.0416 |
| month | 17.3550 | 17.3550 | 17.3550 |


## Reference size = 5000

**Diagnostic only, NOT ground truth** (`reference_detectable_effect`) — the OLD reference-vs-production computation, kept for reference but never used below to grade detections:

| Feature | "Drifted" vs. this reference | Detail |
|---|---|---|
| pickup_longitude | True | D=0.0294, p=3.41e-04 |
| pickup_latitude | True | D=0.0254, p=3.09e-03 |
| dropoff_longitude | False | D=0.0180, p=7.91e-02 |
| dropoff_latitude | True | D=0.0227, p=1.12e-02 |
| trip_duration | True | D=0.0945, p=2.85e-39 |
| gender_id | False | PSI=0.0422 |
| month | True | PSI=16.2644 |

**Sweep, pooled, D_gt=0.01 (PSI>=0.2), WITH `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |
|---|---|---|---|---|---|---|---|---|
| 1000 | 66 | 0 | 78 | 24 | 1.000 | 0.458 | 0.629 | 168 |
| 3000 | 85 | 0 | 59 | 24 | 1.000 | 0.590 | 0.742 | 168 |
| 5000 | 94 | 0 | 50 | 24 | 1.000 | 0.653 | 0.790 | 168 |
| 10000 | 113 | 0 | 31 | 24 | 1.000 | 0.785 | 0.879 | 168 |
| 15000 | 110 | 0 | 34 | 24 | 1.000 | 0.764 | 0.866 | 168 |
| 20000 | 114 | 0 | 30 | 24 | 1.000 | 0.792 | 0.884 | 168 |
| 50000 | 121 | 0 | 23 | 24 | 1.000 | 0.840 | 0.913 | 168 |

**Same, D_gt=0.01, WITHOUT `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1000 | 42 | 0 | 78 | 24 | 1.000 | 0.350 | 0.519 |
| 3000 | 61 | 0 | 59 | 24 | 1.000 | 0.508 | 0.674 |
| 5000 | 70 | 0 | 50 | 24 | 1.000 | 0.583 | 0.737 |
| 10000 | 89 | 0 | 31 | 24 | 1.000 | 0.742 | 0.852 |
| 15000 | 86 | 0 | 34 | 24 | 1.000 | 0.717 | 0.835 |
| 20000 | 90 | 0 | 30 | 24 | 1.000 | 0.750 | 0.857 |
| 50000 | 97 | 0 | 23 | 24 | 1.000 | 0.808 | 0.894 |

**Sweep, pooled, D_gt=0.02 (PSI>=0.2), WITH `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |
|---|---|---|---|---|---|---|---|---|
| 1000 | 58 | 8 | 38 | 64 | 0.879 | 0.604 | 0.716 | 168 |
| 3000 | 63 | 22 | 33 | 50 | 0.741 | 0.656 | 0.696 | 168 |
| 5000 | 67 | 27 | 29 | 45 | 0.713 | 0.698 | 0.705 | 168 |
| 10000 | 80 | 33 | 16 | 39 | 0.708 | 0.833 | 0.766 | 168 |
| 15000 | 73 | 37 | 23 | 35 | 0.664 | 0.760 | 0.709 | 168 |
| 20000 | 75 | 39 | 21 | 33 | 0.658 | 0.781 | 0.714 | 168 |
| 50000 | 84 | 37 | 12 | 35 | 0.694 | 0.875 | 0.774 | 168 |

**Same, D_gt=0.02, WITHOUT `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1000 | 34 | 8 | 38 | 64 | 0.810 | 0.472 | 0.596 |
| 3000 | 39 | 22 | 33 | 50 | 0.639 | 0.542 | 0.586 |
| 5000 | 43 | 27 | 29 | 45 | 0.614 | 0.597 | 0.606 |
| 10000 | 56 | 33 | 16 | 39 | 0.629 | 0.778 | 0.696 |
| 15000 | 49 | 37 | 23 | 35 | 0.570 | 0.681 | 0.620 |
| 20000 | 51 | 39 | 21 | 33 | 0.567 | 0.708 | 0.630 |
| 50000 | 60 | 37 | 12 | 35 | 0.619 | 0.833 | 0.710 |

**Sweep, pooled, D_gt=0.05 (PSI>=0.2), WITH `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |
|---|---|---|---|---|---|---|---|---|
| 1000 | 48 | 18 | 0 | 102 | 0.727 | 1.000 | 0.842 | 168 |
| 3000 | 48 | 37 | 0 | 83 | 0.565 | 1.000 | 0.722 | 168 |
| 5000 | 48 | 46 | 0 | 74 | 0.511 | 1.000 | 0.676 | 168 |
| 10000 | 48 | 65 | 0 | 55 | 0.425 | 1.000 | 0.596 | 168 |
| 15000 | 48 | 62 | 0 | 58 | 0.436 | 1.000 | 0.608 | 168 |
| 20000 | 48 | 66 | 0 | 54 | 0.421 | 1.000 | 0.593 | 168 |
| 50000 | 48 | 73 | 0 | 47 | 0.397 | 1.000 | 0.568 | 168 |

**Same, D_gt=0.05, WITHOUT `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1000 | 24 | 18 | 0 | 102 | 0.571 | 1.000 | 0.727 |
| 3000 | 24 | 37 | 0 | 83 | 0.393 | 1.000 | 0.565 |
| 5000 | 24 | 46 | 0 | 74 | 0.343 | 1.000 | 0.511 |
| 10000 | 24 | 65 | 0 | 55 | 0.270 | 1.000 | 0.425 |
| 15000 | 24 | 62 | 0 | 58 | 0.279 | 1.000 | 0.436 |
| 20000 | 24 | 66 | 0 | 54 | 0.267 | 1.000 | 0.421 |
| 50000 | 24 | 73 | 0 | 47 | 0.247 | 1.000 | 0.397 |

**Categorical ground truth sensitivity (D_gt fixed at 0.02 for continuous), PSI>=0.1 vs PSI>=0.2, pooled, batch size=50000:**

| PSI threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.1 | 84 | 37 | 12 | 35 | 0.694 | 0.875 | 0.774 |
| 0.2 | 84 | 37 | 12 | 35 | 0.694 | 0.875 | 0.774 |

**Per-month breakdown at batch size=50000, D_gt=0.02/PSI>=0.2, ground truth = that month's OWN population truth (full baseline vs that month's full data), not pooled:**

| Month | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 04 | 16 | 13 | 0 | 27 | 0.552 | 1.000 | 0.711 |
| 05 | 44 | 0 | 4 | 8 | 1.000 | 0.917 | 0.957 |
| 06 | 48 | 0 | 0 | 8 | 1.000 | 1.000 | 1.000 |

**Per-feature batch statistic (D for continuous, PSI for categorical) mean/std at batch size=50000, vs. FIXED population value:**

| Feature | Population D/PSI (fixed) | Per-batch mean | Per-batch std |
|---|---|---|---|
| pickup_longitude | 0.0219 | 0.0296 | 0.0048 |
| pickup_latitude | 0.0189 | 0.0253 | 0.0045 |
| dropoff_longitude | 0.0206 | 0.0181 | 0.0049 |
| dropoff_latitude | 0.0195 | 0.0236 | 0.0042 |
| trip_duration | 0.0924 | 0.0916 | 0.0166 |
| gender_id | 0.0432 | 0.0423 | 0.0086 |
| month | 16.2660 | 17.3534 | 0.0000 |

**A/A test (iid test-calibration check — both holdout and reference drawn at random from Jan-Mar; does NOT capture month-to-month variation within the baseline period):**

| Batch size | System false-alarm rate | pickup_longitude | pickup_latitude | dropoff_longitude | dropoff_latitude | trip_duration | gender_id | month |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.120 | 0.100 | 0.010 | 0.040 | 0.000 | 0.010 | 0.000 | 0.000 |
| 3000 | 0.180 | 0.160 | 0.000 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5000 | 0.270 | 0.240 | 0.000 | 0.030 | 0.010 | 0.020 | 0.000 | 0.000 |
| 10000 | 0.210 | 0.130 | 0.000 | 0.090 | 0.010 | 0.010 | 0.000 | 0.000 |
| 15000 | 0.140 | 0.100 | 0.000 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 |
| 20000 | 0.020 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Alpha-predicted system false-alarm rate for 7 uncorrected tests at alpha=0.05: `1-(1-0.05)^7` = **0.302**.


## Reference size = 50000

**Diagnostic only, NOT ground truth** (`reference_detectable_effect`) — the OLD reference-vs-production computation, kept for reference but never used below to grade detections:

| Feature | "Drifted" vs. this reference | Detail |
|---|---|---|
| pickup_longitude | True | D=0.0223, p=1.02e-21 |
| pickup_latitude | True | D=0.0211, p=1.76e-19 |
| dropoff_longitude | True | D=0.0247, p=1.46e-26 |
| dropoff_latitude | True | D=0.0206, p=1.28e-18 |
| trip_duration | True | D=0.0963, p=0.00e+00 |
| gender_id | False | PSI=0.0453 |
| month | True | PSI=16.2648 |

**Sweep, pooled, D_gt=0.01 (PSI>=0.2), WITH `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |
|---|---|---|---|---|---|---|---|---|
| 1000 | 70 | 0 | 74 | 24 | 1.000 | 0.486 | 0.654 | 168 |
| 3000 | 123 | 0 | 21 | 24 | 1.000 | 0.854 | 0.921 | 168 |
| 5000 | 131 | 0 | 13 | 24 | 1.000 | 0.910 | 0.953 | 168 |
| 10000 | 140 | 0 | 4 | 24 | 1.000 | 0.972 | 0.986 | 168 |
| 15000 | 144 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 | 168 |
| 20000 | 144 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 | 168 |
| 50000 | 144 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 | 168 |

**Same, D_gt=0.01, WITHOUT `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1000 | 46 | 0 | 74 | 24 | 1.000 | 0.383 | 0.554 |
| 3000 | 99 | 0 | 21 | 24 | 1.000 | 0.825 | 0.904 |
| 5000 | 107 | 0 | 13 | 24 | 1.000 | 0.892 | 0.943 |
| 10000 | 116 | 0 | 4 | 24 | 1.000 | 0.967 | 0.983 |
| 15000 | 120 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |
| 20000 | 120 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |
| 50000 | 120 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |

**Sweep, pooled, D_gt=0.02 (PSI>=0.2), WITH `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |
|---|---|---|---|---|---|---|---|---|
| 1000 | 57 | 13 | 39 | 59 | 0.814 | 0.594 | 0.687 | 168 |
| 3000 | 86 | 37 | 10 | 35 | 0.699 | 0.896 | 0.785 | 168 |
| 5000 | 88 | 43 | 8 | 29 | 0.672 | 0.917 | 0.775 | 168 |
| 10000 | 93 | 47 | 3 | 25 | 0.664 | 0.969 | 0.788 | 168 |
| 15000 | 96 | 48 | 0 | 24 | 0.667 | 1.000 | 0.800 | 168 |
| 20000 | 96 | 48 | 0 | 24 | 0.667 | 1.000 | 0.800 | 168 |
| 50000 | 96 | 48 | 0 | 24 | 0.667 | 1.000 | 0.800 | 168 |

**Same, D_gt=0.02, WITHOUT `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1000 | 33 | 13 | 39 | 59 | 0.717 | 0.458 | 0.559 |
| 3000 | 62 | 37 | 10 | 35 | 0.626 | 0.861 | 0.725 |
| 5000 | 64 | 43 | 8 | 29 | 0.598 | 0.889 | 0.715 |
| 10000 | 69 | 47 | 3 | 25 | 0.595 | 0.958 | 0.734 |
| 15000 | 72 | 48 | 0 | 24 | 0.600 | 1.000 | 0.750 |
| 20000 | 72 | 48 | 0 | 24 | 0.600 | 1.000 | 0.750 |
| 50000 | 72 | 48 | 0 | 24 | 0.600 | 1.000 | 0.750 |

**Sweep, pooled, D_gt=0.05 (PSI>=0.2), WITH `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |
|---|---|---|---|---|---|---|---|---|
| 1000 | 48 | 22 | 0 | 98 | 0.686 | 1.000 | 0.814 | 168 |
| 3000 | 48 | 75 | 0 | 45 | 0.390 | 1.000 | 0.561 | 168 |
| 5000 | 48 | 83 | 0 | 37 | 0.366 | 1.000 | 0.536 | 168 |
| 10000 | 48 | 92 | 0 | 28 | 0.343 | 1.000 | 0.511 | 168 |
| 15000 | 48 | 96 | 0 | 24 | 0.333 | 1.000 | 0.500 | 168 |
| 20000 | 48 | 96 | 0 | 24 | 0.333 | 1.000 | 0.500 | 168 |
| 50000 | 48 | 96 | 0 | 24 | 0.333 | 1.000 | 0.500 | 168 |

**Same, D_gt=0.05, WITHOUT `month`:**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1000 | 24 | 22 | 0 | 98 | 0.522 | 1.000 | 0.686 |
| 3000 | 24 | 75 | 0 | 45 | 0.242 | 1.000 | 0.390 |
| 5000 | 24 | 83 | 0 | 37 | 0.224 | 1.000 | 0.366 |
| 10000 | 24 | 92 | 0 | 28 | 0.207 | 1.000 | 0.343 |
| 15000 | 24 | 96 | 0 | 24 | 0.200 | 1.000 | 0.333 |
| 20000 | 24 | 96 | 0 | 24 | 0.200 | 1.000 | 0.333 |
| 50000 | 24 | 96 | 0 | 24 | 0.200 | 1.000 | 0.333 |

**Categorical ground truth sensitivity (D_gt fixed at 0.02 for continuous), PSI>=0.1 vs PSI>=0.2, pooled, batch size=50000:**

| PSI threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.1 | 96 | 48 | 0 | 24 | 0.667 | 1.000 | 0.800 |
| 0.2 | 96 | 48 | 0 | 24 | 0.667 | 1.000 | 0.800 |

**Per-month breakdown at batch size=50000, D_gt=0.02/PSI>=0.2, ground truth = that month's OWN population truth (full baseline vs that month's full data), not pooled:**

| Month | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 04 | 16 | 32 | 0 | 8 | 0.333 | 1.000 | 0.500 |
| 05 | 48 | 0 | 0 | 8 | 1.000 | 1.000 | 1.000 |
| 06 | 48 | 0 | 0 | 8 | 1.000 | 1.000 | 1.000 |

**Per-feature batch statistic (D for continuous, PSI for categorical) mean/std at batch size=50000, vs. FIXED population value:**

| Feature | Population D/PSI (fixed) | Per-batch mean | Per-batch std |
|---|---|---|---|
| pickup_longitude | 0.0219 | 0.0229 | 0.0050 |
| pickup_latitude | 0.0189 | 0.0218 | 0.0035 |
| dropoff_longitude | 0.0206 | 0.0249 | 0.0052 |
| dropoff_latitude | 0.0195 | 0.0221 | 0.0042 |
| trip_duration | 0.0924 | 0.0937 | 0.0165 |
| gender_id | 0.0432 | 0.0456 | 0.0091 |
| month | 16.2660 | 17.3539 | 0.0000 |

**A/A test (iid test-calibration check — both holdout and reference drawn at random from Jan-Mar; does NOT capture month-to-month variation within the baseline period):**

| Batch size | System false-alarm rate | pickup_longitude | pickup_latitude | dropoff_longitude | dropoff_latitude | trip_duration | gender_id | month |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.170 | 0.050 | 0.050 | 0.050 | 0.010 | 0.030 | 0.000 | 0.000 |
| 3000 | 0.160 | 0.040 | 0.040 | 0.040 | 0.010 | 0.030 | 0.000 | 0.000 |
| 5000 | 0.200 | 0.020 | 0.060 | 0.120 | 0.010 | 0.030 | 0.000 | 0.000 |
| 10000 | 0.090 | 0.010 | 0.020 | 0.040 | 0.010 | 0.010 | 0.000 | 0.000 |
| 15000 | 0.020 | 0.000 | 0.000 | 0.010 | 0.010 | 0.000 | 0.000 | 0.000 |
| 20000 | 0.020 | 0.000 | 0.000 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 |

Alpha-predicted system false-alarm rate for 7 uncorrected tests at alpha=0.05: `1-(1-0.05)^7` = **0.302**.


## Reference size vs. batch size

- m=5000: asymptotic KS critical-value floor as n->infinity: c(alpha)/sqrt(m) = 1.36/sqrt(5000) = **0.0192**
- m=50000: asymptotic KS critical-value floor as n->infinity: c(alpha)/sqrt(m) = 1.36/sqrt(50000) = **0.0061**
- At n=20,000, m=5,000: c(alpha)*sqrt((n+m)/(nm)) = **0.0215**

**pickup_latitude fixed population D = 0.0189** (same for both reference sizes now, per the fix above) vs. m=5,000 floor (0.0192) and m=50,000 floor (0.0061).

**pickup_latitude recall by batch size, at both reference sizes (pooled, with month, FIXED ground truth, D_gt=0.01 -- chosen because pickup_latitude's population D=0.0189 is below 0.02, so D_gt=0.02 would make it a ground-truth negative with no recall to measure):**

| Batch size | Recall @ m=5000 | Recall @ m=50000 |
|---|---|---|
| 1000 | 0.167 | 0.250 |
| 3000 | 0.500 | 0.708 |
| 5000 | 0.667 | 0.833 |
| 10000 | 0.792 | 1.000 |
| 15000 | 0.833 | 1.000 |
| 20000 | 0.833 | 1.000 |
| 50000 | 0.792 | 1.000 |

**Same conclusion as before, now on a stable ground truth**: pickup_latitude's population D (0.0189) sits close to the m=5,000 floor. At m=5,000, recall plateaus well below 1.0 at large batch sizes; at m=50,000, recall reaches 1.000. Reference size, not batch size, is the binding constraint — and this time the ground truth used to measure it doesn't move when the reference does.

**Note on `month`'s per-batch std=0.0000**: every row drawn from a single production month file has `month` equal to that one value, so every batch's PSI for `month` compares against an identical 100%-one-category distribution — no batch-to-batch variation to produce nonzero std. Expected, not a bug.
