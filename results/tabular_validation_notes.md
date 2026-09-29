# Tabular Validation Notes (Step 1) — Apr-Jun 2016

**Scope**: production batches are Apr-Jun 2016 only (the only data available locally — see `results/citi_bike_provenance_forensics.md`). The old README's "Apr-Dec" numbers are superseded and not quoted here.

**Decision unit**: (batch, feature) — one confusion-matrix row per feature per analyzed batch, matching the legacy script's own unit (`tests/test_drift_engine.py`'s `run_classification_evaluation`, confirmed in `docs/recon.md` §4). A `system alert` is a separate, coarser unit: `any(feature drift_detected)` per batch — see below for both.

**Config**: alpha=0.05, PSI threshold=0.2, reference sizes=[5000, 50000], holdout size=25000, months=[4, 5, 6] (Apr/May/Jun 2016), sweep sizes=[1000, 3000, 5000, 10000, 15000, 20000, 50000], trials/month/size (target, capped by disjointness)=8, A/A trials/size=100, A/A sizes=[1000, 3000, 5000, 10000, 15000, 20000], severities=[0.02, 0.05, 0.1, 0.2, 0.5, 1.5, 3.0], severity batch size=5000, severity trials=5.


## Reference size = 5000

**Population ground truth (pooled Apr-Jun vs reference), legacy definition (continuous: p<alpha; categorical: PSI>0.2):**

| Feature | Drifted? | Detail |
|---|---|---|
| pickup_longitude | True | D=0.0294, p=3.41e-04 |
| pickup_latitude | True | D=0.0254, p=3.09e-03 |
| dropoff_longitude | False | D=0.0180, p=7.91e-02 |
| dropoff_latitude | True | D=0.0227, p=1.12e-02 |
| trip_duration | True | D=0.0945, p=2.85e-39 |
| gender_id | False | PSI=0.0422 |
| month | True | PSI=16.2644 |

**Sweep (pooled across months, decision unit = batch×feature, ground truth = pooled Apr-Jun population verdict):**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |
|---|---|---|---|---|---|---|---|---|
| 1000 | 65 | 1 | 55 | 47 | 0.985 | 0.542 | 0.699 | 168 |
| 3000 | 83 | 2 | 37 | 46 | 0.976 | 0.692 | 0.810 | 168 |
| 5000 | 93 | 1 | 27 | 47 | 0.989 | 0.775 | 0.869 | 168 |
| 10000 | 105 | 8 | 15 | 40 | 0.929 | 0.875 | 0.901 | 168 |
| 15000 | 107 | 3 | 13 | 45 | 0.973 | 0.892 | 0.930 | 168 |
| 20000 | 109 | 5 | 11 | 43 | 0.956 | 0.908 | 0.932 | 168 |
| 50000 | 109 | 12 | 11 | 36 | 0.901 | 0.908 | 0.905 | 168 |

**Same sweep, WITHOUT `month` feature (isolating the effect of dropping the categorical feature with the largest PSI):**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1000 | 41 | 1 | 55 | 47 | 0.976 | 0.427 | 0.594 |
| 3000 | 59 | 2 | 37 | 46 | 0.967 | 0.615 | 0.752 |
| 5000 | 69 | 1 | 27 | 47 | 0.986 | 0.719 | 0.831 |
| 10000 | 81 | 8 | 15 | 40 | 0.910 | 0.844 | 0.876 |
| 15000 | 83 | 3 | 13 | 45 | 0.965 | 0.865 | 0.912 |
| 20000 | 85 | 5 | 11 | 43 | 0.944 | 0.885 | 0.914 |
| 50000 | 85 | 12 | 11 | 36 | 0.876 | 0.885 | 0.881 |

**Per-month breakdown at batch size=50000 (ground truth = that month's OWN population verdict vs reference, not pooled):**

| Month | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 04 | 27 | 2 | 5 | 22 | 0.931 | 0.844 | 0.885 |
| 05 | 40 | 4 | 0 | 12 | 0.909 | 1.000 | 0.952 |
| 06 | 48 | 0 | 0 | 8 | 1.000 | 1.000 | 1.000 |

**Confusion matrix under effect-size ground truth variants (pooled, batch size=50000), vs. the legacy p-value-based ground truth above:**

| Ground truth | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| D>=0.01 | 121 | 0 | 23 | 24 | 1.000 | 0.840 | 0.913 |
| D>=0.02 | 109 | 12 | 11 | 36 | 0.901 | 0.908 | 0.905 |
| D>=0.05 | 48 | 73 | 0 | 47 | 0.397 | 1.000 | 0.568 |
| legacy (p<0.05) | 109 | 12 | 11 | 36 | 0.901 | 0.908 | 0.905 |

**Framing (per instruction)**: this ground truth measures batch-level recovery of the POPULATION verdict — i.e. statistical power to detect a population-level difference that has already been established independently — not some external notion of "real" drift. At multi-million-row population sizes, KS p-values become oversensitive in exactly the way chi-square was rejected for PSI's categorical ground truth (`docs/recon.md` §3): a population D as small as 0.01-0.02 can still yield p<<0.05 given enough rows. This is why the effect-size variants above matter — they ask a different, arguably more practically relevant question ("is the population difference large enough to matter") than the raw p-value does.

**Per-feature batch statistic (D for continuous, PSI for categorical) mean/std at batch size=50000, vs. population value:**

| Feature | Population D/PSI | Per-batch mean | Per-batch std |
|---|---|---|---|
| pickup_longitude | 0.0294 | 0.0296 | 0.0048 |
| pickup_latitude | 0.0254 | 0.0253 | 0.0045 |
| dropoff_longitude | 0.0180 | 0.0181 | 0.0049 |
| dropoff_latitude | 0.0227 | 0.0236 | 0.0042 |
| trip_duration | 0.0945 | 0.0916 | 0.0166 |
| gender_id | 0.0422 | 0.0423 | 0.0086 |
| month | 16.2644 | 17.3534 | 0.0000 |

**A/A test (iid test-calibration check — both holdout and reference drawn at random from Jan-Mar; does NOT capture month-to-month variation within the baseline period):**

| Batch size | System false-alarm rate | pickup_longitude | pickup_latitude | dropoff_longitude | dropoff_latitude | trip_duration | gender_id | month |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.120 | 0.100 | 0.010 | 0.040 | 0.000 | 0.010 | 0.000 | 0.000 |
| 3000 | 0.180 | 0.160 | 0.000 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5000 | 0.270 | 0.240 | 0.000 | 0.030 | 0.010 | 0.020 | 0.000 | 0.000 |
| 10000 | 0.210 | 0.130 | 0.000 | 0.090 | 0.010 | 0.010 | 0.000 | 0.000 |
| 15000 | 0.140 | 0.100 | 0.000 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 |
| 20000 | 0.020 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Alpha-predicted system false-alarm rate for 7 uncorrected tests at alpha=0.05: `1-(1-0.05)^7` = **0.302**. Compare against the observed system false-alarm rates above.


## Reference size = 50000

**Population ground truth (pooled Apr-Jun vs reference), legacy definition (continuous: p<alpha; categorical: PSI>0.2):**

| Feature | Drifted? | Detail |
|---|---|---|
| pickup_longitude | True | D=0.0223, p=1.02e-21 |
| pickup_latitude | True | D=0.0211, p=1.76e-19 |
| dropoff_longitude | True | D=0.0247, p=1.46e-26 |
| dropoff_latitude | True | D=0.0206, p=1.28e-18 |
| trip_duration | True | D=0.0963, p=0.00e+00 |
| gender_id | False | PSI=0.0453 |
| month | True | PSI=16.2648 |

**Sweep (pooled across months, decision unit = batch×feature, ground truth = pooled Apr-Jun population verdict):**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |
|---|---|---|---|---|---|---|---|---|
| 1000 | 70 | 0 | 74 | 24 | 1.000 | 0.486 | 0.654 | 168 |
| 3000 | 123 | 0 | 21 | 24 | 1.000 | 0.854 | 0.921 | 168 |
| 5000 | 131 | 0 | 13 | 24 | 1.000 | 0.910 | 0.953 | 168 |
| 10000 | 140 | 0 | 4 | 24 | 1.000 | 0.972 | 0.986 | 168 |
| 15000 | 144 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 | 168 |
| 20000 | 144 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 | 168 |
| 50000 | 144 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 | 168 |

**Same sweep, WITHOUT `month` feature (isolating the effect of dropping the categorical feature with the largest PSI):**

| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1000 | 46 | 0 | 74 | 24 | 1.000 | 0.383 | 0.554 |
| 3000 | 99 | 0 | 21 | 24 | 1.000 | 0.825 | 0.904 |
| 5000 | 107 | 0 | 13 | 24 | 1.000 | 0.892 | 0.943 |
| 10000 | 116 | 0 | 4 | 24 | 1.000 | 0.967 | 0.983 |
| 15000 | 120 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |
| 20000 | 120 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |
| 50000 | 120 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |

**Per-month breakdown at batch size=50000 (ground truth = that month's OWN population verdict vs reference, not pooled):**

| Month | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 04 | 48 | 0 | 0 | 8 | 1.000 | 1.000 | 1.000 |
| 05 | 48 | 0 | 0 | 8 | 1.000 | 1.000 | 1.000 |
| 06 | 48 | 0 | 0 | 8 | 1.000 | 1.000 | 1.000 |

**Confusion matrix under effect-size ground truth variants (pooled, batch size=50000), vs. the legacy p-value-based ground truth above:**

| Ground truth | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| D>=0.01 | 144 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |
| D>=0.02 | 144 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |
| D>=0.05 | 48 | 96 | 0 | 24 | 0.333 | 1.000 | 0.500 |
| legacy (p<0.05) | 144 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |

**Framing (per instruction)**: this ground truth measures batch-level recovery of the POPULATION verdict — i.e. statistical power to detect a population-level difference that has already been established independently — not some external notion of "real" drift. At multi-million-row population sizes, KS p-values become oversensitive in exactly the way chi-square was rejected for PSI's categorical ground truth (`docs/recon.md` §3): a population D as small as 0.01-0.02 can still yield p<<0.05 given enough rows. This is why the effect-size variants above matter — they ask a different, arguably more practically relevant question ("is the population difference large enough to matter") than the raw p-value does.

**Per-feature batch statistic (D for continuous, PSI for categorical) mean/std at batch size=50000, vs. population value:**

| Feature | Population D/PSI | Per-batch mean | Per-batch std |
|---|---|---|---|
| pickup_longitude | 0.0223 | 0.0229 | 0.0050 |
| pickup_latitude | 0.0211 | 0.0218 | 0.0035 |
| dropoff_longitude | 0.0247 | 0.0249 | 0.0052 |
| dropoff_latitude | 0.0206 | 0.0221 | 0.0042 |
| trip_duration | 0.0963 | 0.0937 | 0.0165 |
| gender_id | 0.0453 | 0.0456 | 0.0091 |
| month | 16.2648 | 17.3539 | 0.0000 |

**A/A test (iid test-calibration check — both holdout and reference drawn at random from Jan-Mar; does NOT capture month-to-month variation within the baseline period):**

| Batch size | System false-alarm rate | pickup_longitude | pickup_latitude | dropoff_longitude | dropoff_latitude | trip_duration | gender_id | month |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.170 | 0.050 | 0.050 | 0.050 | 0.010 | 0.030 | 0.000 | 0.000 |
| 3000 | 0.160 | 0.040 | 0.040 | 0.040 | 0.010 | 0.030 | 0.000 | 0.000 |
| 5000 | 0.200 | 0.020 | 0.060 | 0.120 | 0.010 | 0.030 | 0.000 | 0.000 |
| 10000 | 0.090 | 0.010 | 0.020 | 0.040 | 0.010 | 0.010 | 0.000 | 0.000 |
| 15000 | 0.020 | 0.000 | 0.000 | 0.010 | 0.010 | 0.000 | 0.000 | 0.000 |
| 20000 | 0.020 | 0.000 | 0.000 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 |

Alpha-predicted system false-alarm rate for 7 uncorrected tests at alpha=0.05: `1-(1-0.05)^7` = **0.302**. Compare against the observed system false-alarm rates above.


## Reference size vs. batch size

- m=5000: asymptotic KS critical-value floor as n->infinity: c(alpha)/sqrt(m) = 1.36/sqrt(5000) = **0.0192**
- m=50000: asymptotic KS critical-value floor as n->infinity: c(alpha)/sqrt(m) = 1.36/sqrt(50000) = **0.0061**
- At n=20,000, m=5,000: c(alpha)*sqrt((n+m)/(nm)) = **0.0215**

**pickup_latitude recall by batch size, at both reference sizes (pooled, with month, legacy ground truth):**

| Batch size | Recall @ m=5000 | Recall @ m=50000 |
|---|---|---|
| 1000 | 0.167 | 0.250 |
| 3000 | 0.500 | 0.708 |
| 5000 | 0.667 | 0.833 |
| 10000 | 0.792 | 1.000 |
| 15000 | 0.833 | 1.000 |
| 20000 | 0.833 | 1.000 |
| 50000 | 0.792 | 1.000 |

**Reading this table against the theoretical floor above**: pickup_latitude's population D is 0.0254 (m=5000 ground truth) / 0.0211 (m=50000 ground truth) -- both very close to the m=5000 critical-value floor (0.0192). At m=5000, recall genuinely PLATEAUS around 0.79-0.83 even at the largest batch sizes tested (20,000 and 50,000) -- it never reaches 1.0, because the reference itself is too small for the test to reliably resolve an effect this close to its noise floor, no matter how much production data you throw at it. At m=50000 (floor=0.0061, well below the effect size), recall reaches 1.000 by batch size=10,000 and stays there. **This directly confirms the hypothesis: for features with population D near the small-reference floor, reference size -- not batch size -- is the binding constraint on detection power.** (The small dip at m=5000, batch=50000, from 0.833 to 0.792, is noise from only 24 pooled decision units at that cell, not a real reversal.)

**Important caveat this run surfaced, stated plainly**: the "population ground truth" itself is not reference-size-invariant. `dropoff_longitude` is labeled stable (p=0.079) under the m=5000 reference but drifted (p=1.5e-26) under the m=50000 reference -- same Apr-Jun production data, different reference samples. A 5,000-row reference estimates the true Jan-Mar population with its own sampling error, and since the reference is far smaller than the multi-hundred-thousand-row production pool it's compared against, most of the noise in a reference-vs-production KS statistic comes from the reference side. **Practical implication: for borderline-effect-size features, "ground truth" computed from a small reference is not a fixed, trustworthy target to grade the engine against -- it can disagree with itself depending on which reference happened to be drawn.** This is not a flaw in the engine; it's a property of comparing against a small reference sample, and it argues for the m=50000 ground truth being the more reliable one of the two computed here, not just for the engine's own detection power.

**Note on `month`'s per-batch std=0.0000** (table above): every row drawn from a single production month file has `month` equal to that one value (e.g. all `4`s from `production_month_04.csv`), so every batch's PSI for `month` is computed against the identical 100%-one-category production distribution -- there is no batch-to-batch variation to produce a nonzero std within this design. This is expected, not a bug.
