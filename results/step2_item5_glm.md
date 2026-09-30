# Step 2 review item 5 (revised): binomial GLM data-collapse analysis, Gate 1 only

Data from item 4's power-curve run: 42000 trials, 10 reference draws x 2 reference sizes x 5 features x 7 D_pop targets x 3 batch sizes x 20 trials.

**Outcome = `significant` (Gate 1) only.** Materiality is `effect_size >= floor` against a FIXED configured floor -- not a function of x, so asymptotic KS collapse theory makes no prediction about it; testing it here would be a category error (this corrects the prior version of this analysis, which incorrectly used `material` as the primary collapse target).

x = sqrt(n*m/(n+m)) * D_pop (achieved population D, exact weighted-KS, never the observed per-batch statistic). Logit link, weak L2 penalty C=1e6 in place of unpenalized MLE (avoids quasi-separation/unbounded coefficients in the near-deterministic tails).

**Cluster-robust inference**: trials sharing a reference draw are not independent -- they share that draw's own finite-sample deviation from the true population (demonstrated directly in `results/step2_aa_multidraw_report.md`: the same reference, tested against many clean batches, produces a correlated pattern of false alarms, not independent ones). Standard errors below are Huber-White sandwich estimates clustered by the physical reference draw (20 clusters: 10 draws x 2 reference sizes), with the standard small-cluster correction `(G/(G-1))*((N-1)/(N-K))`. Naive (IID-assumed) SEs are shown alongside for comparison -- they are anti-conservative here. **Caveat**: with G=20 clusters, cluster-robust SEs are themselves below the usual rule-of-thumb comfort zone (~30-50+ clusters) for reliable asymptotic coverage; treat these as indicative, not exact.

n=42000, positive rate (any feature Holm-significant at its own drift level)=0.8360

| Model | Covariates | k (params) | log-lik | AIC | BIC |
|---|---|---|---|---|---|
| M0 | x only | 2 | -8416.42 | 16836.84 | 16854.13 |
| M1 | x + feature | 6 | -8069.70 | 16151.39 | 16203.26 |
| M2 | x + feature + m | 7 | -8046.54 | 16107.09 | 16167.61 |

**Naive LRT M0->M1 (feature, beyond x)**: chi2=693.450, df=4, p=0.0000 -- IID-assumed, anti-conservative; see cluster-robust coefficient table below for the corrected read.

**Naive LRT M1->M2 (reference size, beyond x+feature)**: chi2=46.304, df=1, p=0.0000


### M2 coefficients: naive vs. cluster-robust SE (G=20 clusters)

| Covariate | Coef | Naive SE | Naive p | Cluster-robust SE | Cluster-robust p |
|---|---|---|---|---|---|
| intercept | -3.8096 | 0.0779 | 0 | 0.2338 | 0 |
| x | +3.8683 | 0.0531 | 0 | 0.1117 | 0 |
| feat_dropoff_longitude | -0.4850 | 0.0643 | 4.508e-14 | 0.2389 | 0.04231 |
| feat_pickup_latitude | -0.4023 | 0.0644 | 4.15e-10 | 0.1439 | 0.005172 |
| feat_pickup_longitude | -1.3825 | 0.0643 | 0 | 0.2696 | 2.925e-07 |
| feat_trip_duration | -1.2248 | 0.0641 | 0 | 0.3022 | 5.055e-05 |
| m_50000 | -0.2742 | 0.0403 | 1.068e-11 | 0.1784 | 0.1243 |

The reference-size coefficient (-0.2742 for m=50000 vs. the m=5000 baseline) is NO LONGER significant at the conventional 0.05 level once clustering is accounted for (naive p=1.07e-11, cluster-robust p=0.124) -- the naive fit's apparent significance was driven by treating 42,000 correlated observations (20 clusters) as if they were 42,000 independent ones. Cluster-robust SE (0.1784) vs. naive SE (0.0403), a 4.4x inflation -- the anti-conservative-naive-SE pattern expected once shared-draw correlation is accounted for. Given this, the predicted-probability-difference analysis below (not the p-value) is the number to actually rely on for judging whether this effect matters in practice.


## Predicted-probability differences (the headline result, not p-values)

**Max gap between features at matched x** (model M1, pooled across reference sizes): 0.331 (i.e. 33.1 percentage points), at x=1.203, between `pickup_longitude` (p=0.344) and `dropoff_latitude` (p=0.675). This is the practical size of the earlier-flagged 'feature effect' -- at the SAME x, the least- and most-detectable features differ by up to about 33 points of detection probability in the transition region; away from the transition (x far from ~1) all features are pinned near 0 or 1 regardless, so the gap there is necessarily small.

**Max gap between reference sizes at matched x** (model M2, baseline feature `dropoff_latitude`): 0.068 (6.8 points), at x=1.032 -- m=5000: p=0.545, m=50000: p=0.477. **This is what the raw coefficient of -0.274 actually means in practice**: at the x where the gap is largest (near the transition, where both curves are most sensitive to a log-odds shift), a batch from the m=5000 reference is about 7 percentage points more likely to register as Gate-1-significant than an equivalent-x batch from the m=50000 reference. Away from the transition region the gap shrinks toward 0 in probability terms even though the log-odds shift (-0.274) is constant -- the same log-odds coefficient means much less in probability terms when either curve is already saturated near 0 or 1. Given the cluster-robust p-value on this coefficient (0.124), treat this 7-point gap as the practical upper bound on the reference-size effect observed in this data, not as a precisely-estimated, statistically ironclad one.
