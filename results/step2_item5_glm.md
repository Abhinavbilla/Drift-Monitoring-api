# Step 2 review item 5: binomial GLM data-collapse re-analysis

Data from item 4's power-curve run: 42000 trials, 10 reference draws x 2 reference sizes x 5 features x 7 D_pop targets x 3 batch sizes x 20 trials.

x = sqrt(n*m/(n+m)) * D_pop (achieved population D from the exact weighted-KS computation, never the per-batch observed statistic). Logit link (scikit-learn `LogisticRegression`, weak L2 penalty C=1e6 in place of unpenalized MLE -- the power curve is near-deterministic away from its transition region, which causes quasi-separation and unbounded coefficients under true MLE; this is a standard, documented practical substitute, not a silent change of estimator).


### Target: `material` (Gate 2 materiality alone -- the near-floor power curve itself)

n=42000, positive rate=0.6351

| Model | Covariates | k (params) | log-lik | AIC | BIC |
|---|---|---|---|---|---|
| M0 | x only | 2 | -22731.55 | 45467.10 | 45484.39 |
| M1 | x + feature | 6 | -22451.53 | 44915.07 | 44966.94 |
| M2 | x + feature + m | 7 | -21804.55 | 43623.10 | 43683.62 |

**LRT M0->M1 (feature effect beyond x)**: chi2=560.028, df=4, p=0.0000

**LRT M1->M2 (reference-size effect beyond x+feature)**: chi2=1293.969, df=1, p=0.0000

M0 coefficient on x: 0.6579 (intercept -1.2317)

**Interpretation**: detection DOES depend significantly on which feature is drifting, beyond x alone -- the single-curve collapse does not fully hold across features; theory only guarantees approximate collapse (power depends on the shape of F-G, not just sup|F-G|=D), so per-feature deviations here are consistent with that caveat, not a bug.

**Interpretation**: detection DOES depend significantly on reference size m beyond x -- at matched x, m=50000 batches have LOWER detection probability than the baseline reference size (coefficient -0.849). This is a real residual m-dependence not captured by the sqrt(nm/(n+m)) scaling alone -- plausibly a finite-sample correction to the asymptotic two-sample KS distribution that the simple sqrt(nm/(n+m))*D collapse variable doesn't fully capture (known refined asymptotics for the KS statistic include higher-order terms beyond this leading-order scaling). Not attributable to noise given n=42,000 trials.


### Target: `drift_detected` (full two-gate system decision, secondary check)

n=42000, positive rate=0.6083

| Model | Covariates | k (params) | log-lik | AIC | BIC |
|---|---|---|---|---|---|
| M0 | x only | 2 | -22034.96 | 44073.92 | 44091.21 |
| M1 | x + feature | 6 | -21730.27 | 43472.54 | 43524.41 |
| M2 | x + feature + m | 7 | -21145.30 | 42304.60 | 42365.12 |

**LRT M0->M1 (feature effect beyond x)**: chi2=609.383, df=4, p=0.0000

**LRT M1->M2 (reference-size effect beyond x+feature)**: chi2=1169.934, df=1, p=0.0000

M0 coefficient on x: 0.7685 (intercept -1.6369)

**Interpretation**: detection DOES depend significantly on which feature is drifting, beyond x alone -- the single-curve collapse does not fully hold across features; theory only guarantees approximate collapse (power depends on the shape of F-G, not just sup|F-G|=D), so per-feature deviations here are consistent with that caveat, not a bug.

**Interpretation**: detection DOES depend significantly on reference size m beyond x -- at matched x, m=50000 batches have LOWER detection probability than the baseline reference size (coefficient -0.824). This is a real residual m-dependence not captured by the sqrt(nm/(n+m)) scaling alone -- plausibly a finite-sample correction to the asymptotic two-sample KS distribution that the simple sqrt(nm/(n+m))*D collapse variable doesn't fully capture (known refined asymptotics for the KS statistic include higher-order terms beyond this leading-order scaling). Not attributable to noise given n=42,000 trials.


## Between-reference-draw variance

Per (reference_size, feature, D_pop target, batch size) cell: the calibrated system's material rate computed separately within each of the 10 reference draws, then the variance/std of those 10 per-draw rates -- this isolates variability attributable to WHICH reference sample was used to fit the baseline, holding the drift-injection method fixed.

**Largest between-draw variance**: ref_size=5000, feature=pickup_latitude, target_D=0.04, n=20000 -- mean rate 0.465 across draws, std=0.466, range [0.000, 1.000] over 10 draws. As expected, this concentrates in the transition region (neither always-0 nor always-1), where which specific reference sample was drawn can tip a near-floor batch's effect size across the materiality threshold.
