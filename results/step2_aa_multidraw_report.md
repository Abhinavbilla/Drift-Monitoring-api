# Step 2 evidence fix 1: A/A over multiple independent reference draws

1200 trials: [5000, 50000] reference sizes x 10 independent, mutually disjoint reference draws (reused from item 4's power-curve run, not refit) x batch sizes [1000, 5000, 20000] x 20 independently-drawn clean (untilted) batches per (ref_size, draw, n) cell.

Gate-1-only = any feature Holm-significant. Gate-2-only = any feature material. Combined = any feature drift_detected (both gates). All three computed per draw, then averaged across the 10 draws, with the between-draw standard deviation reported alongside.


## ref_size=5000, batch_size(n)=1000

| Draw | Gate-1-only k/N (rate, 95% CI) | Gate-2-only k/N (rate, 95% CI) | Combined k/N (rate, 95% CI) |
|---|---|---|---|
| 0 | 1/20 (0.050, 0.001-0.249) | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) |
| 1 | 0/20 (0.000, 0.000-0.168) | 4/20 (0.200, 0.057-0.437) | 0/20 (0.000, 0.000-0.168) |
| 2 | 0/20 (0.000, 0.000-0.168) | 8/20 (0.400, 0.191-0.639) | 0/20 (0.000, 0.000-0.168) |
| 3 | 3/20 (0.150, 0.032-0.379) | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) |
| 4 | 2/20 (0.100, 0.012-0.317) | 2/20 (0.100, 0.012-0.317) | 1/20 (0.050, 0.001-0.249) |
| 5 | 1/20 (0.050, 0.001-0.249) | 3/20 (0.150, 0.032-0.379) | 1/20 (0.050, 0.001-0.249) |
| 6 | 0/20 (0.000, 0.000-0.168) | 2/20 (0.100, 0.012-0.317) | 0/20 (0.000, 0.000-0.168) |
| 7 | 1/20 (0.050, 0.001-0.249) | 3/20 (0.150, 0.032-0.379) | 0/20 (0.000, 0.000-0.168) |
| 8 | 0/20 (0.000, 0.000-0.168) | 2/20 (0.100, 0.012-0.317) | 0/20 (0.000, 0.000-0.168) |
| 9 | 1/20 (0.050, 0.001-0.249) | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) |

**Averaged over 10 draws**: Gate-1-only mean=0.045 (between-draw std=0.050, range [0.000, 0.150]), pooled 9/200 (0.021-0.084). Gate-2-only mean=0.135 (std=0.106). Combined mean=0.010 (std=0.021).


## ref_size=5000, batch_size(n)=5000

| Draw | Gate-1-only k/N (rate, 95% CI) | Gate-2-only k/N (rate, 95% CI) | Combined k/N (rate, 95% CI) |
|---|---|---|---|
| 0 | 4/20 (0.200, 0.057-0.437) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 1 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 2 | 6/20 (0.300, 0.119-0.543) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 3 | 10/20 (0.500, 0.272-0.728) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 4 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 5 | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 6 | 6/20 (0.300, 0.119-0.543) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 7 | 9/20 (0.450, 0.231-0.685) | 1/20 (0.050, 0.001-0.249) | 1/20 (0.050, 0.001-0.249) |
| 8 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 9 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |

**Averaged over 10 draws**: Gate-1-only mean=0.180 (between-draw std=0.197, range [0.000, 0.500]), pooled 36/200 (0.129-0.240). Gate-2-only mean=0.005 (std=0.016). Combined mean=0.005 (std=0.016).


## ref_size=5000, batch_size(n)=20000

| Draw | Gate-1-only k/N (rate, 95% CI) | Gate-2-only k/N (rate, 95% CI) | Combined k/N (rate, 95% CI) |
|---|---|---|---|
| 0 | 20/20 (1.000, 0.832-1.000) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 1 | 20/20 (1.000, 0.832-1.000) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 2 | 11/20 (0.550, 0.315-0.769) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 3 | 20/20 (1.000, 0.832-1.000) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 4 | 7/20 (0.350, 0.154-0.592) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 5 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 6 | 4/20 (0.200, 0.057-0.437) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 7 | 20/20 (1.000, 0.832-1.000) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 8 | 13/20 (0.650, 0.408-0.846) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 9 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |

**Averaged over 10 draws**: Gate-1-only mean=0.575 (between-draw std=0.419, range [0.000, 1.000]), pooled 115/200 (0.503-0.644). Gate-2-only mean=0.000 (std=0.000). Combined mean=0.000 (std=0.000).


## ref_size=50000, batch_size(n)=1000

| Draw | Gate-1-only k/N (rate, 95% CI) | Gate-2-only k/N (rate, 95% CI) | Combined k/N (rate, 95% CI) |
|---|---|---|---|
| 0 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 1 | 1/20 (0.050, 0.001-0.249) | 1/20 (0.050, 0.001-0.249) | 1/20 (0.050, 0.001-0.249) |
| 2 | 1/20 (0.050, 0.001-0.249) | 2/20 (0.100, 0.012-0.317) | 1/20 (0.050, 0.001-0.249) |
| 3 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 4 | 2/20 (0.100, 0.012-0.317) | 1/20 (0.050, 0.001-0.249) | 1/20 (0.050, 0.001-0.249) |
| 5 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 6 | 1/20 (0.050, 0.001-0.249) | 2/20 (0.100, 0.012-0.317) | 1/20 (0.050, 0.001-0.249) |
| 7 | 1/20 (0.050, 0.001-0.249) | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) |
| 8 | 1/20 (0.050, 0.001-0.249) | 1/20 (0.050, 0.001-0.249) | 1/20 (0.050, 0.001-0.249) |
| 9 | 1/20 (0.050, 0.001-0.249) | 2/20 (0.100, 0.012-0.317) | 0/20 (0.000, 0.000-0.168) |

**Averaged over 10 draws**: Gate-1-only mean=0.040 (between-draw std=0.032, range [0.000, 0.100]), pooled 8/200 (0.017-0.077). Gate-2-only mean=0.050 (std=0.041). Combined mean=0.025 (std=0.026).


## ref_size=50000, batch_size(n)=5000

| Draw | Gate-1-only k/N (rate, 95% CI) | Gate-2-only k/N (rate, 95% CI) | Combined k/N (rate, 95% CI) |
|---|---|---|---|
| 0 | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 1 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 2 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 3 | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 4 | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 5 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 6 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 7 | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 8 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 9 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |

**Averaged over 10 draws**: Gate-1-only mean=0.020 (between-draw std=0.026, range [0.000, 0.050]), pooled 4/200 (0.005-0.050). Gate-2-only mean=0.000 (std=0.000). Combined mean=0.000 (std=0.000).


## ref_size=50000, batch_size(n)=20000

| Draw | Gate-1-only k/N (rate, 95% CI) | Gate-2-only k/N (rate, 95% CI) | Combined k/N (rate, 95% CI) |
|---|---|---|---|
| 0 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 1 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 2 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 3 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 4 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 5 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 6 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 7 | 1/20 (0.050, 0.001-0.249) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 8 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |
| 9 | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) | 0/20 (0.000, 0.000-0.168) |

**Averaged over 10 draws**: Gate-1-only mean=0.005 (between-draw std=0.016, range [0.000, 0.050]), pooled 1/200 (0.000-0.028). Gate-2-only mean=0.000 (std=0.000). Combined mean=0.000 (std=0.000).


## Explaining the earlier single-draw figure (Holm=on, Floor=off, ref=5000, batch=10000: rate=0.430)

That earlier number came from `scripts/step2_analyze.py`'s 2x2 ablation, itself computed from the ORIGINAL single-reference-draw calibrated run (`step2_val_ref5000_calibrated`, one fixed m=5000 reference). n=10000 sits between the two batch sizes tested here; at n=5000 this multi-draw run's Gate-1-only rate averages 0.180 (between-draw std 0.197, draws ranging [0.000, 0.500]), and at n=20000 it averages 0.575 (std 0.419, range [0.000, 1.000]) -- 0.430 at n=10000 sits plausibly on the trend between these two, but which specific single draw you'd land on could easily read anywhere across that between-draw range, not just the trend's middle.

**Why the rate is elevated at all, and why it grows with n**: at m=5000, the reference's OWN empirical CDF differs from the true population CDF by a random amount whose typical size is on the order of `c(alpha)/sqrt(m)` = 0.0192 (exactly the `minimum_detectable_d_at_fit_time` quantity from item 6). This is NOT measurement error that averages out across A/A trials -- it is a FIXED property of that one reference draw, shared by every trial tested against it. A larger batch (bigger n) makes the KS test MORE powerful, so it becomes increasingly likely to detect even this small, fixed, reference-specific deviation as 'significant' -- which is exactly why the ORIGINAL single-draw Gate-1-only (Holm=on, Floor=off) rate climbed steeply with n at ref_size=5000 (`step2_side_by_side.md`'s ablation: 0.03 at n=1000, 0.12 at n=3000, 0.21 at n=5000, 0.43 at n=10000, 0.81 at n=15000, 1.00 at n=20000) even though the TRUE population-level null (batch and reference drawn from the identical distribution) should keep the Holm-corrected false-alarm rate near alpha regardless of n. Averaging over independent draws (this run) is what actually estimates the system's true expected behavior; a single draw estimates one (possibly unlucky) realization of it -- and this run's own n=20000 average, 0.575 (between-draw range [0.000, 1.000]), shows the ORIGINAL single draw landed within that range rather than being a uniquely broken sample -- the instability is structural (a property of the design at small m, growing with n), not a one-off fluke of that particular reference.

**Implication for design**: for a stable false-alarm rate, the materiality floor must be set at least around `c(alpha)/sqrt(m)` -- 0.0192 at m=5,000, 0.0061 at m=50,000 -- i.e. at least as large as the reference's own inherent finite-sample noise. Below that bound, Gate 2 cannot reliably distinguish a real drift effect from the reference's own sampling error, and Gate 1 alone becomes MORE likely to false-alarm on that noise as batch size n grows (since higher n means higher power to detect even a tiny, spurious, reference-specific deviation). The locked default floor (0.05) sits comfortably above both bounds tested here (0.0192 and 0.0061), which is exactly why this project's combined (two-gate) A/A rate stays near 0 in every table above and in the original single-draw run alike -- but a user who configures a smaller floor for a smaller reference (m below roughly (c(alpha)/floor)^2) would lose this protection and should be warned; `reference_too_small_for_floor` (item 6) already flags exactly this condition at fit time.
