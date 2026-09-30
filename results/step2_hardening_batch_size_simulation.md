# Hardening pass item 4: recommended_batch_size redefinition, simulation verification

Method: reference ~ Uniform(0,1) (m samples), batch ~ Uniform(D, 1+D) (n samples) -- the population KS distance between these two is EXACTLY D (for D<1, a closed-form result, not an approximation), so 'true D' is controlled exactly. 2000 independent trials per cell, floor=0.05, seed=42.


## m=5000: new n (recommended_batch_size) = 7252, old n (min_batch_size_at_floor) = 869


### n=7252 (new definition)

- true D=0.0: mean(D_obs)=0.0158, std=0.0048, median=0.0151, P(D_obs>0.05)=0.0000 (0/2000, 95% CI 0.0000-0.0018)
- true D=0.02: mean(D_obs)=0.0313, std=0.0060, median=0.0307, P(D_obs>0.05)=0.0060 (12/2000, 95% CI 0.0031-0.0105)
- true D=0.05: mean(D_obs)=0.0613, std=0.0060, median=0.0607, P(D_obs>0.05)=0.9930 (1986/2000, 95% CI 0.9883-0.9962)

### n=869 (old definition)

- true D=0.0: mean(D_obs)=0.0322, std=0.0099, median=0.0307, P(D_obs>0.05)=0.0545 (109/2000, 95% CI 0.0450-0.0654)
- true D=0.02: mean(D_obs)=0.0432, std=0.0118, median=0.0417, P(D_obs>0.05)=0.2615 (523/2000, 95% CI 0.2424-0.2814)
- true D=0.05: mean(D_obs)=0.0728, std=0.0121, median=0.0720, P(D_obs>0.05)=0.9930 (1986/2000, 95% CI 0.9883-0.9962)

## m=50000: new n (recommended_batch_size) = 3146, old n (min_batch_size_at_floor) = 751


### n=3146 (new definition)

- true D=0.0: mean(D_obs)=0.0159, std=0.0047, median=0.0153, P(D_obs>0.05)=0.0000 (0/2000, 95% CI 0.0000-0.0018)
- true D=0.02: mean(D_obs)=0.0315, std=0.0060, median=0.0309, P(D_obs>0.05)=0.0050 (10/2000, 95% CI 0.0024-0.0092)
- true D=0.05: mean(D_obs)=0.0615, std=0.0060, median=0.0609, P(D_obs>0.05)=0.9980 (1996/2000, 95% CI 0.9949-0.9995)

### n=751 (old definition)

- true D=0.0: mean(D_obs)=0.0315, std=0.0094, median=0.0300, P(D_obs>0.05)=0.0465 (93/2000, 95% CI 0.0377-0.0567)
- true D=0.02: mean(D_obs)=0.0430, std=0.0116, median=0.0415, P(D_obs>0.05)=0.2625 (525/2000, 95% CI 0.2433-0.2824)
- true D=0.05: mean(D_obs)=0.0722, std=0.0123, median=0.0707, P(D_obs>0.05)=0.9970 (1994/2000, 95% CI 0.9935-0.9989)

**Runtime**: 122.2s for 24000 total ks_2samp calls.
