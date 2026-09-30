"""
Step 2 review item 5 (2026-09-30 instruction, revised 2026-09-30 per
evidence-fix review): binomial GLM data-collapse analysis, using item 4's
multi-reference-draw run (results/tabular_validation_item4_power_curve_raw.json).

REVISION (evidence fix 2): the collapse test now uses ONLY the Gate-1
(`significant`) outcome. Materiality (Gate 2) is `effect_size >= floor`
against a FIXED, configured floor -- it is not a function of
x=sqrt(nm/(n+m))*D_pop at all (a given x can correspond to many different
(n, m, D_pop) combinations with different observed effect sizes relative
to the same fixed floor), so asymptotic KS collapse theory makes no
prediction about it and testing it for x-collapse is a category error.
Only significance (a p-value threshold on the KS statistic) is the
quantity asymptotic KS theory actually describes as a function of x.

x = sqrt(n*m/(n+m)) * D_pop (achieved population D via exact weighted-KS,
never the observed per-batch statistic).

Two nested logistic models (logit link; scikit-learn LogisticRegression,
weak L2 penalty C=1e6 in place of unpenalized MLE -- avoids quasi-
separation/unbounded coefficients in the near-deterministic tails;
statsmodels is not installed and this project doesn't add dependencies
without sign-off):
    M1: significant ~ x + feature (5 dummies)
    M2: significant ~ x + feature + reference_size (m, 2 levels)

REVISION (evidence fix 2): trials sharing a reference draw are NOT
independent (they share that draw's own sampling error -- see
scripts/step2_aa_multidraw.py), so naive (IID-assumed) standard errors
and LRT p-values are anti-conservative. Cluster-robust (Huber-White
sandwich) standard errors, clustered by the physical reference draw (20
clusters: 10 draws x 2 reference sizes), are computed by hand (no
statsmodels) and reported alongside the naive ones. With only 20
clusters, cluster-robust SEs are themselves not fully reliable by the
usual rule of thumb (~30-50+ clusters wanted) -- flagged explicitly, not
silently trusted.

REVISION (evidence fix 2): rather than leaning on p-values (which are
close to meaningless as an effect-size signal at n=42,000 regardless of
clustering), the headline results are PREDICTED-PROBABILITY differences:
the max gap between features' fitted curves at matched x, and the max gap
between reference sizes' fitted curves at matched x -- reported in actual
probability units, translating the raw log-odds coefficients (e.g. the
m=50000 vs m=5000 coefficient of about -0.85) into what they mean in
practice.

Run:
    python scripts/step2_item5_glm.py
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

IN_JSON = "results/tabular_validation_item4_power_curve_raw.json"
OUT_JSON = "results/step2_item5_glm.json"
OUT_MD = "results/step2_item5_glm.md"

N_CLUSTERS_EXPECTED = 20  # 10 draws x 2 reference sizes


def log_likelihood(y, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit(X, y):
    """X excludes the intercept column; sklearn fits its own intercept."""
    model = LogisticRegression(penalty="l2", C=1e6, max_iter=5000, solver="lbfgs")
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    ll = log_likelihood(y, p)
    k = X.shape[1] + 1
    aic = 2 * k - 2 * ll
    bic = k * math.log(len(y)) - 2 * ll
    return model, ll, k, aic, bic, p


def cluster_robust_se(X_with_intercept, y, p, coef_full, cluster_ids):
    """Huber-White sandwich covariance, clustered by cluster_ids, around a
    (weakly L2-regularized) point estimate -- the penalty term is O(1e-6)
    relative to the Fisher information here and is ignored in the bread
    matrix, a standard practical approximation. coef_full includes the
    intercept as its first entry, matching X_with_intercept's first column."""
    W = p * (1 - p)
    XtWX = (X_with_intercept * W[:, None]).T @ X_with_intercept
    bread = np.linalg.inv(XtWX)

    resid = y - p
    scores = X_with_intercept * resid[:, None]  # per-row score contribution

    clusters = pd.unique(cluster_ids)
    G = len(clusters)
    K = X_with_intercept.shape[1]
    N = len(y)
    meat = np.zeros((K, K))
    for c in clusters:
        mask = cluster_ids == c
        s_c = scores[mask].sum(axis=0)
        meat += np.outer(s_c, s_c)

    correction = (G / (G - 1)) * ((N - 1) / (N - K))
    V = correction * bread @ meat @ bread
    se = np.sqrt(np.diag(V))
    z = coef_full / se
    p_values = 2 * (1 - norm.cdf(np.abs(z)))
    return se, z, p_values, G


def lrt(ll_reduced, k_reduced, ll_full, k_full):
    stat = 2 * (ll_full - ll_reduced)
    df = k_full - k_reduced
    p = float(chi2.sf(stat, df)) if df > 0 else float("nan")
    return {"lr_stat": float(stat), "df": df, "p_value": p}


def build_design(df, include_m=False):
    dummies = pd.get_dummies(df["feature"], prefix="feat", drop_first=True)
    feat_names = dummies.columns.tolist()
    cols = [df[["x"]].values, dummies.values.astype(float)]
    names = ["x"] + feat_names
    if include_m:
        m_dummy = pd.get_dummies(df["ref_size"], prefix="m", drop_first=True)
        cols.append(m_dummy.values.astype(float))
        names.extend(m_dummy.columns.tolist())
    X = np.hstack(cols)
    return X, names, dummies.columns.tolist()


def main():
    raw = json.load(open(IN_JSON, encoding="utf-8"))
    records = raw["raw_records"]
    df = pd.DataFrame.from_records(records)
    df["significant"] = df["significant"].astype(bool)
    df["x"] = np.sqrt(df["n"] * df["ref_size"] / (df["n"] + df["ref_size"])) * df["achieved_d"]
    df["cluster_id"] = df["ref_size"].astype(str) + "_" + df["draw"].astype(str)

    notes = []
    notes.append("# Step 2 review item 5 (revised): binomial GLM data-collapse analysis, Gate 1 only\n")
    notes.append(
        f"Data from item 4's power-curve run: {len(df)} trials, "
        f"{raw['config']['n_reference_draws']} reference draws x {len(raw['config']['reference_sizes'])} "
        f"reference sizes x {len(raw['config']['continuous_features'])} features x "
        f"{len(raw['config']['d_pop_targets'])} D_pop targets x {len(raw['config']['batch_sizes'])} "
        f"batch sizes x {raw['config']['n_trials']} trials.\n"
    )
    notes.append(
        "**Outcome = `significant` (Gate 1) only.** Materiality is `effect_size >= floor` against a "
        "FIXED configured floor -- not a function of x, so asymptotic KS collapse theory makes no "
        "prediction about it; testing it here would be a category error (this corrects the prior "
        "version of this analysis, which incorrectly used `material` as the primary collapse target).\n"
    )
    notes.append(
        "x = sqrt(n*m/(n+m)) * D_pop (achieved population D, exact weighted-KS, never the observed "
        "per-batch statistic). Logit link, weak L2 penalty C=1e6 in place of unpenalized MLE (avoids "
        "quasi-separation/unbounded coefficients in the near-deterministic tails).\n"
    )
    notes.append(
        "**Cluster-robust inference**: trials sharing a reference draw are not independent -- they "
        "share that draw's own finite-sample deviation from the true population (demonstrated "
        "directly in `results/step2_aa_multidraw_report.md`: the same reference, tested against many "
        "clean batches, produces a correlated pattern of false alarms, not independent ones). Standard "
        "errors below are Huber-White sandwich estimates clustered by the physical reference draw (20 "
        "clusters: 10 draws x 2 reference sizes), with the standard small-cluster correction "
        "`(G/(G-1))*((N-1)/(N-K))`. Naive (IID-assumed) SEs are shown alongside for comparison -- they "
        "are anti-conservative here. **Caveat**: with G=20 clusters, cluster-robust SEs are themselves "
        "below the usual rule-of-thumb comfort zone (~30-50+ clusters) for reliable asymptotic coverage; "
        "treat these as indicative, not exact.\n"
    )

    y = df["significant"].astype(int).values
    notes.append(f"n={len(df)}, positive rate (any feature Holm-significant at its own drift level)={y.mean():.4f}\n")

    X1, names1, feat_names = build_design(df, include_m=False)
    m1, ll1, k1, aic1, bic1, p1 = fit_logit(X1, y)
    X2, names2, _ = build_design(df, include_m=True)
    m2, ll2, k2, aic2, bic2, p2 = fit_logit(X2, y)

    X0 = df[["x"]].values
    model0 = LogisticRegression(penalty="l2", C=1e6, max_iter=5000, solver="lbfgs")
    model0.fit(X0, y)
    p0 = model0.predict_proba(X0)[:, 1]
    ll0 = log_likelihood(y, p0)
    k0 = 2

    notes.append("| Model | Covariates | k (params) | log-lik | AIC | BIC |\n|---|---|---|---|---|---|")
    notes.append(f"| M0 | x only | {k0} | {ll0:.2f} | {2*k0-2*ll0:.2f} | {k0*math.log(len(y))-2*ll0:.2f} |")
    notes.append(f"| M1 | x + feature | {k1} | {ll1:.2f} | {aic1:.2f} | {bic1:.2f} |")
    notes.append(f"| M2 | x + feature + m | {k2} | {ll2:.2f} | {aic2:.2f} | {bic2:.2f} |")

    lrt_01 = lrt(ll0, k0, ll1, k1)
    lrt_12 = lrt(ll1, k1, ll2, k2)
    notes.append(f"\n**Naive LRT M0->M1 (feature, beyond x)**: chi2={lrt_01['lr_stat']:.3f}, "
                 f"df={lrt_01['df']}, p={lrt_01['p_value']:.4f} -- IID-assumed, anti-conservative; "
                 f"see cluster-robust coefficient table below for the corrected read.")
    notes.append(f"\n**Naive LRT M1->M2 (reference size, beyond x+feature)**: chi2={lrt_12['lr_stat']:.3f}, "
                 f"df={lrt_12['df']}, p={lrt_12['p_value']:.4f}\n")

    # ---------- cluster-robust SE for M2 (the full model) ----------
    X2_full = np.hstack([np.ones((len(y), 1)), X2])
    coef_full = np.concatenate([[m2.intercept_[0]], m2.coef_[0]])
    se_robust, z_robust, p_robust, G = cluster_robust_se(X2_full, y, p2, coef_full, df["cluster_id"].values)
    se_naive_approx = np.sqrt(np.diag(np.linalg.inv((X2_full * (p2 * (1 - p2))[:, None]).T @ X2_full)))

    all_names = ["intercept"] + names2
    notes.append(f"\n### M2 coefficients: naive vs. cluster-robust SE (G={G} clusters)\n")
    notes.append("| Covariate | Coef | Naive SE | Naive p | Cluster-robust SE | Cluster-robust p |\n"
                 "|---|---|---|---|---|---|")
    for i, name in enumerate(all_names):
        notes.append(f"| {name} | {coef_full[i]:+.4f} | {se_naive_approx[i]:.4f} | "
                     f"{2*(1-norm.cdf(abs(coef_full[i]/se_naive_approx[i]))):.4g} | "
                     f"{se_robust[i]:.4f} | {p_robust[i]:.4g} |")

    m_idx = all_names.index([n for n in all_names if n.startswith("m_")][0])
    m_level = all_names[m_idx].replace("m_", "")
    m_coef = coef_full[m_idx]
    still_significant = p_robust[m_idx] < 0.05
    se_inflation = se_robust[m_idx] / se_naive_approx[m_idx]
    if still_significant:
        sig_note = (f"remains significant under the cluster-robust SE (p={p_robust[m_idx]:.4g})"
                    f" despite the {se_inflation:.1f}x larger standard error")
    else:
        sig_note = (f"is NO LONGER significant at the conventional 0.05 level once clustering is "
                    f"accounted for (naive p={2*(1-norm.cdf(abs(m_coef/se_naive_approx[m_idx]))):.3g}, "
                    f"cluster-robust p={p_robust[m_idx]:.3g}) -- the naive fit's apparent significance "
                    f"was driven by treating 42,000 correlated observations (20 clusters) as if they "
                    f"were 42,000 independent ones")
    notes.append(
        f"\nThe reference-size coefficient ({m_coef:+.4f} for m={m_level} vs. the m=5000 baseline) "
        f"{sig_note}. Cluster-robust SE ({se_robust[m_idx]:.4f}) vs. naive SE "
        f"({se_naive_approx[m_idx]:.4f}), a {se_inflation:.1f}x inflation -- the anti-conservative-naive-SE "
        f"pattern expected once shared-draw correlation is accounted for. Given this, the predicted-"
        f"probability-difference analysis below (not the p-value) is the number to actually rely on for "
        f"judging whether this effect matters in practice.\n"
    )

    # ---------- predicted-probability differences ----------
    notes.append("\n## Predicted-probability differences (the headline result, not p-values)\n")
    x_grid = np.linspace(df["x"].min(), df["x"].max(), 400)

    # Max gap between features, at matched x, using M1 (no m -- pooled across reference sizes)
    intercept1 = m1.intercept_[0]
    coef1 = m1.coef_[0]
    x_coef1 = coef1[0]
    feat_coefs1 = {name.replace("feat_", "", 1): coef for name, coef in zip(feat_names, coef1[1:])}
    all_features_sorted = sorted(set(df["feature"]))
    stripped_feat_names = set(feat_coefs1.keys())
    baseline_feature = [f for f in all_features_sorted if f not in stripped_feat_names][0]

    max_feat_gap, max_feat_gap_x, max_feat_pair = -1, None, None
    for x in x_grid:
        probs = {}
        for feat in all_features_sorted:
            logit_val = intercept1 + x_coef1 * x + (feat_coefs1.get(feat, 0.0) if feat != baseline_feature else 0.0)
            probs[feat] = sigmoid(logit_val)
        lo_feat = min(probs, key=probs.get)
        hi_feat = max(probs, key=probs.get)
        gap = probs[hi_feat] - probs[lo_feat]
        if gap > max_feat_gap:
            max_feat_gap, max_feat_gap_x, max_feat_pair = gap, x, (lo_feat, hi_feat, probs[lo_feat], probs[hi_feat])

    notes.append(
        f"**Max gap between features at matched x** (model M1, pooled across reference sizes): "
        f"{max_feat_gap:.3f} (i.e. {max_feat_gap*100:.1f} percentage points), at x={max_feat_gap_x:.3f}, "
        f"between `{max_feat_pair[0]}` (p={max_feat_pair[2]:.3f}) and `{max_feat_pair[1]}` "
        f"(p={max_feat_pair[3]:.3f}). This is the practical size of the earlier-flagged 'feature effect' "
        f"-- at the SAME x, the least- and most-detectable features differ by up to about "
        f"{max_feat_gap*100:.0f} points of detection probability in the transition region; away from "
        f"the transition (x far from ~1) all features are pinned near 0 or 1 regardless, so the gap "
        f"there is necessarily small.\n"
    )

    # Max gap between reference sizes, at matched x, using M2 (baseline feature)
    intercept2 = m2.intercept_[0]
    coef2 = m2.coef_[0]
    x_coef2 = coef2[0]
    m_coef2 = coef2[names2.index([n for n in names2 if n.startswith("m_")][0])]

    max_m_gap, max_m_gap_x, max_m_probs = -1, None, None
    for x in x_grid:
        logit_5000 = intercept2 + x_coef2 * x
        logit_other = logit_5000 + m_coef2
        p_5000, p_other = sigmoid(logit_5000), sigmoid(logit_other)
        gap = abs(p_5000 - p_other)
        if gap > max_m_gap:
            max_m_gap, max_m_gap_x, max_m_probs = gap, x, (p_5000, p_other)

    notes.append(
        f"**Max gap between reference sizes at matched x** (model M2, baseline feature "
        f"`{baseline_feature}`): {max_m_gap:.3f} ({max_m_gap*100:.1f} points), at x={max_m_gap_x:.3f} "
        f"-- m=5000: p={max_m_probs[0]:.3f}, m={m_level}: p={max_m_probs[1]:.3f}. **This is what the "
        f"raw coefficient of {m_coef2:+.3f} actually means in practice**: at the x where the gap is "
        f"largest (near the transition, where both curves are most sensitive to a log-odds shift), a "
        f"batch from the m=5000 reference is about {max_m_gap*100:.0f} percentage points more likely "
        f"to register as Gate-1-significant than an equivalent-x batch from the m={m_level} reference. "
        f"Away from the transition region the gap shrinks toward 0 in probability terms even though the "
        f"log-odds shift ({m_coef2:+.3f}) is constant -- the same log-odds coefficient means much less "
        f"in probability terms when either curve is already saturated near 0 or 1. Given the "
        f"cluster-robust p-value on this coefficient ({p_robust[m_idx]:.3g}), treat this "
        f"{max_m_gap*100:.0f}-point gap as the practical upper bound on the reference-size effect "
        f"observed in this data, not as a precisely-estimated, statistically ironclad one.\n"
    )

    result = {
        "n": len(df), "positive_rate": float(y.mean()),
        "models": {
            "M0": {"k": k0, "log_lik": ll0},
            "M1": {"covariates": names1, "k": k1, "log_lik": ll1, "aic": aic1, "bic": bic1,
                   "coef": m1.coef_[0].tolist(), "intercept": float(m1.intercept_[0])},
            "M2": {"covariates": names2, "k": k2, "log_lik": ll2, "aic": aic2, "bic": bic2,
                   "coef": m2.coef_[0].tolist(), "intercept": float(m2.intercept_[0]),
                   "cluster_robust_se": se_robust.tolist(), "cluster_robust_p": p_robust.tolist(),
                   "naive_se": se_naive_approx.tolist(), "n_clusters": int(G)},
        },
        "lrt_feature_naive": lrt_01, "lrt_reference_size_naive": lrt_12,
        "max_feature_gap": {"gap": float(max_feat_gap), "x": float(max_feat_gap_x),
                             "low_feature": max_feat_pair[0], "high_feature": max_feat_pair[1],
                             "low_p": float(max_feat_pair[2]), "high_p": float(max_feat_pair[3])},
        "max_reference_size_gap": {"gap": float(max_m_gap), "x": float(max_m_gap_x),
                                    "p_m5000": float(max_m_probs[0]), "p_m_other": float(max_m_probs[1]),
                                    "m_other_level": m_level},
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(notes))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
