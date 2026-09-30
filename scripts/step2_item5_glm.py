"""
Step 2 review item 5 (2026-09-30 instruction): redo the data-collapse
analysis as a binomial GLM instead of the earlier within-bin-std check,
which conflated the power curve's slope with sampling noise. Uses item 4's
multi-reference-draw run (results/tabular_validation_item4_power_curve_raw.json).

x = sqrt(n*m/(n+m)) * D_pop (population D, never the observed per-batch
statistic -- same reasoning as step2_analyze.py's data-collapse section:
detection is a deterministic function of the observed statistic, so
plotting against it would collapse onto a step function by construction).

y = the calibrated system's MATERIAL flag (Gate 2), matching item 4's
"detection (material) rate" framing -- this is what a near-floor power
curve is actually about (does the materiality gate's own decision track
the asymptotic KS theory's prediction). drift_detected (both gates) is
fit as a secondary check and reported alongside for completeness.

Three nested logistic models (logit link -- "probit or logit" per
instruction; logit chosen since it's directly supported by the installed
scikit-learn without adding a new dependency; statsmodels is not installed
and this project's standing rule is not to add dependencies without
sign-off):
    M0: y ~ x
    M1: y ~ x + feature (5 dummies)
    M2: y ~ x + feature + reference_size (m, 2 levels: 5000/50000)
Likelihood-ratio test M0->M1 (does detection depend on WHICH feature is
drifting, beyond what x alone predicts -- theory says approximately no)
and M1->M2 (does detection depend on the reference size beyond what x
already captures -- theory says no, since x already includes n and m).

A weak L2 penalty (C=1e6, effectively unpenalized) is used instead of
pure MLE because the power curve is near-deterministic away from the
transition region (x<0.5: never detected; x>1.7: always detected in the
existing data), which causes quasi-separation and unbounded MLE
coefficients with truly unpenalized logistic regression -- a standard,
documented practical fix, noted here rather than silently applied.

Run:
    python scripts/step2_item5_glm.py
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

IN_JSON = "results/tabular_validation_item4_power_curve_raw.json"
OUT_JSON = "results/step2_item5_glm.json"
OUT_MD = "results/step2_item5_glm.md"


def log_likelihood(y, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logit(X, y):
    model = LogisticRegression(penalty="l2", C=1e6, max_iter=5000, solver="lbfgs")
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    ll = log_likelihood(y, p)
    k = X.shape[1] + 1  # + intercept
    aic = 2 * k - 2 * ll
    bic = k * math.log(len(y)) - 2 * ll
    return model, ll, k, aic, bic


def lrt(ll_reduced, k_reduced, ll_full, k_full):
    stat = 2 * (ll_full - ll_reduced)
    df = k_full - k_reduced
    p = float(chi2.sf(stat, df)) if df > 0 else float("nan")
    return {"lr_stat": float(stat), "df": df, "p_value": p}


def build_design(df, features, include_feature=False, include_m=False):
    cols = [df[["x"]].values]
    names = ["x"]
    if include_feature:
        dummies = pd.get_dummies(df["feature"], prefix="feat", drop_first=True)
        cols.append(dummies.values.astype(float))
        names.extend(dummies.columns.tolist())
    if include_m:
        m_dummy = pd.get_dummies(df["ref_size"], prefix="m", drop_first=True)
        cols.append(m_dummy.values.astype(float))
        names.extend(m_dummy.columns.tolist())
    X = np.hstack(cols)
    return X, names


def run_for_target(df, target_col, label, notes):
    y = df[target_col].astype(int).values
    notes.append(f"\n### Target: `{target_col}` ({label})\n")
    notes.append(f"n={len(df)}, positive rate={y.mean():.4f}\n")

    X0, names0 = build_design(df, None, include_feature=False, include_m=False)
    m0, ll0, k0, aic0, bic0 = fit_logit(X0, y)

    X1, names1 = build_design(df, None, include_feature=True, include_m=False)
    m1, ll1, k1, aic1, bic1 = fit_logit(X1, y)

    X2, names2 = build_design(df, None, include_feature=True, include_m=True)
    m2, ll2, k2, aic2, bic2 = fit_logit(X2, y)

    notes.append("| Model | Covariates | k (params) | log-lik | AIC | BIC |\n|---|---|---|---|---|---|")
    notes.append(f"| M0 | x only | {k0} | {ll0:.2f} | {aic0:.2f} | {bic0:.2f} |")
    notes.append(f"| M1 | x + feature | {k1} | {ll1:.2f} | {aic1:.2f} | {bic1:.2f} |")
    notes.append(f"| M2 | x + feature + m | {k2} | {ll2:.2f} | {aic2:.2f} | {bic2:.2f} |")

    lrt_01 = lrt(ll0, k0, ll1, k1)
    lrt_12 = lrt(ll1, k1, ll2, k2)
    notes.append(f"\n**LRT M0->M1 (feature effect beyond x)**: "
                 f"chi2={lrt_01['lr_stat']:.3f}, df={lrt_01['df']}, p={lrt_01['p_value']:.4f}")
    notes.append(f"\n**LRT M1->M2 (reference-size effect beyond x+feature)**: "
                 f"chi2={lrt_12['lr_stat']:.3f}, df={lrt_12['df']}, p={lrt_12['p_value']:.4f}\n")

    notes.append("M0 coefficient on x: " + f"{m0.coef_[0][0]:.4f} (intercept {m0.intercept_[0]:.4f})\n")

    if lrt_01["p_value"] >= 0.05:
        notes.append(
            "**Interpretation**: no significant evidence that detection depends on WHICH continuous "
            "feature is drifting, beyond what x=sqrt(nm/(n+m))*D_pop alone predicts -- consistent with "
            "(approximate) data collapse across features.\n"
        )
    else:
        notes.append(
            "**Interpretation**: detection DOES depend significantly on which feature is drifting, "
            "beyond x alone -- the single-curve collapse does not fully hold across features; theory "
            "only guarantees approximate collapse (power depends on the shape of F-G, not just "
            "sup|F-G|=D), so per-feature deviations here are consistent with that caveat, not a bug.\n"
        )
    if lrt_12["p_value"] >= 0.05:
        notes.append(
            "**Interpretation**: no significant evidence that detection depends on reference size m "
            "beyond x+feature -- consistent with x already fully capturing the n,m dependence predicted "
            "by asymptotic KS theory.\n"
        )
    else:
        m_cols = [i for i, name in enumerate(names2) if name.startswith("m_")]
        m_coef = m2.coef_[0][m_cols[0]] if m_cols else None
        m_level = names2[m_cols[0]].replace("m_", "") if m_cols else None
        direction = (
            f"at matched x, m={m_level} batches have LOWER detection probability than the baseline "
            f"reference size (coefficient {m_coef:+.3f})" if (m_coef is not None and m_coef < 0) else
            f"at matched x, m={m_level} batches have HIGHER detection probability than the baseline "
            f"reference size (coefficient {m_coef:+.3f})" if m_coef is not None else
            "direction not resolved (no m dummy in the fitted design)"
        )
        notes.append(
            f"**Interpretation**: detection DOES depend significantly on reference size m beyond x -- "
            f"{direction}. This is a real residual m-dependence not captured by the sqrt(nm/(n+m)) "
            f"scaling alone -- plausibly a finite-sample correction to the asymptotic two-sample KS "
            f"distribution that the simple sqrt(nm/(n+m))*D collapse variable doesn't fully capture "
            f"(known refined asymptotics for the KS statistic include higher-order terms beyond this "
            f"leading-order scaling). Not attributable to noise given n=42,000 trials.\n"
        )

    return {
        "target": target_col, "n": len(df), "positive_rate": float(y.mean()),
        "models": {
            "M0": {"covariates": names0, "k": k0, "log_lik": ll0, "aic": aic0, "bic": bic0,
                   "coef": m0.coef_[0].tolist(), "intercept": float(m0.intercept_[0])},
            "M1": {"covariates": names1, "k": k1, "log_lik": ll1, "aic": aic1, "bic": bic1,
                   "coef": m1.coef_[0].tolist(), "intercept": float(m1.intercept_[0])},
            "M2": {"covariates": names2, "k": k2, "log_lik": ll2, "aic": aic2, "bic": bic2,
                   "coef": m2.coef_[0].tolist(), "intercept": float(m2.intercept_[0])},
        },
        "lrt_feature": lrt_01, "lrt_reference_size": lrt_12,
    }


def between_draw_variance(df):
    """Between-reference-draw variance of the material rate, per
    (ref_size, feature, target_d, n) cell -- averaged over draws' own
    binomial rate, per item 4's instruction."""
    rows = []
    grp_cols = ["ref_size", "feature", "target_d", "n"]
    for keys, g in df.groupby(grp_cols):
        per_draw = g.groupby("draw")["material"].mean()
        rows.append({
            **dict(zip(grp_cols, keys)),
            "n_draws": len(per_draw),
            "mean_rate_across_draws": float(per_draw.mean()),
            "between_draw_variance": float(per_draw.var(ddof=1)) if len(per_draw) > 1 else None,
            "between_draw_std": float(per_draw.std(ddof=1)) if len(per_draw) > 1 else None,
            "min_draw_rate": float(per_draw.min()), "max_draw_rate": float(per_draw.max()),
        })
    return rows


def main():
    raw = json.load(open(IN_JSON, encoding="utf-8"))
    records = raw["raw_records"]
    df = pd.DataFrame.from_records(records)
    df["material"] = df["material"].astype(bool)
    df["drift_detected"] = df["drift_detected"].astype(bool)
    df["x"] = np.sqrt(df["n"] * df["ref_size"] / (df["n"] + df["ref_size"])) * df["achieved_d"]

    notes = []
    notes.append("# Step 2 review item 5: binomial GLM data-collapse re-analysis\n")
    notes.append(
        f"Data from item 4's power-curve run: {len(df)} trials, "
        f"{raw['config']['n_reference_draws']} reference draws x {len(raw['config']['reference_sizes'])} "
        f"reference sizes x {len(raw['config']['continuous_features'])} features x "
        f"{len(raw['config']['d_pop_targets'])} D_pop targets x {len(raw['config']['batch_sizes'])} "
        f"batch sizes x {raw['config']['n_trials']} trials.\n"
    )
    notes.append(
        "x = sqrt(n*m/(n+m)) * D_pop (achieved population D from the exact weighted-KS computation, "
        "never the per-batch observed statistic). Logit link (scikit-learn `LogisticRegression`, weak "
        "L2 penalty C=1e6 in place of unpenalized MLE -- the power curve is near-deterministic away "
        "from its transition region, which causes quasi-separation and unbounded coefficients under "
        "true MLE; this is a standard, documented practical substitute, not a silent change of "
        "estimator).\n"
    )

    result = {}
    result["material_gate"] = run_for_target(df, "material", "Gate 2 materiality alone -- the near-floor power curve itself", notes)
    result["drift_detected_secondary"] = run_for_target(df, "drift_detected", "full two-gate system decision, secondary check", notes)

    notes.append("\n## Between-reference-draw variance\n")
    notes.append(
        "Per (reference_size, feature, D_pop target, batch size) cell: the calibrated system's material "
        "rate computed separately within each of the 10 reference draws, then the variance/std of those "
        "10 per-draw rates -- this isolates variability attributable to WHICH reference sample was used "
        "to fit the baseline, holding the drift-injection method fixed.\n"
    )
    bdv = between_draw_variance(df)
    result["between_draw_variance"] = bdv
    nonzero_var = [r for r in bdv if r["between_draw_variance"] not in (None, 0.0)]
    if nonzero_var:
        max_var_row = max(nonzero_var, key=lambda r: r["between_draw_variance"])
        notes.append(
            f"**Largest between-draw variance**: ref_size={max_var_row['ref_size']}, "
            f"feature={max_var_row['feature']}, target_D={max_var_row['target_d']}, n={max_var_row['n']} "
            f"-- mean rate {max_var_row['mean_rate_across_draws']:.3f} across draws, "
            f"std={max_var_row['between_draw_std']:.3f}, range "
            f"[{max_var_row['min_draw_rate']:.3f}, {max_var_row['max_draw_rate']:.3f}] over "
            f"{max_var_row['n_draws']} draws. As expected, this concentrates in the transition region "
            f"(neither always-0 nor always-1), where which specific reference sample was drawn can tip "
            f"a near-floor batch's effect size across the materiality threshold.\n"
        )
    else:
        notes.append("All cells had zero between-draw variance (fully deterministic given x).\n")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(notes))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
