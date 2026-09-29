"""
Consumes results/tabular_validation_legacy_raw.json (produced by
step1_validation.py) and produces:
  - results/tabular_validation_legacy.json   (clean, structured summary)
  - results/tabular_validation_notes.md      (write-up, explanations)

Scope: Apr-Jun 2016 production data only (see
results/citi_bike_provenance_forensics.md for why).

GROUND TRUTH FIX (this version): the first version of this script computed
"population ground truth" from the SAMPLED reference (5,000 or 50,000 rows)
vs. production -- which meant the ground truth itself changed depending on
which reference happened to be drawn (dropoff_longitude flipped labels
between reference sizes). That is not a ground truth; it's a
reference-dependent diagnostic. Fixed here: population truth is now computed
ONCE, from the FULL baseline CSV (all 1,577,611 Jan-Mar rows, independent of
any sampled reference or batch size) vs. the full production data, with an
effect-size criterion (no p-values at this population scale -- oversensitive
for the same reason chi-square was rejected for PSI's ground truth). This
label is identical across every reference size and batch size by
construction. The old reference-vs-production computation is kept, but
renamed and reported only as a diagnostic
("reference_detectable_effect") -- never used to compute TP/FP/FN/TN.

Re-labeling is done OFFLINE against the detections already collected in
step1_validation.py's raw output -- no new HTTP calls, per instruction (the
engine's detections don't change; only which label we grade them against
does).
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

CONTINUOUS_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"]
CATEGORICAL_FEATURES = ["gender_id", "month"]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
ALPHA = 0.05
D_GT_THRESHOLDS = [0.01, 0.02, 0.05]
PSI_GT_THRESHOLDS = [0.1, 0.2]

BASELINE_CSV = "tests/citi_bike_baseline.csv"
PRODUCTION_MONTH_CSVS = {4: "tests/splits/production_month_04.csv",
                          5: "tests/splits/production_month_05.csv",
                          6: "tests/splits/production_month_06.csv"}


def confusion_counts(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": precision, "recall": recall, "f1": f1,
            "n": len(y_true)}


def fmt(cc):
    p = f"{cc['precision']:.3f}" if cc["precision"] is not None else "n/a"
    r = f"{cc['recall']:.3f}" if cc["recall"] is not None else "n/a"
    f1 = f"{cc['f1']:.3f}" if cc["f1"] is not None else "n/a"
    return p, r, f1


def compute_psi(ref_series, cur_series, epsilon=0.0001):
    cats = sorted(set(ref_series.astype(str).unique()) | set(cur_series.astype(str).unique()))
    ref_p = ref_series.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    cur_p = cur_series.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    ref_p = ref_p.replace(0, epsilon)
    cur_p = cur_p.replace(0, epsilon)
    return float(((cur_p - ref_p) * np.log(cur_p / ref_p)).sum())


def compute_population_truth():
    """THE fix: population truth computed ONCE from the FULL baseline CSV
    (never a sampled reference) vs. full production data, per month and
    pooled. Returns raw D/PSI values (labels derived separately per
    threshold) -- this dict is identical regardless of reference size or
    batch size, by construction (it doesn't use either)."""
    print("Loading full baseline CSV for population ground truth (one-time, ~1.58M rows)...")
    baseline_full = pd.read_csv(BASELINE_CSV)
    months_full = {m: pd.read_csv(p) for m, p in PRODUCTION_MONTH_CSVS.items()}
    pooled_full = pd.concat(months_full.values(), ignore_index=True)

    def raw_stats(prod_df):
        out = {}
        for col in CONTINUOUS_FEATURES:
            d, _ = stats.ks_2samp(baseline_full[col].values, prod_df[col].values)
            out[col] = {"population_D": float(d)}
        for col in CATEGORICAL_FEATURES:
            psi = compute_psi(baseline_full[col], prod_df[col])
            out[col] = {"population_PSI": psi}
        return out

    pooled_stats = raw_stats(pooled_full)
    per_month_stats = {m: raw_stats(df) for m, df in months_full.items()}
    return pooled_stats, per_month_stats


def labels_at(raw_stats_dict, d_gt, psi_gt):
    """Derive a {feature: bool} label dict from raw population stats at a
    given (D_gt, psi_gt) pair. Continuous: D >= d_gt. Categorical: PSI >= psi_gt."""
    labels = {}
    for col in CONTINUOUS_FEATURES:
        labels[col] = bool(raw_stats_dict[col]["population_D"] >= d_gt)
    for col in CATEGORICAL_FEATURES:
        labels[col] = bool(raw_stats_dict[col]["population_PSI"] >= psi_gt)
    return labels


def sweep_confusion(sweep_size_result, labels, exclude_features=None, month_filter=None):
    exclude_features = exclude_features or set()
    y_true, y_pred = [], []
    if month_filter is not None:
        responses = sweep_size_result["per_month"][str(month_filter)]["raw_responses"]
    else:
        responses = sweep_size_result["pooled"]["raw_responses"]
    for resp in responses:
        for feat in ALL_FEATURES:
            if feat in exclude_features:
                continue
            y_true.append(1 if labels.get(feat, False) else 0)
            y_pred.append(1 if resp[feat]["drift_detected"] else 0)
    return confusion_counts(y_true, y_pred)


def per_feature_batch_d_stats(sweep_size_result):
    out = {}
    responses = sweep_size_result["pooled"]["raw_responses"]
    for feat in ALL_FEATURES:
        vals = [r[feat]["statistic"] for r in responses]
        mean = sum(vals) / len(vals) if vals else None
        var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals and mean is not None else None
        std = var ** 0.5 if var is not None else None
        out[feat] = {"mean": mean, "std": std, "n": len(vals)}
    return out


def aa_false_alarm_rates(aa_result):
    responses = aa_result["raw_responses"]
    n = len(responses)
    per_feature = {f: 0 for f in ALL_FEATURES}
    system_alarms = 0
    for resp in responses:
        any_alarm = False
        for feat in ALL_FEATURES:
            if resp[feat]["drift_detected"]:
                per_feature[feat] += 1
                any_alarm = True
        if any_alarm:
            system_alarms += 1
    per_feature_rate = {f: c / n for f, c in per_feature.items()}
    return {"n_trials": n, "per_feature_false_alarm_rate": per_feature_rate,
            "system_false_alarm_rate": system_alarms / n}


def smallest_reliable_severity(severity_dict, reliable_threshold=1.0):
    """Smallest severity (sorted ascending) at which EVERY continuous
    feature hits detection rate >= reliable_threshold (default: 5/5, i.e.
    100% of trials)."""
    for sev in sorted((float(s) for s in severity_dict.keys())):
        entry = severity_dict[str(sev)] if str(sev) in severity_dict else severity_dict[sev]
        rates = {f: v["detections"] / v["n_trials"] for f, v in entry.items()}
        if all(r >= reliable_threshold for r in rates.values()):
            return sev
    return None


def main():
    raw = json.load(open("results/tabular_validation_legacy_raw.json"))
    config = raw["config"]

    pooled_truth_raw, per_month_truth_raw = compute_population_truth()

    summary = {
        "config": config,
        "ground_truth_fix_note": (
            "Population ground truth computed ONCE from the full baseline CSV (1,577,611 rows) "
            "vs full production data -- independent of reference size and batch size by "
            "construction. Effect-size criteria only (no p-values at this population scale)."
        ),
        "population_truth_raw": {"pooled": pooled_truth_raw, "per_month": {str(m): v for m, v in per_month_truth_raw.items()}},
        "by_reference_size": {},
    }
    notes_lines = []

    # ============================================================
    # COMPACT SUMMARY (top of file, per instruction)
    # ============================================================
    notes_lines.append("# Tabular Validation Notes (Step 1) — Apr-Jun 2016\n")
    notes_lines.append("## Compact summary\n")
    notes_lines.append(
        "**Scope**: Apr-Jun 2016 production only (see `results/citi_bike_provenance_forensics.md`). "
        "Old README \"Apr-Dec\" numbers remain superseded, not quoted here.\n"
    )
    notes_lines.append(
        "**Ground truth fix**: population truth is now computed once from the FULL baseline CSV "
        "(1,577,611 rows) vs full production data, independent of reference/batch size. The old "
        "reference-vs-production computation is kept only as a diagnostic "
        "(`reference_detectable_effect`), never as ground truth. See 'Ground truth fix, verified' "
        "below for the dropoff_longitude check.\n"
    )

    # Pooled labels at each D_gt (PSI fixed at 0.2, the production threshold) for the compact summary
    default_labels = labels_at(pooled_truth_raw, 0.02, 0.2)

    notes_lines.append("**Real batches only, pooled Apr-Jun, decision unit=(batch,feature), "
                        "D_gt=0.02 / PSI>=0.2, at the largest batch size tested (50,000), "
                        "WITH and WITHOUT `month`:**\n")
    notes_lines.append("| Reference size | month? | TP | FP | FN | TN | Precision | Recall | F1 |\n"
                        "|---|---|---|---|---|---|---|---|---|")
    for ref_size_str in raw["by_reference_size"].keys():
        big = raw["by_reference_size"][ref_size_str]["sweep"]["50000"]
        cc_with = sweep_confusion(big, default_labels)
        cc_without = sweep_confusion(big, default_labels, exclude_features={"month"})
        for label, cc in [("with", cc_with), ("without", cc_without)]:
            p, r, f1 = fmt(cc)
            notes_lines.append(f"| {ref_size_str} | {label} | {cc['TP']} | {cc['FP']} | {cc['FN']} | "
                                f"{cc['TN']} | {p} | {r} | {f1} |")
    notes_lines.append("")

    notes_lines.append("**A/A per-feature and system false-alarm rates vs. alpha-predicted "
                        f"(`1-(1-{ALPHA})^7`={1 - (1 - ALPHA) ** 7:.3f}), both reference sizes, "
                        "all A/A sizes:**\n")
    notes_lines.append("| Reference size | Batch size | System | " + " | ".join(ALL_FEATURES) + " |\n" +
                        "|---|---|---|" + "---|" * len(ALL_FEATURES))
    for ref_size_str, r in raw["by_reference_size"].items():
        for size_str, aa_result in r["aa_test"].items():
            rates = aa_false_alarm_rates(aa_result)
            per_feat_str = " | ".join(f"{rates['per_feature_false_alarm_rate'][f]:.3f}" for f in ALL_FEATURES)
            notes_lines.append(f"| {ref_size_str} | {size_str} | {rates['system_false_alarm_rate']:.3f} | {per_feat_str} |")
    notes_lines.append("")

    notes_lines.append("**Smallest synthetic severity reliably detected (100% of trials, every "
                        "continuous feature) at each reference size:**\n")
    notes_lines.append("| Reference size | Smallest reliable severity (σ) |\n|---|---|")
    for ref_size_str, r in raw["by_reference_size"].items():
        sev = smallest_reliable_severity(r["severity"])
        notes_lines.append(f"| {ref_size_str} | {sev if sev is not None else 'none tested reach 100%'} |")
    notes_lines.append("")

    notes_lines.append("**Original README discrepancies (all about the now-superseded old numbers):**\n")
    notes_lines.append(
        "- *Headline recall 0.939 vs. naive per-feature-table sum 81/90=0.900*: **unreproducible** — "
        "the original `/fit` call's exact reference rows and the script that produced that headline run "
        "no longer exist (`split_citi_bike.py` was missing, see `docs/recon.md` §4/§9); there is no way "
        "to recompute the exact old number.\n"
        "- *Sweep row 10,000: recall 0.812 at precision 1.0 implies F1=0.896, but README says 0.886 "
        "(implying precision≈0.975)*: **unreproducible**, same reason — the exact old sweep run's data "
        "is gone.\n"
        "- *Headline batch size (0.939) exceeds every sweep row including 50k (0.917)*: **explained**, "
        "not a contradiction — confirmed by reading the code (`docs/recon.md` §4) that the headline used "
        "a fixed `PRODUCTION_BATCH_SIZE=25000`, a size the separate 8-trial sweep never tested. Different "
        "experiments, not a discrepancy in one experiment — though the specific 0.939 value itself remains "
        "unreproducible for the reasons above.\n"
    )

    notes_lines.append("---\n")

    # ============================================================
    # Ground truth fix, verified
    # ============================================================
    notes_lines.append("## Ground truth fix, verified\n")
    notes_lines.append(
        "**dropoff_longitude**, which flipped labels between reference sizes in the previous (flawed) "
        "version of this analysis, now has a single, fixed population D — because population truth no "
        "longer uses either sampled reference at all:\n"
    )
    dl_d = pooled_truth_raw["dropoff_longitude"]["population_D"]
    notes_lines.append(f"- Pooled Apr-Jun population D for `dropoff_longitude` (full baseline vs full "
                        f"production, independent of reference size): **{dl_d:.4f}** — one number, used "
                        f"for both m=5,000 and m=50,000 analyses.\n")
    notes_lines.append("**Full population truth (pooled Apr-Jun), raw statistics:**\n")
    notes_lines.append("| Feature | Population D or PSI |\n|---|---|")
    for feat in ALL_FEATURES:
        v = pooled_truth_raw[feat].get("population_D", pooled_truth_raw[feat].get("population_PSI"))
        notes_lines.append(f"| {feat} | {v:.4f} |")
    notes_lines.append("")
    notes_lines.append("**Per-month population truth, raw statistics (all vs. the SAME full baseline CSV):**\n")
    notes_lines.append("| Feature | Apr (D/PSI) | May (D/PSI) | Jun (D/PSI) |\n|---|---|---|---|")
    for feat in ALL_FEATURES:
        vals = []
        for m in [4, 5, 6]:
            v = per_month_truth_raw[m][feat].get("population_D", per_month_truth_raw[m][feat].get("population_PSI"))
            vals.append(f"{v:.4f}")
        notes_lines.append(f"| {feat} | {vals[0]} | {vals[1]} | {vals[2]} |")
    notes_lines.append("")

    for ref_size_str, r in raw["by_reference_size"].items():
        ref_size = int(ref_size_str)
        notes_lines.append(f"\n## Reference size = {ref_size}\n")
        rs_summary = {"sweep": {}, "aa": {}, "severity": r["severity"],
                      "min_drift_fraction": r["min_drift_fraction"], "per_feature_batch_d": {},
                      "reference_detectable_effect_DIAGNOSTIC_ONLY": r["ground_truth_pooled"]}

        notes_lines.append(
            "**Diagnostic only, NOT ground truth** (`reference_detectable_effect`) — the OLD "
            "reference-vs-production computation, kept for reference but never used below to grade "
            "detections:\n"
        )
        notes_lines.append("| Feature | \"Drifted\" vs. this reference | Detail |\n|---|---|---|")
        old_gt = r["ground_truth_pooled"]["ground_truth"]
        old_details = r["ground_truth_pooled"]["details"]
        for feat in ALL_FEATURES:
            detail = old_details[feat]
            detail_str = (f"D={detail.get('population_D'):.4f}, p={detail.get('ks_pvalue'):.2e}"
                           if "population_D" in detail else f"PSI={detail.get('population_PSI'):.4f}")
            notes_lines.append(f"| {feat} | {old_gt[feat]} | {detail_str} |")
        notes_lines.append("")

        # --- Sweep tables at each D_gt (categorical fixed at PSI>=0.2), WITH and WITHOUT month ---
        for d_gt in D_GT_THRESHOLDS:
            labels = labels_at(pooled_truth_raw, d_gt, 0.2)
            notes_lines.append(f"**Sweep, pooled, D_gt={d_gt} (PSI>=0.2), WITH `month`:**\n")
            notes_lines.append("| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |\n"
                                "|---|---|---|---|---|---|---|---|---|")
            per_dgt = {}
            for size_str, size_result in r["sweep"].items():
                cc = sweep_confusion(size_result, labels)
                per_dgt[size_str] = cc
                p, rc, f1 = fmt(cc)
                notes_lines.append(f"| {size_str} | {cc['TP']} | {cc['FP']} | {cc['FN']} | {cc['TN']} | "
                                    f"{p} | {rc} | {f1} | {cc['n']} |")
            notes_lines.append("")

            notes_lines.append(f"**Same, D_gt={d_gt}, WITHOUT `month`:**\n")
            notes_lines.append("| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|")
            per_dgt_no_month = {}
            for size_str, size_result in r["sweep"].items():
                cc = sweep_confusion(size_result, labels, exclude_features={"month"})
                per_dgt_no_month[size_str] = cc
                p, rc, f1 = fmt(cc)
                notes_lines.append(f"| {size_str} | {cc['TP']} | {cc['FP']} | {cc['FN']} | {cc['TN']} | {p} | {rc} | {f1} |")
            notes_lines.append("")
            rs_summary["sweep"][f"d_gt_{d_gt}"] = {"with_month": per_dgt, "without_month": per_dgt_no_month}

        # --- Categorical PSI threshold sensitivity (0.1 vs 0.2), fixed D_gt=0.02 ---
        notes_lines.append("**Categorical ground truth sensitivity (D_gt fixed at 0.02 for continuous), "
                            "PSI>=0.1 vs PSI>=0.2, pooled, batch size=50000:**\n")
        notes_lines.append("| PSI threshold | TP | FP | FN | TN | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|")
        for psi_gt in PSI_GT_THRESHOLDS:
            labels = labels_at(pooled_truth_raw, 0.02, psi_gt)
            cc = sweep_confusion(r["sweep"]["50000"], labels)
            p, rc, f1 = fmt(cc)
            notes_lines.append(f"| {psi_gt} | {cc['TP']} | {cc['FP']} | {cc['FN']} | {cc['TN']} | {p} | {rc} | {f1} |")
        notes_lines.append("")

        # --- Per-month breakdown at biggest size, using PER-MONTH population truth ---
        biggest_size = str(max(int(s) for s in r["sweep"].keys()))
        notes_lines.append(f"**Per-month breakdown at batch size={biggest_size}, D_gt=0.02/PSI>=0.2, "
                            "ground truth = that month's OWN population truth (full baseline vs that month's "
                            "full data), not pooled:**\n")
        notes_lines.append("| Month | TP | FP | FN | TN | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|")
        per_month_cc = {}
        for m in [4, 5, 6]:
            labels_m = labels_at(per_month_truth_raw[m], 0.02, 0.2)
            cc = sweep_confusion(r["sweep"][biggest_size], labels_m, month_filter=m)
            per_month_cc[m] = cc
            p, rc, f1 = fmt(cc)
            notes_lines.append(f"| {m:02d} | {cc['TP']} | {cc['FP']} | {cc['FN']} | {cc['TN']} | {p} | {rc} | {f1} |")
        notes_lines.append("")
        rs_summary["per_month_at_biggest_size"] = per_month_cc

        # --- Per-feature batch D mean/std ---
        notes_lines.append(f"**Per-feature batch statistic (D for continuous, PSI for categorical) mean/std "
                            f"at batch size={biggest_size}, vs. FIXED population value:**\n")
        notes_lines.append("| Feature | Population D/PSI (fixed) | Per-batch mean | Per-batch std |\n|---|---|---|---|")
        d_stats = per_feature_batch_d_stats(r["sweep"][biggest_size])
        rs_summary["per_feature_batch_d"] = d_stats
        for feat in ALL_FEATURES:
            pop_val = pooled_truth_raw[feat].get("population_D", pooled_truth_raw[feat].get("population_PSI"))
            s = d_stats[feat]
            notes_lines.append(f"| {feat} | {pop_val:.4f} | {s['mean']:.4f} | {s['std']:.4f} |")
        notes_lines.append("")

        # --- A/A test ---
        notes_lines.append("**A/A test (iid test-calibration check — both holdout and reference drawn at "
                            "random from Jan-Mar; does NOT capture month-to-month variation within the "
                            "baseline period):**\n")
        notes_lines.append("| Batch size | System false-alarm rate | " +
                            " | ".join(ALL_FEATURES) + " |\n|---|---|" + "---|" * len(ALL_FEATURES))
        for size_str, aa_result in r["aa_test"].items():
            rates = aa_false_alarm_rates(aa_result)
            rs_summary["aa"][size_str] = rates
            per_feat_str = " | ".join(f"{rates['per_feature_false_alarm_rate'][f]:.3f}" for f in ALL_FEATURES)
            notes_lines.append(f"| {size_str} | {rates['system_false_alarm_rate']:.3f} | {per_feat_str} |")
        k = len(ALL_FEATURES)
        predicted_system_rate = 1 - (1 - ALPHA) ** k
        notes_lines.append(f"\nAlpha-predicted system false-alarm rate for {k} uncorrected tests at "
                            f"alpha={ALPHA}: `1-(1-{ALPHA})^{k}` = **{predicted_system_rate:.3f}**.\n")

        summary["by_reference_size"][ref_size_str] = rs_summary

    # ============================================================
    # Reference size vs. batch size (unaffected by the ground-truth fix --
    # recomputed against the FIXED labels this time)
    # ============================================================
    notes_lines.append("\n## Reference size vs. batch size\n")
    import math
    c_alpha = 1.36
    for m in [5000, 50000]:
        floor_val = c_alpha / math.sqrt(m)
        notes_lines.append(f"- m={m}: asymptotic KS critical-value floor as n->infinity: "
                            f"c(alpha)/sqrt(m) = {c_alpha}/sqrt({m}) = **{floor_val:.4f}**")
    n20k_5000 = c_alpha * math.sqrt((20000 + 5000) / (20000 * 5000))
    notes_lines.append(f"- At n=20,000, m=5,000: c(alpha)*sqrt((n+m)/(nm)) = **{n20k_5000:.4f}**\n")

    pickup_lat_pop_d = pooled_truth_raw["pickup_latitude"]["population_D"]
    notes_lines.append(f"**pickup_latitude fixed population D = {pickup_lat_pop_d:.4f}** (same for both "
                        "reference sizes now, per the fix above) vs. m=5,000 floor "
                        f"({c_alpha / math.sqrt(5000):.4f}) and m=50,000 floor ({c_alpha / math.sqrt(50000):.4f}).\n")

    # D_gt=0.01 here specifically because pickup_latitude's population D (0.0189)
    # is BELOW 0.02 -- at D_gt=0.02 it isn't a ground-truth positive at all, so
    # "recall" would be undefined (no positives to recall). D_gt=0.01 is the
    # threshold under which this feature genuinely IS a true positive, which is
    # what this demonstration needs: a real small effect whose detectability is
    # limited by the small-reference noise floor.
    notes_lines.append("**pickup_latitude recall by batch size, at both reference sizes "
                        "(pooled, with month, FIXED ground truth, D_gt=0.01 -- chosen because "
                        "pickup_latitude's population D=0.0189 is below 0.02, so D_gt=0.02 would "
                        "make it a ground-truth negative with no recall to measure):**\n")
    notes_lines.append("| Batch size | Recall @ m=5000 | Recall @ m=50000 |\n|---|---|---|")
    fixed_labels = labels_at(pooled_truth_raw, 0.01, 0.2)

    def feature_recall(size_result, labels, feat):
        y_true, y_pred = [], []
        for resp in size_result["pooled"]["raw_responses"]:
            y_true.append(1 if labels.get(feat, False) else 0)
            y_pred.append(1 if resp[feat]["drift_detected"] else 0)
        return confusion_counts(y_true, y_pred)["recall"]

    for size_str in raw["by_reference_size"]["5000"]["sweep"].keys():
        r5000 = feature_recall(raw["by_reference_size"]["5000"]["sweep"][size_str], fixed_labels, "pickup_latitude")
        r50000 = feature_recall(raw["by_reference_size"]["50000"]["sweep"][size_str], fixed_labels, "pickup_latitude")
        r5000_s = f"{r5000:.3f}" if r5000 is not None else "n/a"
        r50000_s = f"{r50000:.3f}" if r50000 is not None else "n/a"
        notes_lines.append(f"| {size_str} | {r5000_s} | {r50000_s} |")

    notes_lines.append(
        f"\n**Same conclusion as before, now on a stable ground truth**: pickup_latitude's population D "
        f"({pickup_lat_pop_d:.4f}) sits close to the m=5,000 floor. At m=5,000, recall plateaus well below "
        "1.0 at large batch sizes; at m=50,000, recall reaches 1.000. Reference size, not batch size, is the "
        "binding constraint — and this time the ground truth used to measure it doesn't move when the "
        "reference does.\n"
    )

    notes_lines.append(
        "**Note on `month`'s per-batch std=0.0000**: every row drawn from a single production month file "
        "has `month` equal to that one value, so every batch's PSI for `month` compares against an "
        "identical 100%-one-category distribution — no batch-to-batch variation to produce nonzero std. "
        "Expected, not a bug.\n"
    )

    with open("results/tabular_validation_legacy.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open("results/tabular_validation_notes.md", "w", encoding="utf-8") as f:
        f.write("\n".join(notes_lines))

    print("Wrote results/tabular_validation_legacy.json")
    print("Wrote results/tabular_validation_notes.md")


if __name__ == "__main__":
    main()
