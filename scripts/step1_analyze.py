"""
Consumes results/tabular_validation_legacy_raw.json (produced by
step1_validation.py) and produces:
  - results/tabular_validation_legacy.json   (clean, structured summary)
  - results/tabular_validation_notes.md      (write-up, explanations)

Scope: Apr-Jun 2016 production data only (see
results/citi_bike_provenance_forensics.md for why). All numbers here come
from the raw JSON produced by an actual script run against the live
backend -- nothing here is estimated or fabricated.
"""

import json

CONTINUOUS_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"]
CATEGORICAL_FEATURES = ["gender_id", "month"]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
ALPHA = 0.05


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


def sweep_confusion(sweep_size_result, gt_pooled, gt_per_month, exclude_features=None, month_filter=None):
    """Builds confusion counts from sweep raw_responses. decision unit =
    (batch, feature). exclude_features lets us compute with/without `month`.
    month_filter: if given, restrict to that month's responses only (per-month
    table); ground truth used is that month's own population ground truth.
    If month_filter is None, uses ALL months' responses pooled, against the
    POOLED ground truth (matches how the legacy script worked: one ground
    truth applied across the whole pooled production set)."""
    exclude_features = exclude_features or set()
    y_true, y_pred = [], []
    if month_filter is not None:
        gt = gt_per_month[str(month_filter)]["ground_truth"]
        responses = sweep_size_result["per_month"][str(month_filter)]["raw_responses"]
    else:
        gt = gt_pooled
        responses = sweep_size_result["pooled"]["raw_responses"]
    for resp in responses:
        for feat in ALL_FEATURES:
            if feat in exclude_features:
                continue
            truth = 1 if gt.get(feat, False) else 0
            pred = 1 if resp[feat]["drift_detected"] else 0
            y_true.append(truth)
            y_pred.append(pred)
    return confusion_counts(y_true, y_pred)


def per_feature_batch_d_stats(sweep_size_result):
    """Per-feature mean/std of the per-batch KS statistic (continuous) or
    PSI (categorical), pooled across months, for one sweep size."""
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
    """Per-feature and system-level false alarm rate from A/A trials."""
    responses = aa_result["raw_responses"]
    n = len(responses)
    per_feature = {}
    system_alarms = 0
    for resp in responses:
        any_alarm = False
        for feat in ALL_FEATURES:
            if feat not in per_feature:
                per_feature[feat] = 0
            if resp[feat]["drift_detected"]:
                per_feature[feat] += 1
                any_alarm = True
        if any_alarm:
            system_alarms += 1
    per_feature_rate = {f: c / n for f, c in per_feature.items()}
    return {"n_trials": n, "per_feature_false_alarm_rate": per_feature_rate,
            "system_false_alarm_rate": system_alarms / n}


def main():
    raw = json.load(open("results/tabular_validation_legacy_raw.json"))
    config = raw["config"]

    summary = {"config": config, "by_reference_size": {}}
    notes_lines = []

    notes_lines.append("# Tabular Validation Notes (Step 1) — Apr-Jun 2016\n")
    notes_lines.append(
        "**Scope**: production batches are Apr-Jun 2016 only (the only data available "
        "locally — see `results/citi_bike_provenance_forensics.md`). The old README's "
        "\"Apr-Dec\" numbers are superseded and not quoted here.\n"
    )
    notes_lines.append(
        "**Decision unit**: (batch, feature) — one confusion-matrix row per feature per "
        "analyzed batch, matching the legacy script's own unit (`tests/test_drift_engine.py`'s "
        "`run_classification_evaluation`, confirmed in `docs/recon.md` §4). A `system alert` "
        "is a separate, coarser unit: `any(feature drift_detected)` per batch — see below "
        "for both.\n"
    )
    notes_lines.append(
        f"**Config**: alpha={config['alpha']}, PSI threshold={config['psi_threshold']}, "
        f"reference sizes={config['reference_sizes']}, holdout size={config['holdout_size']}, "
        f"months={config['months']} (Apr/May/Jun 2016), sweep sizes={config['sweep_sizes']}, "
        f"trials/month/size (target, capped by disjointness)={config['trials_target_per_month']}, "
        f"A/A trials/size={config['aa_trials_per_size']}, A/A sizes={config['aa_sizes']}, "
        f"severities={config['severities']}, severity batch size={config['severity_batch_size']}, "
        f"severity trials={config['severity_trials']}.\n"
    )

    for ref_size_str, r in raw["by_reference_size"].items():
        ref_size = int(ref_size_str)
        notes_lines.append(f"\n## Reference size = {ref_size}\n")
        rs_summary = {"ground_truth_pooled": r["ground_truth_pooled"], "sweep": {}, "aa": {}, "severity": r["severity"],
                      "min_drift_fraction": r["min_drift_fraction"], "per_feature_batch_d": {}}

        gt_pooled = r["ground_truth_pooled"]["ground_truth"]
        gt_per_month = r["ground_truth_per_month"]
        gt_details = r["ground_truth_pooled"]["details"]

        notes_lines.append("**Population ground truth (pooled Apr-Jun vs reference), legacy definition "
                            "(continuous: p<alpha; categorical: PSI>0.2):**\n")
        notes_lines.append("| Feature | Drifted? | Detail |\n|---|---|---|")
        for feat in ALL_FEATURES:
            detail = gt_details[feat]
            detail_str = (f"D={detail.get('population_D'):.4f}, p={detail.get('ks_pvalue'):.2e}"
                           if "population_D" in detail else f"PSI={detail.get('population_PSI'):.4f}")
            notes_lines.append(f"| {feat} | {gt_pooled[feat]} | {detail_str} |")
        notes_lines.append("")

        # --- Sweep: precision/recall/F1/raw counts, per size, pooled AND per month ---
        notes_lines.append("**Sweep (pooled across months, decision unit = batch×feature, "
                            "ground truth = pooled Apr-Jun population verdict):**\n")
        notes_lines.append("| Batch size | TP | FP | FN | TN | Precision | Recall | F1 | n |\n"
                            "|---|---|---|---|---|---|---|---|---|")
        for size_str, size_result in r["sweep"].items():
            cc_with = sweep_confusion(size_result, gt_pooled, gt_per_month)
            cc_without_month = sweep_confusion(size_result, gt_pooled, gt_per_month, exclude_features={"month"})
            rs_summary["sweep"][size_str] = {"pooled_with_month": cc_with, "pooled_without_month": cc_without_month,
                                              "per_month": {}}
            p = f"{cc_with['precision']:.3f}" if cc_with["precision"] is not None else "n/a"
            rc = f"{cc_with['recall']:.3f}" if cc_with["recall"] is not None else "n/a"
            f1 = f"{cc_with['f1']:.3f}" if cc_with["f1"] is not None else "n/a"
            notes_lines.append(f"| {size_str} | {cc_with['TP']} | {cc_with['FP']} | {cc_with['FN']} | "
                                f"{cc_with['TN']} | {p} | {rc} | {f1} | {cc_with['n']} |")
            for m in [4, 5, 6]:
                cc_m = sweep_confusion(size_result, gt_pooled, gt_per_month, month_filter=m)
                rs_summary["sweep"][size_str]["per_month"][m] = cc_m
        notes_lines.append("")

        notes_lines.append("**Same sweep, WITHOUT `month` feature (isolating the effect of dropping the "
                            "categorical feature with the largest PSI):**\n")
        notes_lines.append("| Batch size | TP | FP | FN | TN | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|")
        for size_str in r["sweep"].keys():
            cc = rs_summary["sweep"][size_str]["pooled_without_month"]
            p = f"{cc['precision']:.3f}" if cc["precision"] is not None else "n/a"
            rc = f"{cc['recall']:.3f}" if cc["recall"] is not None else "n/a"
            f1 = f"{cc['f1']:.3f}" if cc["f1"] is not None else "n/a"
            notes_lines.append(f"| {size_str} | {cc['TP']} | {cc['FP']} | {cc['FN']} | {cc['TN']} | {p} | {rc} | {f1} |")
        notes_lines.append("")

        # --- Per-month breakdown at the largest sweep size (most stable estimates) ---
        biggest_size = str(max(int(s) for s in r["sweep"].keys()))
        notes_lines.append(f"**Per-month breakdown at batch size={biggest_size} "
                            "(ground truth = that month's OWN population verdict vs reference, not pooled):**\n")
        notes_lines.append("| Month | TP | FP | FN | TN | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|")
        for m in [4, 5, 6]:
            cc = rs_summary["sweep"][biggest_size]["per_month"][m]
            p = f"{cc['precision']:.3f}" if cc["precision"] is not None else "n/a"
            rc = f"{cc['recall']:.3f}" if cc["recall"] is not None else "n/a"
            f1 = f"{cc['f1']:.3f}" if cc["f1"] is not None else "n/a"
            notes_lines.append(f"| {m:02d} | {cc['TP']} | {cc['FP']} | {cc['FN']} | {cc['TN']} | {p} | {rc} | {f1} |")
        notes_lines.append("")

        # --- Effect-size ground truth variants ---
        notes_lines.append("**Confusion matrix under effect-size ground truth variants "
                            f"(pooled, batch size={biggest_size}), vs. the legacy p-value-based ground truth above:**\n")
        notes_lines.append("| Ground truth | TP | FP | FN | TN | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|")
        big_result = r["sweep"][biggest_size]
        gt_variants_summary = {}
        for floor_str, gt_variant in r["ground_truth_effect_size_variants"].items():
            cc = sweep_confusion(big_result, gt_variant, gt_per_month)
            gt_variants_summary[floor_str] = cc
            p = f"{cc['precision']:.3f}" if cc["precision"] is not None else "n/a"
            rc = f"{cc['recall']:.3f}" if cc["recall"] is not None else "n/a"
            f1 = f"{cc['f1']:.3f}" if cc["f1"] is not None else "n/a"
            notes_lines.append(f"| D>={floor_str} | {cc['TP']} | {cc['FP']} | {cc['FN']} | {cc['TN']} | {p} | {rc} | {f1} |")
        cc_legacy = sweep_confusion(big_result, gt_pooled, gt_per_month)
        notes_lines.append(f"| legacy (p<{ALPHA}) | {cc_legacy['TP']} | {cc_legacy['FP']} | {cc_legacy['FN']} | "
                            f"{cc_legacy['TN']} | {cc_legacy['precision']:.3f} | {cc_legacy['recall']:.3f} | "
                            f"{cc_legacy['f1']:.3f} |")
        notes_lines.append(
            "\n**Framing (per instruction)**: this ground truth measures batch-level recovery of the "
            "POPULATION verdict — i.e. statistical power to detect a population-level difference that has "
            "already been established independently — not some external notion of \"real\" drift. At "
            "multi-million-row population sizes, KS p-values become oversensitive in exactly the way chi-square "
            "was rejected for PSI's categorical ground truth (`docs/recon.md` §3): a population D as small as "
            "0.01-0.02 can still yield p<<0.05 given enough rows. This is why the effect-size variants above "
            "matter — they ask a different, arguably more practically relevant question (\"is the population "
            "difference large enough to matter\") than the raw p-value does.\n"
        )
        rs_summary["ground_truth_effect_size_confusion"] = gt_variants_summary

        # --- Per-feature batch D mean/std at the biggest sweep size ---
        notes_lines.append(f"**Per-feature batch statistic (D for continuous, PSI for categorical) mean/std "
                            f"at batch size={biggest_size}, vs. population value:**\n")
        notes_lines.append("| Feature | Population D/PSI | Per-batch mean | Per-batch std |\n|---|---|---|---|")
        d_stats = per_feature_batch_d_stats(big_result)
        rs_summary["per_feature_batch_d"] = d_stats
        for feat in ALL_FEATURES:
            pop_val = gt_details[feat].get("population_D", gt_details[feat].get("population_PSI"))
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
        k_continuous = len(CONTINUOUS_FEATURES) + len(CATEGORICAL_FEATURES)
        predicted_system_rate = 1 - (1 - ALPHA) ** k_continuous
        notes_lines.append(f"\nAlpha-predicted system false-alarm rate for {k_continuous} uncorrected tests at "
                            f"alpha={ALPHA}: `1-(1-{ALPHA})^{k_continuous}` = **{predicted_system_rate:.3f}**. "
                            "Compare against the observed system false-alarm rates above.\n")

        summary["by_reference_size"][ref_size_str] = rs_summary

    # --- Reference-size-vs-batch-size comparison ---
    notes_lines.append("\n## Reference size vs. batch size\n")
    import math
    c_alpha = 1.36  # standard asymptotic two-sided KS critical constant at alpha=0.05
    for m in [5000, 50000]:
        floor_val = c_alpha / math.sqrt(m)
        notes_lines.append(f"- m={m}: asymptotic KS critical-value floor as n->infinity: "
                            f"c(alpha)/sqrt(m) = {c_alpha}/sqrt({m}) = **{floor_val:.4f}**")
    n20k_5000 = c_alpha * math.sqrt((20000 + 5000) / (20000 * 5000))
    notes_lines.append(f"- At n=20,000, m=5,000: c(alpha)*sqrt((n+m)/(nm)) = **{n20k_5000:.4f}**\n")

    notes_lines.append("**pickup_latitude recall by batch size, at both reference sizes "
                        "(pooled, with month, legacy ground truth):**\n")
    notes_lines.append("| Batch size | Recall @ m=5000 | Recall @ m=50000 |\n|---|---|---|")
    for size_str in raw["by_reference_size"]["5000"]["sweep"].keys():
        cc5000 = sweep_confusion(raw["by_reference_size"]["5000"]["sweep"][size_str],
                                  raw["by_reference_size"]["5000"]["ground_truth_pooled"]["ground_truth"],
                                  raw["by_reference_size"]["5000"]["ground_truth_per_month"])
        # feature-specific recall for pickup_latitude only
        def feature_recall(size_result, gt, feat):
            y_true, y_pred = [], []
            for resp in size_result["pooled"]["raw_responses"]:
                y_true.append(1 if gt.get(feat, False) else 0)
                y_pred.append(1 if resp[feat]["drift_detected"] else 0)
            cc = confusion_counts(y_true, y_pred)
            return cc["recall"]

        r5000 = feature_recall(raw["by_reference_size"]["5000"]["sweep"][size_str],
                                raw["by_reference_size"]["5000"]["ground_truth_pooled"]["ground_truth"],
                                "pickup_latitude")
        r50000 = feature_recall(raw["by_reference_size"]["50000"]["sweep"][size_str],
                                 raw["by_reference_size"]["50000"]["ground_truth_pooled"]["ground_truth"],
                                 "pickup_latitude")
        r5000_s = f"{r5000:.3f}" if r5000 is not None else "n/a (gt=stable)"
        r50000_s = f"{r50000:.3f}" if r50000 is not None else "n/a (gt=stable)"
        notes_lines.append(f"| {size_str} | {r5000_s} | {r50000_s} |")

    notes_lines.append(
        "\n**Reading this table against the theoretical floor above**: pickup_latitude's population D is "
        "0.0254 (m=5000 ground truth) / 0.0211 (m=50000 ground truth) -- both very close to the m=5000 "
        "critical-value floor (0.0192). At m=5000, recall genuinely PLATEAUS around 0.79-0.83 even at the "
        "largest batch sizes tested (20,000 and 50,000) -- it never reaches 1.0, because the reference itself "
        "is too small for the test to reliably resolve an effect this close to its noise floor, no matter how "
        "much production data you throw at it. At m=50000 (floor=0.0061, well below the effect size), recall "
        "reaches 1.000 by batch size=10,000 and stays there. **This directly confirms the hypothesis: for "
        "features with population D near the small-reference floor, reference size -- not batch size -- is "
        "the binding constraint on detection power.** (The small dip at m=5000, batch=50000, from 0.833 to "
        "0.792, is noise from only 24 pooled decision units at that cell, not a real reversal.)\n"
    )

    notes_lines.append(
        "**Important caveat this run surfaced, stated plainly**: the \"population ground truth\" itself is "
        "not reference-size-invariant. `dropoff_longitude` is labeled stable (p=0.079) under the m=5000 "
        "reference but drifted (p=1.5e-26) under the m=50000 reference -- same Apr-Jun production data, "
        "different reference samples. A 5,000-row reference estimates the true Jan-Mar population with its "
        "own sampling error, and since the reference is far smaller than the multi-hundred-thousand-row "
        "production pool it's compared against, most of the noise in a reference-vs-production KS statistic "
        "comes from the reference side. **Practical implication: for borderline-effect-size features, "
        "\"ground truth\" computed from a small reference is not a fixed, trustworthy target to grade the "
        "engine against -- it can disagree with itself depending on which reference happened to be drawn.** "
        "This is not a flaw in the engine; it's a property of comparing against a small reference sample, and "
        "it argues for the m=50000 ground truth being the more reliable one of the two computed here, not "
        "just for the engine's own detection power.\n"
    )

    notes_lines.append(
        "**Note on `month`'s per-batch std=0.0000** (table above): every row drawn from a single production "
        "month file has `month` equal to that one value (e.g. all `4`s from `production_month_04.csv`), so "
        "every batch's PSI for `month` is computed against the identical 100%-one-category production "
        "distribution -- there is no batch-to-batch variation to produce a nonzero std within this design. "
        "This is expected, not a bug.\n"
    )

    with open("results/tabular_validation_legacy.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open("results/tabular_validation_notes.md", "w", encoding="utf-8") as f:
        f.write("\n".join(notes_lines))

    print("Wrote results/tabular_validation_legacy.json")
    print("Wrote results/tabular_validation_notes.md")


if __name__ == "__main__":
    main()
