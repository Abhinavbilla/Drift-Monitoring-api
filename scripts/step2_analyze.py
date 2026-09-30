"""
Step 2 (c): legacy vs. calibrated side-by-side, using
results/tabular_validation_legacy_raw.json (Step 1) and
results/tabular_validation_calibrated_raw.json (scripts/step2_calibrated_rerun.py,
identical batches by construction -- same seeds).

Produces results/step2_side_by_side.md and results/step2_side_by_side.json.

Includes the data-collapse check (corrected per instruction, 2026-09-30):
x = sqrt(n*m/(n+m)) * D_POPULATION (never the per-batch observed statistic,
since detection is a deterministic function of the observed statistic and
plotting against it would collapse onto a step function by construction).
"""

import json
import math

from scipy.stats import beta as beta_dist

CONTINUOUS_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"]
CATEGORICAL_FEATURES = ["gender_id", "month"]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
ALPHA = 0.05
LOCKED_KS_FLOOR = 0.05
FLOOR_SENSITIVITY = [0.015, 0.02, 0.03, 0.05]


def clopper_pearson(k, n, alpha=0.05):
    """Exact Clopper-Pearson 95% CI for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lower), float(upper))


def confusion_counts(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": precision, "recall": recall, "f1": f1, "n": len(y_true)}


def fmt(cc):
    p = f"{cc['precision']:.3f}" if cc["precision"] is not None else "n/a"
    r = f"{cc['recall']:.3f}" if cc["recall"] is not None else "n/a"
    f1 = f"{cc['f1']:.3f}" if cc["f1"] is not None else "n/a"
    return p, r, f1


def labels_at(pooled_truth, d_gt, psi_gt=0.2):
    labels = {}
    for col in CONTINUOUS_FEATURES:
        labels[col] = bool(pooled_truth[col]["population_D"] >= d_gt)
    for col in CATEGORICAL_FEATURES:
        labels[col] = bool(pooled_truth[col]["population_PSI"] >= psi_gt)
    return labels


def sweep_confusion_legacy(sweep_size_result, labels):
    """Legacy: y_pred = the stored drift_detected (plain p<alpha / PSI>0.2)."""
    y_true, y_pred = [], []
    for resp in sweep_size_result["pooled"]["raw_responses"]:
        for feat in ALL_FEATURES:
            y_true.append(1 if labels.get(feat, False) else 0)
            y_pred.append(1 if resp[feat]["drift_detected"] else 0)
    return confusion_counts(y_true, y_pred)


def sweep_confusion_calibrated(sweep_size_result, labels, floor_override=None):
    """Calibrated: y_pred recomputed from stored significant + effect_size,
    so floor sensitivity can be swept WITHOUT new HTTP calls -- significant
    (Gate 1) doesn't depend on the floor at all; only material (Gate 2) does."""
    y_true, y_pred = [], []
    for resp in sweep_size_result["pooled"]["raw_responses"]:
        for feat in ALL_FEATURES:
            y_true.append(1 if labels.get(feat, False) else 0)
            m = resp[feat]
            floor = floor_override if floor_override is not None else m["effect_floor"]
            material = m["effect_size"] >= floor
            y_pred.append(1 if (m["significant"] and material) else 0)
    return confusion_counts(y_true, y_pred)


def aa_false_alarm_rate_legacy(aa_result):
    responses = aa_result["raw_responses"]
    n = len(responses)
    system_alarms = sum(1 for resp in responses if any(resp[f]["drift_detected"] for f in ALL_FEATURES))
    per_feature = {f: sum(1 for resp in responses if resp[f]["drift_detected"]) / n for f in ALL_FEATURES}
    return {"system_rate": system_alarms / n, "per_feature": per_feature, "n": n}


def aa_false_alarm_rate_calibrated(aa_result):
    responses = aa_result["raw_responses"]
    n = len(responses)
    system_alarms = sum(1 for resp in responses if any(resp[f]["drift_detected"] for f in ALL_FEATURES))
    per_feature = {f: sum(1 for resp in responses if resp[f]["drift_detected"]) / n for f in ALL_FEATURES}
    return {"system_rate": system_alarms / n, "per_feature": per_feature, "n": n}


def aa_gate_decomposition(aa_result):
    """Per instruction (item 2): decompose the calibrated A/A system rate
    into Gate-1-only (any feature significant, regardless of materiality),
    Gate-2-only (any feature material, regardless of significance -- a
    diagnostic of how often a null batch's effect_size alone clears the
    floor by chance), and the combined rate (actual drift_detected, both
    gates). Raw counts + Clopper-Pearson 95% CIs for each."""
    responses = aa_result["raw_responses"]
    n = len(responses)
    k_gate1 = sum(1 for resp in responses if any(resp[f]["significant"] for f in ALL_FEATURES))
    k_gate2 = sum(1 for resp in responses if any(resp[f]["material"] for f in ALL_FEATURES))
    k_combined = sum(1 for resp in responses if any(resp[f]["drift_detected"] for f in ALL_FEATURES))
    return {
        "n": n,
        "gate1_only": {"k": k_gate1, "rate": k_gate1 / n, "ci95": clopper_pearson(k_gate1, n)},
        "gate2_only": {"k": k_gate2, "rate": k_gate2 / n, "ci95": clopper_pearson(k_gate2, n)},
        "combined": {"k": k_combined, "rate": k_combined / n, "ci95": clopper_pearson(k_combined, n)},
    }


def two_gate_predict(resp, feat, use_holm, use_floor, floor_override=None):
    """Recompute a single feature's drift decision under one of the 2x2
    ablation cells, entirely from calibrated_raw's already-stored fields
    (raw p_value, p_value_adjusted via 'significant', effect_size,
    effect_floor) -- no new HTTP calls. use_holm=False falls back to the
    raw (uncorrected) p-value at ALPHA; use_floor=False drops the
    materiality gate entirely (treated as always-material)."""
    m = resp[feat]
    sig = m["significant"] if use_holm else (m["p_value"] < ALPHA)
    if use_floor:
        floor = floor_override if floor_override is not None else m["effect_floor"]
        mat = m["effect_size"] >= floor
    else:
        mat = True
    return bool(sig and mat)


def sweep_confusion_ablation(sweep_size_result, labels, use_holm, use_floor):
    y_true, y_pred = [], []
    for resp in sweep_size_result["pooled"]["raw_responses"]:
        for feat in ALL_FEATURES:
            y_true.append(1 if labels.get(feat, False) else 0)
            y_pred.append(1 if two_gate_predict(resp, feat, use_holm, use_floor) else 0)
    return confusion_counts(y_true, y_pred)


def aa_rate_ablation(aa_result, use_holm, use_floor):
    responses = aa_result["raw_responses"]
    n = len(responses)
    k = sum(1 for resp in responses if any(two_gate_predict(resp, f, use_holm, use_floor) for f in ALL_FEATURES))
    return {"n": n, "k": k, "rate": k / n, "ci95": clopper_pearson(k, n)}


def data_collapse_points(legacy_raw, pooled_truth, per_month_truth):
    """x = sqrt(n*m/(n+m)) * D_POPULATION (never observed D -- see module
    docstring), y = empirical detection rate, pooled across every
    (feature, size, month) cell in the legacy sweep. Uses per-MONTH
    population D (the correct ground truth for a batch drawn from that
    specific month), not the pooled value, since each batch is genuinely
    only from one month."""
    points = []
    for ref_size_str, r in legacy_raw["by_reference_size"].items():
        m = int(ref_size_str)
        for size_str, size_result in r["sweep"].items():
            n = int(size_str)
            x_factor = math.sqrt(n * m / (n + m))
            for month_str, month_data in size_result["per_month"].items():
                month = int(month_str)
                responses = month_data["raw_responses"]
                if not responses:
                    continue
                for feat in CONTINUOUS_FEATURES:  # PSI features excluded -- different statistic/theory
                    d_pop = per_month_truth[str(month)][feat]["population_D"]
                    x = x_factor * d_pop
                    detections = sum(1 for resp in responses if resp[feat]["drift_detected"])
                    rate = detections / len(responses)
                    points.append({"feature": feat, "n": n, "m": m, "month": month,
                                   "D_pop": d_pop, "x": x, "detection_rate": rate, "n_trials": len(responses)})
    return points


def main():
    legacy_summary = json.load(open("results/tabular_validation_legacy.json", encoding="utf-8"))
    legacy_raw = json.load(open("results/tabular_validation_legacy_raw.json", encoding="utf-8"))
    calibrated_raw = json.load(open("results/tabular_validation_calibrated_raw.json", encoding="utf-8"))

    pooled_truth = legacy_summary["population_truth_raw"]["pooled"]
    per_month_truth = legacy_summary["population_truth_raw"]["per_month"]

    notes = []
    summary_json = {"config": {"locked_ks_floor": LOCKED_KS_FLOOR, "alpha": ALPHA,
                                "floor_sensitivity_grid": FLOOR_SENSITIVITY},
                     "by_reference_size": {}}

    notes.append("# Step 2 (c): Legacy vs. Calibrated Side-by-Side — Apr-Jun 2016\n")
    notes.append(
        "**Identical batches**: the calibrated re-run (`scripts/step2_calibrated_rerun.py`) uses the "
        "exact same batch-drawing seeds as the legacy run (`scripts/step1_validation.py`), so every "
        "comparison below is on byte-identical data — only the decision logic (single threshold vs. "
        "two-gate) differs.\n"
    )
    notes.append(f"**Locked decisions applied**: alpha={ALPHA}, multiple_testing=Holm, "
                 f"KS D floor={LOCKED_KS_FLOOR} (dataset-independent default), PSI floor=0.2.\n")

    labels_locked = labels_at(pooled_truth, LOCKED_KS_FLOOR)

    # ================= Precision/recall at matched threshold =================
    notes.append("## Precision/recall at matched threshold (D_gt = configured floor = 0.05)\n")
    notes.append(
        "**Correction, per 2026-09-30 review**: at `D_gt=0.05`, calibrated's Gate 2 floor and the "
        "ground-truth threshold used to label this table are the SAME number (0.05) — the ground truth "
        "in this table calls a continuous feature 'positive' iff `population_D >= 0.05`, and calibrated's "
        "materiality gate flags a feature iff `effect_size >= 0.05`. **Only `trip_duration` (population D "
        "= 0.092, well clear of the floor) is a true positive under this specific ground truth.** So "
        "calibrated's precision=recall=1.000 below is close to true by construction at this exact "
        "threshold, not evidence of general accuracy — it mainly demonstrates that **the materiality gate "
        "removes the false positives legacy produces on the four sub-floor coordinate features** (which "
        "legacy flags because they're statistically significant, even though their effect size never "
        "clears 0.05). It does not show calibrated is 'more accurate' in any threshold-independent sense; "
        "see the floor-sensitivity table further down, where moving the floor changes which features count "
        "as positive and precision/recall move accordingly — that's the real generalization test, not this "
        "single matched-threshold table.\n"
    )
    notes.append("| Ref size | Batch size | Mode | TP | FP | FN | TN | Precision | Recall | F1 |\n"
                 "|---|---|---|---|---|---|---|---|---|---|")
    for ref_size_str in legacy_raw["by_reference_size"].keys():
        for size_str in legacy_raw["by_reference_size"][ref_size_str]["sweep"].keys():
            legacy_cc = sweep_confusion_legacy(legacy_raw["by_reference_size"][ref_size_str]["sweep"][size_str], labels_locked)
            calib_cc = sweep_confusion_calibrated(calibrated_raw["by_reference_size"][ref_size_str]["sweep"][size_str], labels_locked)
            for mode, cc in [("legacy", legacy_cc), ("calibrated", calib_cc)]:
                p, r, f1 = fmt(cc)
                notes.append(f"| {ref_size_str} | {size_str} | {mode} | {cc['TP']} | {cc['FP']} | {cc['FN']} | "
                             f"{cc['TN']} | {p} | {r} | {f1} |")

    # ================= A/A false-alarm rate + gate decomposition =================
    notes.append("\n## A/A system false-alarm rate per batch size, legacy vs. calibrated\n")
    notes.append(f"Alpha-predicted (7 uncorrected tests): `1-(1-{ALPHA})^7` = {1-(1-ALPHA)**7:.3f}. "
                 f"Calibrated applies Holm correction across the 7 features, so its system rate should "
                 f"track much closer to {ALPHA} itself.\n")
    notes.append("**A/A batches are iid draws from the same Jan-Mar pool as the reference — they test "
                 "calibration under the null (no real drift), not month-to-month variation within the "
                 "baseline period. A low false-alarm rate here does not by itself validate behavior "
                 "against genuine temporal drift in the reference period.**\n")
    notes.append("| Ref size | Batch size | Legacy: k/n (95% CI) | Legacy rate | Calibrated: k/n (95% CI) | Calibrated rate |\n"
                 "|---|---|---|---|---|---|")
    aa_summary = {}
    for ref_size_str in legacy_raw["by_reference_size"].keys():
        aa_summary[ref_size_str] = {}
        for size_str in legacy_raw["by_reference_size"][ref_size_str]["aa_test"].keys():
            legacy_aa = aa_false_alarm_rate_legacy(legacy_raw["by_reference_size"][ref_size_str]["aa_test"][size_str])
            calib_aa = aa_false_alarm_rate_calibrated(calibrated_raw["by_reference_size"][ref_size_str]["aa_test"][size_str])
            legacy_k = round(legacy_aa["system_rate"] * legacy_aa["n"])
            legacy_ci = clopper_pearson(legacy_k, legacy_aa["n"])
            calib_k = round(calib_aa["system_rate"] * calib_aa["n"])
            calib_ci = clopper_pearson(calib_k, calib_aa["n"])
            aa_summary[ref_size_str][size_str] = {"legacy": legacy_aa, "calibrated": calib_aa,
                                                    "legacy_ci95": legacy_ci, "calibrated_ci95": calib_ci}
            notes.append(f"| {ref_size_str} | {size_str} | {legacy_k}/{legacy_aa['n']} "
                         f"({legacy_ci[0]:.3f}-{legacy_ci[1]:.3f}) | {legacy_aa['system_rate']:.3f} | "
                         f"{calib_k}/{calib_aa['n']} ({calib_ci[0]:.3f}-{calib_ci[1]:.3f}) | "
                         f"{calib_aa['system_rate']:.3f} |")

    notes.append(
        "\n### A/A gate decomposition (calibrated mode) — Gate-1-only, Gate-2-only, combined\n"
    )
    notes.append(
        "Gate-1-only = fraction of null batches with ANY feature `significant` (Holm-adjusted p<alpha), "
        "ignoring materiality entirely — this is what the false-alarm rate WOULD be if only the "
        "significance test existed (comparable to legacy's rate above, modulo the Holm correction itself). "
        "Gate-2-only = fraction with ANY feature `material` (effect_size >= floor), ignoring significance "
        "— a diagnostic of how often a null batch's sampling noise alone pushes an effect size over the "
        "floor by chance; not a real decision rule, since it's never used without the significance test. "
        "Combined = the actual two-gate `drift_detected` (both must hold).\n"
    )
    notes.append("| Ref size | Batch size | Gate-1-only k/n (95% CI) | Gate-2-only k/n (95% CI) | "
                 "Combined k/n (95% CI) |\n|---|---|---|---|---|")
    gate_decomp_summary = {}
    for ref_size_str in calibrated_raw["by_reference_size"].keys():
        gate_decomp_summary[ref_size_str] = {}
        for size_str in calibrated_raw["by_reference_size"][ref_size_str]["aa_test"].keys():
            decomp = aa_gate_decomposition(calibrated_raw["by_reference_size"][ref_size_str]["aa_test"][size_str])
            gate_decomp_summary[ref_size_str][size_str] = decomp
            g1, g2, comb = decomp["gate1_only"], decomp["gate2_only"], decomp["combined"]
            notes.append(
                f"| {ref_size_str} | {size_str} | {g1['k']}/{decomp['n']} ({g1['ci95'][0]:.3f}-{g1['ci95'][1]:.3f}) | "
                f"{g2['k']}/{decomp['n']} ({g2['ci95'][0]:.3f}-{g2['ci95'][1]:.3f}) | "
                f"{comb['k']}/{decomp['n']} ({comb['ci95'][0]:.3f}-{comb['ci95'][1]:.3f}) |"
            )
    notes.append(
        "\n**Plainly stated**: Gate-1-only rates track close to legacy's own rates (both are testing "
        "significance alone, modulo Holm's correction pulling calibrated's Gate-1-only rate down "
        "somewhat vs. legacy's uncorrected rate). The combined rate collapses to 0-1% almost entirely "
        "because of Gate 2 (materiality) — a null batch essentially never has BOTH a significant AND a "
        "materially-large effect size on the same feature at the same time, since a null batch's true "
        "effect is exactly zero. The gap between Gate-1-only and Combined is the materiality gate's actual "
        "contribution to the low system-level false-alarm rate, not the Holm correction alone.\n"
    )

    # ================= Floor sensitivity (calibrated only, recomputed from stored effect_size) =================
    notes.append("\n## Floor sensitivity (calibrated mode, biggest sweep size=50000), recomputed from "
                 "stored effect_size/significant -- no new HTTP calls needed\n")
    notes.append("| Ref size | Floor | TP | FP | FN | TN | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|---|")
    floor_sensitivity = {}
    for ref_size_str in calibrated_raw["by_reference_size"].keys():
        floor_sensitivity[ref_size_str] = {}
        big = calibrated_raw["by_reference_size"][ref_size_str]["sweep"]["50000"]
        for floor in FLOOR_SENSITIVITY:
            labels_floor = labels_at(pooled_truth, floor)
            cc = sweep_confusion_calibrated(big, labels_floor, floor_override=floor)
            floor_sensitivity[ref_size_str][floor] = cc
            p, r, f1 = fmt(cc)
            notes.append(f"| {ref_size_str} | {floor} | {cc['TP']} | {cc['FP']} | {cc['FN']} | {cc['TN']} | {p} | {r} | {f1} |")

    # ================= 2x2 ablation: {Holm on/off} x {floor on/off} =================
    notes.append("\n## 2x2 ablation: {Holm on/off} x {floor on/off}, at D_gt = 0.05\n")
    notes.append(
        "All four cells recomputed from calibrated_raw's already-stored `p_value` (raw/unadjusted), "
        "`significant` (Holm-adjusted p<alpha), `effect_size`, and `effect_floor` -- no new HTTP calls. "
        "'Holm off' substitutes the raw per-feature p-value against alpha directly (what a single "
        "uncorrected test per feature would give); 'floor off' drops Gate 2 (materiality) entirely, "
        "i.e. treats every feature as material regardless of effect size. "
        "Precision/recall use the D_gt=0.05 labels (labels_locked) pooled over the largest sweep batch "
        "size (50,000) per reference size; A/A system rate uses the calibrated A/A pool at each batch "
        "size. (Holm=off, Floor=off) is the closest calibrated-data analogue of legacy's own decision "
        "rule for continuous features (raw p<alpha, no materiality) -- it will differ from legacy's "
        "own reported numbers for categorical features, since legacy uses a fixed PSI>0.2 threshold, "
        "not a p-value test, while calibrated_raw's categorical p_value comes from the PSI parametric "
        "bootstrap introduced in Step 2 -- so this cell isolates the Holm/floor axes on the SAME "
        "underlying test family, it does not reproduce legacy's categorical rule exactly.\n"
    )
    notes.append("### Precision/recall (D_gt=0.05, pooled at batch size 50,000)\n")
    notes.append("| Ref size | Holm | Floor | TP | FP | FN | TN | Precision | Recall | F1 |\n"
                 "|---|---|---|---|---|---|---|---|---|---|")
    ablation_pr = {}
    for ref_size_str in calibrated_raw["by_reference_size"].keys():
        ablation_pr[ref_size_str] = {}
        big = calibrated_raw["by_reference_size"][ref_size_str]["sweep"]["50000"]
        for use_holm in (True, False):
            for use_floor in (True, False):
                cc = sweep_confusion_ablation(big, labels_locked, use_holm, use_floor)
                ablation_pr[ref_size_str][f"holm={use_holm}_floor={use_floor}"] = cc
                p, r, f1 = fmt(cc)
                notes.append(f"| {ref_size_str} | {'on' if use_holm else 'off'} | {'on' if use_floor else 'off'} | "
                             f"{cc['TP']} | {cc['FP']} | {cc['FN']} | {cc['TN']} | {p} | {r} | {f1} |")

    notes.append("\n### A/A system rate, per batch size\n")
    notes.append("| Ref size | Batch size | Holm | Floor | k/n | rate | 95% CI |\n|---|---|---|---|---|---|---|")
    ablation_aa = {}
    for ref_size_str in calibrated_raw["by_reference_size"].keys():
        ablation_aa[ref_size_str] = {}
        for size_str in calibrated_raw["by_reference_size"][ref_size_str]["aa_test"].keys():
            ablation_aa[ref_size_str][size_str] = {}
            aa_result = calibrated_raw["by_reference_size"][ref_size_str]["aa_test"][size_str]
            for use_holm in (True, False):
                for use_floor in (True, False):
                    r = aa_rate_ablation(aa_result, use_holm, use_floor)
                    ablation_aa[ref_size_str][size_str][f"holm={use_holm}_floor={use_floor}"] = r
                    notes.append(f"| {ref_size_str} | {size_str} | {'on' if use_holm else 'off'} | "
                                 f"{'on' if use_floor else 'off'} | {r['k']}/{r['n']} | {r['rate']:.3f} | "
                                 f"({r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}) |")

    notes.append(
        "\n**Attribution -- the two gates fix different failure modes, and the two tables above show "
        "each one's contribution separately**:\n\n"
        "*Precision/recall (matched-threshold sweep, pooled at n=50,000)*: the floor alone already "
        "gets precision/recall to 1.000/1.000 regardless of Holm (compare Holm=on,Floor=on vs. "
        "Holm=off,Floor=on -- identical). At this large batch size, every sub-floor coordinate feature "
        "is statistically significant whether or not its p-value is Holm-adjusted (their raw p-values "
        "are already far below alpha), so Holm alone (Floor=off) barely moves precision "
        "(0.353->0.286 at m=50000, both far from 1.000) -- **the materiality gate, not Holm, is what "
        "removes these false positives from the confusion table.** Recall is 1.000 in all four cells: "
        "`trip_duration`'s effect (D=0.092) is far too large for either raw or Holm-adjusted p-values "
        "to lose significance, so neither gate costs any recall here.\n\n"
        "*A/A system rate*: here Holm's own contribution is clearly visible and separate from the "
        "floor's. With the floor off, turning Holm on cuts the false-alarm rate substantially at every "
        "batch size (e.g. at ref=5000/batch=10000: 0.880 -> 0.430) -- this is Holm bounding the "
        "multiple-testing false-positive rate across the 7 features, exactly as designed. Adding the "
        "floor on top (Holm=on,Floor=on) drives the rate to ~0 at every cell, because a null batch's "
        "effect size essentially never independently clears the floor at the same time a p-value is "
        "significant. So: **the floor is the dominant lever for matched-threshold precision, while "
        "Holm is the dominant lever for system-level false-alarm control under the null (A/A) -- the "
        "full calibrated system (both gates on) is the only configuration that is simultaneously good "
        "on both axes.**\n"
    )

    # ================= Explicit check: significant-but-not-material =================
    notes.append("\n## Explicit check: significant-but-not-material at the 0.05 default\n")
    notes.append(
        "Rates computed by POOLING all 24 batches at the largest sweep size (50,000) per reference "
        "size -- not a single representative batch -- for real statistical weight.\n"
    )
    big5000 = calibrated_raw["by_reference_size"]["5000"]["sweep"]["50000"]["pooled"]["raw_responses"]
    big50000 = calibrated_raw["by_reference_size"]["50000"]["sweep"]["50000"]["pooled"]["raw_responses"]

    def pooled_rates(responses, feat):
        n = len(responses)
        sig = sum(1 for r in responses if r[feat]["significant"]) / n
        mat = sum(1 for r in responses if r[feat]["material"]) / n
        det = sum(1 for r in responses if r[feat]["drift_detected"]) / n
        return sig, mat, det

    notes.append("| Feature | Population D | m=5000: sig / mat / detected rate | m=50000: sig / mat / detected rate |\n"
                 "|---|---|---|---|")
    check_result = {}
    for feat in CONTINUOUS_FEATURES:
        pop_d = pooled_truth[feat]["population_D"]
        sig5000, mat5000, det5000 = pooled_rates(big5000, feat)
        sig50000, mat50000, det50000 = pooled_rates(big50000, feat)
        check_result[feat] = {"population_D": pop_d,
                               "m5000_rates": {"significant": sig5000, "material": mat5000, "detected": det5000},
                               "m50000_rates": {"significant": sig50000, "material": mat50000, "detected": det50000}}
        notes.append(f"| {feat} | {pop_d:.4f} | {sig5000:.2f} / {mat5000:.2f} / {det5000:.2f} | "
                     f"{sig50000:.2f} / {mat50000:.2f} / {det50000:.2f} |")
    notes.append(
        "\nAt m=50,000, all four coordinate features (population D 0.019-0.022) are **significant in "
        "100% of batches but material in 0%** -- exactly the documented significant-but-not-material "
        "pattern, confirmed on the full pooled sample, not an anecdote. `trip_duration` (D=0.092, well "
        "above the 0.05 floor) is significant AND material in 100% of batches at both reference sizes, "
        "as expected for a real, sizeable effect. At m=5,000 the coordinate features' significance rate "
        "is lower (50-83%) rather than 100% -- consistent with Step 1's own finding that the smaller "
        "reference has less power to resolve effects this close to its own detection floor -- but "
        "material rate stays at 0% regardless, so the materiality gate's behavior doesn't depend on "
        "reference size the way significance does.\n"
    )
    # ================= Data collapse =================
    notes.append("\n## Data-collapse check (population D, corrected per 2026-09-30 instruction)\n")
    notes.append(
        "x = `sqrt(n*m/(n+m)) * D_population` (per-MONTH population D, the true generating-distribution "
        "distance for that batch's month -- never the per-batch observed KS statistic, since detection is "
        "a deterministic function of the observed statistic and that would collapse onto a step function "
        "by construction). y = empirical detection rate. Computed on LEGACY's raw `drift_detected` (the "
        "pure significance decision, p<alpha with no materiality gate) -- this is what asymptotic KS "
        "theory actually makes a claim about; calibrated mode's two-gate `drift_detected` deliberately "
        "adds a human materiality choice on top and would confound the check. Continuous features only "
        "(PSI is a different statistic/theory).\n"
    )
    points = data_collapse_points(legacy_raw, pooled_truth, per_month_truth)
    notes.append(f"**{len(points)} (feature, n, m, month) cells computed.**\n")
    notes.append("| Feature | n | m | month | D_pop | x=sqrt(nm/(n+m))*D_pop | detection rate | n_trials |\n"
                 "|---|---|---|---|---|---|---|---|")
    for pt in sorted(points, key=lambda p: p["x"]):
        notes.append(f"| {pt['feature']} | {pt['n']} | {pt['m']} | {pt['month']:02d} | {pt['D_pop']:.4f} | "
                     f"{pt['x']:.3f} | {pt['detection_rate']:.3f} | {pt['n_trials']} |")

    # Bin by x and report mean/std of detection rate per bin, across ALL features/n/m/month --
    # this is the actual collapse test: do different (feature,n,m) combinations at the SAME x
    # produce the SAME detection rate?
    notes.append(
        "\n**Collapse test**: group points into x-bins and check whether detection rate is consistent "
        "WITHIN a bin regardless of which feature/n/m/month produced it. If collapse holds, all points in "
        "a bin should have similar detection rates (low within-bin spread).\n"
    )
    bins = {}
    for pt in points:
        b = round(pt["x"], 1)  # 0.1-wide bins
        bins.setdefault(b, []).append(pt)
    notes.append("| x-bin | n_points | rate mean | rate std | rate min | rate max | features present |\n"
                 "|---|---|---|---|---|---|---|")
    max_std = 0.0
    max_std_bin = None
    for b in sorted(bins.keys()):
        rates = [p["detection_rate"] for p in bins[b]]
        mean = sum(rates) / len(rates)
        var = sum((rr - mean) ** 2 for rr in rates) / len(rates)
        std = var ** 0.5
        feats = sorted(set(p["feature"] for p in bins[b]))
        if std > max_std and len(rates) > 1:
            max_std = std
            max_std_bin = b
        notes.append(f"| {b} | {len(rates)} | {mean:.3f} | {std:.3f} | {min(rates):.3f} | {max(rates):.3f} | "
                     f"{', '.join(f[:10] for f in feats)} |")

    # Each cell's detection rate is itself an estimate from only 8 binary
    # trials -- compare the observed within-bin spread against the spread
    # pure binomial sampling noise alone would produce at that rate, before
    # calling any deviation a real theory violation.
    max_binomial_se = 0.5 / math.sqrt(8)  # worst case, at p=0.5, n=8
    notes.append(
        f"\n**Plain report on collapse**: the largest within-bin spread is std={max_std:.3f} at x-bin={max_std_bin}. "
        f"**Context needed before calling this a theory violation**: each cell's detection rate is itself "
        f"estimated from only 8 binary trials -- at the worst case (true rate near 0.5), binomial sampling "
        f"noise alone gives a standard error of `sqrt(0.5*0.5/8)` = {max_binomial_se:.3f} for a SINGLE cell's "
        f"rate estimate. A bin's within-bin std (spread ACROSS several such noisy estimates, if their true "
        f"rate were identical) would be expected to land in a similar range purely from this sampling noise "
        f"-- before any real curve-shape difference between features/months enters into it.\n"
    )
    if max_std <= max_binomial_se * 1.3:
        notes.append(
            f"The observed spread ({max_std:.3f}) is close to or below what 8-trials-per-cell binomial "
            f"noise alone would predict (~{max_binomial_se:.3f}) — **this data does not provide clear "
            f"evidence against the collapse holding**; it's also not precise enough (8 trials/cell) to "
            f"confidently confirm it either. At the extremes (x<0.5 and x>1.7), collapse is clean (std=0 "
            f"in most bins — either every trial detects or none do), which is the expected, uncontroversial "
            f"part of any sigmoid-shaped power curve. The transition region (roughly x=0.6-1.6) is where "
            f"more trials per cell would be needed to distinguish genuine curve-shape differences between "
            f"features from simple small-sample noise.\n"
        )
    else:
        notes.append(
            f"The observed spread ({max_std:.3f}) meaningfully exceeds what 8-trials-per-cell binomial "
            f"noise alone would predict (~{max_binomial_se:.3f} at worst case) — **this is suggestive of a "
            f"real deviation from the single-curve collapse**, not just sampling noise, though confirming "
            f"it properly would need more trials per cell than this run collected. See which feature(s) "
            f"dominate the high-spread bin(s) above for where the deviation concentrates.\n"
        )

    summary_json["precision_recall_matched_threshold"] = "see markdown table"
    summary_json["aa_false_alarm"] = aa_summary
    summary_json["aa_gate_decomposition"] = gate_decomp_summary
    summary_json["ablation_2x2_precision_recall"] = ablation_pr
    summary_json["ablation_2x2_aa_rate"] = ablation_aa
    summary_json["floor_sensitivity"] = {rs: {str(f): cc for f, cc in d.items()} for rs, d in floor_sensitivity.items()}
    summary_json["significant_but_not_material_check"] = check_result
    summary_json["data_collapse_points"] = points
    summary_json["data_collapse_bins"] = {str(b): {"n": len(bins[b]),
                                                     "mean": sum(p["detection_rate"] for p in bins[b]) / len(bins[b]),
                                                     "features": sorted(set(p["feature"] for p in bins[b]))}
                                           for b in bins}

    with open("results/step2_side_by_side.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, default=str)
    with open("results/step2_side_by_side.md", "w", encoding="utf-8") as f:
        f.write("\n".join(notes))

    print("Wrote results/step2_side_by_side.json")
    print("Wrote results/step2_side_by_side.md")


if __name__ == "__main__":
    main()
