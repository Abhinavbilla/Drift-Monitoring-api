"""
Step 2 (e): turn results/step2e_text_image_raw.json into tables.
Reads only the raw draws -- every number in the output is computed here from
them, never typed in by hand.

Writes results/step2e_text_image_report.md and .json.
Scoring labels are the PRE-REGISTERED ones in step2e_validate.py's docstring.
"""

import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "step2e_text_image_raw.json")
OUT_MD = os.path.join(ROOT, "results", "step2e_text_image_report.md")
OUT_JSON = os.path.join(ROOT, "results", "step2e_text_image_report.json")

ALPHA = 0.05
LEGACY_CUT = 0.65
FLOORS = [0.55, 0.60, 0.65]
NEG = ["aa"]
POS = ["mix25", "mix50", "mix100"]
MIX_ORDER = ["mix05", "mix10", "mix25", "mix50", "mix100"]
MIX_LABEL = {"mix05": "5%", "mix10": "10%", "mix25": "25%", "mix50": "50%", "mix100": "100%"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rules(r):
    """{rule name: bool detected} for one draw."""
    out = {"legacy": r["auc"] > LEGACY_CUT}
    for f in FLOORS:
        out[f"cal_floor{f:.2f}"] = bool(r["p"] < ALPHA and r["auc"] >= f)
    return out


RULES = ["legacy"] + [f"cal_floor{f:.2f}" for f in FLOORS]


def rate(rows, rule):
    k = sum(rules(r)[rule] for r in rows)
    return k, len(rows)


def fmt(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100*k/n:.1f}% ({lo*100:.0f}-{hi*100:.0f})"


def main():
    raw = json.load(open(RAW))
    rows = raw["rows"]
    by = defaultdict(list)
    for r in rows:
        by[(r["modality"], r["n_ref"], r["n_batch"], r["scenario"])].append(r)
    mods = sorted({r["modality"] for r in rows})
    n_refs = sorted({r["n_ref"] for r in rows})
    n_bs = sorted({r["n_batch"] for r in rows})

    md, summary = [], {"integrity_check": raw["integrity_check"], "skipped": raw["skipped"]}
    md.append("# Step 2 (e) -- text/image validation on PetFinder.my\n")
    md.append(f"Integrity check (real `analyze()` vs script): {raw['integrity_check']['checked'] - raw['integrity_check']['mismatches']}"
              f"/{raw['integrity_check']['checked']} identical.\n")
    md.append("Cells show detection rate with 95% Wilson interval (low-high %). "
              f"alpha={ALPHA}; legacy = AUC>{LEGACY_CUT}; cal_floorX = p<alpha AND AUC>=X.\n")

    # ---- 1. A/A false alarms ----
    md.append("## 1. A/A false-alarm rate (dogs vs dogs; ideal <= 5%)\n")
    summary["aa"] = {}
    for m in mods:
        md.append(f"### {m}\n")
        md.append("| n_ref | n_batch | draws | mean AUC | " + " | ".join(RULES) + " |")
        md.append("|---|---|---|---|" + "---|" * len(RULES))
        for nr in n_refs:
            for nb in n_bs:
                g = by.get((m, nr, nb, "aa"), [])
                if not g:
                    continue
                aucs = [r["auc"] for r in g]
                cells = [fmt(*rate(g, rl)) for rl in RULES]
                md.append(f"| {nr} | {nb} | {len(g)} | {np.mean(aucs):.3f} | " + " | ".join(cells) + " |")
                summary["aa"][f"{m}|{nr}|{nb}"] = {
                    "draws": len(g), "mean_auc": float(np.mean(aucs)),
                    **{rl: rate(g, rl)[0] / len(g) for rl in RULES},
                }
        md.append("")

    # ---- 2. calibration check: are A/A p-values ~uniform? ----
    md.append("## 2. Is the p-value calibrated on real embeddings? (A/A: P(p<0.05) should be ~5%, P(p<0.10) ~10%)\n")
    md.append("| modality | n_ref | n_batch | P(p<0.05) | P(p<0.10) | median p |")
    md.append("|---|---|---|---|---|---|")
    summary["p_calibration"] = {}
    for m in mods:
        for nr in n_refs:
            for nb in n_bs:
                g = by.get((m, nr, nb, "aa"), [])
                if not g:
                    continue
                ps = np.array([r["p"] for r in g])
                md.append(f"| {m} | {nr} | {nb} | {fmt(int((ps<0.05).sum()), len(ps))} | "
                          f"{fmt(int((ps<0.10).sum()), len(ps))} | {np.median(ps):.3f} |")
                summary["p_calibration"][f"{m}|{nr}|{nb}"] = {
                    "p_lt_005": float((ps < 0.05).mean()), "p_lt_010": float((ps < 0.10).mean()),
                    "median_p": float(np.median(ps))}
    md.append("")

    # ---- 3. power curve ----
    md.append("## 3. Power curve: detection rate by % of batch that is cats\n")
    summary["power"] = {}
    for m in mods:
        for nr in n_refs:
            md.append(f"### {m}, n_ref={nr}\n")
            md.append("| rule | n_batch | " + " | ".join(MIX_LABEL[s] for s in MIX_ORDER) + " |")
            md.append("|---|---|" + "---|" * len(MIX_ORDER))
            for rl in RULES:
                for nb in n_bs:
                    cells = []
                    for s in MIX_ORDER:
                        g = by.get((m, nr, nb, s), [])
                        cells.append(fmt(*rate(g, rl)))
                        summary["power"][f"{m}|{nr}|{nb}|{s}|{rl}"] = (rate(g, rl)[0] / len(g)) if g else None
                    md.append(f"| {rl} | {nb} | " + " | ".join(cells) + " |")
            md.append("")

    # ---- 4. classification metrics ----
    md.append("## 4. Precision / recall / F1\n")
    md.append(f"Negatives = {NEG}; positives = {POS} (pre-registered). 5%/10% mixtures and the within-dog "
              "scenarios are excluded from this table.\n")
    summary["classification"] = {}
    for m in mods:
        md.append(f"### {m} (pooled over all n_ref / n_batch)\n")
        md.append("| rule | TN | FP | FN | TP | precision | recall | F1 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for rl in RULES:
            y, yhat = [], []
            for (mm, nr, nb, s), g in by.items():
                if mm != m or s not in NEG + POS:
                    continue
                for r in g:
                    y.append(1 if s in POS else 0)
                    yhat.append(int(rules(r)[rl]))
            tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
            pr = precision_score(y, yhat, zero_division=0)
            rc = recall_score(y, yhat, zero_division=0)
            f1 = f1_score(y, yhat, zero_division=0)
            md.append(f"| {rl} | {tn} | {fp} | {fn} | {tp} | {pr:.3f} | {rc:.3f} | {f1:.3f} |")
            summary["classification"][f"{m}|pooled|{rl}"] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                                                                   precision=pr, recall=rc, f1=f1)
        md.append("")
        md.append(f"### {m}, by batch size (F1 per rule)\n")
        md.append("| n_ref | n_batch | " + " | ".join(RULES) + " |")
        md.append("|---|---|" + "---|" * len(RULES))
        for nr in n_refs:
            for nb in n_bs:
                cells = []
                for rl in RULES:
                    y, yhat = [], []
                    for s in NEG + POS:
                        for r in by.get((m, nr, nb, s), []):
                            y.append(1 if s in POS else 0)
                            yhat.append(int(rules(r)[rl]))
                    cells.append(f"{f1_score(y, yhat, zero_division=0):.3f}" if y else "n/a")
                    summary["classification"][f"{m}|{nr}|{nb}|{rl}"] = {"f1": f1_score(y, yhat, zero_division=0) if y else None}
                md.append(f"| {nr} | {nb} | " + " | ".join(cells) + " |")
        md.append("")

    # ---- 5. exploratory ----
    md.append("## 5. Exploratory (NOT scored as drift/no-drift): subtle within-dog differences\n")
    md.append("`age`: young dogs (<=6 mo) as reference vs adult dogs (>=24 mo) as batch. "
              "`adopt`: fast-adopted (speed 0-1) vs not-adopted (speed 4). Whether these differ in text/photos is "
              "unknown a priori; these are detection rates, not accuracy.\n")
    md.append("| modality | scenario | n_ref | n_batch | draws | mean AUC | " + " | ".join(RULES) + " |")
    md.append("|---|---|---|---|---|---|" + "---|" * len(RULES))
    for m in mods:
        for s in ["age", "adopt"]:
            for nr in n_refs:
                for nb in n_bs:
                    g = by.get((m, nr, nb, s), [])
                    if not g:
                        continue
                    md.append(f"| {m} | {s} | {nr} | {nb} | {len(g)} | {np.mean([r['auc'] for r in g]):.3f} | "
                              + " | ".join(fmt(*rate(g, rl)) for rl in RULES) + " |")
    md.append("")
    if raw["skipped"]:
        md.append("## Skipped cells\n")
        for s in raw["skipped"]:
            md.append(f"- {s}")

    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(summary, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
