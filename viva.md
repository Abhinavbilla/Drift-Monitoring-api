# Viva Guide — Drift Sentinel (Drift Monitoring API)

Questions an examiner is likely to ask, with answers grounded in this repository. Numbers come from the README's validation section and the committed outputs in `results/`. Where something was not achieved or has a known weakness, the answer says so; examiners respect an honest limitation more than a vague claim.

**How to use this:** read Part 1 until you can say it without notes. Parts 2–6 are the technical core. Part 9 is the hard questions; practise those out loud.

---

## Contents

1. [The 60-second pitch](#1-the-60-second-pitch)
2. [Problem and motivation](#2-problem-and-motivation)
3. [Core concepts](#3-core-concepts)
4. [Statistical tests for columns](#4-statistical-tests-for-columns)
5. [Multiple testing and the two-gate decision](#5-multiple-testing-and-the-two-gate-decision)
6. [Relationship drift](#6-relationship-drift)
7. [Text and image drift](#7-text-and-image-drift)
8. [Validation and results](#8-validation-and-results)
9. [Hard and critical questions](#9-hard-and-critical-questions)
10. [Tech stack, system design and architecture](#10-tech-stack-system-design-and-architecture)
11. [Security and privacy](#11-security-and-privacy)
12. [Deployment](#12-deployment)
13. [Frontend and user experience](#13-frontend-and-user-experience)
14. [Testing and engineering process](#14-testing-and-engineering-process)
15. [Limitations and future work](#15-limitations-and-future-work)
16. [Quick-fire facts](#16-quick-fire-facts)
17. [Formula sheet](#17-formula-sheet)

---

## 1. The 60-second pitch

> Machine-learning models fail quietly. They are validated on training data, deployed, and then the real data slowly changes: new customers, a new season, a broken upstream pipeline. Accuracy drops, and nobody notices until a business metric falls.
>
> Drift Sentinel watches for that. You give it the data your model learned from; it saves a **baseline**. Then every production **batch** is compared with that baseline using statistical tests, and it tells you, in plain words, **which columns changed and which relationships between columns broke**.
>
> It handles tables that mix numbers, categories, free text and images. Every decision is a calibrated hypothesis test, corrected for multiple testing (Holm) and gated by a minimum effect size, so it doesn't drown you in false alarms. It sends email and signed webhooks, keeps a history, and versions baselines.
>
> It's a FastAPI backend with a React dashboard, deployed on Azure, and I validated it on three real datasets: 4.5 million Citi Bike trips, PetFinder pet listings with photos, and Inside Airbnb Edinburgh.

---

## 2. Problem and motivation

**Q: What is data drift?**
A change between the distribution of data a model was trained on and the data it sees in production. Formally, P_train(X) ≠ P_prod(X). The model was optimised for one distribution and is now being used on another, so its guarantees no longer hold.

**Q: What is the difference between data drift and concept drift?**
- **Data (covariate) drift:** the inputs P(X) change. Example: more young customers than before.
- **Concept drift:** the relationship between inputs and the target, P(Y | X), changes. Example: the same customer profile now defaults more often because of a recession.

This project detects **data drift**, plus **relationship drift between input columns**. Concept drift needs ground-truth labels, which usually arrive late or never, so it is out of scope.

**Q: Why not just monitor model accuracy?**
Labels arrive late (a loan default takes months) or never. Input drift can be measured **immediately**, with no labels, so it's an early warning. It doesn't prove accuracy dropped, but it says "the model is now working outside what it was checked on".

**Q: What was the starting point of the project?**
A paper on out-of-distribution detection and statistical process control. The first version was a tabular engine (KS, PSI, IQR); it grew in stages: text/image/joint workflows, calibrated decisions with Holm, hardening, history/versioning/webhooks, a React frontend, and finally unified table monitoring.

**Q: Who is the user?**
An ML or data team that has a model in production. The dashboard is for setting up and reading results; in production, their pipeline sends batches through the API with an access token.

**Q: What makes this project different from a simple "run KS on every column" script?**
1. **Calibrated decisions:** a family-wise false-alarm rate (Holm) plus an effect-size floor, instead of an uncorrected p < 0.05 per column.
2. **Relationship drift:** catches the case where every column looks normal but the link between columns broke.
3. **Mixed data types in one table:** numbers, categories, text and images, all in one statistical family.
4. **Per-project null distributions:** p-values for complex tests come from the project's own reference data, not textbook formulas that break on real data.
5. **A product around it:** auth, isolation, background jobs, history, baseline versions, alert state machine, signed webhooks.

---

## 3. Core concepts

**Q: What is a baseline?**
A saved description of what "normal" data looks like, built from the reference (training) data: per-column distributions, category frequencies, embeddings for text/image, relationship statistics and the confirmed schema. Every fit creates a new **version**; old versions are kept and can be re-activated.

**Q: What is a batch?**
A set of production rows checked together. Drift is a property of a distribution, so you need many rows; one row can only be an outlier, not drift.

**Q: What four kinds of change does the system distinguish?**

| Kind | Meaning | Checked by |
|---|---|---|
| Marginal (column) drift | One column's own distribution changed | KS, PSI, Domain Classifier Test |
| Relationship drift | How two columns relate changed | Spearman, log-linear G², mid-rank shift, probes, matching |
| Joint drift | The combination of all columns changed | Not built in the unified path (see limitations) |
| Point anomaly | One row is unusual | IQR fences (`/predict`) |

**Q: Why does "no column drifted" not mean "no drift"?**
Because two columns can each keep exactly the same distribution while their pairing changes. Example: house prices and house sizes both look normal, but big houses are no longer the expensive ones. Shuffling a column is the extreme case: its distribution is unchanged, every relationship is destroyed. That's why the system has relationship tests.

**Q: What is the workflow from the user's point of view?**
1. **Profile:** upload the training table (plus an image ZIP). The profiler proposes a type for every column (numeric, categorical, text, image, ignore) with a confidence and a reason, and proposes relationships.
2. **Human review:** the user confirms or changes every type and relationship. Nothing is monitored until confirmed.
3. **Fit:** a baseline version is saved.
4. **Analyze:** each production batch goes through schema checks, column tests and relationship tests, then one Holm-corrected decision and a report.
5. **History, alerts, email, webhooks.**

**Q: Why a human review step instead of fully automatic?**
Column type inference is heuristic. An ID column that looks numeric, or a zip code that looks numeric but is categorical, would produce nonsense tests. A wrong type silently gives wrong answers, so the human confirms. The profiler's `confidence` is a heuristic agreement score, not a calibrated probability, and the UI says so.

**Q: What does the profiler look at?**
Numeric coercion, cardinality, dominant value, monotonic or index-like columns, timestamps, structured ID patterns. On top: **image** if values are filenames matching the ZIP or embedded images; **text** if values are highly unique, multi-word and long; **ignore** for URLs, identifiers and date strings.

---

## 4. Statistical tests for columns

**Q: Which test is used for numeric columns and why?**
The **two-sample Kolmogorov–Smirnov test**. Its statistic D is the largest vertical gap between the two empirical cumulative distribution functions.
- **Non-parametric:** no normality assumption. Real columns (prices, durations) are skewed.
- **Sensitive to any change:** location, spread or shape, unlike a t-test (mean only).
- **Interpretable effect size:** D is between 0 and 1 and has a direct meaning, so it also serves as the materiality measure.
- **Analytic p-value and a closed-form minimum detectable effect**, used for fit-time warnings.

**Q: Weaknesses of KS?**
Less sensitive in the tails than Anderson–Darling; with huge samples it flags tiny, meaningless differences (handled by the effect floor); it assumes continuous data (ties reduce power).

**Q: Which test is used for categorical columns and why?**
**PSI (Population Stability Index)** with a **parametric-bootstrap p-value**.
PSI = Σ (actual% − expected%) · ln(actual% / expected%), with a small epsilon for empty categories.
- Industry standard from credit-risk monitoring with known rules of thumb: below 0.1 stable, 0.1–0.2 moderate, above 0.2 significant.
- Weights changes by importance: a big category moving matters more than a tiny one.

**Q: PSI has no p-value. How do you get one?**
Parametric bootstrap at the actual batch size: draw many multinomial samples of size n from the reference category frequencies (that simulates "no drift"), compute PSI for each, and p = (1 + #{null PSI ≥ observed}) / (draws + 1). The "+1" avoids a p-value of exactly zero. This fixes the old flat "PSI > 0.2" rule, which means different things at n = 100 and n = 100,000.

**Q: Why not chi-square for categories?**
It is unreliable with rare categories and small expected counts, and at millions of rows it flags negligible shifts. In the Citi Bike validation, chi-square ground truth mislabelled `gender_id`, which was one of the bugs found.

**Q: Why not Wasserstein or Jensen–Shannon distance?**
They're good distances but give no p-value directly; you'd need the same resampling machinery, and the number is harder to explain than KS's D. Wasserstein is listed in future plans as an extra numeric metric.

**Q: What does `/predict` do?**
A single-row check. For numbers, the **IQR rule**: flagged if below Q1 − 3·IQR or above Q3 + 3·IQR. For categories, flagged if never seen in the reference. 3× rather than the textbook 1.5× keeps false alarms rare. One row cannot show drift, only an outlier.

**Q: What is the minimum detectable D?**
For a batch of size n against a reference of size m at significance α: D_min = c(α) · √((n + m) / (n·m)), with c(0.05) = 1.36. The dashboard warns at fit time if the reference is too small to resolve the 0.05 floor.

**Q: What is the DKW check?**
The Dvoretzky–Kiefer–Wolfowitz inequality bounds how far an empirical CDF from m samples can be from the true CDF: with probability ≥ 1 − α, sup |F_m − F| ≤ √(ln(2/α) / 2m). If that bound is larger than the KS floor, the reference itself is too noisy to judge drift at that size, and a warning is shown.

---

## 5. Multiple testing and the two-gate decision

**Q: What is the multiple-testing problem here?**
If you test 30 columns at α = 0.05 each and nothing changed, the chance of at least one false alarm is 1 − 0.95³⁰ ≈ 79%. A monitor that cries wolf every batch gets ignored.

**Q: How does Holm's method work?**
Sort the m p-values ascending: p(1) ≤ … ≤ p(m). Compare p(i) with α / (m − i + 1). Reject in order until the first one that fails; stop there. It controls the **family-wise error rate** (probability of any false alarm) at α, and is uniformly more powerful than Bonferroni.

**Q: Why Holm and not Benjamini–Hochberg?**
BH controls the false discovery **rate** (the expected share of alarms that are false), which tolerates some false alarms. For alerting, a single false page is costly, so FWER is the right target. BH is implemented (`bh_adjust`) and can be configured.

**Q: What is the "family"?**
All tests in one batch: every monitored column test **and** every relationship test, together. Adding more relationships lowers the per-test sensitivity, and the UI tells the user this ("adding pairs lowers per-test sensitivity").

**Q: What is the two-gate decision?**
A test is marked DRIFT only if:
1. **Statistical gate:** Holm-adjusted p < 0.05, and
2. **Materiality gate:** the effect is at least a floor: KS D ≥ 0.05, PSI ≥ 0.2, text/image AUC ≥ 0.55, Spearman Δρ ≥ 0.10, categorical interaction RMS ≥ ln 1.25, numeric↔categorical shift ≥ 0.05, probe drop ≥ 0.10.

**Q: Why do you need the effect floor if you already have a p-value?**
With big batches, every tiny difference becomes "significant". On Citi Bike, across independent reference draws, **significance alone alarmed up to 57.5% of the time with no real change** (reference 5,000, batch 20,000), because every finite reference differs slightly from the true population and large batches detect that. The full two-gate rule alarmed 0–2.5%. The floor asks "is it big enough to matter?", the p-value asks "is it real?". You need both.

**Q: What did the two-gate rule achieve on real data?**
On Citi Bike, no-change false alarms went from 2–27% per cell (single threshold, uncorrected) to 0–1%. On real drift, precision rose from 0.33–0.73 to 0.87–1.00 with recall ≥ 0.979.

**Q: Isn't that result circular?**
Partly, and the README says so: the ground truth used the same 0.05 cut-off as the floor, so that comparison mostly shows the floor removing sub-threshold alarms. The cleaner evidence is the A/A (no-change) false-alarm rate, which doesn't depend on that definition.

**Q: What are "legacy" and "calibrated" decision modes?**
Legacy is the original rule (per-feature p < 0.05 for KS, PSI > 0.2, no correction). Calibrated is two-gate + Holm. New projects default to calibrated; projects created before the change stay legacy permanently, so their behaviour never changes under them. A test proves legacy mode reproduces the old code path exactly.

**Q: What is the recommended batch size?**
The smallest batch at which the minimum detectable D is half the floor: 7,252 rows for a 5,000-row reference, 3,146 for a 50,000-row reference. In simulation (2,000 trials per cell) that gave 0% floor-crossings with no drift, versus 4.7–5.5% under the old definition, while still catching true D = 0.05 99.3–99.8% of the time.

**Q: What is the sustained alert policy?**
A per-project `{k, m}` rule: alert if k of the last m batches drifted (default 1 of 1). Alert state changes are logged only on transitions: `opened`, `still_open`, `resolved`. This avoids re-alerting every batch.

---

## 6. Relationship drift

**Q: What is relationship drift?**
A change in how two columns relate, beyond changes in each column's own distribution. The design requirement: **a change in a column's own distribution must not register as relationship drift**, otherwise you double-count what the column test already reports.

**Q: Test for numeric ↔ numeric?**
Change in **Spearman rank correlation**: |ρ_batch − ρ_ref|, floor 0.10. Rank-based, so monotone transformations and rescaling of either column leave it unchanged, and it captures any monotone relationship, not just linear (unlike Pearson).

**Q: Test for categorical ↔ categorical?**
**G² of the log-linear model with no three-way interaction** [AC][AS][CS], where A and C are the two columns and S is reference vs batch. The model says: the odds ratios between A and C are the same in reference and batch, each with its own marginals. G² measures how badly that fits. Odds ratios don't depend on category proportions, so a change in the category mix alone doesn't trigger it; it also catches re-pairing at the same strength. A plain chi-square of independence would fire whenever one column's mix shifted.

**Q: Test for numeric ↔ categorical?**
Weighted shift in each category's **mean mid-rank PIT** (probability integral transform), with the batch re-weighted to the reference category mix. A common monotone shift of the numeric column cancels, and a category-mix change cancels. Mid-ranks keep it stable under ties.

**Q: Why mid-ranks? What went wrong before?**
The first version used KS on the PIT values. Under heavy ties the null distribution of the statistic was around 0.6, i.e. the test was badly miscalibrated. Validation found it, and it was replaced by the mean mid-rank shift. That's an example of validation catching a real bug.

**Q: How do you get p-values for these relationship tests?**
**Disjoint-split null from the project's own reference data:** repeatedly split the reference into two disjoint parts, one of them the size of the production batch, compute the statistic, and see how big it gets with no drift. p = (1 + #exceed) / (K + 1), with K = max(1000, 2m/α) where m is the family size. No asymptotic formula is assumed, because chi-square asymptotics break on sparse tables and ties.

**Q: Why compute nulls at analysis time instead of at fit time?**
The plan proposed a fit-time grid of batch sizes with conservative bracketing. The implementation computes the null at the exact batch size at analysis time instead: exact and simpler, at a cost of a few seconds per analysis. A deliberate, documented deviation from the plan.

**Q: How are relationships chosen?**
The profiler proposes pairs that are clearly related in the reference: up to 25 numeric/categorical pairs and 15 embedding pairs. Weaker pairs are listed for manual adding. An informational "emergence screen" lists unwatched pairs that became strongly related in production; it never alerts.

**Q: What relationships involve text and images?**
- **Probe (text/image → categorical):** a classifier trained on reference embeddings predicts a category (e.g. description → pet type). Statistic: balanced accuracy, which ignores class-prior shift. A drop of ≥ 0.10 means the pairing broke.
- **Matching (text ↔ image):** a ridge map from image to text embeddings; AUC of "own text is more similar than another row's text". Measures pairing only.
- **Probe (text/image → numeric):** Spearman of ridge predictions. **Report-only:** shown but never alerts, because it failed validation (fooled by population-mix shifts: 26% alarms in the "mix" scenario).

**Q: What does `confounded_by` mean?**
If a probe's source column itself drifted, a probe drop might come from the column change, not a broken relationship. The probe is flagged `confounded_by` that column.

---

## 7. Text and image drift

**Q: How do you test drift on text or images?**
Convert each item to a vector with a frozen pretrained model, then test whether reference and batch vectors are distinguishable.
- Text: `all-MiniLM-L6-v2` (384 dimensions).
- Image: `resnet18`, penultimate layer (512 dimensions).

**Q: What is the Domain Classifier Test (DCT)?**
Label reference vectors 0 and batch vectors 1, train a logistic regression with 5-fold stratified cross-validation, and measure ROC-AUC on the held-out predictions. If the two sets are from the same distribution, the classifier can't separate them: AUC ≈ 0.5. If AUC is well above 0.5, the batch is distinguishable, i.e. drifted. AUC is also the effect size.

**Q: Why a classifier and not a distance like cosine between mean embeddings?**
The mean can stay the same while the distribution changes (e.g. spread or a sub-cluster). A classifier uses the whole distribution, gives an interpretable effect size (AUC), and the cross-validation protects against overfitting. MMD and energy distance are alternatives that also need a resampling null.

**Q: Why PCA to 64 dimensions?**
Two reasons: the per-project null needs many repeated classifier fits, so it must be affordable; and high dimension with small samples hurts power. Validation compared PCA-64 with raw embeddings: equal or better detection and 0% false alarms on unchanged data for both, so PCA-64 was adopted. PCA is fitted on the reference only and stored.

**Q: How is the text/image p-value calibrated?**
From a **real-embedding null**: 200 disjoint splits of the project's own reference embeddings at the batch size, with a Gaussian tail beyond the draws. Validation on PetFinder: with no drift, P(p < 0.05) was 1–5% in every cell (target 5%).

**Q: What was wrong with the earlier calibration?**
The first calibrated version used a synthetic-Gaussian null grid. On real embeddings it was **over-confident: up to 14.5% false alarms** where 5% was the target. Real embeddings aren't Gaussian. That finding led to the real-embedding null. The old legacy rule (AUC > 0.65) was safe (0–2% false alarms) but weak: recall only about 0.68.

**Q: What did the real-embedding null gain?**
Batches that were 25% cats (in a dog reference) were detected **100%** of the time, versus 6–10% with the old AUC > 0.65 rule. That's also why the AUC floor was lowered to 0.55: at 0.65 the floor would have cancelled most of the power gain, and false alarms are controlled by the p-value, not the floor.

**Q: Why these two embedding models?**
Small, fast on CPU, widely used general-purpose encoders. The limitation is real: specialised domains (medical images, technical text) may separate worse, and there's no model selection.

**Q: What is the "joint" workflow?**
An older endpoint for records with tabular + text + image together. It concatenates [tabular | text | image | interaction terms | presence mask] and runs a DCT with an L1-regularised logistic regression. The interaction block (element-wise products) was added to close the "XOR / correlation-inversion" gap, where a linear classifier can't see a change that only exists in the combination. It's kept for backward compatibility, but its accuracy figures were measured with a mis-specified classifier and never re-measured, and the unified table path doesn't use it.

---

## 8. Validation and results

**Q: Which datasets did you validate on?**

| Dataset | What it tested | Headline |
|---|---|---|
| NYC Citi Bike 2016 (~4.5 M trips) | Column tests, two-gate rule | Precision 1.000, recall 0.939, F1 0.969 on real seasonal batches; no-change false alarms 0–1% vs 2–27% |
| PetFinder.my (Kaggle) | Text/image drift, calibration, relationships, full table | False alarms 1–5%; 25%-cats detected 100%; shuffled column detected 100%; full table 0% no-change alarms |
| Inside Airbnb Edinburgh | Second dataset for the full table workflow | Most scenarios 100%; text shuffle 54%; 8% no-change alarms |

**Q: How was the Citi Bike validation designed?**
January–March 2016 trips as the reference, April–December as production. That is natural seasonal drift, not simulated. Plus synthetic shifts of 0.5σ, 1.5σ and 3σ (each detected 50/50). Ground truth was computed independently with `scipy.stats.ks_2samp` and PSI on the full populations.

**Q: Per-feature Citi Bike results?**
`trip_duration` (D 0.0924) and `month` (PSI 16.27) detected 100%; coordinates with D around 0.019–0.022 detected 53–100%; `gender_id` (PSI 0.043) correctly stayed stable (below the 0.2 floor).

**Q: How does batch size affect recall?**
On Citi Bike: recall 0.458 at 1,000 rows, 0.729 at 3,000, 0.812 at 10,000, 0.896 at 20,000, 0.917 at 50,000. For this dataset, 15,000–20,000 rows per batch is the knee of the curve.

**Q: What is detection latency here?**
The share of a batch that must be drifted before an alert fires: `trip_duration` 10%, `dropoff_longitude` 10%, `dropoff_latitude` 20%, the pickup coordinates 30%.

**Q: How do you get ground truth for text, image and relationship drift?**
**By construction.** You can't know the "true" drift in real data, so you create it: mix cats into a dog reference (population drift), or shuffle a column across rows (destroys its relationships while keeping its own distribution). Scenarios and labels were fixed in each script's docstring **before** running, so the results couldn't be tuned after the fact.

**Q: What is an A/A test?**
Reference and batch drawn from the same population, so any alarm is false. It measures the false-alarm rate. Results: Citi Bike 0–1%, PetFinder relationships 0–1%, PetFinder full table 0%, Airbnb full table 8%.

**Q: What were the relationship-drift results?**
PetFinder, 2,000-row reference, 100 draws per cell, batches of 300 and 1,000:
- No change: 0–1% relationship alarms.
- Monotone transform of numeric columns: relationship alarms 1–2%, while the column test catches it 100%. Exactly the intended separation.
- Category mix shifted: relationship 1–3%, column 100%.
- A column shuffled: relationship **100%**, column 0%.
- 25% of rows shuffled: 2–3%, because the effect (~0.03) is below the 0.05 floor, by design.

**Q: Full-table (M4) results?**
Reference 1,500 rows, batch 300. PetFinder: 0% no-change alarms; shuffles of numeric, categorical, text and image columns detected 99–100%; flagged relationships involved the changed column 97–100% of the time. Batch-level relationship precision/recall/F1: 0.998/0.80/0.89 (PetFinder), 0.96/0.89/0.92 (Airbnb).

**Q: Why did Airbnb give 8% false alarms?**
All involved `property_type`. Hosts with many near-identical listings make rows **clustered, not independent**. The disjoint-split nulls assume exchangeable rows; clustering narrows the null and inflates false alarms. The fix (future work) is block or cluster-aware resampling.

**Q: Why was Airbnb text shuffle only 54%?**
Airbnb descriptions predict property and room type only moderately, so breaking the pairing changes the probe accuracy less. A weak relationship is hard to detect breaking.

**Q: Did any validation scenario fail by design?**
Yes. PetFinder's "shuffle within one category" scenario tested nothing: it left the only monitored relationship of that column intact. It's reported as a scenario-design flaw, not hidden.

**Q: What bugs did validation find?**
1. Categorical keys stored as strings but arriving as integers, so lookups silently said "no drift".
2. Chi-square ground truth mislabelled `gender_id`.
3. KS-on-PIT for numeric↔categorical broke under heavy ties (null around 0.6).
4. The M1 fit deduplicated on monitored columns only, collapsing distinct records.
5. The synthetic-Gaussian null was over-confident on real embeddings (up to 14.5% false alarms).

---

## 9. Hard and critical questions

**Q: Your ground truth is constructed. How do you know it works on real drift?**
Only the Citi Bike seasonal batches are natural drift. Text, image and relationship scenarios are constructed (mixing, shuffling). Constructed drift lets you know the answer and control the effect size, which is the only way to measure false-alarm rates and power precisely. But it's not a guarantee for every kind of real drift, and the README says that explicitly.

**Q: Is there anything wrong with the Citi Bike baseline?**
A forensic check (`results/citi_bike_provenance_forensics.md`) found the stored live baseline was drawn from roughly the first 1% of rows of the baseline file, not a random sample. It was documented, not redone. The calibrated re-runs drew their own references, including 10 disjoint reference draws.

**Q: Why trust your false-alarm rate on a new dataset?**
Because the nulls for relationship and embedding tests come from that dataset's own reference, not from assumptions. The main risk is non-independent rows (clustering, time order), which is exactly what happened on Airbnb, and it's listed as a limitation.

**Q: The p-value tells you if drift is real. Does drift mean the model is worse?**
No. Drift means the model is operating outside the data it was validated on. Some drift is harmless (a feature the model barely uses). Linking drift to performance needs labels or feature importance; that is out of scope. The system's wording is deliberately "out-of-distribution relative to the configured reference".

**Q: Couldn't you just give the training and production data to an LLM and ask if it drifted?**
That puts the representation and the decision inside one black box: no controlled false-alarm rate, not reproducible, not explainable. This system keeps them separate: a frozen encoder is just a measuring instrument, and the decision is a calibrated test with a known error rate.

**Q: Does this work for LLMs?**
Not directly, and I investigated it. Comparing user prompts with an LLM's training data is meaningless: a prompt like "yes, go ahead" never resembles internet-scale training data, so everything looks drifted. The right framing is **prompt drift against a known-good window**: compare this week's prompts with the prompts the system handled well, or with its evaluation set, using context-aware embeddings for short replies. The same statistical machinery applies; the baseline definition changes. Not built.

**Q: What did you set out to do that you didn't achieve?**
**Whole-record joint drift.** The original goal was to catch changes across a whole multimodal record (table + text + image) even when each part looks normal. The unified path tests columns and pairs; the optional whole-row joint signal (plan section I.7) was not built, and three-way interactions aren't covered. The old joint endpoint exists but is only reliable near the effect size it was tuned on. That's why the older per-modality tools are kept under "Advanced" rather than removed.

**Q: Why are the old tools still there?**
1. The specification for the unified workflow required the old text/image/joint endpoints to stay for backward compatibility; retiring them was explicitly a separate, later decision.
2. Projects created with the old setup can only be checked with the old tools.
3. The joint tool covers something the new path doesn't (whole-record signal), even if weakly.

**Q: Why is PSI's floor 0.2 but KS's floor 0.05? Aren't these arbitrary?**
They're conventions, chosen before seeing results. PSI 0.2 is the established industry threshold for "significant shift". KS 0.05 was chosen to be dataset-independent and deliberately not tuned to the Citi Bike coordinate cluster (around 0.019–0.022). Floors are per-project configurable. The honest answer: a floor is a business decision about "how big a change matters", which statistics can't make for you.

**Q: What happens with very small batches?**
Power drops (Citi Bike recall 0.458 at 1,000 rows). PSI false-alarms on many-category columns at small sizes (PetFinder smoke test: `State` PSI 0.39 at N = 100, resolved by N = 300). Tiny references (about 20 rows) give unreliable embedding tests (10–14% false alarms). The fit page warns about minimum detectable effect and recommended batch size.

**Q: Why not test every pair of columns?**
With 30 columns that's 435 pairs. Every extra test joins the Holm family and lowers sensitivity for all tests. So only pairs clearly related in the reference are proposed (capped at 25 + 15), the user confirms them, and the UI shows the family size.

**Q: Why is the text→numeric probe report-only?**
It failed a pre-registered check: in the "category mix shifted" scenario it alarmed 26% of the time, i.e. it confused a population change with a broken relationship. Rather than ship an alert that's wrong a quarter of the time, it's shown but never alerts.

**Q: What would you do differently if you started again?**
1. Design for clustered data from the start (block resampling).
2. Use random samples for every baseline and record provenance automatically.
3. Build the bring-your-own-embeddings path early, so model teams can monitor their model's own representation.
4. Plan the joint signal as a first-class deliverable, not optional.

---

## 10. Tech stack, system design and architecture

**Q: What tech stack did you use?**
Short version (20 seconds):
> The backend is **Python with FastAPI**, and the statistics use **SciPy, NumPy, pandas and scikit-learn**. Text and images are turned into vectors with **sentence-transformers (MiniLM)** and **PyTorch/torchvision (ResNet-18)**. Data is stored in **SQLite** plus files on disk. The frontend is **React with TypeScript, Vite and Tailwind**. Everything is packaged in **Docker** with **nginx**, and deployed on **Azure App Service** through **GitHub Actions**.

If asked for detail, layer by layer:

| Layer | What I used | Why, in one line |
|---|---|---|
| API | FastAPI, Uvicorn, Pydantic | Fast, typed request checks, and the docs page is generated automatically |
| Statistics | SciPy (KS test), NumPy, pandas, scikit-learn (logistic regression, PCA, ridge) | Standard, well-tested scientific libraries; no need to write the maths from scratch |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (text), torchvision `resnet18` (images) | Small, ready-made models that run on CPU, with no training needed |
| Storage | SQLite in WAL mode, plus a file store (NumPy `.npz`, Parquet) | No setup, transactional, enough for moderate traffic; large arrays live in files, never pickles |
| Background work | Own job queue: a worker thread with jobs saved in the database | Embedding thousands of images takes minutes, too long for one web request |
| Auth | Google Identity Services, email + password (scrypt), JWT session tokens, hashed personal access tokens | People sign in with a browser; scripts use revocable tokens |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, TanStack Query, Recharts | A modern, typed UI; TanStack Query handles data loading and caching; Recharts draws the trend charts |
| Packaging | Docker (multi-stage build), nginx, supervisord | One container: nginx serves the React app and forwards `/api` to FastAPI |
| Cloud and CI/CD | Azure App Service, Azure Container Registry, GitHub Actions | Every push to main builds the image and redeploys automatically |
| Testing | pytest (about 380 tests) | Unit, end-to-end, security and statistical-property tests |
| Legacy | Streamlit | The first dashboard; kept running, but React replaced it |

**Q: Why FastAPI and not Flask or Django?**
FastAPI gives typed validation and automatic API docs, and handles file uploads well. Django is heavier than this API needs.

**Q: Why not PostgreSQL?**
SQLite was enough for one server and needs no setup. All database access goes through one helper, so moving to PostgreSQL later is a contained change.

**Q: Why not Celery or Redis for background jobs?**
It would mean one more service to run and pay for. A database-backed queue survives restarts and was enough for this scale, and the worker can be split out later.

**Q: Why not a bigger embedding model?**
These two run on CPU in minutes, and the tests are calibrated on whatever model is used. The trade-off, which is a known limitation, is weaker results on specialised data such as medical images.

**Tip:** pair every tool with its reason ("FastAPI because…, SQLite because…"). Examiners care more about why you chose something than what you chose.

**Q: Describe the architecture.**
- **FastAPI** (`main.py`): auth, isolation, endpoints, tabular core.
- **Background job worker** (`jobs.py`): one in-process thread, FIFO queue, restart recovery, 24 h staged-upload expiry.
- **Orchestration** (`drift/table_monitor.py`): profile, fit, analyze.
- **Detectors:** `drift/detector.py` (KS, PSI, Holm, two-gate), `drift/relationship_detector.py`, `drift/embedding_tests.py`.
- **Ingestion:** `ingest/readers.py` (formats), `ingest/images.py` (safe ZIP handling).
- **Storage:** SQLite in WAL mode (`db/crud.py`) for metadata; a filesystem blob store (`db/blob_store.py`) for embeddings (npz, float16), PCA and probe arrays, and reference rows (Parquet).
- **Webhook delivery thread** (`drift/webhooks.py`).
- **React** frontend; a Python client in `clients/python`.

**Q: Why background jobs?**
Embedding ~15,000 descriptions took about 4 minutes and ~14,650 photos about 6 minutes on a 16-core desktop. An HTTP request can't wait that long. The API returns a `job_id` immediately and the client polls `GET /jobs/{id}`.

**Q: What happens if the server restarts mid-job?**
Running jobs are marked `interrupted` on restart and must be resubmitted. Queued jobs are recovered.

**Q: Why SQLite?**
Zero setup, a single file, transactional, and enough for moderate traffic. WAL mode allows concurrent reads during writes. All DB access goes through one helper (`crud.get_connection()`), a seam for moving to PostgreSQL. The trade-off: one process, no horizontal scaling.

**Q: Why store embeddings as npz and not pickle?**
Pickle can execute arbitrary code when loaded. Arrays in npz are pure data. Probe models are stored as arrays too, "never pickles".

**Q: What is idempotency and why add it?**
A pipeline that retries after a timeout might send the same batch twice, recording a duplicate analysis and possibly a duplicate alert. With an `Idempotency-Key` header, a replay with the same data returns the earlier result; with different data it returns `409 Conflict`. Keys expire after 7 days. Table analyses key on a hash of the uploaded files.

**Q: How does baseline versioning work?**
Every fit creates a new version; old versions are kept. You can activate any older version or analyze against a specific one (`baseline_version`). History records which version was actually used.

**Q: What does the schema report check?**
Missing or unexpected columns, dtype changes, null-rate increases, unseen categories, constant columns, and for tables also missing/corrupt images and empty text. Severity comes from a per-project `schema_policy`. Valid columns are always analyzed; nothing is dropped silently.

**Q: How are duplicate rows handled?**
Rows identical in every column are removed only if the table has an identifier-like column (non-float, complete, unique, 20+ distinct). Otherwise they're kept and reported as `identical_rows_kept`, because they may be different records that share values (e.g. two identical pets). That was a decision made after validation showed the old rule dropping legitimate rows.

**Q: How do webhooks work?**
HMAC-SHA256 signed (`X-Drift-Signature-256`), sent on alert `opened` and `resolved` (`still_open` is opt-in), retried with 1/4/16/64 s backoff up to 5 attempts, statistics-only payload. The receiver recomputes the HMAC with the shared secret and compares in constant time.

**Q: Why a signature?**
Anyone who knows the webhook URL could post fake alerts. The HMAC proves the message came from the server holding the secret and wasn't modified.

**Q: How is a numeric/categorical-only table guaranteed to behave like the old tabular endpoint?**
Its numeric/categorical part reuses the tabular storage and code, and a regression test checks byte-identical results between the two paths.

---

## 11. Security and privacy

**Q: How does authentication work?**
- **Google sign-in:** Google Identity Services gives an ID token; the backend verifies it and issues its own session token.
- **Email + password:** sign-up sends a 6-digit code; the account works only after confirming it.
- **Session token:** a JWT signed with HS256 using `COOKIE_KEY`, valid 1 hour.
- **Personal access tokens** for scripts: `dm_<prefix>_<secret>`, stored as SHA-256 hashes, looked up by prefix, revocable, optionally scoped to specific projects, expiring after 30/90/365 days.

**Q: How are passwords stored?**
scrypt (memory-hard) with a per-user random salt and a constant-time comparison. Verification codes are stored only as keyed hashes, expire after 15 minutes and allow 5 tries. Five wrong passwords lock sign-in for 15 minutes.

**Q: How do you avoid revealing whether an account exists?**
Sign-in responses are identical for unknown and known accounts, and a dummy scrypt check runs for unknown accounts so the response time doesn't leak it either.

**Q: Why must emails be verified?**
Projects belong to an email address. Without verification, anyone could register someone else's address and see their projects. The same confirmed address sees the same projects via Google or password.

**Q: Why can't an access token create tokens?**
Token management endpoints accept signed-in sessions only. If a token leaked from a CI script, it can't mint new long-lived tokens for itself.

**Q: How is one user prevented from seeing another's data?**
Project IDs are namespaced per owner. Every endpoint checks ownership. Another user's project, stage or job returns `404`, not `403`, so the attacker can't even tell it exists.

**Q: What raw data is kept?**
Only staged uploads, for at most 24 hours, deleted as soon as a fit or analysis uses them. History, job results, webhooks and alert events hold statistics only. Job errors are sanitized, and the server log records exception type and stack, not messages, because messages can echo cell values. A test plants a sentinel value and checks every table for it.

**Q: How are uploads secured?**
Streamed to disk with size caps; ZIPs are never extracted to disk; unsafe paths (`..`, absolute, drive letters), symlinks, encrypted entries and archive bombs are rejected; images are decoded with a pixel cap and a format allowlist; damaged entries become a per-row `corrupt` status instead of crashing the job.

**Q: What is SSRF and how is it handled?**
Server-Side Request Forgery: tricking the server into making requests to internal addresses (e.g. cloud metadata endpoints). Webhook URLs resolving to private, loopback or link-local addresses are rejected. The system never fetches remote image URLs.

**Q: Any known security issue?**
OAuth secrets were committed to the git history in the past. Removing them from the code isn't enough because history is public; they had to be rotated. The old OAuth client has since been deleted and replaced.

---

## 12. Deployment

**Q: How is it deployed?**
A single Docker container on **Azure App Service** (Linux, B1 plan). Inside, supervisord runs nginx, uvicorn (FastAPI) and the legacy Streamlit dashboard. nginx serves the React build at `/`, proxies `/api/` to FastAPI and `/streamlit/` to Streamlit, and allows uploads up to 710 MB.

**Q: What is the CI/CD pipeline?**
On every push to `main`, a GitHub Actions workflow builds the image and pushes it to Azure Container Registry. App Service's continuous deployment pulls `:latest` and restarts.

**Q: Why does FastAPI run with `--root-path /api`?**
nginx strips the `/api` prefix before forwarding. Without telling FastAPI its public prefix, the Swagger docs page asked for `/openapi.json` instead of `/api/openapi.json` and rendered blank.

**Q: How is data kept across restarts on Azure?**
The container's disk is wiped on restart. Uploaded files and embeddings live on the persistent `/home` share. SQLite isn't safe on that network share (file locking), so the live database stays on local disk and `db/snapshot.py` saves a consistent copy (SQLite's online backup API) to `/home` every 5 minutes and on shutdown, restoring it on startup. Trade-off: a crash can lose the last few minutes of writes.

**Q: Why one container instead of microservices?**
Simplicity and cost for a student deployment. The seams for scaling are documented: `BlobStore` → object storage, the `jobs` table → a separate worker, the DB helper → PostgreSQL.

---

## 13. Frontend and user experience

**Q: What is the frontend stack?**
React 19, TypeScript, Vite, Tailwind CSS, TanStack Query for server state, Recharts for charts.

**Q: What are the dashboard sections?**
- **All models / Connect your model** (global).
- Per model: **Overview** (latest verdict, key numbers), **Check my data** (setup + manual check), **Past checks** (history and trend charts), **Saved baselines** (versions), **Alerts & webhooks**, **Activity log** (single-row checks).
- **Advanced:** older per-modality tools and single-row check.

**Q: Why does the dashboard say "model" but the API says "project"?**
The users think in terms of the ML model being watched, so the UI says "model". The API keeps `project_id` so existing integrations and tokens don't break.

**Q: What is the Connect your model page?**
The guide for automating it: create an access token, install the Python client, send the baseline once, then send each batch on a schedule; Python and curl code is pre-filled with the model name and API address. It also explains when to send a batch.

**Q: How did you design the UI for non-technical users?**
Plain verdicts first ("Price and size no longer move together"), statistics behind a "Technical details" toggle, guided steps, and the human review step with evidence for every proposal.

---

## 14. Testing and engineering process

**Q: How is it tested?**
pytest, around 380 tests across 30+ modules (the last full run passed 380; one more token test was added afterwards). Highlights:
- `test_table_unified.py`: end-to-end jobs, equivalence with the tabular path, isolation, idempotency, no raw values persisted.
- `test_table_hardening.py`: fuzzed tables and ZIPs, damaged entries, limits.
- `test_relationship_detector.py`: invariance to marginal changes, re-pairing, ties, Holm membership.
- `test_embedding_tests.py`: null calibration, probes, matching.
- Plus history, idempotency, schema report, versioning, alert state machine, webhook signing and SSRF, cross-user isolation, email auth, tokens, database snapshot.

**Q: Testing a statistical system: how do you test correctness, not just "it runs"?**
Property tests: monotone transforms must not trigger relationship drift; shuffles must; A/A runs must stay near the nominal rate; legacy mode must reproduce old outputs exactly; numeric-only tables must match the tabular path byte for byte.

**Q: What engineering process did you follow?**
Read the current code before changing it; write the plan before the code (`docs/unified_table_plan.md`, approved milestone by milestone M1–M4); pre-register validation scenarios; keep a running engineering log (`docs/PROGRESS.md`); correct the docs when a finding is narrower than hoped (e.g. the over-confident synthetic null, the failed numeric probe).

**Q: What is pre-registration and why do it?**
Writing down the scenarios, metrics and pass criteria **before** running the experiment. It stops you from adjusting thresholds after seeing results until they look good.

---

## 15. Limitations and future work

**Known limitations:**
- **Exchangeability:** nulls assume roughly independent rows; clustered or time-ordered data inflates false alarms (Airbnb 8%).
- **Relationship coverage:** Spearman misses non-monotone dependence; numeric↔categorical tracks position, not spread; only pairs, not three-way interactions; probes can't detect relationships that appear only in production.
- **No whole-row joint signal** in the unified path.
- **Fixed embedding models**, general-purpose.
- **Small data:** PSI on many-category columns at small batch sizes; tiny references for embedding tests.
- **Scale:** one process, SQLite, local files; background jobs die with the instance.
- **No retention cap** for baseline versions.
- **Validated on three datasets**, drift mostly constructed.
- **Text/image 20-Newsgroups/CIFAR-10 suite** was written but never run to completion, so it has no published numbers.
- **No concept drift** (needs labels).

**Future work:**
- Optional whole-row joint signal (experimental, off by default).
- Cluster/block-aware resampling for clustered data.
- Bring-your-own embeddings (monitor the model's own representation).
- Prompt drift for LLM applications against a known-good window.
- Baseline retention and pruning.
- PostgreSQL, object storage, separate worker.
- Webhook delivery history and a "send test event" action.
- Wasserstein as an extra metric; rolling-window baselines.
- Audio and video via embedding tests.

---

## 16. Quick-fire facts

| Question | Answer |
|---|---|
| Backend framework | FastAPI + Uvicorn + Pydantic |
| Database | SQLite (WAL) + filesystem blob store |
| Frontend | React 19, TypeScript, Vite, Tailwind, TanStack Query, Recharts |
| Numeric test | Two-sample KS, floor D ≥ 0.05 |
| Categorical test | PSI + parametric bootstrap, floor 0.2 |
| Text/image test | Domain Classifier Test, PCA-64, real-embedding null (200 splits), floor AUC ≥ 0.55 |
| Single-row test | IQR fences, 3 × IQR |
| Multiple testing | Holm (FWER), α = 0.05 |
| Text model | all-MiniLM-L6-v2, 384-dim |
| Image model | resnet18 penultimate layer, 512-dim |
| Relationship cap | 25 numeric/categorical + 15 embedding pairs |
| Session token | JWT HS256, 1 hour |
| Access token | `dm_<prefix>_<secret>`, SHA-256 hashed, 30/90/365 days, max 20 active |
| Password hashing | scrypt, per-user salt |
| Webhook signing | HMAC-SHA256, `X-Drift-Signature-256` |
| Webhook retries | 1/4/16/64 s, up to 5 attempts |
| Idempotency key lifetime | 7 days |
| Staged upload lifetime | ≤ 24 h |
| Upload limits | table 200 MB; ZIP 500 MB, 20,000 entries |
| Deployment | Docker on Azure App Service B1, GitHub Actions → ACR |
| DB backup interval | every 5 minutes + on shutdown |
| Citi Bike headline | P 1.000 / R 0.939 / F1 0.969 |
| Two-gate effect on false alarms | 2–27% → 0–1% |
| Significance-only worst case | 57.5% false alarms |
| Embedding calibration | 1–5% false alarms; 25% cats detected 100% (vs 6–10%) |
| Airbnb false alarms | 8%, all `property_type` (clustering) |

---

## 17. Formula sheet

**KS statistic:** D = sup_x | F_ref(x) − F_batch(x) |

**KS minimum detectable D:** D_min = c(α) · √((n + m) / (n·m)), c(0.05) = 1.36

**DKW bound:** sup | F_m − F | ≤ √( ln(2/α) / (2m) ) with probability ≥ 1 − α

**PSI:** Σ_k (a_k − e_k) · ln(a_k / e_k), where a = batch share, e = reference share

**Bootstrap / split-null p-value:** p = (1 + #{null statistic ≥ observed}) / (B + 1)

**Holm:** sort p(1) ≤ … ≤ p(m); reject H(i) while p(i) ≤ α / (m − i + 1); stop at the first failure

**Family-wise error without correction:** 1 − (1 − α)^m, e.g. 30 tests → ≈ 79%

**IQR fences:** [Q1 − 3·IQR, Q3 + 3·IQR], IQR = Q3 − Q1

**Spearman relationship statistic:** | ρ_batch − ρ_ref |

**Domain Classifier Test:** AUC of a cross-validated classifier separating reference (0) from batch (1); 0.5 = indistinguishable

**HMAC check:** "sha256=" + HMAC_SHA256(secret, raw_body) compared in constant time with the header
