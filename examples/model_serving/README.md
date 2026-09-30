# Model-serving drift monitoring example

End-to-end example (Step 3e, corrected 2026-09-30 hardening pass item 5):
a small scikit-learn model trained on Citi Bike data, served with
FastAPI, with its input features logged asynchronously and periodically
sent to the Drift Monitoring API for analysis.

**Correction (2026-09-30, revised 2026-10-01)**: the original version of
this example mischaracterized a flagged window from the real Apr-Jun
replay as "real drift." It is not that either. The monitored coordinate
features' pooled population D between the Jan-Mar reference and Apr-Jun
production is ~0.02 (see "The real replay" below), below the 0.05
materiality floor -- but the flagged window's cause is **not
established**: it is not explained by pure sampling noise either (ruled
out by this project's own null simulation -- see below), and the
window's own local population D, while still sub-floor, is meaningfully
higher than the pooled figure. This version uses the current
(50,000-row) reference and the redefined `recommended_batch_size`, and
adds a separately-labeled synthetic scenario with an actual, substantial
injected drift so the reader can see a confirmed-drift case for contrast
-- but the real replay's one flagged window is reported as unresolved,
not as either "real drift" or "noise."

## Pipeline

1. **`train_model.py`** -- trains a `RandomForestRegressor` predicting
   `trip_duration` from the model's five real input features
   (`pickup_longitude`, `pickup_latitude`, `dropoff_longitude`,
   `dropoff_latitude`, `gender_id`) on `tests/splits/reference_n50000.csv`
   (50,000 rows). Writes `artifacts/model.pkl`.
2. **Fit the drift baseline** -- the same five input features (never the
   `trip_duration` target) are fit as the Drift Monitoring API's
   reference, with `decision_mode="calibrated"`.
3. **`serve.py`** -- a separate FastAPI app (its own port) that loads the
   trained model and exposes `POST /predict`. Every prediction's
   post-preprocessing input features are logged to a JSONL file via a
   `BackgroundTasks` call, so logging never adds latency to the
   prediction response. `MODEL_SERVE_LOG_PATH` selects the log file, so a
   second instance can serve the synthetic scenario on its own port
   without touching the real replay's log.
4. **`replay.py`** -- replays a sample of the real Apr-Jun production
   data through `serve.py`'s `/predict` endpoint.
5. **`replay_synthetic.py`** -- generates a SEPARATE, clearly-labeled
   synthetic batch with a real, substantial injected drift (normal-score
   exponential tilting, the same method as
   `scripts/step2_item4_power_curve.py`, verified via exact weighted-KS)
   and replays it through a second `serve.py` instance.
6. **`scheduled_job.py`** -- reads newly-logged rows, groups them into
   fixed-size windows (**at or above `/fit`'s `recommended_batch_size`**,
   per instruction), and calls `DriftClient.analyze()` for each complete
   window, saving the report to `reports/real/` or `reports/synthetic/`.
   Tracks a line-count cursor so repeated runs only process new rows.
7. **`generate_readme_tables.py`** -- regenerates the two tables below
   directly from the saved `reports/*/window_*.json` files. No number in
   this README is typed by hand; re-run the pipeline and this script to
   refresh them.

## Running it

```bash
# 1. Train the model
python examples/model_serving/train_model.py

# 2. Mint a token scoped to this example's project
python scripts/create_token.py create --email you@example.com \
    --name "model-serving-example" --project model_serving_example

# 3. Fit the drift baseline (with the token from step 2) -- note the
#    recommended_batch_size in the response; that's your window size.
python -c "
import sys; sys.path.insert(0, 'clients/python')
from drift_monitor_client import DriftClient
import pandas as pd
df = pd.read_csv('tests/splits/reference_n50000.csv')
features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','gender_id']
client = DriftClient('http://127.0.0.1:8000', '<TOKEN>')
result = client.fit('model_serving_example', df[features], calibration_config={'decision_mode': 'calibrated'})
print(result['calibration_info'])
"

# 4. Start the real serving app
python -m uvicorn examples.model_serving.serve:app --port 8001

# 5. Replay real production traffic (separate terminal)
python examples/model_serving/replay.py

# 6. Run the scheduled job on the real log
python examples/model_serving/scheduled_job.py \
    --project-id model_serving_example --token <TOKEN> --window-size 3146

# 7. Start a SECOND serving app for the synthetic scenario (separate terminal)
MODEL_SERVE_LOG_PATH=examples/model_serving/logs/synthetic_features.jsonl \
    python -m uvicorn examples.model_serving.serve:app --port 8002

# 8. Replay the synthetic drift scenario
python examples/model_serving/replay_synthetic.py

# 9. Run the scheduled job on the synthetic log
python examples/model_serving/scheduled_job.py \
    --project-id model_serving_example --token <TOKEN> --window-size 3146 \
    --log-path examples/model_serving/logs/synthetic_features.jsonl \
    --cursor-path examples/model_serving/logs/.cursor_synthetic \
    --reports-dir examples/model_serving/reports/synthetic

# 10. Regenerate this README's tables
python examples/model_serving/generate_readme_tables.py
```

## The real replay: flagged; cause not established

**Correction (2026-10-01):** this section previously called window 3's
alert "consistent with noise." That claim contradicted this project's own
simulation (`results/step2_hardening_batch_size_simulation.md`), which
found a 0% rate of exceeding the floor under the true null at this n/m —
D_obs=0.0550 is roughly an 8-standard-deviation event against that null
(mean 0.0159, std 0.0047), not a plausible noise fluctuation. Findings
from re-examining this window:

1. Windows are **ordered chronological slices** of the replay log, not
   iid draws — window 3 spans 2016-05-28 through 2016-06-17, straddling
   May and June, not a random sample of the full Apr-Jun period.
2. Population D for `dropoff_longitude` (baseline vs. the REAL
   population in that exact May 28-Jun 17 date range, all rows, not a
   sample) is **0.0309** — notably higher than the pooled Apr-Jun figure
   below (0.0206) used in the original "noise" claim, and reflecting
   real temporal variation the pooled average hides.
3. Under the pure null (true D=0) at n=3,146/m=50,000, D_obs=0.0550 is
   effectively impossible (~8σ) — so "just sampling noise on a near-zero
   effect" is ruled out.
4. That window's elevated local population D (0.031, still below the
   0.05 floor) is a more plausible partial explanation, but wasn't
   itself simulated at this n/m, so no exact P(D_obs>=0.0550 | true
   D=0.031) is available to confirm it's sufficient on its own.
5. **Conclusion: this window's alert is flagged, not explained.** It is
   not evidence of a real drift event in the sense this system is meant
   to catch (a shift from the FIT-TIME reference), and it is also not
   cleanly dismissible as pure noise — real temporal structure within
   the Apr-Jun period, not represented in the pooled population-D table,
   is the leading candidate but unconfirmed.

Population D between the Jan-Mar reference and the POOLED Apr-Jun
production (from `results/tabular_validation_legacy.json`'s stored
population truth) -- this is an average over the whole period and, per
finding 2 above, can differ substantially from any specific sub-period:

| Feature | Population D (pooled Apr-Jun) | vs. 0.05 floor |
|---|---|---|
| `pickup_longitude` | 0.0219 | 0.44x |
| `pickup_latitude` | 0.0189 | 0.38x |
| `dropoff_longitude` | 0.0206 | 0.41x |
| `dropoff_latitude` | 0.0195 | 0.39x |

One real run (2026-09-30), 15,000 replayed predictions (5,000/month,
Apr/May/Jun) through the actual served model, split into complete
3,146-row windows (`recommended_batch_size` for this 50,000-row
reference):

<!-- BEGIN REAL_REPLAY_TABLE -->
| Window | Alert | Drifted feature(s) (statistic, p_value_adjusted) |
|---|---|---|
| 0 | False | none |
| 1 | False | none |
| 2 | False | none |
| 3 | **True** | `dropoff_longitude` (D=0.0550, p_adj=1.614e-07) |
<!-- END REAL_REPLAY_TABLE -->

See the corrected framing above -- window 3's alert is flagged with its
cause not established, neither confirmed real drift nor cleanly
dismissible as noise. Four windows from one run is also not enough data
to characterize this on its own; that's what the dedicated simulation
(same document) is for, and it doesn't cover this window's actual local
population D.

## The synthetic scenario: real, substantial drift

`replay_synthetic.py` tilts `pickup_longitude` (normal-score exponential
tilting against the holdout pool) to a target population D of 0.10 --
double the materiality floor, and about 5x the real replay's actual
population D above. Because the tilting resamples whole rows by
importance weight on `pickup_longitude`, and pickup/dropoff longitude are
naturally correlated in real trip data (trips starting further west tend
to also end further west, given NYC's geography), `dropoff_longitude`
picks up a correlated shift too -- reported honestly below, not hidden.
One real run (2026-09-30), 12,584 tilted predictions (4 trials x 3,146
rows) through a second served-model instance, same window size:

<!-- BEGIN SYNTHETIC_TABLE -->
| Window | Alert | `pickup_longitude` (tilted feature) | `dropoff_longitude` (correlated) |
|---|---|---|---|
| 0 | **True** | D=0.1068, p_adj=3.814e-29, material=True | D=0.0694, p_adj=2.973e-12, material=True |
| 1 | **True** | D=0.1117, p_adj=5.972e-32, material=True | D=0.0840, p_adj=5.232e-18, material=True |
| 2 | **True** | D=0.1060, p_adj=9.955e-29, material=True | D=0.0722, p_adj=3.021e-13, material=True |
| 3 | **True** | D=0.0976, p_adj=2.695e-24, material=True | D=0.0591, p_adj=8.185e-09, material=True |

**4/4 windows alerted.**
<!-- END SYNTHETIC_TABLE -->

This is what the system is actually supposed to catch -- contrast this
against the real replay's table above, where the same system correctly
stays mostly quiet on an effect that's genuinely below the floor.

## Files not committed

`logs/*.jsonl`, `logs/.cursor*`, and `artifacts/*.pkl` are regenerated by
re-running the pipeline above and are not committed (see `.gitignore` in
this directory) -- `reports/real/` and `reports/synthetic/` are the
committed, reproducible evidence of what the calibrated engine actually
did, and are what `generate_readme_tables.py` reads.
