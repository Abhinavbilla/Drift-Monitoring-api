# Model-serving drift monitoring example

End-to-end example (Step 3e, corrected 2026-09-30 hardening pass item 5):
a small scikit-learn model trained on Citi Bike data, served with
FastAPI, with its input features logged asynchronously and periodically
sent to the Drift Monitoring API for analysis.

**Correction (2026-09-30)**: the original version of this example
mischaracterized two flagged windows from the real Apr-Jun replay as
"real drift." They were not. The monitored coordinate features' true
population D between the Jan-Mar reference and Apr-Jun production is
**~0.02** (see "The real replay" below) -- comfortably BELOW the 0.05
materiality floor. Any window from that replay that crosses the floor is
doing so from sampling noise on a genuinely sub-floor effect, not from
detecting a real distributional shift. This version fixes that framing,
uses the current (50,000-row) reference and the redefined
`recommended_batch_size`, and adds a separately-labeled synthetic
scenario with an actual, substantial injected drift so the reader can
see both cases side by side.

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

## The real replay: noise, not drift

Population D between the Jan-Mar reference and Apr-Jun production, for
each monitored continuous feature (from `results/tabular_validation_legacy.json`'s
stored population truth -- computed once, independent of any batch draw):

| Feature | Population D | vs. 0.05 floor |
|---|---|---|
| `pickup_longitude` | 0.0219 | 0.44x |
| `pickup_latitude` | 0.0189 | 0.38x |
| `dropoff_longitude` | 0.0206 | 0.41x |
| `dropoff_latitude` | 0.0195 | 0.39x |

Every one of these is well below the 0.05 materiality floor. A run
of this replay should mostly show `alert=False`; an occasional window
crossing the floor from sampling noise is expected and consistent with
the false-material rates characterized in
`results/step2_hardening_batch_size_simulation.md` at true D near this
range -- not evidence of real drift. One real run (2026-09-30), 15,000
replayed predictions (5,000/month, Apr/May/Jun) through the actual served
model, split into complete 3,146-row windows (`recommended_batch_size`
for this 50,000-row reference):

<!-- BEGIN REAL_REPLAY_TABLE -->
| Window | Alert | Drifted feature(s) (statistic, p_value_adjusted) |
|---|---|---|
| 0 | False | none |
| 1 | False | none |
| 2 | False | none |
| 3 | **True** | `dropoff_longitude` (D=0.0550, p_adj=1.614e-07) |
<!-- END REAL_REPLAY_TABLE -->

Any window above showing `alert=True` has an observed statistic close to
the 0.05 floor -- read it as "consistent with the floor-adjacent noise
this hardening pass's simulation already characterizes," not as a
detected real-world shift. Four windows from one run is not enough data
to precisely re-estimate that false-material rate; that's what the
dedicated simulation is for.

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
