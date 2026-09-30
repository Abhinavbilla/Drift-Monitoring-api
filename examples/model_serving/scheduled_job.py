"""
Step 3e end-to-end example, part 4: the scheduled-job script. Reads
logs/served_features.jsonl (written by serve.py's async post-prediction
logging), groups newly-logged rows into fixed-size windows, and calls
DriftClient.analyze() for each complete window against the Drift
Monitoring API, saving each window's report to reports/. A line-count
cursor (logs/.cursor) means repeated invocations -- as a real cron job
would make -- only process rows not already processed; a partial,
not-yet-window-sized tail is left for the next run.

Run:
    python examples/model_serving/scheduled_job.py \
        --project-id model_serving_example --token <PAT> --window-size 3146

For the labeled synthetic scenario (item 5), pass --log-path/--cursor-
path/--reports-dir to point at a separate log/report set instead of
overwriting the real replay's:
    python examples/model_serving/scheduled_job.py \
        --project-id model_serving_example --token <PAT> --window-size 3146 \
        --log-path logs/synthetic_features.jsonl --cursor-path logs/.cursor_synthetic \
        --reports-dir reports/synthetic
"""

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "clients", "python"))
from drift_monitor_client import DriftClient  # noqa: E402

DEFAULT_LOG_PATH = os.path.join(HERE, "logs", "served_features.jsonl")
DEFAULT_CURSOR_PATH = os.path.join(HERE, "logs", ".cursor")
DEFAULT_REPORTS_DIR = os.path.join(HERE, "reports", "real")
INPUT_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "gender_id"]


def _read_cursor(cursor_path) -> int:
    if os.path.exists(cursor_path):
        with open(cursor_path) as f:
            return int(f.read().strip())
    return 0


def _write_cursor(cursor_path, n: int):
    with open(cursor_path, "w") as f:
        f.write(str(n))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--window-size", type=int, required=True,
                         help="Rows per analyze() window -- should be >= /fit's recommended_batch_size.")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--cursor-path", default=DEFAULT_CURSOR_PATH)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        print("No log file yet -- nothing to process.")
        return

    with open(args.log_path, encoding="utf-8") as f:
        all_lines = f.readlines()

    start = _read_cursor(args.cursor_path)
    new_lines = all_lines[start:]
    client = DriftClient(args.base_url, args.token)
    os.makedirs(args.reports_dir, exist_ok=True)

    n_windows = len(new_lines) // args.window_size
    if n_windows == 0:
        print(f"{len(new_lines)} new row(s) logged, below window size {args.window_size} -- "
              f"nothing to analyze yet, will pick up next run.")
        return

    processed = 0
    for w in range(n_windows):
        window_lines = new_lines[w * args.window_size:(w + 1) * args.window_size]
        records = [json.loads(line) for line in window_lines]
        df = pd.DataFrame(records)[INPUT_FEATURES]
        report = client.analyze(args.project_id, df)

        window_idx = (start // args.window_size) + w
        report_path = os.path.join(args.reports_dir, f"window_{window_idx:04d}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        drifted = [feat for feat, m in report["feature_metrics"].items() if m["drift_detected"]]
        print(f"Window {window_idx}: {len(df)} rows -> alert={report['system_alert_triggered']} "
              f"drifted={drifted or 'none'} -> {report_path}")
        processed += len(window_lines)

    _write_cursor(args.cursor_path, start + processed)
    leftover = len(new_lines) - processed
    print(f"Processed {processed} new row(s) into {n_windows} window(s). "
          f"{leftover} row(s) left over (incomplete window, picked up next run).")


if __name__ == "__main__":
    main()
