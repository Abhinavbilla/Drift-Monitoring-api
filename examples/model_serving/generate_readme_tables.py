"""
Step 3e end-to-end example, item 5 (hardening pass): generates the
markdown tables in README.md directly from reports/real/*.json and
reports/synthetic/*.json -- so every number in the README comes from an
actual saved report, never typed by hand. Replaces the content between
the <!-- BEGIN ... --> / <!-- END ... --> marker pairs in README.md;
everything outside those markers is left untouched.

Run:
    python examples/model_serving/generate_readme_tables.py
"""

import glob
import json
import os
import re

HERE = os.path.dirname(__file__)
README_PATH = os.path.join(HERE, "README.md")


def _load_reports(subdir):
    paths = sorted(glob.glob(os.path.join(HERE, "reports", subdir, "window_*.json")))
    reports = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            reports.append((os.path.basename(p), json.load(f)))
    return reports


def _real_table():
    reports = _load_reports("real")
    lines = ["| Window | Alert | Drifted feature(s) (statistic, p_value_adjusted) |", "|---|---|---|"]
    for fname, report in reports:
        window_idx = int(re.search(r"window_(\d+)", fname).group(1))
        alert = report["system_alert_triggered"]
        drifted = [
            f"`{feat}` (D={m['statistic']:.4f}, p_adj={m['p_value_adjusted']:.4g})"
            for feat, m in report["feature_metrics"].items() if m["drift_detected"]
        ]
        lines.append(f"| {window_idx} | {'**True**' if alert else 'False'} | {', '.join(drifted) or 'none'} |")
    return "\n".join(lines)


def _synthetic_table():
    reports = _load_reports("synthetic")
    lines = ["| Window | Alert | `pickup_longitude` (tilted feature) | `dropoff_longitude` (correlated) |",
             "|---|---|---|---|"]
    for fname, report in reports:
        window_idx = int(re.search(r"window_(\d+)", fname).group(1))
        alert = report["system_alert_triggered"]
        pl = report["feature_metrics"]["pickup_longitude"]
        dl = report["feature_metrics"]["dropoff_longitude"]
        lines.append(
            f"| {window_idx} | {'**True**' if alert else 'False'} | "
            f"D={pl['statistic']:.4f}, p_adj={pl['p_value_adjusted']:.4g}, material={pl['material']} | "
            f"D={dl['statistic']:.4f}, p_adj={dl['p_value_adjusted']:.4g}, material={dl['material']} |"
        )
    n_alerted = sum(1 for _, r in reports if r["system_alert_triggered"])
    lines.append(f"\n**{n_alerted}/{len(reports)} windows alerted.**")
    return "\n".join(lines)


def _replace_between_markers(content, marker_name, new_body):
    pattern = re.compile(
        rf"(<!-- BEGIN {marker_name} -->\n)(.*?)(\n<!-- END {marker_name} -->)", re.DOTALL
    )
    if not pattern.search(content):
        raise ValueError(f"Markers for '{marker_name}' not found in README.md")
    return pattern.sub(lambda m: m.group(1) + new_body + m.group(3), content)


def main():
    with open(README_PATH, encoding="utf-8") as f:
        content = f.read()

    content = _replace_between_markers(content, "REAL_REPLAY_TABLE", _real_table())
    content = _replace_between_markers(content, "SYNTHETIC_TABLE", _synthetic_table())

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated tables in {README_PATH}")


if __name__ == "__main__":
    main()
