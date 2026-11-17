"""Aggregate independent RelPoseNet loss runs and plot comparable validation metrics.

Example:
    python compare_loss_runs.py --root outputs/loss_comparison

This script intentionally does NOT compare the raw objective values across losses,
because different loss functions have different scales and units.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_COLUMNS = (
    "scene_macro_translation_median_m",
    "scene_macro_rotation_median_deg",
    "pair_within_5cm_5deg_pct",
)


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/loss_comparison",
                        help="Root directory containing one folder per loss run")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Loss-run directory does not exist: {root}")

    runs = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_file = run_dir / "metrics.csv"
        if not metrics_file.is_file():
            continue
        rows = read_rows(metrics_file)
        if not rows:
            continue
        runs.append((run_dir.name, rows))

    if not runs:
        raise RuntimeError(f"No non-empty metrics.csv files found under {root}")

    output_dir = root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for run_name, rows in runs:
        last = rows[-1]
        item = {"run": run_name, "final_step": last.get("step", "")}
        for key in METRIC_COLUMNS:
            item[key] = as_float(last, key)
        item["translation_median_m"] = as_float(last, "translation_median_m")
        item["rotation_median_deg"] = as_float(last, "rotation_median_deg")
        summary.append(item)

    summary.sort(key=lambda x: x[METRIC_COLUMNS[0]])
    summary_file = output_dir / "final_comparison.csv"
    fields = ["run", "final_step", *METRIC_COLUMNS,
              "translation_median_m", "rotation_median_deg"]
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    labels = [x["run"] for x in summary]
    values = [x[METRIC_COLUMNS[0]] for x in summary]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.1), 5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Scene-macro median translation error (m)")
    ax.set_title("Final pairwise translation accuracy by loss")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "final_translation_comparison.png", dpi=160)
    plt.close(fig)

    values = [x[METRIC_COLUMNS[1]] for x in summary]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.1), 5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Scene-macro median rotation error (degrees)")
    ax.set_title("Final pairwise rotation accuracy by loss")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "final_rotation_comparison.png", dpi=160)
    plt.close(fig)

    # Validation accuracy trajectories share units and are directly comparable.
    for metric, filename, ylabel, title in (
        (METRIC_COLUMNS[0], "translation_curves_comparison.png",
         "Scene-macro median translation error (m)",
         "Validation translation accuracy across loss runs"),
        (METRIC_COLUMNS[1], "rotation_curves_comparison.png",
         "Scene-macro median rotation error (degrees)",
         "Validation rotation accuracy across loss runs"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = 0
        for run_name, rows in runs:
            steps, metric_values = [], []
            for row in rows:
                try:
                    steps.append(float(row["step"]))
                    metric_values.append(float(row[metric]))
                except (KeyError, TypeError, ValueError):
                    continue
            if steps:
                ax.plot(steps, metric_values, label=run_name)
                plotted += 1
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    print(f"Read {len(runs)} runs.")
    print(f"Saved summary: {summary_file}")
    print(f"Saved comparative plots in: {output_dir}")
    print("Raw train/validation objective curves are intentionally not overlaid across losses.")


if __name__ == "__main__":
    main()
