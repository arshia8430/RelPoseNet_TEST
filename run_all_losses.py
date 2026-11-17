"""Run every supported RelPoseNet loss sequentially with randomized loss hyperparameters.

Run from the project root (the directory containing main.py):
    python run_all_losses.py --seed 2026

This is intended for an initial smoke/comparison run, NOT hyperparameter tuning
or a scientifically controlled final comparison.
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LOSSES = [
    "mse", "hard", "soft", "softgrid", "dir", "pmse",
    "posenet10", "posenet100", "posenet", "se3", "ray",
    "denseray", "soft_sqrt", "softgrid_sqrt",
]
ZF_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
SIGMA_VALUES = [0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
RHO_VALUES = [4.0, 12.0, 24.0, 48.0]
BETA_VALUES = [1.0, 10.0, 25.0, 50.0, 100.0, 200.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026,
                        help="Seed used only to draw the randomized loss hyperparameters")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for main.py")
    parser.add_argument("--start", default="all",
                        help="Start from this loss name (inclusive), useful after interruption")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands and manifest without launching training")
    args = parser.parse_args()

    root = Path.cwd()
    if not (root / "main.py").is_file():
        raise SystemExit("Run this script from the project root (the folder containing main.py).")
    if args.start not in ("all", *LOSSES):
        raise SystemExit(f"Unknown --start loss {args.start!r}. Choose from: {', '.join(LOSSES)}")

    rng = random.Random(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = root / f"random_loss_manifest_{stamp}.csv"
    selected = LOSSES[LOSSES.index(args.start):] if args.start != "all" else LOSSES
    rows = []

    for index, name in enumerate(selected, start=1):
        zf = rng.choice(ZF_VALUES)
        sigma = rng.choice(SIGMA_VALUES)
        rho = rng.choice(RHO_VALUES)
        beta = rng.choice(BETA_VALUES)
        run_id = f"random{args.seed}_{stamp}"

        # Same training budget/optimizer/scheduler/seed from the existing Hydra config.
        # Only the selected loss and its relevant hyperparameters are overridden.
        cmd = [
            args.python, "main.py",
            f"train_params.loss_name={name}",
            f"train_params.loss_zf={zf}",
            f"train_params.loss_sigma={sigma}",
            f"train_params.loss_rho={rho}",
            f"train_params.loss_beta={beta}",
            f"train_params.run_id={run_id}",
            "model_params.resume_snapshot=null",
        ]
        rows.append({"loss": name, "zf": zf, "sigma": sigma, "rho": rho,
                     "beta": beta, "run_id": run_id, "status": "pending"})

        print("\\n" + "=" * 88, flush=True)
        print(f"[{index}/{len(selected)}] {name} | zf={zf} sigma={sigma} rho={rho} beta={beta}", flush=True)
        print("Command:", " ".join(cmd), flush=True)
        if args.dry_run:
            rows[-1]["status"] = "dry-run"
            continue

        result = subprocess.run(cmd, cwd=root)
        rows[-1]["status"] = "ok" if result.returncode == 0 else f"failed({result.returncode})"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if result.returncode != 0:
            print(f"Run {name} failed. Continuing with the next loss.", flush=True)

    if not args.dry_run:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\\nManifest saved to: {manifest_path}")
        print("After successful runs, aggregate results with:")
        print(f"{args.python} compare_loss_runs.py --root outputs/loss_comparison")
    else:
        print("\\nDry run complete; no training was started.")


if __name__ == "__main__":
    main()
