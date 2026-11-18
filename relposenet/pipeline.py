"""Train one RelPoseNet loss at a time and log comparable pose metrics."""

from __future__ import annotations

import csv
import math
import os
import random
from os import path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm

from relposenet.augmentations import get_pair_augmentations
from relposenet.criterion import RelPoseCriterion
from relposenet.dataset import SevenScenesRelPoseDataset
from relposenet.model import RelPoseNet
from relposenet.utils import set_seed


def _cfg_value(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _rotation_error_deg(q_est, q_gt):
    qe = F.normalize(q_est, p=2, dim=1, eps=1e-12)
    qg = F.normalize(q_gt, p=2, dim=1, eps=1e-12)
    dot = (qe * qg).sum(dim=1).abs().clamp(0.0, 1.0)
    sin_half = torch.sqrt((1.0 - dot.square()).clamp_min(0.0))
    angle = 2.0 * torch.atan2(sin_half, dot)
    return torch.rad2deg(angle)


def _pose_metrics(q_est, t_est, q_gt, t_gt):
    trans = torch.linalg.vector_norm(t_est - t_gt, dim=1)
    rot = _rotation_error_deg(q_est, q_gt)

    ne = torch.linalg.vector_norm(t_est, dim=1)
    ng = torch.linalg.vector_norm(t_gt, dim=1)
    valid = (ne > 1e-8) & (ng > 1e-8)
    # Keep one entry per sample so scene ids stay aligned. A direction angle is
    # undefined for a zero-length vector; represent it as NaN rather than drop it.
    direction = torch.full_like(trans, float("nan"))
    if valid.any():
        a = F.normalize(t_est[valid], p=2, dim=1)
        b = F.normalize(t_gt[valid], p=2, dim=1)
        dot = (a * b).sum(dim=1).clamp(-1.0, 1.0)
        direction[valid] = torch.rad2deg(torch.acos(dot))

    return trans.detach(), rot.detach(), direction.detach()


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reproducible initial weights, data order and augmentation randomness across loss runs.
        seed = int(_cfg_value(cfg, "seed", 1984))
        set_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.train_cfg = cfg.train_params
        self.data_cfg = cfg.data_params
        self.loss_name = str(_cfg_value(self.train_cfg, "loss_name", "mse")).lower()

        self.zf = float(_cfg_value(self.train_cfg, "loss_zf", 1.0))
        self.sigma = float(_cfg_value(self.train_cfg, "loss_sigma", 4.0))
        self.rho = float(_cfg_value(self.train_cfg, "loss_rho", 24.0))
        self.beta = float(_cfg_value(self.train_cfg, "loss_beta", 100.0))
        alpha = float(_cfg_value(self.train_cfg, "alpha", 1.0))

        self.require_geometry = self.loss_name in RelPoseCriterion.GEOMETRIC_NAMES

        camera_cfg = _cfg_value(cfg, "camera_params", None)
        train_aug, val_aug = get_pair_augmentations(camera_cfg)

        # Always validate pair labels against the absolute 7-Scenes poses.
        # This is required for trustworthy metric translation reports, even if
        # the selected objective itself is the original MSE baseline.
        self.train_dataset = SevenScenesRelPoseDataset(
            cfg, split="train", transforms=train_aug,
            require_geometry=True,
        )
        self.val_dataset = SevenScenesRelPoseDataset(
            cfg, split="val", transforms=val_aug,
            require_geometry=True,
        )

        if self.train_dataset.rotation_inverse != self.val_dataset.rotation_inverse:
            raise RuntimeError("Train/validation pair files use different quaternion conventions.")
        if self.train_dataset.translation_sign != self.val_dataset.translation_sign:
            raise RuntimeError("Train/validation pair files use different translation conventions.")
        self.rotation_inverse = self.train_dataset.rotation_inverse
        self.translation_sign = self.train_dataset.translation_sign

        batch_size = int(self.train_cfg.bs)
        workers = int(self.train_cfg.n_workers)
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(self.device.type == "cuda"),
            num_workers=workers,
            drop_last=True,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(self.device.type == "cuda"),
            num_workers=workers,
            drop_last=False,
            worker_init_fn=_seed_worker,
        )
        if len(self.train_loader) == 0 or len(self.val_loader) == 0:
            raise RuntimeError("Train and validation dataloaders must both contain at least one batch.")

        self.train_iterator = self._infinite_loader(self.train_loader)

        self.model = RelPoseNet(cfg.model_params).to(self.device)
        self.criterion = RelPoseCriterion(
            loss_name=self.loss_name,
            alpha=alpha,
            zf=self.zf,
            sigma=self.sigma,
            rho=self.rho,
            beta=self.beta,
            img_size=224,
            rotation_inverse=self.rotation_inverse,
            translation_sign=self.translation_sign,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.train_cfg.lr)
        )
        scheduler_cfg = self.train_cfg.scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(scheduler_cfg.lrate_decay_steps),
            gamma=float(scheduler_cfg.lrate_decay_factor),
        )

        # Loss runs must start from the same pretrained backbone and freshly initialized heads.
        resume = _cfg_value(cfg.model_params, "resume_snapshot", None)
        if resume:
            raise ValueError(
                "For a fair loss comparison, set model_params.resume_snapshot=null. "
                "Each loss must train from a fresh, identically seeded initialization."
            )

        self.run_dir = self._make_run_dir()
        if osp.exists(self.run_dir) and os.listdir(self.run_dir):
            raise FileExistsError(
                f"Output directory already contains a run: {self.run_dir}. "
                "Set train_params.run_id to a new unique value before rerunning, "
                "so metrics from different runs are not mixed."
            )
        self.log_dir = osp.join(self.run_dir, "tensorboard")
        self.checkpoint_dir = osp.join(self.run_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        self.train_curve_steps, self.train_curve_values = [], []
        self.val_curve_steps, self.val_curve_values = [], []
        self.history_rows = []
        self.csv_path = osp.join(self.run_dir, "metrics.csv")
        self.per_scene_csv_path = osp.join(self.run_dir, "per_scene_metrics.csv")

        if self.require_geometry:
            print(
                "WARNING: geometric losses use a shared K approximation unless you "
                "provide a validated RGB calibration. Official 7-Scenes does not provide "
                "calibrated RGB intrinsics; see README_FA.md."
            )

        print(f"Loss: {self.loss_name}; device: {self.device}; output: {self.run_dir}")

    @staticmethod
    def _infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    def _make_run_dir(self):
        out_root = str(_cfg_value(self.cfg.output_params, "out_dir", "outputs"))
        tag = self.loss_name
        if self.loss_name in {"hard", "soft", "softgrid", "soft_sqrt", "softgrid_sqrt"}:
            tag += f"_zf{self.zf:g}_s{self.sigma:g}"
        if self.loss_name in {"softgrid", "softgrid_sqrt"}:
            tag += f"_rho{self.rho:g}"
        if self.loss_name == "posenet":
            tag += f"_beta{self.beta:g}"
        suffix = str(_cfg_value(self.train_cfg, "run_id", ""))
        if suffix:
            tag += "_" + suffix
        return osp.join(out_root, "loss_comparison", tag)

    def _batch_to_device(self, batch):
        return {
            "img1": batch["img1"].to(self.device, non_blocking=True),
            "img2": batch["img2"].to(self.device, non_blocking=True),
            "q_gt": batch["q_gt"].to(self.device, non_blocking=True),
            "t_gt": batch["t_gt"].to(self.device, non_blocking=True),
            "R1": batch["R1"].to(self.device, non_blocking=True),
            "K": batch["K"].to(self.device, non_blocking=True),
            "scene_id": batch["scene_id"].to(self.device, non_blocking=True),
        }

    def _forward_loss(self, batch, training):
        data = self._batch_to_device(batch)
        q_est, t_est = self.model(data["img1"], data["img2"])
        per_sample = self.criterion(
            data["q_gt"], data["t_gt"], q_est, t_est,
            R1=data["R1"], K=data["K"],
            rotation_inverse=data["rotation_inverse"],
        )
        return per_sample, q_est, t_est, data

    def _validate(self):
        self.model.eval()
        losses, trans_all, rot_all, dir_all, scene_all = [], [], [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                per_sample, q_est, t_est, data = self._forward_loss(batch, training=False)
                losses.append(per_sample.detach())
                te, re, de = _pose_metrics(q_est, t_est, data["q_gt"], data["t_gt"])
                trans_all.append(te)
                rot_all.append(re)
                dir_all.append(de)
                scene_all.append(data["scene_id"].detach())

        loss_values = torch.cat(losses).cpu().numpy()
        trans = torch.cat(trans_all).cpu().numpy()
        rot = torch.cat(rot_all).cpu().numpy()
        scenes = torch.cat(scene_all).cpu().numpy()
        direction = torch.cat(dir_all).cpu().numpy()

        per_scene = []
        for sid in sorted(np.unique(scenes).tolist()):
            mask = scenes == sid
            scene_dir = direction[mask]
            finite_scene_dir = scene_dir[np.isfinite(scene_dir)]
            per_scene.append({
                "scene_id": int(sid),
                "scene": ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")[int(sid)],
                "count": int(mask.sum()),
                "translation_median_m": float(np.median(trans[mask])),
                "rotation_median_deg": float(np.median(rot[mask])),
                "direction_median_deg": float(np.median(finite_scene_dir)) if len(finite_scene_dir) else float("nan"),
            })

        result = {
            "val_loss": float(np.mean(loss_values)),
            "translation_mean_m": float(np.mean(trans)),
            "translation_median_m": float(np.median(trans)),
            "rotation_mean_deg": float(np.mean(rot)),
            "rotation_median_deg": float(np.median(rot)),
            "direction_median_deg": float(np.nanmedian(direction)) if np.isfinite(direction).any() else float("nan"),
            "pair_within_5cm_5deg_pct": float(100.0 * np.mean((trans <= 0.05) & (rot <= 5.0))),
            # Macro-average of per-scene medians mirrors the paper's scene-wise reporting.
            "scene_macro_translation_median_m": float(np.mean([x["translation_median_m"] for x in per_scene])),
            "scene_macro_rotation_median_deg": float(np.mean([x["rotation_median_deg"] for x in per_scene])),
            "per_scene": per_scene,
        }
        self.model.train()
        return result

    def _save_checkpoint(self, step, filename="latest.pth"):
        path = osp.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "step": step,
                "loss_name": self.loss_name,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "rotation_inverse": self.rotation_inverse,
                "translation_sign": self.translation_sign,
                "loss_params": {
                    "alpha": float(_cfg_value(self.train_cfg, "alpha", 1.0)),
                    "zf": self.zf,
                    "sigma": self.sigma,
                    "rho": self.rho,
                    "beta": self.beta,
                },
            },
            path,
        )

    def _append_history(self, row):
        self.history_rows.append(row)
        fieldnames = [
            "step", "train_loss_window_mean", "val_loss",
            "translation_mean_m", "translation_median_m",
            "rotation_mean_deg", "rotation_median_deg", "direction_median_deg",
            "pair_within_5cm_5deg_pct",
            "scene_macro_translation_median_m", "scene_macro_rotation_median_deg",
        ]
        exists = osp.isfile(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})

        per_scene = row.get("per_scene", [])
        if per_scene:
            scene_fields = [
                "step", "scene_id", "scene", "count",
                "translation_median_m", "rotation_median_deg", "direction_median_deg",
            ]
            scene_exists = osp.isfile(self.per_scene_csv_path)
            with open(self.per_scene_csv_path, "a", newline="", encoding="utf-8") as f:
                scene_writer = csv.DictWriter(f, fieldnames=scene_fields)
                if not scene_exists:
                    scene_writer.writeheader()
                for item in per_scene:
                    scene_writer.writerow({"step": row["step"], **item})

    def _plot_results(self):
        if self.train_curve_steps or self.val_curve_steps:
            fig, ax = plt.subplots(figsize=(9, 5))
            if self.train_curve_steps:
                ax.plot(self.train_curve_steps, self.train_curve_values, label="Train loss")
            if self.val_curve_steps:
                ax.plot(self.val_curve_steps, self.val_curve_values, label="Validation loss")
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Selected objective value")
            ax.set_title(f"Training curve — {self.loss_name}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(osp.join(self.run_dir, "loss_curves.png"), dpi=160)
            plt.close(fig)

        if self.history_rows:
            steps = [r["step"] for r in self.history_rows]
            trans = [r["scene_macro_translation_median_m"] for r in self.history_rows]
            rot = [r["scene_macro_rotation_median_deg"] for r in self.history_rows]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(steps, trans, marker="o", label="Mean of per-scene median translation errors (m)")
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Median translation error (m)")
            ax.set_title("Validation translation accuracy (scene-macro median)")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(osp.join(self.run_dir, "translation_accuracy.png"), dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(steps, rot, marker="o", label="Mean of per-scene median rotation errors (deg)")
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Median rotation error (degrees)")
            ax.set_title("Validation rotation accuracy (scene-macro median)")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(osp.join(self.run_dir, "rotation_accuracy.png"), dpi=160)
            plt.close(fig)

    def run(self):
        n_iters = int(self.train_cfg.n_train_iters)
        log_interval = int(self.cfg.output_params.log_scalar_interval)
        val_interval = int(self.cfg.output_params.validate_interval)
        running_train_losses = []
        train_loss_window_mean = float("nan")
        last_val_step = None

        self.model.train()
        for step in range(1, n_iters + 1):
            batch = next(self.train_iterator)
            self.optimizer.zero_grad(set_to_none=True)

            per_sample, _, _, _ = self._forward_loss(batch, training=True)
            loss = per_sample.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at step {step} for loss '{self.loss_name}'."
                )
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_train_losses.append(float(loss.detach().cpu()))

            if step % log_interval == 0:
                train_loss_window_mean = float(np.mean(running_train_losses))
                running_train_losses.clear()
                self.train_curve_steps.append(step)
                self.train_curve_values.append(train_loss_window_mean)
                self.writer.add_scalar("loss/train_window_mean", train_loss_window_mean, step)

            if step % val_interval == 0:
                # Record any tail since the last log interval, then start a fresh window.
                if running_train_losses:
                    train_loss_window_mean = float(np.mean(running_train_losses))
                    running_train_losses.clear()
                    self.train_curve_steps.append(step)
                    self.train_curve_values.append(train_loss_window_mean)
                    self.writer.add_scalar("loss/train_window_mean", train_loss_window_mean, step)
                val = self._validate()
                self.val_curve_steps.append(step)
                self.val_curve_values.append(val["val_loss"])
                self.writer.add_scalar("loss/validation", val["val_loss"], step)
                for key in (
                    "translation_mean_m", "translation_median_m",
                    "rotation_mean_deg", "rotation_median_deg", "direction_median_deg",
                    "pair_within_5cm_5deg_pct",
                    "scene_macro_translation_median_m", "scene_macro_rotation_median_deg",
                ):
                    self.writer.add_scalar(f"validation/{key}", val[key], step)

                row = {"step": step, "train_loss_window_mean": train_loss_window_mean, **val}
                self._append_history(row)
                self._save_checkpoint(step, "latest.pth")
                last_val_step = step

                print(
                    f"step {step}/{n_iters} | train={train_loss_window_mean:.6g} "
                    f"val={val['val_loss']:.6g} | "
                    f"macro median t={val['scene_macro_translation_median_m']:.4f} m | "
                    f"macro median R={val['scene_macro_rotation_median_deg']:.3f} deg"
                )
                train_loss_window_mean = float("nan")

        # Always evaluate the final weights, even when n_iters is not divisible by val_interval.
        if last_val_step != n_iters:
            if running_train_losses:
                train_loss_window_mean = float(np.mean(running_train_losses))
                self.train_curve_steps.append(n_iters)
                self.train_curve_values.append(train_loss_window_mean)
                self.writer.add_scalar("loss/train_window_mean", train_loss_window_mean, n_iters)
                running_train_losses.clear()
            val = self._validate()
            if running_train_losses:
                train_loss_window_mean = float(np.mean(running_train_losses))
            self.val_curve_steps.append(n_iters)
            self.val_curve_values.append(val["val_loss"])
            self.writer.add_scalar("loss/validation", val["val_loss"], n_iters)
            for key in (
                "translation_mean_m", "translation_median_m",
                "rotation_mean_deg", "rotation_median_deg", "direction_median_deg",
                "pair_within_5cm_5deg_pct",
                "scene_macro_translation_median_m", "scene_macro_rotation_median_deg",
            ):
                self.writer.add_scalar(f"validation/{key}", val[key], n_iters)
            row = {"step": n_iters, "train_loss_window_mean": train_loss_window_mean, **val}
            self._append_history(row)

        self._save_checkpoint(n_iters, "final.pth")
        self._plot_results()
        self.writer.close()

        final = self._validate()
        print("\nFinal validation metrics")
        for key, value in final.items():
            if key == "per_scene":
                continue
            print(f"  {key}: {value:.6f}")
        print(f"\nSaved results to: {self.run_dir}")
