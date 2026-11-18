"""7-Scenes pair dataset with absolute first-camera orientation for geometric losses."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from os import path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]


def _cfg_value(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def quaternion_wxyz_to_matrix_np(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-10:
        raise ValueError(f"Invalid quaternion in pair file: {q}")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )


def _rotation_error_deg(A, B):
    R = A.T @ B
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


class SevenScenesRelPoseDataset(Dataset):
    """Loads pairs and checks pair labels against official camera-to-world poses.

    The pair-list labels are retained for the original MSE baseline. For our
    geometric losses, R1 from the first image's absolute pose is used to rotate
    the pair translation into camera-1 coordinates.
    """

    def __init__(self, cfg, split="train", transforms=None, require_geometry=False):
        self.cfg = cfg
        self.split = split
        self.transforms = transforms
        self.require_geometry = bool(require_geometry)
        self.data_cfg = cfg.data_params
        self.img_dir = str(self.data_cfg.img_dir)
        self.pose_dir = str(_cfg_value(self.data_cfg, "pose_dir", self.img_dir))

        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(SCENES):
            self.scenes_dict[i] = scene

        pair_file = (
            self.data_cfg.train_pairs_fname
            if split == "train"
            else self.data_cfg.val_pairs_fname
        )
        self.img1_paths, self.img2_paths = [], []
        self.t_gt, self.q_gt = [], []
        self.scene_ids = []
        self.pose1_paths, self.pose2_paths = [], []

        with open(pair_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                chunks = line.split()
                if len(chunks) < 10:
                    raise ValueError(
                        f"Malformed pair file line {line_number}: expected >=10 fields, got {len(chunks)}"
                    )

                scene_id = int(chunks[2])
                if scene_id not in self.scenes_dict:
                    raise ValueError(f"Unknown scene id {scene_id} at line {line_number}")
                scene = self.scenes_dict[scene_id]

                rel1 = chunks[0].replace("\\", "/").lstrip("/")
                rel2 = chunks[1].replace("\\", "/").lstrip("/")
                img1 = osp.join(self.img_dir, scene, rel1)
                img2 = osp.join(self.img_dir, scene, rel2)
                self.img1_paths.append(img1)
                self.img2_paths.append(img2)

                self.t_gt.append(torch.tensor([float(x) for x in chunks[3:6]], dtype=torch.float32))
                self.q_gt.append(torch.tensor([float(x) for x in chunks[6:10]], dtype=torch.float32))
                self.scene_ids.append(scene_id)

                p1 = osp.join(self.pose_dir, scene, rel1.replace(".color.png", ".pose.txt"))
                p2 = osp.join(self.pose_dir, scene, rel2.replace(".color.png", ".pose.txt"))
                self.pose1_paths.append(p1)
                self.pose2_paths.append(p2)

        if not self.img1_paths:
            raise RuntimeError(f"No pairs were loaded from {pair_file}")

        self.rotation_inverse = False
        self.translation_sign = 1.0
        if self.require_geometry:
            self.rotation_inverse, self.translation_sign = self._validate_pair_pose_convention()

    def _read_pose(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError(
                f"Required 7-Scenes pose file not found: {path}\n"
                "For Hard/Soft/SoftGrid/Ray losses, the corresponding "
                "frame-XXXXXX.pose.txt files must be available. Set data_params.pose_dir "
                "if poses are stored separately from img_dir."
            )
        pose = np.loadtxt(path, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 camera-to-world pose in {path}, got {pose.shape}")
        if not np.isfinite(pose).all():
            raise ValueError(f"Non-finite values in pose file: {path}")
        R = pose[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=5e-2) or np.linalg.det(R) < 0.8:
            raise ValueError(f"Rotation in pose file is not a valid SO(3) matrix: {path}")
        return pose

    def _validate_pair_pose_convention(self, max_pairs=96):
        """Check wxyz + relative-rotation order + signed metric centre displacement.

        q_gt is expected to encode either R1.T@R2 or its inverse. t_gt is
        expected to encode either C2-C1 or its negative, in world coordinates.
        We detect the consistent signs, then criterion.py converts them to the
        camera-1 frame. A mismatch stops geometric training rather than silently
        optimizing an invalid geometric objective.
        """
        rot_err_forward, rot_err_inverse = [], []
        trans_err_forward, trans_err_inverse = [], []
        used = 0

        sample_count = min(len(self.img1_paths), max_pairs)
        sample_indices = np.linspace(0, len(self.img1_paths) - 1, sample_count, dtype=int)
        for i in sample_indices:
            P1 = self._read_pose(self.pose1_paths[i])
            P2 = self._read_pose(self.pose2_paths[i])
            R1, C1 = P1[:3, :3], P1[:3, 3]
            R2, C2 = P2[:3, :3], P2[:3, 3]
            delta = C2 - C1
            if np.linalg.norm(delta) < 1e-2:
                continue
            t_label = self.t_gt[i].numpy().astype(np.float64)
            qR = quaternion_wxyz_to_matrix_np(self.q_gt[i].numpy())

            # Official 7-Scenes pose files are camera-to-world. The paper's
            # pair relation is R1^T R2; also test the inverse for robust checking.
            rot_err_forward.append(_rotation_error_deg(qR, R1.T @ R2))
            rot_err_inverse.append(_rotation_error_deg(qR, R2.T @ R1))

            denom = float(np.linalg.norm(delta))
            trans_err_forward.append(float(np.linalg.norm(t_label - delta)) / denom)
            trans_err_inverse.append(float(np.linalg.norm(t_label + delta)) / denom)
            used += 1

        if used < 8:
            raise RuntimeError(
                f"Only {used} valid pairs were available for geometry convention checking; need at least 8."
            )

        med_rot_f = float(np.median(rot_err_forward))
        med_rot_i = float(np.median(rot_err_inverse))
        med_t_f = float(np.median(trans_err_forward))
        med_t_i = float(np.median(trans_err_inverse))

        rotation_inverse = med_rot_i < med_rot_f
        best_rot = min(med_rot_f, med_rot_i)
        translation_sign = -1.0 if med_t_i < med_t_f else 1.0
        best_t = min(med_t_f, med_t_i)

        # A small tolerance allows for text-file rounding, but prevents using
        # a different transform convention or a non-metric/unit-only target.
        if best_rot > 2.0:
            raise RuntimeError(
                "Pair quaternion convention did not match the 7-Scenes poses. "
                f"Median angular errors: R1.T@R2={med_rot_f:.3f} deg, "
                f"R2.T@R1={med_rot_i:.3f} deg. Check quaternion order (this code assumes w,x,y,z) "
                "and the pair-file image order before using geometric losses."
            )
        if best_t > 0.05:
            raise RuntimeError(
                "Pair translation labels did not match the metric camera-centre displacement. "
                f"Median relative errors: +(C2-C1)={med_t_f:.4f}, "
                f"-(C2-C1)={med_t_i:.4f}. The focus-point losses need metric translation. "
                "Do not continue by substituting an arbitrary scale."
            )

        print(
            f"[{self.split}] Pair convention verified on {used} pairs: "
            f"rotation={'inverse' if rotation_inverse else 'R1.T@R2'} "
            f"(median {best_rot:.3f} deg), "
            f"translation={'-(C2-C1)' if translation_sign < 0 else '(C2-C1)'} "
            f"(median relative error {best_t:.4f})."
        )
        return rotation_inverse, translation_sign

    def __getitem__(self, item):
        img1 = Image.open(self.img1_paths[item]).convert("RGB")
        img2 = Image.open(self.img2_paths[item]).convert("RGB")
        t_gt = self.t_gt[item].clone()
        q_gt = self.q_gt[item].clone()
        pair_swapped = False

        if self.require_geometry:
            pose1 = self._read_pose(self.pose1_paths[item])
            pose2 = self._read_pose(self.pose2_paths[item])
            R1 = torch.tensor(pose1[:3, :3], dtype=torch.float32)
            # pose2 is read here as an existence/validity check and to keep the
            # pair-data assumptions explicit; R1 is all criterion needs.
        else:
            R1 = torch.eye(3, dtype=torch.float32)

        # Preserve the original pair-swap augmentation, but only for training.
        # With camera-to-world poses, the first camera orientation must swap too.
        if self.split == "train" and random.random() > 0.5:
            img1, img2 = img2, img1
            t_gt = -t_gt
            # Conjugating wxyz gives the inverse relative rotation. This is
            # different from sign negation, which represents the same rotation.
            q_gt = torch.stack((q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]))
            if self.require_geometry:
                R1 = torch.tensor(pose2[:3, :3], dtype=torch.float32)
            pair_swapped = True

        if self.transforms is None:
            raise RuntimeError("A paired image transform is required; use get_augmentations(camera_params).")

        img1, img2, K = self.transforms(img1, img2)
        return {
            "img1": img1,
            "img2": img2,
            "t_gt": t_gt,
            "q_gt": q_gt,
            "R1": R1,
            "K": K,
            "scene_id": torch.tensor(self.scene_ids[item], dtype=torch.long),
            "rotation_inverse": torch.tensor(
                self.rotation_inverse ^ pair_swapped,
                dtype=torch.bool,
            ),
        }

    def __len__(self):
        return len(self.img1_paths)


class SevenScenesTestDataset(Dataset):
    """Original evaluation dataset kept for compatibility with experiments/."""

    def __init__(self, experiment_cfg, transforms=None):
        self.experiment_cfg = experiment_cfg
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(SCENES):
            self.scenes_dict[i] = scene
        self.img1_paths, self.img2_paths = self._read_pairs_txt()

    def _read_pairs_txt(self):
        img1_paths, img2_paths = [], []
        pair_file = self.experiment_cfg.paths.test_pairs_fname
        img_dir = self.experiment_cfg.paths.img_path
        with open(pair_file, "r", encoding="utf-8") as f:
            for line in f:
                chunks = line.strip().split()
                if not chunks:
                    continue
                scene_id1, scene_id2 = int(chunks[2]), int(chunks[3])
                rel1 = chunks[1].replace("\\", "/").lstrip("/")
                rel2 = chunks[0].replace("\\", "/").lstrip("/")
                img1_paths.append(osp.join(img_dir, self.scenes_dict[scene_id2], rel1))
                img2_paths.append(osp.join(img_dir, self.scenes_dict[scene_id1], rel2))
        return img1_paths, img2_paths

    def __getitem__(self, item):
        img1 = Image.open(self.img1_paths[item]).convert("RGB")
        img2 = Image.open(self.img2_paths[item]).convert("RGB")
        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        return {"img1": img1, "img2": img2}

    def __len__(self):
        return len(self.img1_paths)
