"""Loss selector/adapter between RelPoseNet outputs and geometric losses.

Inputs q_gt/t_gt/q_est/t_est follow the original pair-file convention.
For focus-point losses we convert the relative displacement into camera-1's
coordinate frame and express camera 2 relative to camera 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from relposenet import losses as L


def quaternion_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert (w,x,y,z) quaternions to rotation matrices; normalizes q first."""
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Quaternion must have shape (B,4), got {tuple(q.shape)}")

    q = F.normalize(q, p=2, dim=1, eps=1e-12)
    w, x, y, z = q.unbind(dim=1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    row0 = torch.stack((ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)), dim=1)
    row1 = torch.stack((2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)), dim=1)
    row2 = torch.stack((2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz), dim=1)
    return torch.stack((row0, row1, row2), dim=1)


class RelPoseCriterion(nn.Module):
    """Select one loss per independent training run.

    Supported names:
        mse, hard, soft, softgrid, dir, pmse, posenet10, posenet100,
        se3, ray, denseray, soft_sqrt, softgrid_sqrt

    The original MSE criterion is preserved exactly in its reduction and
    weighting: mean(t squared error) + alpha * mean(q squared error).
    """

    GEOMETRIC_NAMES = {"hard", "soft", "softgrid", "dir", "ray", "denseray", "soft_sqrt", "softgrid_sqrt"}

    def __init__(
        self,
        loss_name: str = "mse",
        alpha: float = 1.0,
        zf: float = 1.0,
        sigma: float = 4.0,
        rho: float = 24.0,
        beta: float = 100.0,
        img_size: int = 224,
        rotation_inverse: bool = False,
        translation_sign: float = 1.0,
    ):
        super().__init__()
        self.loss_name = str(loss_name).lower()
        self.alpha = float(alpha)
        self.zf = float(zf)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)
        self.img_size = int(img_size)
        self.rotation_inverse = bool(rotation_inverse)
        self.translation_sign = float(translation_sign)

        allowed = {
            "mse", "hard", "soft", "softgrid", "dir", "pmse",
            "posenet10", "posenet100", "posenet", "se3", "ray",
            "denseray", "soft_sqrt", "softgrid_sqrt",
        }
        if self.loss_name not in allowed:
            raise ValueError(f"Unknown loss '{loss_name}'. Choose from: {sorted(allowed)}")

        if self.loss_name in self.GEOMETRIC_NAMES:
            if self.loss_name == "softgrid_sqrt":
                self._softgrid_fn = L.make_softgrid(
                    self.rho, zf=self.zf, sigma=self.sigma,
                    root=True, img_size=self.img_size,
                )
            elif self.loss_name == "softgrid":
                self._softgrid_fn = L.make_softgrid(
                    self.rho, zf=self.zf, sigma=self.sigma,
                    root=False, img_size=self.img_size,
                )
            else:
                self._softgrid_fn = None
        else:
            self._softgrid_fn = None

    def _camera_dicts(self, q_gt, t_gt, q_est, t_est, R1, K, rotation_inverse=None):
        """Build camera-2 poses in camera-1 coordinates for geometric losses."""
        if R1 is None or K is None:
            raise ValueError(
                f"Loss '{self.loss_name}' requires R1 and K from the paired dataset."
            )

        B = q_est.shape[0]
        if R1.ndim == 2:
            R1 = R1.unsqueeze(0).expand(B, -1, -1)
        if R1.shape != (B, 3, 3):
            raise ValueError(f"R1 must be (3,3) or (B,3,3), got {tuple(R1.shape)}")
        if K.ndim == 2:
            K = K.unsqueeze(0).expand(B, -1, -1)
        if K.shape != (B, 3, 3):
            raise ValueError(f"K must be (B,3,3), got {tuple(K.shape)}")
        if R1.device != q_est.device or R1.dtype != q_est.dtype:
            raise ValueError("R1, q_est and q_gt must be on the same device and dtype")
        if K.device != q_est.device or K.dtype != q_est.dtype:
            raise ValueError("K, q_est and q_gt must be on the same device and dtype")

        R_est = quaternion_wxyz_to_matrix(q_est)
        R_gt = quaternion_wxyz_to_matrix(q_gt)

        # The data-convention validator determines whether q stores R1^T R2
        # or its inverse. Convert both labels/predictions to camera-2-in-camera-1.
        inverse = self.rotation_inverse if rotation_inverse is None else rotation_inverse
        if torch.is_tensor(inverse):
            inverse = inverse.to(device=q_est.device, dtype=torch.bool)
            if inverse.ndim == 0:
                inverse = inverse.expand(B)
            elif inverse.shape != (B,):
                raise ValueError(f"rotation_inverse must be a scalar or (B,), got {tuple(inverse.shape)}")
        elif isinstance(inverse, bool):
            inverse = torch.full((B,), inverse, dtype=torch.bool, device=q_est.device)
        else:
            raise TypeError("rotation_inverse must be a bool or a (B,) tensor")

        if inverse.any():
            if inverse.all():
                R_est = R_est.transpose(1, 2)
                R_gt = R_gt.transpose(1, 2)
            else:
                R_est = R_est.clone()
                R_gt = R_gt.clone()
                R_est[inverse] = R_est[inverse].transpose(1, 2)
                R_gt[inverse] = R_gt[inverse].transpose(1, 2)

        # Pair-file t is the signed world-frame camera-centre displacement.
        # Rotate it into camera 1's frame before using it as a camera centre.
        d_est_world = self.translation_sign * t_est
        d_gt_world = self.translation_sign * t_gt
        t_est_local = (R1.transpose(1, 2) @ d_est_world.unsqueeze(-1)).squeeze(-1)
        t_gt_local = (R1.transpose(1, 2) @ d_gt_world.unsqueeze(-1)).squeeze(-1)

        cam_est = {"K": K, "R": R_est, "t": t_est_local, "q": q_est}
        cam_gt = {"K": K, "R": R_gt, "t": t_gt_local, "q": q_gt}
        return cam_est, cam_gt

    def forward(
        self,
        q_gt,
        t_gt,
        q_est,
        t_est,
        R1=None,
        K=None,
        rotation_inverse=None,
    ):
        """Return a per-sample loss vector, shape (B,)."""
        if not (torch.is_tensor(q_gt) and torch.is_tensor(t_gt)
                and torch.is_tensor(q_est) and torch.is_tensor(t_est)):
            raise TypeError("q_gt, t_gt, q_est and t_est must be torch tensors")
        if q_gt.shape != q_est.shape or t_gt.shape != t_est.shape:
            raise ValueError("Prediction and target shapes must match")
        if q_est.ndim != 2 or q_est.shape[-1] != 4:
            raise ValueError(f"Quaternion tensors must have shape (B,4), got {tuple(q_est.shape)}")
        if t_est.ndim != 2 or t_est.shape[-1] != 3:
            raise ValueError(f"Translation tensors must have shape (B,3), got {tuple(t_est.shape)}")
        if q_est.shape[0] != t_est.shape[0]:
            raise ValueError("Quaternion and translation batch sizes must match")
        if not (q_est.device == q_gt.device == t_est.device == t_gt.device):
            raise ValueError("Prediction and target tensors must be on the same device")
        if not (q_est.dtype == q_gt.dtype == t_est.dtype == t_gt.dtype):
            raise ValueError("Prediction and target tensors must have the same dtype")

        # Original baseline: keep the original unnormalized regression outputs.
        if self.loss_name == "mse":
            q_loss = (q_est - q_gt).square().mean(dim=1)
            t_loss = (t_est - t_gt).square().mean(dim=1)
            return t_loss + self.alpha * q_loss

        # Baselines that compare the relative output vectors directly.
        if self.loss_name in {"pmse", "posenet", "posenet10", "posenet100", "se3"}:
            if K is None:
                raise ValueError(f"Loss '{self.loss_name}' requires K")
            B = q_est.shape[0]
            if K.ndim == 2:
                K = K.unsqueeze(0).expand(B, -1, -1)
            if K.shape != (B, 3, 3):
                raise ValueError(f"K must be (3,3) or (B,3,3), got {tuple(K.shape)}")
            if K.device != q_est.device or K.dtype != q_est.dtype:
                raise ValueError("K, q_est and q_gt must be on the same device and dtype")
            R_est = quaternion_wxyz_to_matrix(q_est)
            R_gt = quaternion_wxyz_to_matrix(q_gt)
            raw_est = {"K": K, "R": R_est, "t": t_est, "q": q_est}
            raw_gt = {"K": K, "R": R_gt, "t": t_gt, "q": q_gt}

            if self.loss_name == "pmse":
                return L.loss_pmse(raw_est, raw_gt, img_size=self.img_size)
            if self.loss_name == "posenet10":
                return L.loss_posenet(raw_est, raw_gt, beta=10.0)
            if self.loss_name == "posenet100":
                return L.loss_posenet(raw_est, raw_gt, beta=100.0)
            if self.loss_name == "posenet":
                return L.loss_posenet(raw_est, raw_gt, beta=self.beta)
            return L.loss_se3(raw_est, raw_gt)

        # The remaining losses require a geometrically consistent relative camera pair.
        cam_est, cam_gt = self._camera_dicts(
            q_gt,
            t_gt,
            q_est,
            t_est,
            R1,
            K,
            rotation_inverse=(
                self.rotation_inverse
                if rotation_inverse is None
                else rotation_inverse
            ),
        )

        if self.loss_name == "hard":
            return L.loss_hard(cam_est, cam_gt, zf=self.zf, img_size=self.img_size)
        if self.loss_name == "soft":
            return L.loss_soft(cam_est, cam_gt, zf=self.zf, sigma=self.sigma, img_size=self.img_size)
        if self.loss_name == "softgrid":
            return self._softgrid_fn(cam_est, cam_gt)
        if self.loss_name == "dir":
            return L.loss_dir(cam_est, cam_gt, img_size=self.img_size)
        if self.loss_name == "ray":
            return L.loss_ray(cam_est, cam_gt, grid_size=6, img_size=self.img_size)
        if self.loss_name == "denseray":
            return L.loss_denseray(cam_est, cam_gt, grid_size=16, img_size=self.img_size)
        if self.loss_name == "soft_sqrt":
            return L.loss_soft_sqrt(
                cam_est, cam_gt, zf=self.zf, sigma=self.sigma, img_size=self.img_size
            )
        if self.loss_name == "softgrid_sqrt":
            return self._softgrid_fn(cam_est, cam_gt)

        raise RuntimeError(f"Loss '{self.loss_name}' was not dispatched")
