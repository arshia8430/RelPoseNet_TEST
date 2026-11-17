"""Differentiable camera-pose losses used in the RelPoseNet comparison.

All functions return one loss value per sample, shape (B,).
Camera convention inside this module:
    X_world = R @ X_camera + t
where R is camera-to-world and t is the camera centre in the same world frame.

For pairwise training, criterion.py expresses camera 2 in camera 1's frame:
    R_rel = R1.T @ R2
    t_rel = R1.T @ (C2 - C1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Pixel coordinates refer to the 224x224 cropped network input.
IMG = 224


def _batch_K(K: torch.Tensor, batch_size: int) -> torch.Tensor:
    if K.ndim == 2:
        K = K.unsqueeze(0)
    if K.shape[0] == 1 and batch_size > 1:
        K = K.expand(batch_size, -1, -1)
    if K.shape != (batch_size, 3, 3):
        raise ValueError(f"K must be (3,3) or (B,3,3); got {tuple(K.shape)}")
    return K


def _validate_camera(cam: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for key in ("K", "R", "t"):
        if key not in cam:
            raise KeyError(f"Camera dictionary is missing '{key}'")

    R = cam["R"]
    t = cam["t"]
    if R.ndim != 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (B,3,3); got {tuple(R.shape)}")
    B = R.shape[0]
    if t.shape != (B, 3):
        raise ValueError(f"t must have shape (B,3); got {tuple(t.shape)}")

    K = _batch_K(cam["K"], B)
    if not (K.device == R.device == t.device):
        raise ValueError("K, R and t must be on the same device")
    if not (K.dtype == R.dtype == t.dtype):
        raise ValueError("K, R and t must have the same dtype")
    return K, R, t


def pixel_corners(img_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    m = float(img_size - 1)
    return torch.tensor(
        [[0.0, m, m, 0.0], [0.0, 0.0, m, m], [1.0, 1.0, 1.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def pixel_grid(grid_size: int, img_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    u = torch.linspace(0.0, img_size - 1.0, grid_size, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, u, indexing="xy")
    return torch.stack((uu.reshape(-1), vv.reshape(-1), torch.ones_like(uu).reshape(-1)))


def dense_pixel_grid(grid_size: int, img_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    c = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) * img_size / grid_size - 0.5
    uu, vv = torch.meshgrid(c, c, indexing="xy")
    return torch.stack((uu.reshape(-1), vv.reshape(-1), torch.ones_like(uu).reshape(-1)))


def rays_cam(K: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Unproject homogeneous pixels P (3,N) to camera rays (B,3,N)."""
    B = K.shape[0]
    P = P.to(device=K.device, dtype=K.dtype).unsqueeze(0).expand(B, -1, -1)
    return torch.linalg.solve(K, P)


def focus_points(cam: dict, P: torch.Tensor, zf: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-space points at zf along pixel rays, and unit ray directions."""
    K, R, t = _validate_camera(cam)
    q = rays_cam(K, P)
    Rq = R @ q
    points = t.unsqueeze(-1) + float(zf) * Rq
    directions = F.normalize(Rq, p=2, dim=1, eps=1e-12)
    return points, directions


# -----------------------------------------------------------------------------
# Proposed losses
# -----------------------------------------------------------------------------

def loss_hard(cam: dict, tgt: dict, zf: float = 1.0, img_size: int = IMG) -> torch.Tensor:
    """Mean squared distance between the four corresponding focus points."""
    _, R, _ = _validate_camera(cam)
    _validate_camera(tgt)
    P = pixel_corners(img_size, R.device, R.dtype)
    A, _ = focus_points(cam, P, zf)
    B, _ = focus_points(tgt, P, zf)
    return (A - B).square().mean(dim=(1, 2))


def loss_dir(cam: dict, tgt: dict, img_size: int = IMG) -> torch.Tensor:
    """Direction-only ablation, averaged over xyz and four image corners."""
    _, R, _ = _validate_camera(cam)
    _validate_camera(tgt)
    P = pixel_corners(img_size, R.device, R.dtype)
    _, d = focus_points(cam, P)
    _, dt = focus_points(tgt, P)
    return (d - dt).square().mean(dim=(1, 2))


def loss_soft(
    cam: dict,
    tgt: dict,
    zf: float = 1.0,
    sigma: float = 4.0,
    img_size: int = IMG,
) -> torch.Tensor:
    """Closed-form Gaussian-weighted ray loss from the supplied formulation."""
    hard = loss_hard(cam, tgt, zf=zf, img_size=img_size)
    direction = loss_dir(cam, tgt, img_size=img_size)
    return hard + (float(sigma) ** 2 / 2.0) * direction


def loss_soft_sqrt(
    cam: dict,
    tgt: dict,
    zf: float = 1.0,
    sigma: float = 4.0,
    img_size: int = IMG,
) -> torch.Tensor:
    value = loss_soft(cam, tgt, zf=zf, sigma=sigma, img_size=img_size)
    eps = 1e-12
    # Subtracting sqrt(eps) preserves the smoothed gradient but makes L=0
    # exactly when the two poses are identical.
    return torch.sqrt(value + eps) - eps ** 0.5


def make_softgrid(
    rho: float,
    zf: float = 1.0,
    sigma: float = 4.0,
    n: int = 5,
    trunc: float = 2.0,
    inward: bool = False,
    root: bool = False,
    img_size: int = IMG,
):
    """Create a SoftGrid loss with Gaussian-weighted samples around each corner."""
    if rho <= 0:
        raise ValueError("rho must be positive")
    if n < 2:
        raise ValueError("n must be at least 2")

    def fn(cam: dict, tgt: dict) -> torch.Tensor:
        _, R, _ = _validate_camera(cam)
        _validate_camera(tgt)
        device, dtype = R.device, R.dtype

        corners = pixel_corners(img_size, device, dtype)
        offsets = torch.linspace(-trunc * rho, trunc * rho, n, device=device, dtype=dtype)
        ox, oy = torch.meshgrid(offsets, offsets, indexing="xy")
        ox, oy = ox.reshape(-1), oy.reshape(-1)

        weights = torch.exp(-(ox.square() + oy.square()) / (2.0 * rho**2))
        weights = weights / weights.sum().clamp_min(1e-12)

        pixel_chunks = []
        for i in range(4):
            cu, cv = corners[0, i], corners[1, i]
            if inward:
                sx = 1.0 if float(cu) < img_size / 2 else -1.0
                sy = 1.0 if float(cv) < img_size / 2 else -1.0
                dx, dy = sx * ox.abs(), sy * oy.abs()
            else:
                dx, dy = ox, oy
            pixel_chunks.append(torch.stack((cu + dx, cv + dy, torch.ones_like(dx))))

        P = torch.cat(pixel_chunks, dim=1)
        point_weights = torch.cat([weights / 4.0] * 4)

        A, da = focus_points(cam, P, zf=zf)
        B, db = focus_points(tgt, P, zf=zf)

        hard = ((A - B).square().mean(dim=1) * point_weights.unsqueeze(0)).sum(dim=1)
        direction = ((da - db).square().mean(dim=1) * point_weights.unsqueeze(0)).sum(dim=1)
        value = hard + (float(sigma) ** 2 / 2.0) * direction
        if root:
            eps = 1e-12
            return torch.sqrt(value + eps) - eps ** 0.5
        return value

    return fn


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------

def quat_pair(cam: dict, tgt: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and sign-align q/tgt so q and -q represent the same rotation."""
    if "q" not in cam or "q" not in tgt:
        raise KeyError("This baseline requires cam['q'] and tgt['q']")
    qc = F.normalize(cam["q"], p=2, dim=-1, eps=1e-12)
    qt = F.normalize(tgt["q"], p=2, dim=-1, eps=1e-12)
    sign = torch.where(
        (qc * qt).sum(dim=-1, keepdim=True) < 0,
        -torch.ones_like(qc[:, :1]),
        torch.ones_like(qc[:, :1]),
    ).detach()
    return qc, qt * sign


def l2(x: torch.Tensor) -> torch.Tensor:
    # torch.linalg.vector_norm returns exactly zero at x=0 and has a finite
    # zero sub-gradient there in PyTorch.
    return torch.linalg.vector_norm(x, ord=2, dim=-1)


def loss_pmse(cam: dict, tgt: dict, img_size: int = IMG) -> torch.Tensor:
    """Pose MSE on [t, q, fx/IMG, fy/IMG], with quaternion sign alignment."""
    Kc, _, tc = _validate_camera(cam)
    Kt, _, tt = _validate_camera(tgt)
    qc, qt = quat_pair(cam, tgt)
    vc = torch.cat(
        (tc, qc, (Kc[:, 0, 0] / img_size).unsqueeze(1), (Kc[:, 1, 1] / img_size).unsqueeze(1)),
        dim=1,
    )
    vt = torch.cat(
        (tt, qt, (Kt[:, 0, 0] / img_size).unsqueeze(1), (Kt[:, 1, 1] / img_size).unsqueeze(1)),
        dim=1,
    )
    return (vc - vt).square().mean(dim=1)


def loss_posenet(cam: dict, tgt: dict, beta: float = 100.0) -> torch.Tensor:
    Kc, _, tc = _validate_camera(cam)
    Kt, _, tt = _validate_camera(tgt)
    qc, qt = quat_pair(cam, tgt)
    translation = l2(tc - tt)
    rotation = l2(qc - qt)
    focal = (torch.log(Kc[:, 0, 0]) - torch.log(Kt[:, 0, 0])).abs()
    return translation + float(beta) * rotation + focal


def so3_angle(Rm: torch.Tensor) -> torch.Tensor:
    """Geodesic SO(3) angle in radians, using a stable atan2 expression."""
    cos_theta = (Rm.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0
    vee = torch.stack(
        (
            Rm[:, 2, 1] - Rm[:, 1, 2],
            Rm[:, 0, 2] - Rm[:, 2, 0],
            Rm[:, 1, 0] - Rm[:, 0, 1],
        ),
        dim=1,
    ) / 2.0
    sin_theta = torch.linalg.vector_norm(vee, dim=1)
    return torch.atan2(sin_theta, cos_theta.clamp(-1.0, 1.0))


def loss_se3(cam: dict, tgt: dict) -> torch.Tensor:
    Kc, Rc, tc = _validate_camera(cam)
    Kt, Rt, tt = _validate_camera(tgt)
    translation = l2(tc - tt)
    rotation = so3_angle(Rc.transpose(1, 2) @ Rt)
    focal = (torch.log(Kc[:, 0, 0]) - torch.log(Kt[:, 0, 0])).abs()
    return translation + rotation + focal


def _plucker(cam: dict, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, d = focus_points(cam, P)
    _, _, t = _validate_camera(cam)
    m = torch.cross(t.unsqueeze(-1).expand_as(d), d, dim=1)
    return d, m


def _ray_loss(cam: dict, tgt: dict, P: torch.Tensor) -> torch.Tensor:
    d, m = _plucker(cam, P)
    dt, mt = _plucker(tgt, P)
    return (d - dt).square().mean(dim=(1, 2)) + (m - mt).square().mean(dim=(1, 2))


def loss_ray(cam: dict, tgt: dict, grid_size: int = 6, img_size: int = IMG) -> torch.Tensor:
    _, R, _ = _validate_camera(cam)
    P = pixel_grid(grid_size, img_size, R.device, R.dtype)
    return _ray_loss(cam, tgt, P)


def loss_denseray(cam: dict, tgt: dict, grid_size: int = 16, img_size: int = IMG) -> torch.Tensor:
    _, R, _ = _validate_camera(cam)
    P = dense_pixel_grid(grid_size, img_size, R.device, R.dtype)
    return _ray_loss(cam, tgt, P)
