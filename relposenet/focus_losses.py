"""Focus-point losses (Hard / Soft / SoftGrid) and baselines for relative-pose training.

Camera convention of the loss paper: dict(K, R, t) with (R, t) the camera-to-world pose (x right, y down, z forward).
Relative-pose setting: camera 1 is the reference (identity pose); camera 2 has pose (R_rel, t_rel) expressed in
camera-1 coordinates.  Both cameras are images of the same Kinect, so they share ONE intrinsic matrix K.
Every loss has the signature fn(cam, tgt) -> (B,)  (lower is better); cam/tgt carry K, R, t and the quaternions
q (unit, used by quaternion baselines) and q_raw (un-normalised network output, used by the original MSE).
"""
import re

import torch
import torch.nn.functional as F

IMG = 224                                      # network input = 224x224 crop of the 256-pixel-high image
FOCAL_PX_DEFAULT = 525.0 * 256.0 / 480.0       # = 280 px; Kinect RGB default f=525 px at 640x480, resized 480->256


def make_K(B, device, f=FOCAL_PX_DEFAULT):
    K = torch.zeros(B, 3, 3, device=device)
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K[:, 0, 2] = IMG / 2
    K[:, 1, 2] = IMG / 2
    K[:, 2, 2] = 1.0
    return K


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------
def quat_to_R(q):
    """(B,4) quaternion (w,x,y,z), normalised here -> (B,3,3) rotation matrix."""
    q = F.normalize(q, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def so3_angle(Rm):
    cos = (Rm.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    vee = torch.stack([Rm[..., 2, 1] - Rm[..., 1, 2], Rm[..., 0, 2] - Rm[..., 2, 0], Rm[..., 1, 0] - Rm[..., 0, 1]], -1) / 2
    return torch.atan2(torch.sqrt((vee ** 2).sum(-1) + 1e-12), cos)


def l2(x):
    return torch.sqrt((x ** 2).sum(-1) + 1e-12)


def invert_cam(c):
    """Camera 1 expressed in camera-2 coordinates:  R' = R^T,  t' = -R^T t."""
    Rt = c["R"].transpose(1, 2)
    out = dict(K=c["K"], R=Rt, t=-(Rt @ c["t"][:, :, None])[:, :, 0])
    for k in ("q", "q_raw"):
        if k in c:
            out[k] = c[k] * c[k].new_tensor([1.0, -1.0, -1.0, -1.0])
    return out


_PIX = {}


def _pix(name, device):
    """Homogeneous pixel sets (3,N): image corners, 6x6 grid, 16x16 patch centres."""
    key = (name, str(device))
    if key not in _PIX:
        if name == "corn":
            P = torch.tensor([[0, IMG - 1, IMG - 1, 0], [0, 0, IMG - 1, IMG - 1], [1, 1, 1, 1]], dtype=torch.float32)
        elif name == "grid":
            u = torch.linspace(0, IMG - 1, 6)
            uu, vv = torch.meshgrid(u, u, indexing="xy")
            P = torch.stack([uu.flatten(), vv.flatten(), torch.ones(36)])
        else:  # dense
            c = (torch.arange(16, dtype=torch.float32) + 0.5) * IMG / 16 - 0.5
            uu, vv = torch.meshgrid(c, c, indexing="xy")
            P = torch.stack([uu.flatten(), vv.flatten(), torch.ones(256)])
        _PIX[key] = P.to(device)
    return _PIX[key]


def rays_cam(K, P):
    """K^{-1} P  ->  (B,3,N)."""
    fx, fy, cx, cy = K[:, 0, 0, None], K[:, 1, 1, None], K[:, 0, 2, None], K[:, 1, 2, None]
    x = (P[0][None] - cx) / fx
    y = (P[1][None] - cy) / fy
    return torch.stack([x, y, torch.ones_like(x)], 1)


# ----------------------------------------------------------------------------------------------
# proposed losses
# ----------------------------------------------------------------------------------------------
def focus_points(cam, P, zf=1.0):
    """World points at depth zf along the rays of pixels P, and the unit ray directions."""
    q = rays_cam(cam["K"], P)
    Rq = cam["R"] @ q
    return cam["t"][:, :, None] + zf * Rq, Rq / q.norm(dim=1, keepdim=True)


def loss_hard(cam, tgt, zf=1.0):
    P = _pix("corn", cam["t"].device)
    A, _ = focus_points(cam, P, zf)
    B, _ = focus_points(tgt, P, zf)
    return ((A - B) ** 2).mean((1, 2))


def loss_dir(cam, tgt):
    P = _pix("corn", cam["t"].device)
    _, d = focus_points(cam, P)
    _, ds = focus_points(tgt, P)
    return ((d - ds) ** 2).mean((1, 2))


def loss_soft(cam, tgt, zf=1.0, sigma=1.0):
    return loss_hard(cam, tgt, zf) + sigma ** 2 / 2 * loss_dir(cam, tgt)


def loss_soft_sqrt(cam, tgt, zf=1.0, sigma=1.0):
    return torch.sqrt(loss_soft(cam, tgt, zf, sigma) + 1e-12)


def loss_soft_numeric(cam, tgt, zf, sigma, n=4001):
    """Trapezoid evaluation of the Gaussian-weighted ray loss (used only to verify the closed form of Soft)."""
    P = _pix("corn", cam["t"].device)
    s = torch.linspace(-8 * sigma, 8 * sigma, n, device=P.device)
    A, d = focus_points(cam, P, zf)
    B, ds = focus_points(tgt, P, zf)
    X = A[..., None] + s * d[..., None]
    Xs = B[..., None] + s * ds[..., None]
    w = torch.exp(-s ** 2 / sigma ** 2)
    num = torch.trapz(w * ((X - Xs) ** 2).sum(1), s, dim=-1)
    return (num / torch.trapz(w, s)).sum(1) / 12


def make_softgrid(rho, zf=1.0, sigma=4.0, n=5, trunc=2.0, inward=False, root=False):
    """Soft loss averaged over a Gaussian pixel grid around each image corner (rho = Gaussian width in pixels)."""
    off = torch.linspace(-trunc * rho, trunc * rho, n)
    ox, oy = torch.meshgrid(off, off, indexing="xy")
    ox, oy = ox.flatten(), oy.flatten()
    w = torch.exp(-(ox ** 2 + oy ** 2) / (2 * rho ** 2))
    w = w / w.sum()
    pix, wts = [], []
    for cu, cv in ((0, 0), (IMG - 1, 0), (IMG - 1, IMG - 1), (0, IMG - 1)):
        if inward:
            sx = 1.0 if cu < IMG / 2 else -1.0
            sy = 1.0 if cv < IMG / 2 else -1.0
            dx, dy = sx * ox.abs(), sy * oy.abs()
        else:
            dx, dy = ox, oy
        pix.append(torch.stack([cu + dx, cv + dy, torch.ones_like(dx)]))
        wts.append(w / 4)
    Pg, om = torch.cat(pix, 1), torch.cat(wts)                    # (3,4G), (4G,), sum(om) = 1

    def fn(cam, tgt):
        dev = cam["t"].device
        P, o = Pg.to(dev), om.to(dev)
        A, da = focus_points(cam, P, zf)
        B, db = focus_points(tgt, P, zf)
        hard = (((A - B) ** 2).mean(1) * o).sum(1)
        dirr = (((da - db) ** 2).mean(1) * o).sum(1)
        val = hard + sigma ** 2 / 2 * dirr
        return torch.sqrt(val + 1e-12) if root else val

    return fn


# ----------------------------------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------------------------------
def _qpair(cam, tgt):
    qc, qt = cam["q"], tgt["q"]
    s = ((qc * qt).sum(1, keepdim=True) >= 0).float() * 2 - 1      # sign-align (q and -q are the same rotation)
    return qc * s, qt


def _logf(cam, tgt):
    return (torch.log(cam["K"][:, 0, 0]) - torch.log(tgt["K"][:, 0, 0])).abs()


def loss_relpose_mse(cam, tgt, alpha=1.0):
    """The original RelPoseNet criterion: MSE(t) + alpha * MSE(q) on the raw (un-normalised) network output."""
    return ((cam["t"] - tgt["t"]) ** 2).mean(1) + alpha * ((cam["q_raw"] - tgt["q_raw"]) ** 2).mean(1)


def loss_pmse(cam, tgt):
    qc, qt = _qpair(cam, tgt)

    def v(c, q):
        return torch.cat([c["t"], q, c["K"][:, 0, 0, None] / IMG, c["K"][:, 1, 1, None] / IMG], 1)

    return ((v(cam, qc) - v(tgt, qt)) ** 2).mean(1)


def loss_posenet(cam, tgt, beta=100.0):
    qc, qt = _qpair(cam, tgt)
    return l2(cam["t"] - tgt["t"]) + beta * l2(qc - qt) + _logf(cam, tgt)


def loss_se3(cam, tgt):
    return l2(cam["t"] - tgt["t"]) + so3_angle(cam["R"].transpose(1, 2) @ tgt["R"]) + _logf(cam, tgt)


def _plucker(c, P):
    _, d = focus_points(c, P)
    m = torch.cross(c["t"][:, :, None].expand_as(d), d, dim=1)
    return d, m


def _ray_loss(cam, tgt, P):
    d, m = _plucker(cam, P)
    ds, ms = _plucker(tgt, P)
    return ((d - ds) ** 2).mean((1, 2)) + ((m - ms) ** 2).mean((1, 2))


def loss_ray(cam, tgt):
    return _ray_loss(cam, tgt, _pix("grid", cam["t"].device))


def loss_denseray(cam, tgt):
    return _ray_loss(cam, tgt, _pix("dense", cam["t"].device))


# ----------------------------------------------------------------------------------------------
# registry:  name -> (fn, group, needs_cam_frame)
# ----------------------------------------------------------------------------------------------
_F = r"([0-9]*\.?[0-9]+)"
FIXED = {
    "RelPose-MSE": "baseline-param",     # original criterion; works with labels in world or camera-1 frame
    "P-MSE": "baseline-param",
    "SE3-geo": "baseline-geo",
    "Ray-Plucker": "baseline-geo",
    "DenseRay": "baseline-geo",
    "Dir-only": "ablation",              # rotation-only term: cannot learn translation, listed for completeness
}


def get_loss(name, alpha=1.0):
    """Any member of the loss families can be requested by name, e.g. 'Hard(zf=1)', 'Soft(zf=2,s=4)',
    'SoftGrid(zf=1,s=4,r=48)', 'SoftSqrt(zf=1,s=4)', 'SoftGridSqrt(zf=1,s=4,r=48)', 'PoseNet(b=10)'."""
    if name == "RelPose-MSE":
        return (lambda c, t: loss_relpose_mse(c, t, alpha)), FIXED[name], False
    if name == "P-MSE":
        return loss_pmse, FIXED[name], False
    if name == "SE3-geo":
        return loss_se3, FIXED[name], True
    if name == "Ray-Plucker":
        return loss_ray, FIXED[name], True
    if name == "DenseRay":
        return loss_denseray, FIXED[name], True
    if name == "Dir-only":
        return loss_dir, FIXED[name], True
    m = re.fullmatch(rf"PoseNet\(b={_F}\)", name)
    if m:
        b = float(m[1])
        return (lambda c, t: loss_posenet(c, t, b)), "baseline-param", False
    m = re.fullmatch(rf"Hard\(zf={_F}\)", name)
    if m:
        zf = float(m[1])
        return (lambda c, t: loss_hard(c, t, zf)), "ours-hard", True
    m = re.fullmatch(rf"Soft\(zf={_F},s={_F}\)", name)
    if m:
        zf, sg = float(m[1]), float(m[2])
        return (lambda c, t: loss_soft(c, t, zf, sg)), "ours-soft", True
    m = re.fullmatch(rf"SoftSqrt\(zf={_F},s={_F}\)", name)
    if m:
        zf, sg = float(m[1]), float(m[2])
        return (lambda c, t: loss_soft_sqrt(c, t, zf, sg)), "ours-sqrt", True
    m = re.fullmatch(rf"SoftGrid\(zf={_F},s={_F},r={_F}\)", name)
    if m:
        return make_softgrid(float(m[3]), float(m[1]), float(m[2])), "ours-softgrid", True
    m = re.fullmatch(rf"SoftGridSqrt\(zf={_F},s={_F},r={_F}\)", name)
    if m:
        return make_softgrid(float(m[3]), float(m[1]), float(m[2]), root=True), "ours-sqrt", True
    raise KeyError(f"unknown loss '{name}'")


def symmetrized(fn):
    """Average of the loss with camera 1 as reference and with camera 2 as reference."""
    return lambda cam, tgt: 0.5 * (fn(cam, tgt) + fn(invert_cam(cam), invert_cam(tgt)))
