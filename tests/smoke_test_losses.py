"""CPU smoke tests of the losses (needs torch, no data):  python tests/smoke_test_losses.py"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from relposenet.focus_losses import (get_loss, invert_cam, loss_soft, loss_soft_numeric, make_K, make_softgrid,  # noqa: E402
                                     quat_to_R)
from relposenet.labels import R_to_quat_wxyz  # noqa: E402

torch.manual_seed(0)
B = 64


def mk(grad=False):
    q_raw = torch.randn(B, 4, requires_grad=grad)
    t = torch.randn(B, 3, requires_grad=grad)
    qn = F.normalize(q_raw, dim=1)
    return q_raw, t, dict(K=make_K(B, "cpu", 280.0), R=quat_to_R(qn), t=t, q=qn, q_raw=q_raw)


NAMES = ["RelPose-MSE", "P-MSE", "PoseNet(b=10)", "SE3-geo", "Ray-Plucker", "DenseRay", "Dir-only",
         "Hard(zf=1)", "Soft(zf=1,s=4)", "SoftGrid(zf=1,s=4,r=48)", "SoftSqrt(zf=1,s=4)", "SoftGridSqrt(zf=1,s=4,r=48)"]
_, _, cam0 = mk()
tgt0 = {k: v.clone() for k, v in cam0.items()}
for n in NAMES:
    fn, grp, _ = get_loss(n)
    s = fn(cam0, tgt0)
    assert s.shape == (B,) and s.abs().max() < 1e-3, (n, float(s.abs().max()))      # zero loss for identical cameras
    q_raw, t, cam = mk(grad=True)
    _, _, tgt = mk()
    fn(cam, tgt).mean().backward()
    assert torch.isfinite(q_raw.grad).all() and q_raw.grad.abs().sum() > 0, n      # gradients reach the rotation output
    if n != "Dir-only":                                                             # Dir-only has no translation term
        assert torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0, n
    print(f"ok  {n:30s} group={grp}")

_, _, c = mk()
_, _, g = mk()
for zf, sg in [(1, 1.0), (2, 4.0), (0.5, 0.25)]:
    a, b = loss_soft(c, g, zf, sg), loss_soft_numeric(c, g, zf, sg)
    rel = float(((a - b).abs() / b.abs().clamp(min=1e-9)).max())
    print(f"Soft closed form vs numeric (zf={zf}, s={sg}): max rel err {rel:.2e}")
    assert rel < 1e-3
d = float((make_softgrid(1e-6, 1.0, 4.0)(c, g) - loss_soft(c, g, 1.0, 4.0)).abs().max())
print("SoftGrid(rho->0) vs Soft:", d)
assert d < 1e-4
c2 = invert_cam(invert_cam(c))
assert float((c2["R"] - c["R"]).abs().max()) < 1e-5 and float((c2["t"] - c["t"]).abs().max()) < 1e-5
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
Q = Q if np.linalg.det(Q) > 0 else Q * np.array([-1, 1, 1])
Rr = torch.tensor(Q, dtype=torch.float32)
e = float((quat_to_R(torch.tensor(R_to_quat_wxyz(Q), dtype=torch.float32)[None])[0] - Rr).abs().max())
print("R -> q -> R:", e)
assert e < 1e-5
print("ALL LOSS SMOKE TESTS PASSED")
