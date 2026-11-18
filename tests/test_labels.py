"""numpy-only tests of the relative-pose label logic (no torch needed):  python tests/test_labels.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from relposenet.labels import R_to_quat_wxyz, check_label_convention, quat_wxyz_to_R, rel_labels  # noqa: E402


def rand_rot(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


rng = np.random.default_rng(0)
err = 0.0
for _ in range(2000):
    R = rand_rot(rng)
    q = R_to_quat_wxyz(R)
    assert q[0] >= 0
    err = max(err, np.abs(quat_wxyz_to_R(q) - R).max())
print("quaternion <-> matrix round trip, max error:", err)
assert err < 1e-12

poses, recs = {}, []
for i in range(300):
    R1, R2 = rand_rot(rng), rand_rot(rng)
    C1, C2 = rng.normal(size=3), rng.normal(size=3)
    poses[f"a{i}"], poses[f"b{i}"] = (R1, C1), (R2, C2)
    lab = rel_labels(R1, C1, R2, C2)       # file labels simulated under the convention verified on the real data
    recs.append(dict(p1=f"a{i}", p2=f"b{i}", t_world=np.round(lab["t_world"], 6), q=np.round(lab["q"], 6)))
et, eq = check_label_convention(recs, poses)
assert min(et, key=et.get) == "world(C2-C1)" and min(eq, key=eq.get) == "R1^T R2"
print("convention detector picks:", min(et, key=et.get), "/", min(eq, key=eq.get))

# swapping the two images: world  t' = -t, q' = conj(q);  camera-1 frame  t' = -R^T t, R' = R^T
l12 = rel_labels(*poses["a0"], *poses["b0"])
l21 = rel_labels(*poses["b0"], *poses["a0"])
assert np.abs(l21["t_world"] + l12["t_world"]).max() < 1e-12
assert np.abs(l21["q"] - l12["q"] * np.array([1, -1, -1, -1])).max() < 1e-12
assert np.abs(l21["t_cam"] + l12["R"].T @ l12["t_cam"]).max() < 1e-12
assert np.abs(l21["R"] - l12["R"].T).max() < 1e-12
print("swap consistency OK")
print("ALL LABEL TESTS PASSED")
