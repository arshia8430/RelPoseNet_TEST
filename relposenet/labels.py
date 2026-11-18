"""Pure-numpy helpers: 7-Scenes poses, relative-pose labels and a check of the label convention."""
import numpy as np

SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']


def read_pose(path):
    """7-Scenes frame-XXXXXX.pose.txt = 4x4 camera-to-world matrix -> (R (3,3), C (3,)). C = camera centre in world."""
    M = np.loadtxt(path, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"unexpected pose file shape {M.shape}: {path}")
    return M[:3, :3].copy(), M[:3, 3].copy()


def quat_wxyz_to_R(q):
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    """Rotation matrix -> unit quaternion (w,x,y,z) with w >= 0."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rel_labels(R1, C1, R2, C2):
    """Pose of camera 2 relative to camera 1 from absolute camera-to-world poses.
    R      = R1^T R2          rotation of camera 2 expressed in camera-1 axes
    t_cam  = R1^T (C2 - C1)   position of camera 2 in camera-1 coordinates   (frame used by the geometric losses)
    t_world= C2 - C1          centre difference in WORLD coordinates          (frame used by the original RelPoseNet labels)
    """
    R = R1.T @ R2
    d = C2 - C1
    return dict(R=R, q=R_to_quat_wxyz(R), t_world=d, t_cam=R1.T @ d)


def check_label_convention(recs, poses, n=2000, seed=0):
    """Compare the labels stored in the pair file with several hypotheses computed from the absolute poses.
    recs: dicts with p1, p2, t_world (file translation), q (file quaternion).  Returns {hypothesis: mean abs error}."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(recs), min(n, len(recs)), replace=False)
    et, eq = {}, {}
    for i in idx:
        r = recs[i]
        (R1, C1), (R2, C2) = poses[r["p1"]], poses[r["p2"]]
        d = C2 - C1
        for sgn, sn in ((1, "C2-C1"), (-1, "C1-C2")):
            for nm, M in (("world", np.eye(3)), ("R1^T", R1.T), ("R2^T", R2.T)):
                et.setdefault(f"{nm}({sn})", []).append(np.abs(r["t_world"] - sgn * (M @ d)).max())
        for nm, R in (("R1^T R2", R1.T @ R2), ("R2^T R1", R2.T @ R1), ("R1 R2^T", R1 @ R2.T), ("R2 R1^T", R2 @ R1.T)):
            q = R_to_quat_wxyz(R)
            eq.setdefault(nm, []).append(min(np.abs(r["q"] - q).max(), np.abs(r["q"] + q).max()))
    return ({k: float(np.mean(v)) for k, v in et.items()}, {k: float(np.mean(v)) for k, v in eq.items()})
