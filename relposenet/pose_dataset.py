"""Datasets for relative-pose training. Labels are derived from the absolute 7-Scenes poses so that the same pair can be
expressed in the camera-1 frame (needed by the geometric losses) or in the world frame (original RelPoseNet labels)."""
import os.path as osp
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from relposenet.labels import SCENES, quat_wxyz_to_R, read_pose, rel_labels

_IMG_CACHE = {}     # path -> uint8 array (optional RAM cache, shared by forked dataloader workers)


def read_pair_file(txt, img_dir):
    """db_all_med_hard_{train,valid}.txt:  img1 img2 scene  t(3)  q(4)   (t, q = labels stored in the file)."""
    recs = []
    with open(txt) as f:
        for line in f:
            c = line.split()
            sid = int(c[2])
            recs.append(dict(p1=osp.join(img_dir, SCENES[sid], c[0][1:]), p2=osp.join(img_dir, SCENES[sid], c[1][1:]),
                             scene=sid, t_world=np.array(c[3:6], np.float64), q=np.array(c[6:10], np.float64)))
    return recs


def read_test_pairs(txt, img_dir, n=None, seed=0):
    """NN_7scenes.txt: query db scene_q scene_db flag  t_q q_q t_db q_db  (absolute poses, camera-to-world).
    Pairs are (img1 = db image, img2 = query image) exactly like the original test pipeline.
    Rows with flag 0 / different scenes (cross-scene retrieval failures) are dropped."""
    recs, poses = [], {}
    with open(txt) as f:
        for k, line in enumerate(f):
            c = line.split()
            sq, sdb, flag = int(c[2]), int(c[3]), int(c[4])
            if flag != 1 or sq != sdb:
                continue
            v = np.array(c[5:19], np.float64)
            p_q, p_db = osp.join(img_dir, SCENES[sq], c[0][1:]), osp.join(img_dir, SCENES[sdb], c[1][1:])
            poses[p_q] = (quat_wxyz_to_R(v[3:7]), v[0:3])
            poses[p_db] = (quat_wxyz_to_R(v[10:14]), v[7:10])
            recs.append(dict(p1=p_db, p2=p_q, scene=sq, row_id=k))
    if n is not None and n < len(recs):
        keep = np.sort(np.random.default_rng(seed).choice(len(recs), n, replace=False))
        recs = [recs[i] for i in keep]
    return recs, poses


def unique_paths(recs):
    return sorted({p for r in recs for p in (r["p1"], r["p2"])})


def load_poses(img_paths, n_threads=16):
    """Absolute poses (R, C) for every colour image path (reads the neighbouring frame-XXXXXX.pose.txt)."""
    def one(p):
        try:
            return p, read_pose(p.replace(".color.png", ".pose.txt"))
        except OSError:
            return p, None
    with ThreadPoolExecutor(n_threads) as ex:
        res = list(ex.map(one, img_paths))
    missing = [p for p, v in res if v is None]
    if missing:
        raise FileNotFoundError(f"{len(missing)} pose files missing, e.g. {missing[0].replace('.color.png', '.pose.txt')}")
    return dict(res)


def missing_images(recs):
    return [p for p in unique_paths(recs) if not osp.exists(p)]


def preload_images(paths, n_threads=16):
    def one(p):
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"))
    todo = [p for p in paths if p not in _IMG_CACHE]
    with ThreadPoolExecutor(n_threads) as ex:
        for p, a in zip(todo, ex.map(one, todo)):
            _IMG_CACHE[p] = a


class RelPoseDataset(torch.utils.data.Dataset):
    """Returns img1, img2 and the relative pose of camera 2 w.r.t. camera 1:
       q_gt (w,x,y,z), R_rel, t_cam (camera-1 frame), t_world (world frame) and the scene id.
       train=True: random swap of the two images (labels are recomputed consistently after the swap)."""

    def __init__(self, recs, transforms=None, train=False, poses=None, use_poses=True):
        self.recs, self.transforms, self.train = recs, transforms, train
        self.poses, self.use_poses = poses, use_poses
        if use_poses and poses is None:
            raise ValueError("use_poses=True needs the absolute poses")

    def __len__(self):
        return len(self.recs)

    @staticmethod
    def _img(p):
        if p in _IMG_CACHE:
            return Image.fromarray(_IMG_CACHE[p])
        return Image.open(p).convert("RGB")

    def __getitem__(self, i):
        r = self.recs[i]
        p1, p2 = r["p1"], r["p2"]
        flip = self.train and random.random() > 0.5
        if flip:
            p1, p2 = p2, p1
        im1, im2 = self._img(p1), self._img(p2)
        if self.transforms is not None:
            im1, im2 = self.transforms(im1), self.transforms(im2)
        if self.use_poses:
            (R1, C1), (R2, C2) = self.poses[p1], self.poses[p2]
            lab = rel_labels(R1, C1, R2, C2)
        else:                                            # original labels from the pair file (world-frame t only)
            t, q = r["t_world"], r["q"]
            if flip:
                t, q = -t, q * np.array([1.0, -1.0, -1.0, -1.0])
            lab = dict(R=quat_wxyz_to_R(q), q=q, t_world=t, t_cam=np.full(3, np.nan))

        def f(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32)

        return dict(img1=im1, img2=im2, q_gt=f(lab["q"]), t_world=f(lab["t_world"]), t_cam=f(lab["t_cam"]),
                    R_rel=f(lab["R"]), scene=int(r["scene"]))
