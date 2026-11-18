"""Tables, training curves and paired comparisons for the runs stored in <out_dir>/runs."""
import glob
import json
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from relposenet.labels import quat_wxyz_to_R, rel_labels
from relposenet.trainer import make_run_id, summarize

METRICS = ["rot_med", "rot_mean", "t_med", "t_mean", "tdir_med", "se3_mean", "acc_rot5", "acc_rot10", "acc_t010", "acc_t025"]


def load_results(out_dir):
    res = []
    for p in sorted(glob.glob(osp.join(out_dir, "runs", "*", "result.json"))):
        with open(p) as f:
            res.append(json.load(f))
    return res


def author_reference(cfg, data):
    """Metrics of the author's pre-computed predictions (MSE, world-frame t) on the SAME test pairs."""
    est = np.loadtxt(cfg.data.author_est)
    rot, terr, tdir = [], [], []
    for r in data["test"]:
        (R1, C1), (R2, C2) = data["poses"][r["p1"]], data["poses"][r["p2"]]
        lab = rel_labels(R1, C1, R2, C2)
        e = est[r["row_id"]]
        Rh, th, t = quat_wxyz_to_R(e[:4]), e[4:7], lab["t_world"]
        rot.append(np.degrees(np.arccos(np.clip((np.trace(Rh.T @ lab["R"]) - 1) / 2, -1, 1))))
        terr.append(np.linalg.norm(th - t))
        tdir.append(np.degrees(np.arccos(np.clip(th @ t / (np.linalg.norm(th) * np.linalg.norm(t)), -1, 1))))
    m = dict(rot=np.array(rot), t=np.array(terr), tdir=np.array(tdir), loss=np.zeros(len(rot)))
    return summarize(m)


def results_table(out_dir, split="test", which="last", reference=None):
    rows = []
    for r in load_results(out_dir):
        s = r["final"][which][split]
        rows.append(dict(loss=r["loss"], frame=r["frame"], seed=r["seed"], group=r["group"], iters=r["n_iters"],
                         it_s=r["it_per_s"], **{k: s[k] for k in METRICS}))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["loss", "frame", "group"])
    agg = g[METRICS].mean().join(g.size().rename("seeds")).join(g["iters"].first())
    if (df.groupby(["loss", "frame"]).size() > 1).any():
        std = g[["rot_med", "t_med", "tdir_med", "se3_mean"]].std().add_suffix("_std")
        agg = agg.join(std)
    agg = agg.reset_index().sort_values("se3_mean")
    if reference is not None and split == "test":
        ref = dict(loss="Author model (MSE, world-frame t)", frame="world", group="reference", seeds=0, iters=np.nan)
        ref.update({k: reference[k] for k in METRICS})
        agg = pd.concat([agg, pd.DataFrame([ref])], ignore_index=True)
    return agg


def _histories(out_dir):
    out = {}
    for p in sorted(glob.glob(osp.join(out_dir, "runs", "*", "history.csv"))):
        rid = osp.basename(osp.dirname(p))
        label = rid.rsplit("_s", 1)[0]
        out.setdefault(label, []).append(pd.read_csv(p))
    return out


def plot_training(out_dir, labels=None, smooth=5):
    """Curves averaged over seeds. Fig 1: comparable accuracy curves. Fig 2: every run's own loss (train/val)."""
    H = _histories(out_dir)
    if labels is not None:
        H = {k: v for k, v in H.items() if k in labels}

    def curve(hs, split, col):
        d = pd.concat([h[h.split == split][["step", col]] for h in hs]).groupby("step")[col].mean().dropna()
        return d.index.values, (d.rolling(smooth, min_periods=1, center=True).mean().values if smooth > 1 else d.values)

    panels = [("train", "rot", "train: rotation error (deg)"), ("train", "t", "train: translation error (m)"),
              ("val", "rot_med", "val: median rotation error (deg)"), ("val", "t_med", "val: median translation error (m)"),
              ("val", "tdir_med", "val: median translation-direction error (deg)"), ("val", "se3_mean", "val: mean (t + rot[rad])")]
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    for a, (sp, col, ttl) in zip(ax.ravel(), panels):
        for k, hs in H.items():
            x, y = curve(hs, sp, col)
            a.plot(x, y, label=k)
        a.set_title(ttl)
        a.set_xlabel("iteration")
        a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, "training_curves.png"), dpi=140)
    plt.show()

    n = len(H)
    cols = min(3, max(n, 1))
    fig, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(6.5 * cols, 4 * ((n + cols - 1) // cols)), squeeze=False)
    for a, (k, hs) in zip(ax.ravel(), H.items()):
        for sp, c in (("train", "tab:blue"), ("val", "tab:orange")):
            x, y = curve(hs, sp, "loss")
            a.plot(x, y, c=c, label=sp)
        a.set_yscale("log")
        a.set_title(k, fontsize=9)
        a.set_xlabel("iteration")
        a.set_ylabel("own training loss")
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    for a in ax.ravel()[n:]:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, "training_own_loss.png"), dpi=140)
    plt.show()


def paired_compare(out_dir, ref_spec, other_specs, seed, B=2000, rng_seed=0):
    """Bootstrap over test pairs of (other - reference) for rotation / translation / direction error (lower is better).
    NOTE: the CI reflects test-set sampling only, not training randomness (use several seeds for that)."""
    def load(spec):
        return np.load(osp.join(out_dir, "runs", make_run_id(spec, seed), "test_last.npz"))
    ref = load(ref_spec)
    rng = np.random.default_rng(rng_seed)
    rows = []
    for sp in other_specs:
        o = load(sp)
        n = len(ref["rot"])
        idx = rng.integers(0, n, (B, n))
        for k, nm in (("rot", "rot_err_deg"), ("t", "t_err_m"), ("tdir", "tdir_err_deg")):
            d = o[k] - ref[k]
            bm = d[idx].mean(1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            rows.append(dict(vs=ref_spec, run=sp, metric=nm, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
                             better="yes" if hi < 0 else ("worse" if lo > 0 else "-")))
    return pd.DataFrame(rows)
