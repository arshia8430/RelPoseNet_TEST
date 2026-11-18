"""Training / evaluation loop of RelPoseNet with a selectable loss (original MSE, baselines, Hard / Soft / SoftGrid)."""
import json
import os
import os.path as osp
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from relposenet.augmentations import get_augmentations
from relposenet.pose_dataset import (RelPoseDataset, load_poses, missing_images, preload_images, read_pair_file,
                                read_test_pairs, unique_paths)
from relposenet.focus_losses import get_loss, invert_cam, make_K, quat_to_R, so3_angle
from relposenet.labels import SCENES
from relposenet.model import RelPoseNet
from relposenet.utils import cycle, set_seed


# ----------------------------------------------------------------------------------------------
# configuration (mirrors configs/main.yaml of the original repository)
# ----------------------------------------------------------------------------------------------
def ns(d):
    return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def make_cfg(work_dir="/content/RelPoseNet_TEST-main", img_dir="/content/7scenes-light",
             out_dir="/content/drive/MyDrive/relpose_runs"):
    A = f"{work_dir}/assets/data"
    return ns(dict(
        data=dict(img_dir=img_dir, train_pairs=f"{A}/db_all_med_hard_train.txt", val_pairs=f"{A}/db_all_med_hard_valid.txt",
                  test_pairs=f"{A}/NN_7scenes.txt", author_est=f"{A}/est_rel_poses_flips_21_alpha1_dropout_no_grey.txt",
                  use_poses=True, ram_cache=False),
        model=dict(backbone_net="resnet34"),
        train=dict(bs=32, lr=1e-3, alpha=1.0, n_workers=os.cpu_count() or 2, n_train_iters=125000,
                   lrate_decay_steps=15000, lrate_decay_factor=0.5, amp=True),
        output=dict(out_dir=out_dir, validate_interval=1300, log_scalar_interval=200, val_subset=2000, test_n=10000),
        loss=dict(focal_px=525.0 * 256.0 / 480.0, symmetric=False),
        seed=1984))


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def prepare_data(cfg):
    d = cfg.data
    tr, va = read_pair_file(d.train_pairs, d.img_dir), read_pair_file(d.val_pairs, d.img_dir)
    te, te_poses = read_test_pairs(d.test_pairs, d.img_dir, n=cfg.output.test_n)
    miss = missing_images(tr + va + te)
    if miss:
        raise FileNotFoundError(f"{len(miss)} images missing, e.g. {miss[:3]}")
    poses = dict(te_poses)
    if d.use_poses:
        poses.update(load_poses(unique_paths(tr + va)))
    if d.ram_cache:
        preload_images(unique_paths(tr + va))
    print(f"train pairs {len(tr)} | val pairs {len(va)} | test pairs {len(te)} | poses {len(poses)}")
    return dict(train=tr, val=va, test=te, poses=poses)


def make_loaders(cfg, data):
    d, t = cfg.data, cfg.train
    aug_tr, aug_ev = get_augmentations()
    ds_tr = RelPoseDataset(data["train"], aug_tr, train=True, poses=data["poses"], use_poses=d.use_poses)
    ds_va = RelPoseDataset(data["val"], aug_ev, train=False, poses=data["poses"], use_poses=d.use_poses)
    ds_te = RelPoseDataset(data["test"], aug_ev, train=False, poses=data["poses"], use_poses=True)
    nw = t.n_workers
    kw_tr = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        kw_tr.update(persistent_workers=True, prefetch_factor=4)
    kw_ev = dict(num_workers=nw, pin_memory=True)
    sub = np.sort(np.random.default_rng(0).choice(len(ds_va), min(cfg.output.val_subset, len(ds_va)), replace=False))
    return dict(train=DataLoader(ds_tr, batch_size=t.bs, shuffle=True, drop_last=True, **kw_tr),
                val_sub=DataLoader(Subset(ds_va, sub), batch_size=64, shuffle=False, **kw_ev),
                val=DataLoader(ds_va, batch_size=64, shuffle=False, **kw_ev),
                test=DataLoader(ds_te, batch_size=64, shuffle=False, **kw_ev))


# ----------------------------------------------------------------------------------------------
# criterion and metrics
# ----------------------------------------------------------------------------------------------
class Criterion:
    """loss_name: see focus_losses.get_loss.  frame: 'cam1' (translation in camera-1 axes) or 'world' (original labels)."""

    def __init__(self, loss_name, frame, cfg):
        self.name, self.frame = loss_name, frame
        self.fn, self.group, self.needs_cam = get_loss(loss_name, alpha=cfg.train.alpha)
        if frame not in ("cam1", "world"):
            raise ValueError("frame must be 'cam1' or 'world'")
        if self.needs_cam and frame != "cam1":
            raise ValueError(f"{loss_name} compares camera geometry and needs the camera-1 frame (use '{loss_name}@cam1')")
        self.sym = bool(cfg.loss.symmetric) and self.needs_cam
        self.focal = float(cfg.loss.focal_px)

    def per_sample(self, q_raw, t_hat, b):
        B = q_raw.shape[0]
        K = make_K(B, q_raw.device, self.focal)
        qn = F.normalize(q_raw, dim=1, eps=1e-8)
        t_gt = b["t_cam"] if self.frame == "cam1" else b["t_world"]
        cam = dict(K=K, R=quat_to_R(qn), t=t_hat, q=qn, q_raw=q_raw)
        tgt = dict(K=K, R=b["R_rel"], t=t_gt, q=b["q_gt"], q_raw=b["q_gt"])
        if self.sym:
            return 0.5 * (self.fn(cam, tgt) + self.fn(invert_cam(cam), invert_cam(tgt)))
        return self.fn(cam, tgt)

    def __call__(self, q_raw, t_hat, b):
        return self.per_sample(q_raw, t_hat, b).mean()


@torch.no_grad()
def batch_metrics(q_raw, t_hat, b, frame):
    """rotation error (deg), translation error (m), translation-direction error (deg) -- in the run's own frame."""
    R_hat = quat_to_R(q_raw)
    rot = torch.rad2deg(so3_angle(R_hat.transpose(1, 2) @ b["R_rel"]))
    t = b["t_cam"] if frame == "cam1" else b["t_world"]
    terr = (t_hat - t).norm(dim=1)
    cos = (t_hat * t).sum(1) / (t_hat.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-8)
    tdir = torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)))
    return rot, terr, tdir


def _to_dev(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def _autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)


@torch.no_grad()
def evaluate(model, loader, crit, device, amp):
    was = model.training
    model.eval()
    acc = {k: [] for k in ("rot", "t", "tdir", "scene", "loss")}
    for b in loader:
        b = _to_dev(b, device)
        with _autocast(device, amp):
            q, t = model(b["img1"], b["img2"])
        q, t = q.float(), t.float()
        rot, terr, tdir = batch_metrics(q, t, b, crit.frame)
        for k, v in (("rot", rot), ("t", terr), ("tdir", tdir), ("scene", b["scene"]), ("loss", crit.per_sample(q, t, b))):
            acc[k].append(v.detach().cpu().numpy())
    model.train(was)
    return {k: np.concatenate(v) for k, v in acc.items()}


def summarize(m):
    se3 = m["t"] + np.deg2rad(m["rot"])
    return dict(n=int(len(m["rot"])),
                rot_mean=float(np.mean(m["rot"])), rot_med=float(np.median(m["rot"])), rot_p90=float(np.percentile(m["rot"], 90)),
                t_mean=float(np.mean(m["t"])), t_med=float(np.median(m["t"])), t_p90=float(np.percentile(m["t"], 90)),
                tdir_mean=float(np.mean(m["tdir"])), tdir_med=float(np.median(m["tdir"])),
                se3_mean=float(np.mean(se3)),
                acc_rot5=float(np.mean(m["rot"] < 5)), acc_rot10=float(np.mean(m["rot"] < 10)),
                acc_t010=float(np.mean(m["t"] < 0.10)), acc_t025=float(np.mean(m["t"] < 0.25)),
                loss=float(np.mean(m["loss"])))


def per_scene(m):
    out = {}
    for sid, name in enumerate(SCENES):
        k = m["scene"] == sid
        if k.any():
            out[name] = dict(n=int(k.sum()), rot_med=float(np.median(m["rot"][k])),
                             t_med=float(np.median(m["t"][k])), tdir_med=float(np.median(m["tdir"][k])))
    return out


# ----------------------------------------------------------------------------------------------
# run management
# ----------------------------------------------------------------------------------------------
def parse_spec(spec):
    """'Hard(zf=1)@cam1' -> ('Hard(zf=1)', 'cam1');  default frame is cam1."""
    name, _, frame = spec.partition("@")
    return name, (frame or "cam1")


def make_run_id(spec, seed):
    name, frame = parse_spec(spec)
    return re.sub(r"[^A-Za-z0-9_.@\-]+", "_", f"{name}@{frame}_s{seed}")


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def benchmark_speed(cfg, data, spec="Hard(zf=1)", iters=60, warmup=10):
    """Measure training it/s (data loading + GPU) without writing anything; prints the time one full run would take."""
    loss_name, frame = parse_spec(spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.train.amp) and device.type == "cuda"
    set_seed(cfg.seed)
    crit = Criterion(loss_name, frame, cfg)
    loaders = make_loaders(cfg, data)
    model = RelPoseNet(cfg.model).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = _make_scaler(amp)
    it = iter(cycle(loaders["train"]))
    for i in range(warmup + iters):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        b = _to_dev(next(it), device)
        with _autocast(device, amp):
            q, t = model(b["img1"], b["img2"])
        loss = crit(q.float(), t.float(), b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ips = iters / (time.time() - t0)
    N = cfg.train.n_train_iters
    print(f"{ips:.2f} it/s  ->  {N} iterations = {N / ips / 3600:.1f} h per run  (workers={cfg.train.n_workers}, amp={amp})")
    return ips


def run_training(cfg, spec, data, seed=None, resume=True, max_iters=None):
    loss_name, frame = parse_spec(spec)
    seed = cfg.seed if seed is None else seed
    rid = make_run_id(spec, seed)
    out = osp.join(cfg.output.out_dir, "runs", rid)
    os.makedirs(out, exist_ok=True)
    res_path = osp.join(out, "result.json")
    if osp.exists(res_path):
        print(f"[skip] {rid}: finished")
        with open(res_path) as f:
            return json.load(f)
    if frame == "cam1" and not cfg.data.use_poses:
        raise ValueError("frame 'cam1' needs use_poses=True (absolute 7-Scenes poses)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.train.amp) and device.type == "cuda"
    N = int(max_iters or cfg.train.n_train_iters)
    set_seed(seed)
    crit = Criterion(loss_name, frame, cfg)
    loaders = make_loaders(cfg, data)
    model = RelPoseNet(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.train.lrate_decay_steps, gamma=cfg.train.lrate_decay_factor)
    scaler = _make_scaler(amp)
    hist, step0, best_val, best_step = [], 0, float("inf"), 0
    ck = osp.join(out, "last.pth")
    if resume and osp.exists(ck):
        s = _torch_load(ck, device)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"])
        hist, step0, best_val, best_step = s["hist"], s["step"], s["best_val"], s["best_step"]
        print(f"[resume] {rid} from step {step0}")

    li, vi = cfg.output.log_scalar_interval, cfg.output.validate_interval
    it = iter(cycle(loaders["train"]))
    model.train()
    acc, cnt = torch.zeros(3, device=device), 0
    t_start = t_log = time.time()
    for step in range(step0 + 1, N + 1):
        b = _to_dev(next(it), device)
        with _autocast(device, amp):
            q_raw, t_hat = model(b["img1"], b["img2"])
        q_raw, t_hat = q_raw.float(), t_hat.float()
        loss = crit(q_raw, t_hat, b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        rot, terr, _ = batch_metrics(q_raw.detach(), t_hat.detach(), b, frame)
        acc += torch.stack([loss.detach(), rot.mean(), terr.mean()])
        cnt += 1

        if step % li == 0:
            l_, r_, t_ = (acc / cnt).tolist()
            acc.zero_()
            now = time.time()
            ips = cnt / (now - t_log)
            cnt, t_log = 0, now
            if not np.isfinite(l_):
                raise RuntimeError(f"{rid}: loss is not finite at step {step} (diverged)")
            hist.append(dict(step=step, split="train", loss=l_, rot=r_, t=t_, lr=sched.get_last_lr()[0], it_s=ips))
            print(f"[{rid}] step {step}/{N} | loss {l_:.4f} | rot {r_:.1f} deg | t {t_:.3f} m | {ips:.1f} it/s "
                  f"| ETA {(N - step) / ips / 60:.0f} min", flush=True)

        if step % vi == 0 or step == N:
            s = summarize(evaluate(model, loaders["val_sub"], crit, device, amp))
            hist.append(dict(step=step, split="val", **s))
            if s["se3_mean"] < best_val:
                best_val, best_step = s["se3_mean"], step
                torch.save(model.state_dict(), osp.join(out, "best.pth"))
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), hist=hist, step=step, best_val=best_val, best_step=best_step), ck)
            pd.DataFrame(hist).to_csv(osp.join(out, "history.csv"), index=False)
            print(f"[{rid}] VAL step {step}: rot_med {s['rot_med']:.2f} deg | t_med {s['t_med']:.3f} m | "
                  f"tdir_med {s['tdir_med']:.1f} deg | se3 {s['se3_mean']:.3f}", flush=True)

    # ---- final evaluation: last weights (and best-val weights) on the full val set and on the test pairs ----
    final = {}
    row_id = np.array([r["row_id"] for r in data["test"]])
    mv, mt = evaluate(model, loaders["val"], crit, device, amp), evaluate(model, loaders["test"], crit, device, amp)
    final["last"] = dict(val=summarize(mv), test=summarize(mt), test_per_scene=per_scene(mt))
    np.savez(osp.join(out, "test_last.npz"), row_id=row_id, **mt)
    if osp.exists(osp.join(out, "best.pth")) and best_step != N:
        model.load_state_dict(_torch_load(osp.join(out, "best.pth"), device))
        mv, mt = evaluate(model, loaders["val"], crit, device, amp), evaluate(model, loaders["test"], crit, device, amp)
        final["best"] = dict(val=summarize(mv), test=summarize(mt), test_per_scene=per_scene(mt))
    else:
        final["best"] = final["last"]
    its = [h["it_s"] for h in hist if h["split"] == "train" and "it_s" in h]
    result = dict(run_id=rid, loss=loss_name, frame=frame, seed=seed, group=crit.group, n_iters=N, best_step=best_step,
                  it_per_s=float(np.median(its)) if its else None, session_min=(time.time() - t_start) / 60,
                  settings=dict(lr=cfg.train.lr, bs=cfg.train.bs, focal_px=crit.focal, symmetric=crit.sym, amp=amp,
                                decay_steps=cfg.train.lrate_decay_steps, decay_factor=cfg.train.lrate_decay_factor),
                  final=final)
    with open(res_path, "w") as f:
        json.dump(result, f, indent=1)
    pd.DataFrame(hist).to_csv(osp.join(out, "history.csv"), index=False)
    t = final["last"]["test"]
    print(f"[{rid}] DONE | test: rot_med {t['rot_med']:.2f} deg | t_med {t['t_med']:.3f} m | tdir_med {t['tdir_med']:.1f} deg")
    return result
