# RelPoseNet with selectable losses (Hard / Soft / SoftGrid focus-point losses)

Quick start: open `run_colab.ipynb` in Colab (GPU runtime) and run the cells in order.

## What was added / changed
| file | change |
|---|---|
| `relposenet/focus_losses.py` | NEW. Hard / Soft / SoftGrid / SoftSqrt focus-point losses + baselines (RelPose-MSE = original criterion, P-MSE, PoseNet, SE3-geo, Ray-Plucker, DenseRay). Any family member can be requested by name, e.g. `Soft(zf=2,s=4)`. |
| `relposenet/labels.py` | NEW. numpy helpers: 7-Scenes poses, relative-pose labels in the camera-1 and world frames, label-convention check. |
| `relposenet/pose_dataset.py` | NEW. Dataset that derives labels from absolute poses (consistent after the random image swap); test pairs from `NN_7scenes.txt`. |
| `relposenet/trainer.py` | NEW. Training loop with selectable loss, AMP, checkpoint/resume, validation, final val + test evaluation. |
| `relposenet/report.py` | NEW. Tables, training curves, author-model reference, paired bootstrap comparison. |
| `relposenet/model.py` | only the pretrained-weights call was updated to the new torchvision API. |
| `tests/test_labels.py`, `tests/smoke_test_losses.py` | NEW. numpy-only and CPU torch tests. |
| `pipeline.py`, `criterion.py`, `dataset.py`, `main.py`, `experiments/` | unchanged (not used by the new runner). |

## Data facts (verified on the files in `assets/data`)
* Pair file line: `img1 img2 scene t(3) q(4)`; `q = quat(R1^T R2)` (w,x,y,z); `t = C2 - C1` in WORLD coordinates (not camera axes).
* Geometric losses need the translation in camera-1 axes `t_cam = R1^T (C2 - C1)` -> needs the absolute rotation of image 1,
  i.e. the `frame-XXXXXX.pose.txt` files of 7-Scenes (camera-to-world). Without them only `RelPose-MSE@world` can run.
* Intrinsics are not in the data; one shared K is used for both cameras: f = 525 px at 640x480 (Kinect RGB default, NOT calibrated by
  the dataset authors) scaled to the resized image; principal point at the centre of the 224x224 crop.
* `db_all_med_hard_valid.txt` is NOT held out: 99% of its frames also occur in the training pairs and all 28 sequences are shared.
  Final accuracy is therefore reported on the test pairs of `NN_7scenes.txt` (query frames from test sequences, db frames from
  training sequences; 323 cross-scene rows with flag 0 are removed). Test pairs are easier than train pairs
  (median |t| 0.29 m / 13 deg vs 0.97 m / 27.6 deg).
* Reference (author's pre-computed predictions on all 85k pairs): median rotation error 9.4 deg, median translation-direction error 33.8 deg;
  identity rotation alone gives 13.0 deg.

## Conventions
* `spec = "<loss>@<frame>"`; frame `cam1` (translation in camera-1 axes) or `world` (original labels).
  Geometric losses require `cam1`. Compare losses only inside the same frame.
* Swapping the two images re-computes the labels exactly (world: t -> -t, q -> conj(q); cam1: R -> R^T, t -> -R^T t).
* `Reproj-px` is not available here (needs 3D scene points); `Dir-only` cannot learn translation.
