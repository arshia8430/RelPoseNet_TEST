"""Download and prepare 7-Scenes, validate the pair lists, then train losses.

Example:
    python prepare_and_run_losses.py --data-dir /content/drive/MyDrive/7scenes \
        --output-dir /content/drive/MyDrive/RelPoseNet-output \
        --accept-noncommercial-license
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Optional

from PIL import Image


SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
LOSSES = (
    "mse", "hard", "soft", "softgrid", "dir", "pmse", "posenet10",
    "posenet100", "posenet", "se3", "ray", "denseray", "soft_sqrt",
    "softgrid_sqrt",
)
DOWNLOAD_BASE = (
    "https://download.microsoft.com/download/2/8/5/"
    "28564B23-0828-408F-8631-23B1EFF1DAC8"
)


def _download_archive(scene: str, download_dir: Path) -> Path:
    archive = download_dir / f"{scene}.zip"
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as zf:
                if zf.testzip() is None:
                    return archive
        except zipfile.BadZipFile:
            pass

    partial = archive.with_suffix(".zip.part")
    url = f"{DOWNLOAD_BASE}/{scene}.zip"
    print(f"Downloading {scene} from Microsoft...")
    try:
        urllib.request.urlretrieve(url, partial)
        with zipfile.ZipFile(partial) as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Corrupt member {bad_member!r} in {partial}")
        partial.replace(archive)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return archive


def _sequence_relative_path(member_name: str) -> Optional[Path]:
    parts = PurePosixPath(member_name.replace("\\", "/")).parts
    if ".." in parts:
        raise ValueError(f"Unsafe path in dataset archive: {member_name}")
    sequence_index = next(
        (i for i, part in enumerate(parts) if part.lower().startswith("seq-")),
        None,
    )
    if sequence_index is None or sequence_index == len(parts) - 1:
        return None
    return Path(*parts[sequence_index:])


def _prepare_scene(archive: Path, scene: str, image_root: Path, pose_root: Path) -> None:
    image_count = 0
    pose_count = 0
    resampling = getattr(Image, "Resampling", Image)

    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            relative = _sequence_relative_path(info.filename)
            if relative is None:
                continue

            name = relative.name.lower()
            if name.endswith(".color.png"):
                destination = image_root / scene / relative
                if destination.is_file():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as stream, Image.open(stream) as image:
                    width, height = image.size
                    scale = 256.0 / min(width, height)
                    size = (round(width * scale), round(height * scale))
                    image.convert("RGB").resize(
                        size, resampling.LANCZOS
                    ).save(destination, format="PNG")
                image_count += 1
            elif name.endswith(".pose.txt"):
                destination = pose_root / scene / relative
                if destination.is_file():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                pose_count += 1

    print(f"Prepared {scene}: {image_count} resized images, {pose_count} poses.")


def _verify_pair_file(
    pair_file: Path, image_root: Path, pose_root: Path
) -> None:
    missing = []
    with pair_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 10:
                raise ValueError(
                    f"Malformed {pair_file} line {line_number}: expected at least 10 fields"
                )
            scene_id = int(fields[2])
            if scene_id < 0 or scene_id >= len(SCENES):
                raise ValueError(
                    f"Unknown scene id {scene_id} in {pair_file} line {line_number}"
                )
            scene = SCENES[scene_id]
            for entry in fields[:2]:
                relative = Path(entry.replace("\\", "/").lstrip("/"))
                image_path = image_root / scene / relative
                pose_path = pose_root / scene / relative.with_name(
                    relative.name.replace(".color.png", ".pose.txt")
                )
                if not image_path.is_file():
                    missing.append(str(image_path))
                if not pose_path.is_file():
                    missing.append(str(pose_path))

    if missing:
        sample = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} image/pose files referenced by {pair_file} are missing. "
            f"First missing paths:\n{sample}"
        )


def _run(args, project_dir: Path, image_root: Path, pose_root: Path) -> None:
    run_id = datetime.now().strftime("all_losses_%Y%m%d_%H%M%S")
    for loss_name in args.losses:
        command = [
            sys.executable,
            "main.py",
            f"data_params.work_dir={project_dir}",
            f"data_params.img_dir={image_root}",
            f"data_params.pose_dir={pose_root}",
            f"output_params.out_dir={args.output_dir}",
            f"train_params.loss_name={loss_name}",
            f"train_params.loss_zf={args.zf}",
            f"train_params.loss_sigma={args.sigma}",
            f"train_params.loss_rho={args.rho}",
            f"train_params.loss_beta={args.beta}",
            f"train_params.run_id={run_id}",
            "model_params.resume_snapshot=null",
            f"seed={args.seed}",
        ]
        if args.iterations is not None:
            command.append(f"train_params.n_train_iters={args.iterations}")
        if args.batch_size is not None:
            command.append(f"train_params.bs={args.batch_size}")
        if args.workers is not None:
            command.append(f"train_params.n_workers={args.workers}")

        print(f"\n=== Training {loss_name} ===", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        subprocess.run(command, cwd=project_dir, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Persistent directory for prepared data and downloaded archives")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Persistent directory for checkpoints, metrics, and plots")
    parser.add_argument("--accept-noncommercial-license", action="store_true",
                        help="Confirm acceptance of Microsoft's 7-Scenes non-commercial license")
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES),
                        help="Losses to run; defaults to all supported losses")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override train_params.n_train_iters for each loss")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1984)
    parser.add_argument("--zf", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--rho", type=float, default=24.0)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Download, resize and validate data without training")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare data and print training commands without launching them")
    args = parser.parse_args()

    if not args.accept_noncommercial_license:
        parser.error(
            "Read Microsoft's dataset license and pass --accept-noncommercial-license "
            "only if your use complies with it."
        )
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")

    project_dir = args.project_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (project_dir / "main.py").is_file():
        parser.error(f"--project-dir does not contain main.py: {project_dir}")
    for filename in (
        "assets/data/db_all_med_hard_train.txt",
        "assets/data/db_all_med_hard_valid.txt",
    ):
        if not (project_dir / filename).is_file():
            parser.error(f"Required pair list not found: {project_dir / filename}")

    download_dir = data_dir / "downloads"
    image_root = data_dir / "images"
    pose_root = data_dir / "poses"
    download_dir.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        archive = _download_archive(scene, download_dir)
        _prepare_scene(archive, scene, image_root, pose_root)

    for filename in (
        "assets/data/db_all_med_hard_train.txt",
        "assets/data/db_all_med_hard_valid.txt",
    ):
        _verify_pair_file(project_dir / filename, image_root, pose_root)
    print("All image and pose references in the train/validation pair files exist.")

    if args.prepare_only:
        print(f"Prepared images: {image_root}\nPrepared poses: {pose_root}")
        return

    args.output_dir = output_dir
    _run(args, project_dir, image_root, pose_root)
    if not args.dry_run:
        print(f"\nTraining outputs are under: {output_dir / 'loss_comparison'}")


if __name__ == "__main__":
    main()
