"""Paired image transforms that keep the intrinsics consistent with each crop.

Unlike applying RandomCrop independently to img1 and img2, this module samples
one crop window for the pair and updates K for that window. This is required by
pixel/ray-based geometric losses.
"""

from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class PairedImageTransform:
    def __init__(
        self,
        train: bool,
        fx: float = 532.57,
        fy: float = 531.54,
        cx: float = 320.0,
        cy: float = 240.0,
        raw_width: int = 640,
        raw_height: int = 480,
        output_size: int = 224,
        expected_short_side: int = 256,
    ):
        self.train = bool(train)
        self.output_size = int(output_size)
        self.expected_short_side = int(expected_short_side)
        self.raw_width = int(raw_width)
        self.raw_height = int(raw_height)
        self.K_raw = torch.tensor(
            [[float(fx), 0.0, float(cx)],
             [0.0, float(fy), float(cy)],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _scaled_intrinsics(self, width: int, height: int) -> torch.Tensor:
        if min(width, height) != self.expected_short_side:
            raise RuntimeError(
                f"Expected images resized so their short side is "
                f"{self.expected_short_side}px before cropping, got {width}x{height}. "
                "Follow the RelPoseNet preprocessing step (short side 256, "
                "aspect ratio preserved)."
            )
        sx = width / float(self.raw_width)
        sy = height / float(self.raw_height)
        if abs(sx - sy) > 0.01 * max(sx, sy):
            raise RuntimeError(
                f"Image resize is not aspect-ratio preserving relative to configured "
                f"raw size {self.raw_width}x{self.raw_height}: got scales sx={sx:.5f}, sy={sy:.5f}."
            )
        K = self.K_raw.clone()
        K[0, 0] *= sx
        K[1, 1] *= sy
        # Pixel-centre convention for resizing: c' = s * (c + 0.5) - 0.5.
        K[0, 2] = sx * (self.K_raw[0, 2] + 0.5) - 0.5
        K[1, 2] = sy * (self.K_raw[1, 2] + 0.5) - 0.5
        return K

    def __call__(self, img1, img2):
        if img1.size != img2.size:
            raise RuntimeError(
                f"Paired images must have equal dimensions, got {img1.size} and {img2.size}."
            )

        width, height = img1.size
        if width < self.output_size or height < self.output_size:
            raise RuntimeError(
                f"Images ({width}x{height}) are smaller than crop {self.output_size}x{self.output_size}."
            )

        K = self._scaled_intrinsics(width, height)

        if self.train:
            top, left, crop_h, crop_w = transforms.RandomCrop.get_params(
                img1, output_size=(self.output_size, self.output_size)
            )
        else:
            crop_h = crop_w = self.output_size
            top = (height - crop_h) // 2
            left = (width - crop_w) // 2

        # Apply exactly the same crop coordinates to both members of the pair.
        img1 = TF.crop(img1, top, left, crop_h, crop_w)
        img2 = TF.crop(img2, top, left, crop_h, crop_w)

        # Cropping changes the principal point; focal lengths remain unchanged.
        K = K.clone()
        K[0, 2] -= float(left)
        K[1, 2] -= float(top)

        # Photometric augmentation does not alter the camera geometry.
        if self.train:
            img1 = self.color_jitter(img1)
            img2 = self.color_jitter(img2)

        img1 = self.normalize(self.to_tensor(img1))
        img2 = self.normalize(self.to_tensor(img2))
        return img1, img2, K


def _cfg_value(cfg, key, default):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def get_pair_augmentations(camera_cfg=None):
    """Return paired train/eval transforms and per-pair crop-adjusted K.

    Defaults fx=532.57, fy=531.54, cx=320, cy=240 are a third-party RGB
    intrinsic estimate based on vanishing points, not an official Microsoft
    calibration. Override them when using a better calibration.
    """
    params = {
        "fx": _cfg_value(camera_cfg, "fx", 532.57),
        "fy": _cfg_value(camera_cfg, "fy", 531.54),
        "cx": _cfg_value(camera_cfg, "cx", 320.0),
        "cy": _cfg_value(camera_cfg, "cy", 240.0),
        "raw_width": _cfg_value(camera_cfg, "raw_width", 640),
        "raw_height": _cfg_value(camera_cfg, "raw_height", 480),
        "output_size": _cfg_value(camera_cfg, "output_size", 224),
        "expected_short_side": _cfg_value(camera_cfg, "expected_short_side", 256),
    }
    return PairedImageTransform(train=True, **params), PairedImageTransform(train=False, **params)


# Backwards-compatible single-image transforms used by the original test/evaluation code.
def get_augmentations():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_aug = transforms.Compose([
        transforms.RandomCrop(size=224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_aug = transforms.Compose([
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_aug, eval_aug
