import logging
from pathlib import Path
from typing import Optional

import numpy as np
from skimage import color, exposure, filters, io, measure, morphology, transform, util

logger = logging.getLogger(__name__)

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


class ImageCrackMaskProvider:
    """Load camera frames and convert dark/bright crack traces into a DIC-grid mask.

    The mask is only used as an auxiliary crack-position prior. COD still comes from DIC
    displacement jump. No camera homography magic is attempted here. If the camera image
    and DIC grid do not share the same ROI, the user must crop images first. Brutal, but honest.
    """

    def __init__(self, config: dict, mat_path: Path):
        cfg = config.get("image_crack_detection", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.cfg = cfg
        self.mat_path = Path(mat_path)
        self.image_dir = self._resolve_image_dir(cfg)
        self.files = self._collect_files(self.image_dir) if self.enabled and self.image_dir else []
        self.frame_index_offset = int(cfg.get("frame_index_offset", 0))
        self.filename_pattern = cfg.get("filename_pattern")

        if self.enabled and not self.files:
            logger.warning("image_crack_detection is enabled, but no camera image files were found.")

    def _resolve_image_dir(self, cfg: dict) -> Optional[Path]:
        raw_dir = cfg.get("image_dir")
        if raw_dir:
            p = Path(raw_dir)
            if not p.is_absolute():
                p = self.mat_path.parent / p
            return p if p.exists() else None

        for name in cfg.get("auto_discover_dir_names", ["images", "imgs", "frames", "camera", "crack_images"]):
            p = self.mat_path.parent / str(name)
            if p.exists() and p.is_dir():
                return p
        return None

    @staticmethod
    def _collect_files(image_dir: Optional[Path]) -> list[Path]:
        if image_dir is None or not image_dir.exists():
            return []
        files: list[Path] = []
        for ext in IMAGE_EXTS:
            files.extend(image_dir.glob(ext))
            files.extend(image_dir.glob(ext.upper()))
        return sorted(set(files), key=lambda p: p.name.lower())

    def _file_for_frame(self, frame_id: int) -> Optional[Path]:
        if not self.enabled or not self.files:
            return None

        if self.filename_pattern:
            candidates = []
            for frame_value in (frame_id, frame_id + self.frame_index_offset):
                try:
                    candidates.append(self.image_dir / self.filename_pattern.format(frame=frame_value, frame_id=frame_value))
                except Exception:
                    pass
            for candidate in candidates:
                if candidate.exists():
                    return candidate

        idx = frame_id + self.frame_index_offset
        if 0 <= idx < len(self.files):
            return self.files[idx]

        frame_texts = {str(frame_id), f"{frame_id:03d}", f"{frame_id:04d}", f"{frame_id:05d}"}
        for f in self.files:
            stem = f.stem
            if any(token in stem for token in frame_texts):
                return f
        return None

    def mask_for_frame(self, frame_id: int, target_shape: tuple[int, int]) -> tuple[Optional[np.ndarray], str, float, int]:
        image_path = self._file_for_frame(frame_id)
        if image_path is None:
            return None, "image_mask_missing", 0.0, 0

        try:
            mask = build_image_crack_mask(image_path, target_shape, self.cfg)
            if mask is None or not np.any(mask):
                return None, f"image_mask_empty:{image_path.name}", 0.0, 0
            return mask, f"image_mask:{image_path.name}", float(np.count_nonzero(mask)), int(mask.size)
        except Exception as exc:
            logger.warning("Failed to build image crack mask for %s: %s", image_path, exc, exc_info=True)
            return None, f"image_mask_failed:{image_path.name}:{exc}", 0.0, 0


def build_image_crack_mask(image_path: Path, target_shape: tuple[int, int], cfg: dict) -> Optional[np.ndarray]:
    img = io.imread(str(image_path))
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        gray = color.rgb2gray(img)
    else:
        gray = util.img_as_float(img)

    gray = np.asarray(gray, dtype=np.float64)
    if not np.isfinite(gray).any():
        return None

    p_low, p_high = np.nanpercentile(gray, [1, 99])
    if p_high > p_low:
        gray = exposure.rescale_intensity(gray, in_range=(p_low, p_high), out_range=(0.0, 1.0))
    else:
        gray = np.clip(gray, 0.0, 1.0)

    sigma = float(cfg.get("background_sigma_px", 12.0))
    background = filters.gaussian(gray, sigma=max(0.5, sigma), preserve_range=True)
    dark_cracks = bool(cfg.get("dark_cracks", True))
    response = background - gray if dark_cracks else gray - background
    response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
    response = np.clip(response, 0.0, None)

    if float(response.max()) <= 0.0:
        return None

    q = float(cfg.get("threshold_quantile", 0.92))
    q = min(max(q, 0.50), 0.995)
    q_threshold = float(np.quantile(response, q))
    try:
        otsu_threshold = float(filters.threshold_otsu(response))
    except Exception:
        otsu_threshold = q_threshold
    threshold = max(q_threshold, otsu_threshold * float(cfg.get("otsu_weight", 0.75)))

    mask = response > threshold
    min_area_px = int(cfg.get("min_object_area_px", 20))
    if min_area_px > 1:
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area_px)

    closing_radius = int(cfg.get("closing_radius_px", 1))
    if closing_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(closing_radius))

    if target_shape and tuple(mask.shape) != tuple(target_shape):
        mask = transform.resize(
            mask.astype(float),
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) >= 0.5

    min_area_points = int(cfg.get("min_object_area_points", 5))
    if min_area_points > 1:
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area_points)

    return mask.astype(bool)


def image_area_skeleton_width_mm(mask: Optional[np.ndarray], dic_point_spacing_mm: float) -> tuple[float, float, float]:
    """Return area, skeleton length, and ACW using area / skeleton length.

    Units are based on the DIC-grid mask. This is a sanity-check width, not the
    primary physical COD measurement.
    """
    if mask is None or not np.any(mask):
        return 0.0, 0.0, 0.0
    mask = np.asarray(mask, dtype=bool)
    skeleton = morphology.skeletonize(mask)
    labels = measure.label(skeleton, connectivity=2)
    length_mm = _skeleton_length_mm(labels, dic_point_spacing_mm)
    area_mm2 = float(np.count_nonzero(mask)) * float(dic_point_spacing_mm) ** 2
    width_mm = area_mm2 / length_mm if length_mm > 0 else 0.0
    return area_mm2, length_mm, width_mm


def _skeleton_length_mm(labels: np.ndarray, spacing_mm: float) -> float:
    total = 0.0
    for crack_id in np.unique(labels):
        if crack_id == 0:
            continue
        y_coords, x_coords = np.where(labels == crack_id)
        points = set(zip(y_coords.tolist(), x_coords.tolist()))
        for y, x in points:
            if (y, x + 1) in points:
                total += 1.0
            if (y + 1, x) in points:
                total += 1.0
            if (y + 1, x + 1) in points:
                total += np.sqrt(2.0)
            if (y + 1, x - 1) in points:
                total += np.sqrt(2.0)
    return float(total * spacing_mm)
