from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from numba import jit
from skimage import measure, morphology

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def _bilinear_interp(img: np.ndarray, y: float, x: float) -> float:
    h, w = img.shape
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        return np.nan

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    return wa * img[y0, x0] + wb * img[y0, x1] + wc * img[y1, x0] + wd * img[y1, x1]


@jit(nopython=True, cache=True)
def _compute_local_normal_3x3(skeleton: np.ndarray, yc: int, xc: int) -> Tuple[float, float]:
    h, w = skeleton.shape
    sum_x, sum_y, count = 0.0, 0.0, 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            y, x = yc + dy, xc + dx
            if 0 <= y < h and 0 <= x < w and skeleton[y, x]:
                sum_x += dx
                sum_y += dy
                count += 1

    if count < 2:
        return 1.0, 0.0

    mean_x, mean_y = sum_x / count, sum_y / count
    sxx, syy, sxy = 0.0, 0.0, 0.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            y, x = yc + dy, xc + dx
            if 0 <= y < h and 0 <= x < w and skeleton[y, x]:
                local_x, local_y = dx - mean_x, dy - mean_y
                sxx += local_x * local_x
                syy += local_y * local_y
                sxy += local_x * local_y

    if sxx == 0.0 and syy == 0.0:
        return 1.0, 0.0

    theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
    nx, ny = -np.sin(theta), np.cos(theta)
    return (nx, ny) if nx >= 0.0 else (-nx, -ny)


@jit(nopython=True, cache=True)
def _fast_cod_kernel(
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    u_map: np.ndarray,
    v_map: np.ndarray,
    sample_mask: np.ndarray,
    has_v: bool,
    require_v: bool,
    skeleton: np.ndarray,
    delta_points: int,
    max_search_points: int,
    displacement_scale_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    widths = np.zeros(len(y_coords))
    valid_idx = np.zeros(len(y_coords), dtype=np.int64)
    count = 0
    start_d = max(1, delta_points)
    end_d = start_d + max(1, max_search_points)

    for index in range(len(y_coords)):
        yc, xc = y_coords[index], x_coords[index]
        nx, ny = _compute_local_normal_3x3(skeleton, yc, xc)
        u_pos, u_neg = np.nan, np.nan
        v_pos, v_neg = np.nan, np.nan

        for distance in range(start_d, end_d):
            y, x = yc + ny * distance, xc + nx * distance
            if _bilinear_interp(sample_mask, y, x) < 0.999:
                continue
            value = _bilinear_interp(u_map, y, x)
            if np.isnan(value):
                continue
            if has_v:
                v_value = _bilinear_interp(v_map, y, x)
                if np.isnan(v_value):
                    continue
                v_pos = v_value
            u_pos = value
            break

        for distance in range(start_d, end_d):
            y, x = yc - ny * distance, xc - nx * distance
            if _bilinear_interp(sample_mask, y, x) < 0.999:
                continue
            value = _bilinear_interp(u_map, y, x)
            if np.isnan(value):
                continue
            if has_v:
                v_value = _bilinear_interp(v_map, y, x)
                if np.isnan(v_value):
                    continue
                v_neg = v_value
            u_neg = value
            break

        if np.isnan(u_pos) or np.isnan(u_neg) or (require_v and not has_v):
            continue

        du = u_pos - u_neg
        if has_v:
            if np.isnan(v_pos) or np.isnan(v_neg):
                continue
            width = abs(du * nx + (v_pos - v_neg) * ny)
        else:
            width = abs(du * nx)

        widths[count] = width * displacement_scale_mm
        valid_idx[count] = index
        count += 1

    return widths[:count], valid_idx[:count]


class CrackPhysicsEngine:
    def __init__(self, config: dict):
        self.config = config
        physics = config.get("physics", {})
        sampling = physics.get("cod_sampling", {})
        quality = config.get("quality", {})
        detection = config.get("crack_detection", {}) or {}

        self.k = float(physics.get("strain_threshold_k", physics.get("mad_k", 2.0)))
        self.min_s = float(physics.get("min_cracking_strain", 1.5e-4))
        self.max_s = float(physics.get("max_cracking_strain_threshold", 0.03))
        self.min_area = int(physics.get("min_crack_area_points", physics.get("min_crack_area_px", 10)))
        self.cod_min = float(physics.get("cod_min_mm", 0.005))
        self.cod_max = float(physics.get("cod_max_mm", 5.0))
        self.delta_points = int(sampling.get("delta_points", sampling.get("delta_px", 3)))
        self.max_search_points = int(sampling.get("max_search_points", sampling.get("max_search_px", 15)))
        self.delta_mm = sampling.get("delta_mm")
        self.max_search_mm = sampling.get("max_search_mm")
        self.closing_radius = int(
            physics.get("morphology_closing_radius_points", physics.get("morphology_closing_radius", 0))
        )
        self.min_length_mm = float(physics.get("min_crack_length_mm", 0.2))
        self.min_mean_cod_mm = float(physics.get("cod_min_mean_mm", min(0.002, self.cod_min)))
        self.require_v = bool(physics.get("require_v_map_for_cod", True))
        self.quality_enabled = bool(quality.get("enabled", True))
        self.quality_mode = str(quality.get("metric_mode", "finite_only"))
        self.quality_threshold = quality.get("threshold")
        self.min_valid_fraction = float(quality.get("min_valid_fraction", 0.2))
        self.fusion_mode = str(detection.get("fusion_mode", "strain_or_image")).lower()
        self.image_dilation_radius = int(detection.get("image_dilation_radius_points", 1))
        self.strain_dilation_radius = int(detection.get("strain_dilation_radius_points", 0))
        self.require_strain_support = bool(detection.get("require_strain_support", False))

        if self.cod_max <= 0:
            raise ValueError("physics.cod_max_mm must be greater than zero.")
        if self.cod_min < 0:
            raise ValueError("physics.cod_min_mm cannot be negative.")
        if not 0 <= self.min_valid_fraction <= 1:
            raise ValueError("quality.min_valid_fraction must be within [0, 1].")

    def build_quality_mask(
        self,
        mask: np.ndarray,
        u_map: np.ndarray,
        v_map: np.ndarray | None,
        quality_map: np.ndarray | None,
    ) -> tuple[np.ndarray, float, str]:
        finite = np.asarray(mask, dtype=bool) & np.isfinite(u_map)
        if v_map is not None:
            finite &= np.isfinite(v_map)
        reason = "finite_u" if v_map is None else "finite_uv"

        if self.quality_enabled and quality_map is not None:
            quality = np.asarray(quality_map, dtype=np.float64)
            quality_mask = np.isfinite(quality)
            if self.quality_threshold is not None:
                threshold = float(self.quality_threshold)
                if self.quality_mode == "lower_is_better":
                    quality_mask &= quality <= threshold
                elif self.quality_mode in {"higher_is_better", "finite_only"}:
                    quality_mask &= quality >= threshold
                else:
                    logger.warning("Unknown quality.metric_mode=%s; using finite values.", self.quality_mode)
                reason = f"quality_{self.quality_mode}_{threshold:g}"
            else:
                reason = "quality_finite_only"
            finite &= quality_mask

        base = np.count_nonzero(mask)
        fraction = float(np.count_nonzero(finite) / base) if base else 0.0
        return finite, fraction, reason

    def extract_skeleton(
        self,
        exx: np.ndarray,
        valid_mask: np.ndarray,
        image_crack_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, str, float]:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(exx)
        clean_exx = exx[valid]
        if clean_exx.size == 0:
            return np.zeros_like(exx, dtype=bool), 0.0, "empty_valid_exx", 0.0

        median = float(np.median(clean_exx))
        mad = float(np.median(np.abs(clean_exx - median)))
        threshold = float(np.clip(median + self.k * mad * 1.4826, self.min_s, self.max_s))
        strain_zone = (exx > threshold) & valid
        if self.strain_dilation_radius > 0:
            strain_zone = morphology.binary_dilation(
                strain_zone,
                morphology.disk(self.strain_dilation_radius),
            ) & valid

        image_zone = None
        image_fraction = 0.0
        if image_crack_mask is not None:
            image_zone = np.asarray(image_crack_mask, dtype=bool)
            if image_zone.shape != valid.shape:
                raise ValueError(
                    f"image_crack_mask shape {image_zone.shape} does not match DIC shape {valid.shape}."
                )
            if self.image_dilation_radius > 0:
                image_zone = morphology.binary_dilation(
                    image_zone,
                    morphology.disk(self.image_dilation_radius),
                )
            image_zone &= valid
            image_fraction = float(np.count_nonzero(image_zone) / max(1, np.count_nonzero(valid)))

        internal_nans = np.zeros_like(valid, dtype=bool)
        if self.closing_radius > 0:
            closed = morphology.closing(valid_mask, morphology.disk(self.closing_radius))
            internal_nans = closed & ~np.asarray(valid_mask, dtype=bool)

        if image_zone is None or not np.any(image_zone):
            damage_zone = strain_zone | internal_nans
            source = "strain_only"
        elif self.fusion_mode in {"strain_only", "strain"}:
            damage_zone = strain_zone | internal_nans
            source = "strain_only"
        elif self.fusion_mode in {"image_only", "image"}:
            damage_zone = image_zone | internal_nans
            source = "image_only"
        elif self.fusion_mode in {"intersection", "and", "strain_and_image"}:
            damage_zone = (strain_zone & image_zone) | internal_nans
            source = "strain_and_image"
        elif self.fusion_mode in {"image_near_strain", "supported_image"}:
            support = morphology.binary_dilation(
                strain_zone,
                morphology.disk(max(1, self.image_dilation_radius)),
            )
            damage_zone = (strain_zone | (image_zone & support)) | internal_nans
            source = "image_near_strain"
        else:
            # 两路都不算稳，先取并集，后面再让 COD 和长度过滤收口。
            damage_zone = (strain_zone | image_zone) | internal_nans
            source = "strain_or_image"

        if self.require_strain_support and image_zone is not None and np.any(image_zone):
            support = morphology.binary_dilation(
                strain_zone,
                morphology.disk(max(1, self.image_dilation_radius)),
            )
            damage_zone &= support
            source += "+strain_support"

        cleaned = morphology.remove_small_objects(damage_zone.astype(bool), min_size=self.min_area)
        return morphology.skeletonize(cleaned), threshold, source, image_fraction

    def _sampling_points(self, dic_point_spacing_mm: float) -> tuple[int, int]:
        if not np.isfinite(dic_point_spacing_mm) or dic_point_spacing_mm <= 0:
            raise ValueError("dic_point_spacing_mm must be finite and greater than zero.")

        delta_points = self.delta_points
        max_search_points = self.max_search_points
        if self.delta_mm is not None:
            delta_points = max(1, int(round(float(self.delta_mm) / dic_point_spacing_mm)))
        if self.max_search_mm is not None:
            max_search_points = max(1, int(round(float(self.max_search_mm) / dic_point_spacing_mm)))
        return delta_points, max_search_points

    def compute_cod(
        self,
        u_map: np.ndarray,
        skeleton: np.ndarray,
        displacement_scale_mm: float,
        dic_point_spacing_mm: float,
        v_map: np.ndarray | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        u_map = np.asarray(u_map, dtype=float)
        skeleton = np.asarray(skeleton, dtype=bool)
        if u_map.shape != skeleton.shape:
            raise ValueError("u_map and skeleton must have the same shape.")
        if v_map is not None and np.asarray(v_map).shape != u_map.shape:
            raise ValueError("v_map must have the same shape as u_map.")

        labels = measure.label(skeleton, connectivity=2)
        y_coords, x_coords = np.where(skeleton)
        delta_points, max_search_points = self._sampling_points(dic_point_spacing_mm)
        if len(y_coords) == 0:
            return self._empty("no_skeleton", delta_points, max_search_points)

        has_v = v_map is not None
        if self.require_v and not has_v:
            return self._empty("missing_v_map_required", delta_points, max_search_points)

        v_data = np.ascontiguousarray(v_map) if has_v else np.zeros_like(u_map)
        if sample_mask is None:
            mask = np.isfinite(u_map)
            if has_v:
                mask &= np.isfinite(v_map)
        else:
            mask = np.asarray(sample_mask, dtype=bool) & np.isfinite(u_map)
            if mask.shape != u_map.shape:
                raise ValueError("sample_mask must have the same shape as u_map.")
            if has_v:
                mask &= np.isfinite(v_map)

        widths, valid_idx = _fast_cod_kernel(
            np.ascontiguousarray(y_coords),
            np.ascontiguousarray(x_coords),
            np.ascontiguousarray(u_map),
            v_data,
            np.ascontiguousarray(mask.astype(np.float64)),
            has_v,
            self.require_v,
            np.ascontiguousarray(skeleton),
            delta_points,
            max_search_points,
            displacement_scale_mm,
        )
        if widths.size < 3:
            return self._empty("insufficient_cod_samples", delta_points, max_search_points)

        finite = np.isfinite(widths)
        widths, valid_idx = widths[finite], valid_idx[finite]
        physical = (widths >= 0.0) & (widths <= self.cod_max)
        widths, valid_idx = widths[physical], valid_idx[physical]
        if widths.size < 3:
            return self._empty("cod_out_of_range", delta_points, max_search_points)

        frame = pd.DataFrame(
            {
                "Crack_ID": labels[y_coords, x_coords][valid_idx],
                "W": widths,
            }
        )
        summary = frame.groupby("Crack_ID")["W"].agg(["mean", "median", "max", "count"]).reset_index()
        summary["p95"] = frame.groupby("Crack_ID")["W"].quantile(0.95).to_numpy()
        lengths = _crack_lengths_mm(labels, float(dic_point_spacing_mm))
        summary["L_mm"] = summary["Crack_ID"].map(lengths).fillna(
            summary["count"] * float(dic_point_spacing_mm)
        )
        summary = summary[
            (summary["L_mm"] >= self.min_length_mm)
            & (summary["max"] >= self.cod_min)
            & (summary["mean"] >= self.min_mean_cod_mm)
        ]
        if summary.empty:
            return self._empty("object_filter_removed_all", delta_points, max_search_points)

        valid_ids = summary["Crack_ID"].to_numpy()
        raw = frame[frame["Crack_ID"].isin(valid_ids)]["W"].to_numpy()
        raw = raw[(raw >= self.cod_min * 0.5) & (raw <= self.cod_max)]
        if raw.size == 0:
            return self._empty("cod_floor_removed_all", delta_points, max_search_points)

        details = summary.rename(
            columns={
                "mean": "W_avg_mm",
                "median": "W_median_mm",
                "p95": "W_95_mm",
                "max": "W_max_mm",
                "L_mm": "Length_mm",
            }
        )
        return {
            "crack_count": int(len(summary)),
            "w_avg": float(summary["mean"].mean()),
            "w_median": float(summary["median"].median()),
            "w_95": float(np.percentile(raw, 95)),
            "w_max": float(summary["max"].max()),
            "w_99": float(np.percentile(raw, 99)),
            "raw_widths": raw,
            "per_crack_details": details,
            "cod_sample_count": int(raw.size),
            "cod_vector_mode": "u_v_normal" if has_v else "u_only_projection",
            "cod_status": "ok",
            "delta_points": int(delta_points),
            "max_search_points": int(max_search_points),
        }

    def _empty(
        self,
        status: str = "empty",
        delta_points: int | None = None,
        max_search_points: int | None = None,
    ) -> Dict[str, Any]:
        measured_zero = status in {"no_skeleton", "object_filter_removed_all"}
        value = 0.0 if measured_zero else float("nan")
        return {
            "crack_count": 0,
            "w_avg": value,
            "w_median": value,
            "w_95": value,
            "w_max": value,
            "w_99": value,
            "raw_widths": np.array([], dtype=float),
            "per_crack_details": pd.DataFrame(),
            "cod_sample_count": 0,
            "cod_vector_mode": "none",
            "cod_status": status,
            "delta_points": int(self.delta_points if delta_points is None else delta_points),
            "max_search_points": int(
                self.max_search_points if max_search_points is None else max_search_points
            ),
        }


def _crack_lengths_mm(labels: np.ndarray, dic_point_spacing_mm: float) -> dict[int, float]:
    lengths: dict[int, float] = {}
    for crack_id in np.unique(labels):
        if crack_id == 0:
            continue

        y_coords, x_coords = np.where(labels == crack_id)
        points = set(zip(y_coords.tolist(), x_coords.tolist()))
        total_steps = 0.0
        for y, x in points:
            if (y, x + 1) in points:
                total_steps += 1.0
            if (y + 1, x) in points:
                total_steps += 1.0
            if (y + 1, x + 1) in points:
                total_steps += np.sqrt(2.0)
            if (y + 1, x - 1) in points:
                total_steps += np.sqrt(2.0)
        lengths[int(crack_id)] = float(total_steps * dic_point_spacing_mm)
    return lengths
