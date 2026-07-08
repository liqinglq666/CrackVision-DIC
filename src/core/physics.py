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
    for i in range(-1, 2):
        for j in range(-1, 2):
            yy, xx = yc + i, xc + j
            if 0 <= yy < h and 0 <= xx < w and skeleton[yy, xx]:
                sum_x += j
                sum_y += i
                count += 1

    if count < 2:
        return 1.0, 0.0

    mx, my = sum_x / count, sum_y / count
    sxx, syy, sxy = 0.0, 0.0, 0.0
    for i in range(-1, 2):
        for j in range(-1, 2):
            yy, xx = yc + i, xc + j
            if 0 <= yy < h and 0 <= xx < w and skeleton[yy, xx]:
                dx, dy = j - mx, i - my
                sxx += dx * dx
                syy += dy * dy
                sxy += dx * dy

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
    has_v: bool,
    require_v: bool,
    skeleton: np.ndarray,
    delta_points: int,
    max_search_points: int,
    displacement_scale_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y_coords)
    widths = np.zeros(n)
    valid_idx = np.zeros(n, dtype=np.int64)
    cnt = 0

    start_d = max(1, delta_points)
    end_d = start_d + max(1, max_search_points)

    for i in range(n):
        yc, xc = y_coords[i], x_coords[i]
        nx, ny = _compute_local_normal_3x3(skeleton, yc, xc)

        u_p, u_n = np.nan, np.nan
        v_p, v_n = np.nan, np.nan

        for d in range(start_d, end_d):
            u_val = _bilinear_interp(u_map, yc + ny * d, xc + nx * d)
            if not np.isnan(u_val):
                if has_v:
                    v_val = _bilinear_interp(v_map, yc + ny * d, xc + nx * d)
                    if np.isnan(v_val):
                        continue
                    v_p = v_val
                u_p = u_val
                break

        for d in range(start_d, end_d):
            u_val = _bilinear_interp(u_map, yc - ny * d, xc - nx * d)
            if not np.isnan(u_val):
                if has_v:
                    v_val = _bilinear_interp(v_map, yc - ny * d, xc - nx * d)
                    if np.isnan(v_val):
                        continue
                    v_n = v_val
                u_n = u_val
                break

        if np.isnan(u_p) or np.isnan(u_n):
            continue

        if require_v and not has_v:
            continue

        du = u_p - u_n
        if has_v:
            if np.isnan(v_p) or np.isnan(v_n):
                continue
            dv = v_p - v_n
            widths[cnt] = abs(du * nx + dv * ny) * displacement_scale_mm
        else:
            widths[cnt] = abs(du * nx) * displacement_scale_mm
        valid_idx[cnt] = i
        cnt += 1

    return widths[:cnt], valid_idx[:cnt]


class CrackPhysicsEngine:
    def __init__(self, config: dict):
        phys = config.get("physics", {})
        sampling = phys.get("cod_sampling", {})
        quality = config.get("quality", {})
        self.k = float(phys.get("strain_threshold_k", phys.get("mad_k", 2.0)))
        self.min_s = float(phys.get("min_cracking_strain", 1.5e-4))
        self.max_s = float(phys.get("max_cracking_strain_threshold", 0.03))
        self.min_area = int(phys.get("min_crack_area_points", phys.get("min_crack_area_px", 10)))
        self.cod_min = float(phys.get("cod_min_mm", 0.005))
        self.cod_max = float(phys.get("cod_max_mm", 5.0))
        self.delta_points = int(sampling.get("delta_points", sampling.get("delta_px", 3)))
        self.max_search_points = int(sampling.get("max_search_points", sampling.get("max_search_px", 15)))
        self.delta_mm = sampling.get("delta_mm")
        self.max_search_mm = sampling.get("max_search_mm")
        self.closing_radius = int(phys.get("morphology_closing_radius_points", phys.get("morphology_closing_radius", 0)))
        self.min_length_mm = float(phys.get("min_crack_length_mm", 0.2))
        self.min_mean_cod_mm = float(phys.get("cod_min_mean_mm", min(0.002, self.cod_min)))
        self.require_v = bool(phys.get("require_v_map_for_cod", True))
        self.quality_enabled = bool(quality.get("enabled", True))
        self.quality_mode = str(quality.get("metric_mode", "finite_only"))
        self.quality_threshold = quality.get("threshold")
        self.min_valid_fraction = float(quality.get("min_valid_fraction", 0.2))

    def build_quality_mask(
        self, mask: np.ndarray, u_map: np.ndarray, v_map: np.ndarray | None, quality_map: np.ndarray | None
    ) -> tuple[np.ndarray, float, str]:
        finite = mask & np.isfinite(u_map)
        if v_map is not None:
            finite &= np.isfinite(v_map)
        reason = "finite_uv"

        if self.quality_enabled and quality_map is not None:
            q = np.asarray(quality_map, dtype=np.float64)
            q_mask = np.isfinite(q)
            if self.quality_threshold is not None:
                threshold = float(self.quality_threshold)
                if self.quality_mode == "lower_is_better":
                    q_mask &= q <= threshold
                else:
                    q_mask &= q >= threshold
                reason = f"quality_{self.quality_mode}_{threshold:g}"
            else:
                reason = "quality_finite_only"
            finite &= q_mask

        base = np.count_nonzero(mask)
        fraction = float(np.count_nonzero(finite) / base) if base > 0 else 0.0
        return finite, fraction, reason

    def extract_skeleton(self, exx: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
        valid = valid_mask & np.isfinite(exx)
        clean_exx = exx[valid]
        if clean_exx.size == 0:
            return np.zeros_like(exx, dtype=bool), 0.0

        median = float(np.median(clean_exx))
        mad = float(np.median(np.abs(clean_exx - median)))
        robust_sigma = mad * 1.4826
        threshold = float(np.clip(median + self.k * robust_sigma, self.min_s, self.max_s))

        internal_nans = np.zeros_like(valid_mask, dtype=bool)
        if self.closing_radius > 0:
            closed_mask = morphology.closing(valid_mask, morphology.disk(self.closing_radius))
            internal_nans = closed_mask & (~valid_mask)

        damage_zone = ((exx > threshold) & valid) | internal_nans
        cleaned = morphology.remove_small_objects(damage_zone, min_size=self.min_area)
        return morphology.skeletonize(cleaned), threshold

    def _sampling_points(self, dic_point_spacing_mm: float) -> tuple[int, int]:
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
    ) -> Dict[str, Any]:
        labels = measure.label(skeleton, connectivity=2)
        y_c, x_c = np.where(skeleton)
        if len(y_c) == 0:
            return self._empty("no_skeleton")

        has_v = v_map is not None
        v_data = np.ascontiguousarray(v_map) if has_v else np.zeros_like(u_map)
        delta_points, max_search_points = self._sampling_points(dic_point_spacing_mm)

        widths, valid_idx = _fast_cod_kernel(
            np.ascontiguousarray(y_c),
            np.ascontiguousarray(x_c),
            np.ascontiguousarray(u_map),
            v_data,
            has_v,
            self.require_v,
            np.ascontiguousarray(skeleton),
            delta_points,
            max_search_points,
            displacement_scale_mm,
        )
        if widths.size < 3:
            return self._empty("insufficient_cod_samples")

        finite = np.isfinite(widths)
        widths = widths[finite]
        valid_idx = valid_idx[finite]
        physical = (widths >= 0.0) & (widths <= self.cod_max)
        widths = widths[physical]
        valid_idx = valid_idx[physical]
        if widths.size < 3:
            return self._empty("cod_out_of_range")

        df = pd.DataFrame({"Crack_ID": labels[y_c, x_c][valid_idx], "W": widths})
        summary = df.groupby("Crack_ID")["W"].agg(["mean", "max", "count"]).reset_index()
        summary["L_mm"] = summary["count"] * float(dic_point_spacing_mm)
        summary = summary[
            (summary["L_mm"] >= self.min_length_mm)
            & (summary["max"] >= self.cod_min)
            & (summary["mean"] >= self.min_mean_cod_mm)
        ]

        if summary.empty:
            return self._empty("object_filter_removed_all")

        valid_ids = summary["Crack_ID"].to_numpy()
        raw = df[df["Crack_ID"].isin(valid_ids)]["W"].to_numpy()
        raw = raw[(raw >= self.cod_min * 0.5) & (raw <= self.cod_max)]

        details = summary.rename(columns={"mean": "W_avg_mm", "max": "W_max_mm", "L_mm": "Length_mm"})
        return {
            "crack_count": int(len(summary)),
            "w_avg": float(summary["mean"].mean()),
            "w_max": float(summary["max"].max()),
            "w_99": float(np.percentile(raw, 99)) if raw.size > 0 else 0.0,
            "raw_widths": raw,
            "per_crack_details": details,
            "cod_sample_count": int(raw.size),
            "cod_vector_mode": "u_v_normal" if has_v else "u_only_projection",
            "cod_status": "ok",
            "delta_points": int(delta_points),
            "max_search_points": int(max_search_points),
        }

    def _empty(self, status: str = "empty") -> Dict[str, Any]:
        return {
            "crack_count": 0,
            "w_avg": 0.0,
            "w_max": 0.0,
            "w_99": 0.0,
            "raw_widths": np.array([], dtype=float),
            "per_crack_details": pd.DataFrame(),
            "cod_sample_count": 0,
            "cod_vector_mode": "none",
            "cod_status": status,
            "delta_points": int(self.delta_points),
            "max_search_points": int(self.max_search_points),
        }
