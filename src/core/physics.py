from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from skimage import measure, morphology

from .models import FrameData


@dataclass(frozen=True)
class LineFit:
    slope: float
    intercept: float
    r2: float
    n: int


class CrackPhysicsEngine:
    """Coordinate-invariant crack detection + regression-based COD measurement."""

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        phys = cfg.get("physics", {})
        det = phys.get("detection", {})
        cod = phys.get("cod", {})

        self.shear_is_engineering = bool(phys.get("shear_component_is_engineering", False))
        self.threshold_k = float(det.get("threshold_k", 2.0))
        self.min_tensile_strain = float(det.get("min_tensile_strain", 2.0e-4))
        self.max_threshold = float(det.get("max_threshold", 0.05))
        self.min_area_points = int(det.get("min_crack_area_points", 4))
        self.closing_radius_points = int(det.get("closing_radius_points", 0))
        self.min_crack_length_mm = float(det.get("min_crack_length_mm", 0.15))
        self.normal_radius_points = int(det.get("normal_radius_points", 4))

        self.near_mm = float(cod.get("near_mm", 0.15))
        self.far_mm = float(cod.get("far_mm", 0.75))
        self.samples_per_side = int(cod.get("samples_per_side", 7))
        self.min_valid_per_side = int(cod.get("min_valid_per_side", 3))
        self.sample_stride_points = max(1, int(cod.get("sample_stride_points", 1)))
        self.min_samples_per_crack = max(1, int(cod.get("min_samples_per_crack", 3)))
        self.min_width_mm = float(cod.get("min_width_mm", 0.001))
        self.max_width_mm = float(cod.get("max_width_mm", 2.0))
        self.robust_sigma = float(cod.get("robust_sigma", 3.5))
        self.min_fit_r2 = float(cod.get("min_fit_r2", -1.0))

        if self.near_mm <= 0 or self.far_mm <= self.near_mm:
            raise ValueError("physics.cod requires 0 < near_mm < far_mm")
        if self.samples_per_side < self.min_valid_per_side or self.min_valid_per_side < 2:
            raise ValueError("COD sampling requires samples_per_side >= min_valid_per_side >= 2")
        if self.max_width_mm <= 0 or self.min_width_mm < 0 or self.min_width_mm >= self.max_width_mm:
            raise ValueError("Invalid COD width limits")

    def maximum_principal_tensile_strain(
        self, exx: np.ndarray, eyy: np.ndarray, exy: np.ndarray
    ) -> np.ndarray:
        exx = np.asarray(exx, dtype=float)
        eyy = np.asarray(eyy, dtype=float)
        exy = np.asarray(exy, dtype=float)
        shear = exy * 0.5 if self.shear_is_engineering else exy
        mean = 0.5 * (exx + eyy)
        radius = np.sqrt((0.5 * (exx - eyy)) ** 2 + shear**2)
        return mean + radius

    def detect_skeleton(self, frame: FrameData) -> tuple[np.ndarray, np.ndarray, float, int]:
        eps1 = self.maximum_principal_tensile_strain(frame.exx_map, frame.eyy_map, frame.exy_map)
        valid = frame.mask & np.isfinite(eps1)
        values = eps1[valid]
        if values.size == 0:
            return np.zeros_like(valid), eps1, float("nan"), 0

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = 1.4826 * mad
        raw_threshold = median + self.threshold_k * robust_sigma
        threshold = float(np.clip(raw_threshold, self.min_tensile_strain, self.max_threshold))
        candidate = valid & (eps1 >= threshold) & (eps1 > 0)

        if self.closing_radius_points > 0:
            candidate = morphology.binary_closing(candidate, morphology.disk(self.closing_radius_points)) & valid
        if self.min_area_points > 1:
            candidate = morphology.remove_small_objects(candidate, min_size=self.min_area_points)

        skeleton = morphology.skeletonize(candidate)
        return skeleton, eps1, threshold, int(np.count_nonzero(candidate))

    def analyze_frame(self, frame: FrameData) -> tuple[dict[str, Any], pd.DataFrame]:
        skeleton, eps1, threshold, candidate_points = self.detect_skeleton(frame)
        valid_fraction = float(np.count_nonzero(frame.mask) / frame.mask.size) if frame.mask.size else 0.0
        base = {
            "Frame": int(frame.frame_id),
            "Time_s": float(frame.time_s),
            "pixel_size_mm": float(frame.pixel_size_mm),
            "dic_step_px": float(frame.dic_step_px) if frame.dic_step_px is not None else float("nan"),
            "dic_point_spacing_mm": float(frame.dic_point_spacing_mm),
            "metadata_source": frame.metadata_source,
            "valid_fraction": valid_fraction,
            "principal_strain_threshold": threshold,
            "candidate_points": candidate_points,
            "skeleton_points": int(np.count_nonzero(skeleton)),
        }

        if not np.any(skeleton):
            return self._empty_summary(base, "no_crack_candidate"), self._empty_details()

        labels = measure.label(skeleton, connectivity=2)
        point_rows: list[dict[str, float | int]] = []
        crack_lengths = self._crack_lengths_mm(labels, frame.dic_point_spacing_mm)

        for crack_id in range(1, int(labels.max()) + 1):
            if crack_lengths.get(crack_id, 0.0) < self.min_crack_length_mm:
                continue
            coords = np.column_stack(np.where(labels == crack_id))
            if len(coords) == 0:
                continue
            for idx in range(0, len(coords), self.sample_stride_points):
                y, x = (int(coords[idx, 0]), int(coords[idx, 1]))
                normal = self._local_normal(coords, y, x)
                sample = self._cod_at_point(frame, y, x, normal)
                if sample is None:
                    continue
                width_mm, slip_mm, fit_r2 = sample
                if not np.isfinite(width_mm) or width_mm < 0 or width_mm > self.max_width_mm:
                    continue
                point_rows.append(
                    {
                        "Crack_ID": crack_id,
                        "Y": y,
                        "X": x,
                        "Width_mm": width_mm,
                        "Slip_mm": slip_mm,
                        "Fit_R2": fit_r2,
                    }
                )

        if not point_rows:
            return self._empty_summary(base, "insufficient_cod_samples"), self._empty_details()

        points = pd.DataFrame(point_rows)
        details_rows: list[dict[str, Any]] = []
        for crack_id, group in points.groupby("Crack_ID", sort=True):
            widths = group["Width_mm"].to_numpy(dtype=float)
            widths = widths[np.isfinite(widths)]
            if widths.size < self.min_samples_per_crack:
                continue
            median_width = float(np.median(widths))
            if median_width < self.min_width_mm:
                continue
            details_rows.append(
                {
                    "Frame": int(frame.frame_id),
                    "Crack_ID": int(crack_id),
                    "Length_mm": float(crack_lengths.get(int(crack_id), 0.0)),
                    "COD_samples": int(widths.size),
                    "W_median_mm": median_width,
                    "W_avg_mm": float(np.mean(widths)),
                    "W_95_mm": float(np.percentile(widths, 95)),
                    "W_max_mm": float(np.max(widths)),
                    "Slip_median_mm": float(np.nanmedian(group["Slip_mm"].to_numpy(dtype=float))),
                    "Fit_R2_median": float(np.nanmedian(group["Fit_R2"].to_numpy(dtype=float))),
                }
            )

        details = pd.DataFrame(details_rows)
        if details.empty:
            return self._empty_summary(base, "crack_filter_removed_all"), self._empty_details()

        accepted_ids = set(details["Crack_ID"].astype(int))
        accepted_points = points[points["Crack_ID"].isin(accepted_ids)]
        raw = accepted_points["Width_mm"].to_numpy(dtype=float)
        raw = raw[np.isfinite(raw)]

        summary = {
            **base,
            "cod_status": "ok",
            "crack_count": int(len(details)),
            "cod_sample_count": int(raw.size),
            "W_median_mm": float(np.median(raw)),
            "W_avg_mm": float(np.mean(raw)),
            "W_95_mm": float(np.percentile(raw, 95)),
            "W_max_mm": float(np.max(raw)),
        }
        return summary, details

    def _cod_at_point(
        self, frame: FrameData, y: int, x: int, normal: tuple[float, float]
    ) -> Optional[tuple[float, float, float]]:
        nx, ny = normal
        tx, ty = -ny, nx
        step_mm = frame.dic_point_spacing_mm
        distances_mm = np.linspace(self.near_mm, self.far_mm, self.samples_per_side)

        pos_s: list[float] = []
        neg_s: list[float] = []
        pos_n: list[float] = []
        neg_n: list[float] = []
        pos_t: list[float] = []
        neg_t: list[float] = []

        for d_mm in distances_mm:
            d_points = d_mm / step_mm
            for sign, s_list, n_list, t_list in (
                (1.0, pos_s, pos_n, pos_t),
                (-1.0, neg_s, neg_n, neg_t),
            ):
                yy = y + sign * ny * d_points
                xx = x + sign * nx * d_points
                if self._bilinear_mask(frame.mask, yy, xx) < 0.999:
                    continue
                u = self._bilinear(frame.u_map, yy, xx)
                v = self._bilinear(frame.v_map, yy, xx)
                if not np.isfinite(u) or not np.isfinite(v):
                    continue
                signed_d = sign * float(d_mm)
                s_list.append(signed_d)
                n_list.append(float(u * nx + v * ny))
                t_list.append(float(u * tx + v * ty))

        if len(pos_s) < self.min_valid_per_side or len(neg_s) < self.min_valid_per_side:
            return None

        fit_pos_n = self._robust_line_fit(np.asarray(pos_s), np.asarray(pos_n))
        fit_neg_n = self._robust_line_fit(np.asarray(neg_s), np.asarray(neg_n))
        fit_pos_t = self._robust_line_fit(np.asarray(pos_s), np.asarray(pos_t))
        fit_neg_t = self._robust_line_fit(np.asarray(neg_s), np.asarray(neg_t))
        if None in (fit_pos_n, fit_neg_n, fit_pos_t, fit_neg_t):
            return None

        assert fit_pos_n and fit_neg_n and fit_pos_t and fit_neg_t
        r2 = min(fit_pos_n.r2, fit_neg_n.r2)
        if r2 < self.min_fit_r2:
            return None

        width_px = abs(fit_pos_n.intercept - fit_neg_n.intercept)
        slip_px = abs(fit_pos_t.intercept - fit_neg_t.intercept)
        return width_px * frame.pixel_size_mm, slip_px * frame.pixel_size_mm, r2

    def _robust_line_fit(self, x: np.ndarray, y: np.ndarray) -> Optional[LineFit]:
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size < self.min_valid_per_side or np.ptp(x) <= 0:
            return None

        keep = np.ones(x.size, dtype=bool)
        for _ in range(3):
            if np.count_nonzero(keep) < self.min_valid_per_side:
                return None
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            residual = y - (slope * x + intercept)
            center = np.median(residual[keep])
            mad = np.median(np.abs(residual[keep] - center))
            if mad <= 1e-12:
                break
            scale = 1.4826 * mad
            new_keep = np.abs(residual - center) <= self.robust_sigma * scale
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep

        if np.count_nonzero(keep) < self.min_valid_per_side:
            return None
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        pred = slope * x[keep] + intercept
        ss_res = float(np.sum((y[keep] - pred) ** 2))
        ss_tot = float(np.sum((y[keep] - np.mean(y[keep])) ** 2))
        r2 = 1.0 if ss_tot <= 1e-15 else 1.0 - ss_res / ss_tot
        return LineFit(float(slope), float(intercept), float(r2), int(np.count_nonzero(keep)))

    def _local_normal(self, coords_yx: np.ndarray, y: int, x: int) -> tuple[float, float]:
        delta = coords_yx.astype(float) - np.array([y, x], dtype=float)
        radius2 = float(self.normal_radius_points**2)
        local = coords_yx[np.sum(delta**2, axis=1) <= radius2]
        if len(local) < 3:
            # Fallback to the whole short crack segment.
            local = coords_yx
        if len(local) < 2:
            return 1.0, 0.0

        xy = np.column_stack((local[:, 1], local[:, 0])).astype(float)
        xy -= xy.mean(axis=0)
        cov = xy.T @ xy
        eigvals, eigvecs = np.linalg.eigh(cov)
        tangent = eigvecs[:, int(np.argmax(eigvals))]
        tx, ty = float(tangent[0]), float(tangent[1])
        norm = math.hypot(tx, ty)
        if norm <= 1e-12:
            return 1.0, 0.0
        tx, ty = tx / norm, ty / norm
        nx, ny = -ty, tx
        if nx < 0 or (abs(nx) < 1e-12 and ny < 0):
            nx, ny = -nx, -ny
        return nx, ny

    @staticmethod
    def _bilinear(arr: np.ndarray, y: float, x: float) -> float:
        h, w = arr.shape
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = x0 + 1, y0 + 1
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            return float("nan")
        dx, dy = x - x0, y - y0
        return float(
            arr[y0, x0] * (1 - dx) * (1 - dy)
            + arr[y0, x1] * dx * (1 - dy)
            + arr[y1, x0] * (1 - dx) * dy
            + arr[y1, x1] * dx * dy
        )

    @classmethod
    def _bilinear_mask(cls, mask: np.ndarray, y: float, x: float) -> float:
        return cls._bilinear(mask.astype(float), y, x)

    @staticmethod
    def _crack_lengths_mm(labels: np.ndarray, spacing_mm: float) -> dict[int, float]:
        out: dict[int, float] = {}
        for crack_id in range(1, int(labels.max()) + 1):
            ys, xs = np.where(labels == crack_id)
            points = set(zip(ys.tolist(), xs.tolist()))
            total = 0.0
            for y, x in points:
                if (y, x + 1) in points:
                    total += 1.0
                if (y + 1, x) in points:
                    total += 1.0
                if (y + 1, x + 1) in points:
                    total += math.sqrt(2.0)
                if (y + 1, x - 1) in points:
                    total += math.sqrt(2.0)
            out[crack_id] = total * float(spacing_mm)
        return out

    @staticmethod
    def _empty_details() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "Frame",
                "Crack_ID",
                "Length_mm",
                "COD_samples",
                "W_median_mm",
                "W_avg_mm",
                "W_95_mm",
                "W_max_mm",
                "Slip_median_mm",
                "Fit_R2_median",
            ]
        )

    @staticmethod
    def _empty_summary(base: dict[str, Any], status: str) -> dict[str, Any]:
        return {
            **base,
            "cod_status": status,
            "crack_count": 0,
            "cod_sample_count": 0,
            "W_median_mm": float("nan"),
            "W_avg_mm": float("nan"),
            "W_95_mm": float("nan"),
            "W_max_mm": float("nan"),
        }
