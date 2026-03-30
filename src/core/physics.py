import logging
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import morphology, measure
from typing import Dict, Any, Tuple
from numba import jit

logger = logging.getLogger(__name__)


# ==========================================
# 高性能 Numba 算子
# ==========================================

@jit(nopython=True, cache=True)
def _bilinear_interp(img: np.ndarray, y: float, x: float) -> float:
    h, w = img.shape
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h: return np.nan
    wa, wb, wc, wd = (x1 - x) * (y1 - y), (x - x0) * (y1 - y), (x1 - x) * (y - y0), (x - x0) * (y - y0)
    return wa * img[y0, x0] + wb * img[y0, x1] + wc * img[y1, x0] + wd * img[y1, x1]


@jit(nopython=True, cache=True)
def _compute_local_normal_3x3(skeleton: np.ndarray, yc: int, xc: int) -> Tuple[float, float]:
    h, w = skeleton.shape
    sum_x, sum_y, count = 0.0, 0.0, 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            yy, xx = yc + i, xc + j
            if 0 <= yy < h and 0 <= xx < w and skeleton[yy, xx]:
                sum_x += j;
                sum_y += i;
                count += 1
    if count < 2: return 1.0, 0.0
    mx, my = sum_x / count, sum_y / count
    sxx, syy, sxy = 0.0, 0.0, 0.0
    for i in range(-1, 2):
        for j in range(-1, 2):
            yy, xx = yc + i, xc + j
            if 0 <= yy < h and 0 <= xx < w and skeleton[yy, xx]:
                dx, dy = j - mx, i - my
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy
    if sxx == 0 and syy == 0: return 1.0, 0.0
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    nx, ny = -np.sin(theta), np.cos(theta)
    return (nx, ny) if nx >= 0 else (-nx, -ny)


@jit(nopython=True, cache=True)
def _fast_cod_rigorous_kernel(y_coords: np.ndarray, x_coords: np.ndarray, u_map: np.ndarray, skeleton: np.ndarray,
                              delta_px: float, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y_coords)
    ws, v_idx = np.zeros(n), np.zeros(n, dtype=np.int64)
    cnt = 0

    # 🌟 核心防线：雷达探测深度。向外最多延伸 15 像素跨越 DIC 失相关黑洞
    max_search = 15

    for i in range(n):
        yc, xc = y_coords[i], x_coords[i]
        nx, ny = _compute_local_normal_3x3(skeleton, yc, xc)

        # 沿法向寻找有效的左侧位移
        val_p = np.nan
        for d in range(int(delta_px), int(delta_px) + max_search):
            v = _bilinear_interp(u_map, yc + ny * d, xc + nx * d)
            if not np.isnan(v):
                val_p = v
                break

        # 沿法向寻找有效的右侧位移
        val_n = np.nan
        for d in range(int(delta_px), int(delta_px) + max_search):
            v = _bilinear_interp(u_map, yc - ny * d, xc - nx * d)
            if not np.isnan(v):
                val_n = v
                break

        if np.isnan(val_p) or np.isnan(val_n): continue

        ws[cnt] = abs((val_p - val_n) * nx) * ratio
        v_idx[cnt], cnt = i, cnt + 1

    return ws[:cnt], v_idx[:cnt]


# ==========================================
# 物理核心层
# ==========================================

class CrackPhysicsEngine:
    def __init__(self, config: dict):
        phys = config.get('physics', {})
        self.k = float(phys.get('strain_threshold_k', 2.0))
        self.min_s = float(phys.get('min_cracking_strain', 1.5e-4))
        self.min_area = int(phys.get('min_crack_area_px', 10))
        self.cod_min = float(phys.get('cod_min_mm', 0.005))
        self.delta = float(phys.get('cod_sampling', {}).get('delta_px', 3.0))

    def extract_skeleton(self, exx: np.ndarray, mask: np.ndarray) -> np.ndarray:
        clean_exx = exx[mask & ~np.isnan(exx)]
        if clean_exx.size == 0: return np.zeros_like(exx, dtype=bool)
        med, mad = np.median(clean_exx), np.median(np.abs(clean_exx - np.median(clean_exx)))
        thresh = min(max(med + self.k * (mad * 1.4826), self.min_s), 0.005)

        # 🌟 形态学愈合：缝合贯穿性主裂缝造成的 NaN 断层
        # 对有效区域进行膨胀闭合，找回那些因为裂缝过宽而被排掉的盲区
        closed_mask = morphology.closing(mask, morphology.disk(10))
        internal_nans = closed_mask & (~mask)

        # 只要是应变超限的，或者处于失相关盲区内的，通通视为裂缝网络
        damage_zone = ((exx > thresh) & mask) | internal_nans
        cleaned = morphology.remove_small_objects(damage_zone, min_size=self.min_area)
        return morphology.skeletonize(cleaned)

    def compute_cod(self, u_map: np.ndarray, skeleton: np.ndarray, ratio: float) -> Dict[str, Any]:
        labels = measure.label(skeleton, connectivity=2)
        y_c, x_c = np.where(skeleton)
        if len(y_c) == 0: return self._empty()

        ws, v_idx = _fast_cod_rigorous_kernel(np.ascontiguousarray(y_c), np.ascontiguousarray(x_c),
                                              np.ascontiguousarray(u_map), np.ascontiguousarray(skeleton),
                                              self.delta, ratio)
        if ws.size < 3: return self._empty()

        df = pd.DataFrame({'Crack_ID': labels[y_c, x_c][v_idx], 'W': ws})
        sum_df = df.groupby('Crack_ID')['W'].agg(['mean', 'max', 'count']).reset_index()
        sum_df['L_mm'] = sum_df['count'] * ratio * 1.128

        sum_df = sum_df[(sum_df['L_mm'] >= 0.2) & (sum_df['max'] >= self.cod_min) & (sum_df['mean'] >= 0.002)]

        if sum_df.empty: return self._empty()

        valid_ids = sum_df['Crack_ID'].values
        clean_raw = df[df['Crack_ID'].isin(valid_ids)]['W'].values
        clean_raw = clean_raw[clean_raw >= (self.cod_min * 0.5)]

        return {
            "crack_count": len(sum_df),
            "w_avg": sum_df['mean'].mean(),
            "w_max": sum_df['max'].max(),
            "w_99": np.percentile(clean_raw, 99) if clean_raw.size > 0 else 0.0,
            "raw_widths": clean_raw,
            "per_crack_details": sum_df.rename(columns={'mean': 'W_avg_mm', 'max': 'W_max_mm', 'L_mm': 'Length_mm'})
        }

    def _empty(self):
        return {"crack_count": 0, "w_avg": 0.0, "w_max": 0.0, "w_99": 0.0, "raw_widths": np.array([], dtype=float),
                "per_crack_details": pd.DataFrame()}