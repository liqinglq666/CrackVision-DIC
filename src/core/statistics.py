import logging
import numpy as np
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class CrackStatisticsEngine:
    """
    微观统计引擎：推断高延性复合材料的概率密度分布。
    NOTE: 提取大应变下的特征缝宽 (w_50, w_90, w_99)，为评估恶劣环境下的自愈合临界条件提供定量依据。
    """

    def __init__(self, bins: int = 100) -> None:
        if bins <= 0:
            raise ValueError("KDE 分箱数量 bins 必须为正整数。")
        self.bins = int(bins)

    def compute_distribution(self, raw_widths: np.ndarray) -> Dict[str, Any]:
        """计算一维位移数组的 PDF、CDF 及其核心分位数。"""
        if raw_widths is None or raw_widths.size < 2:
            return self._empty_distribution()

        # 🌟 性能优化：将 4 次独立的排序合并为 1 次底层 C 级查询，性能提升数倍！
        percentiles = np.percentile(raw_widths, [50, 90, 99, 99.9])
        w_50, w_90, w_99, w_max_limit = float(percentiles[0]), float(percentiles[1]), float(percentiles[2]), float(
            percentiles[3])

        # NOTE: 过滤 99.9% 以上的极端飞点，防止少数离群噪点将 KDE 核密度宽带灾难性拉伸
        valid_widths = raw_widths[raw_widths < w_max_limit]

        if valid_widths.size < 2:
            return self._empty_distribution(w_50, w_90, w_99)

        # 预分配安全初值，防止作用域逃逸
        x_grid = np.linspace(0, max(0.1, w_99), self.bins)
        pdf_y = np.zeros_like(x_grid, dtype=np.float64)
        cdf_y = np.zeros_like(x_grid, dtype=np.float64)

        try:
            kde = gaussian_kde(valid_widths, bw_method='scott')
            x_grid = np.linspace(0, max(0.1, float(np.max(valid_widths)) * 1.1), self.bins)
            pdf_y = kde(x_grid)

            dx = x_grid[1] - x_grid[0]
            cdf_y = np.cumsum(pdf_y) * dx

            if cdf_y[-1] > 0:
                cdf_y = np.clip(cdf_y / cdf_y[-1], 0.0, 1.0)

        except (LinAlgError, ValueError) as e:
            # 容错：当方差奇异时，平滑降级为零分布
            logger.warning(f"KDE 拟合发生奇异 (Error: {e})，已安全降级。")

        return {
            "w_50": w_50,
            "w_90": w_90,
            "w_99": w_99,
            "pdf_x": x_grid,
            "pdf_y": pdf_y,
            "cdf_y": cdf_y
        }

    def _empty_distribution(self, w_50: float = 0.0, w_90: float = 0.0, w_99: float = 0.0) -> Dict[str, Any]:
        """返回安全的空分布结构体"""
        return {
            "w_50": w_50,
            "w_90": w_90,
            "w_99": w_99,
            "pdf_x": np.array([], dtype=np.float64),
            "pdf_y": np.array([], dtype=np.float64),
            "cdf_y": np.array([], dtype=np.float64)
        }


class FractureMechanicsEvaluator:
    """
    断裂力学评估引擎：负责宏观统计推断、CDM 损伤评估与断裂能求解。
    🌟 架构修复：重命名以彻底解决与时域同步模块的类名冲突。
    """

    def __init__(self, gauge_length_mm: float) -> None:
        if gauge_length_mm <= 0:
            raise ValueError("标距长度 gauge_length_mm 必须严格大于零。")
        self.L = float(gauge_length_mm)

    def compute_damage_index(self, total_crack_width: float, global_strain: float) -> float:
        """
        计算连续损伤因子 (Continuum Damage Mechanics)。
        反映微裂缝对宏观应变的物理贡献度: $D = \frac{\sum w_i}{\varepsilon L}$
        """
        if global_strain <= 1e-6 or total_crack_width <= 0.0:
            return 0.0

        damage_d = total_crack_width / (global_strain * self.L)
        return float(np.clip(damage_d, 0.0, 1.0))

    def compute_uniformity(self, crack_positions_x: np.ndarray) -> float:
        """
        计算多缝空间均匀性指数: $Uniformity = 1 - CV_{spacing}$
        """
        if crack_positions_x is None or crack_positions_x.size < 2:
            return 0.0

        sorted_x = np.sort(crack_positions_x)
        spacings = np.diff(sorted_x)
        mean_s = float(np.mean(spacings))

        if mean_s <= 1e-6:
            return 0.0

        cv_spacing = float(np.std(spacings)) / mean_s
        return float(np.clip(1.0 - cv_spacing, 0.0, 1.0))

    def estimate_fracture_energy(self, stress_history: np.ndarray, w_history: np.ndarray) -> float:
        """
        估算断裂耗能: $G_f \approx \int \sigma \, dw$

        NOTE: 未使用原生的 np.trapz 或 scipy.integrate.trapezoid，
        是因为我们必须在积分前滤除卸载/闭合引起的负位移增量，手动梯形积分是唯一兼顾过滤与性能的最优解。
        """
        if stress_history.size < 2 or w_history.size < 2:
            return 0.0

        if stress_history.shape != w_history.shape:
            logger.error(f"断裂能估算失败：应力阵列 {stress_history.shape} 与裂宽阵列 {w_history.shape} 维度不匹配。")
            return 0.0

        w_diff = np.diff(w_history)
        avg_stress = (stress_history[:-1] + stress_history[1:]) / 2.0

        # 滤除闭合/噪声引起的负增量
        valid_idx = w_diff > 0
        if not np.any(valid_idx):
            return 0.0

        g_f = np.sum(avg_stress[valid_idx] * w_diff[valid_idx])
        return float(g_f)