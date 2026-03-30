import logging
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from typing import Dict

# NOTE: 统一使用模块级 logger 替代静默处理
logger = logging.getLogger(__name__)


class EvolutionSegmenter:
    """
    基于物理演化特征的 ECC 拉伸全过程时序分段引擎。
    通过微观特征（裂宽、条数）的一阶导数突变，精准识别材料的本构生命周期。
    """

    def __init__(self,
                 smoothing_sigma: float = 2.0,
                 loc_multiplier: float = 3.0,
                 sat_threshold_ratio: float = 0.1) -> None:
        """
        初始化分段引擎，注入物理超参数。

        Args:
            smoothing_sigma: 高斯滤波的 sigma 值。消除 DIC 亚像素离散跳跃导致的导数毛刺。
            loc_multiplier: 局部化突变判定倍率。当裂宽扩展速率超过基线此倍数时，判定为主裂缝开启。
            sat_threshold_ratio: 饱和衰退阈值。当裂缝萌生速率降至峰值速率的该比例下时，判定为多缝饱和。
        """
        if smoothing_sigma <= 0:
            raise ValueError("高斯滤波参数 smoothing_sigma 必须严格大于 0。")

        self.sigma = smoothing_sigma
        self.loc_multiplier = loc_multiplier
        self.sat_threshold_ratio = sat_threshold_ratio

    def _compute_derivatives(self, strain: np.ndarray, feature: np.ndarray) -> np.ndarray:
        """
        计算平滑后的物理特征关于全局应变的导数 (dFeature/dStrain)。
        使用前向差分并进行末位补零，严密防范真实试验机数据中出现的应变停滞奇异性。
        """
        smoothed_feature = gaussian_filter1d(feature, self.sigma)

        # 使用一阶前向差分
        d_strain = np.diff(strain)
        d_feature = np.diff(smoothed_feature)

        # 🛡️ 防御：屏蔽传感器采样精度极限导致的应变停滞 (d_strain == 0)
        # 将其限制在微小的正数安全阈值上，防止除以零引发 RuntimeWarning 和 inf
        d_strain_safe = np.where(d_strain <= 1e-6, 1e-6, d_strain)

        derivative = d_feature / d_strain_safe

        # 补齐数组长度以对齐原始 DataFrame 的 Index
        return np.append(derivative, 0.0)

    def segment(self, df: pd.DataFrame,
                col_strain: str = 'Strain_pct',
                col_stress: str = 'Stress_MPa',
                col_count: str = 'crack_count',
                col_wmax: str = 'W_max_um') -> Dict[str, int]:
        """
        自动寻找特征突变点，将拉伸过程划分为四个核心物理阶段。
        提供灵活的列名映射机制，无缝对接上游的宽表输出。

        Returns:
            Dict[str, int]: 各阶段起始点在 DataFrame 中的整数索引 (iloc)。未触发的阶段返回 -1。
        """
        stages = {
            "01_Elastic": 0,
            "02_First_Cracking": -1,
            "03_Saturation": -1,
            "04_Localization": -1
        }

        required_cols = [col_strain, col_stress, col_count, col_wmax]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"演化分段失败，输入数据帧缺失映射的物理特征列: {missing_cols}")

        if len(df) < 10:
            logger.warning("数据帧总长度极短 (< 10)，无法执行微观演化阶段的数值划分。")
            return stages

        strain = df[col_strain].to_numpy(copy=True)
        stress = df[col_stress].to_numpy(copy=True)
        count = df[col_count].to_numpy(copy=True)
        max_w = df[col_wmax].to_numpy(copy=True)

        # ==========================================
        # 阶段 A: 寻找初裂点 (First Cracking)
        # 物理判据：微观检测管线首次确认裂缝存在
        # ==========================================
        crack_indices = np.where(count > 0)[0]
        if crack_indices.size > 0:
            stages["02_First_Cracking"] = int(crack_indices[0])
        else:
            logger.info("试件在全生命周期内未触发开裂判据 (可能为纯弹性或极脆断)。")
            return stages

        # ==========================================
        # 阶段 B: 寻找局部化破坏点 (Localization / Failure)
        # 物理判据：裂宽扩展率 dWmax/dStrain 发生非线性突涨，或宏观应力达到 UTS
        # ==========================================
        peak_idx = int(np.argmax(stress))
        dw_dE = self._compute_derivatives(strain, max_w)

        # NOTE: 以宏观峰值点前序窗口作为基线搜索域，防止晚期数据污染
        search_start = max(0, peak_idx - 10)
        search_domain = dw_dE[search_start:]

        if len(search_domain) > 5:
            baseline_rate = float(np.nanmean(search_domain))

            with np.errstate(invalid='ignore'):
                mutation_candidates = np.where(search_domain > self.loc_multiplier * baseline_rate)[0]

            if mutation_candidates.size > 0:
                stages["04_Localization"] = int(search_start + mutation_candidates[0])
            else:
                stages["04_Localization"] = peak_idx
        else:
            stages["04_Localization"] = peak_idx

        # 🛡️ 强制物理法则：应变局部化绝不可能发生在材料初裂之前。如有，必为计算或噪点所致。
        if stages["04_Localization"] <= stages["02_First_Cracking"]:
            logger.debug("修正：检测到局部化时序倒置，强制采用宏观应力峰值点 (UTS) 作为局部化锚点。")
            stages["04_Localization"] = peak_idx

        # ==========================================
        # 阶段 C: 寻找裂缝饱和点 (Crack Saturation)
        # 物理判据：在应变硬化区间内，新裂缝萌生率 dN/dE 趋于枯竭
        # ==========================================
        idx_start = stages["02_First_Cracking"]
        idx_end = stages["04_Localization"]

        if idx_end > idx_start:
            dN_dE = self._compute_derivatives(strain, count)
            dN_dE_hardening = dN_dE[idx_start:idx_end]

            if dN_dE_hardening.size > 0:
                peak_rate = float(np.nanmax(dN_dE_hardening))

                with np.errstate(invalid='ignore'):
                    saturated_candidates = np.where(dN_dE_hardening < self.sat_threshold_ratio * peak_rate)[0]

                if saturated_candidates.size > 0:
                    stages["03_Saturation"] = int(idx_start + saturated_candidates[0])
                else:
                    # NOTE: 试件可能直至破坏都未彻底饱和。采用工程退化策略 (硬化段 80% 处)。
                    stages["03_Saturation"] = int(idx_start + 0.8 * (idx_end - idx_start))

        return stages