import logging
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

# NOTE: 模块级日志配置
logger = logging.getLogger(__name__)


# 🌟 核心升级：同时开启 slots=True 与 frozen=True。
# slots 负责榨干内存开销，frozen 负责从语法层面上阻断任何下游对物理场源数据的非法篡改。
@dataclass(slots=True, frozen=True)
class FrameData:
    """
    数据层：单帧统一标准数据结构。
    用于在 IO 层、物理引擎层和统计层之间安全地流转物理场与宏观力学信号。
    """
    frame_id: int
    u_map: NDArray[np.float64]    # X 轴位移场矩阵 (双精度)
    exx_map: NDArray[np.float64]  # X 轴主应变场矩阵 (双精度)
    mask: NDArray[np.bool_]       # 试件有效区域掩码 (布尔型)
    ratio: float                  # 空间标定系数 (mm/pixel)
    time_s: float                 # 当前帧绝对或相对时间戳

    # 宏观力学信号 (允许为空，以兼容只有单源 DIC 图像而无 MTS 数据的回溯测试)
    load_n: Optional[float] = None
    stress_mpa: Optional[float] = None

    def __post_init__(self) -> None:
        """
        防御性编程：在对象实例化的瞬间对物理域和边界条件进行硬阻断，
        严防下游发生难以溯源的 Numpy Broadcasting 报错。
        """
        # 1. 物理标量越界防御
        if self.frame_id < 0:
            raise ValueError(f"非法 Frame ID: {self.frame_id}。帧序号必须为非负整数。")
        if self.ratio <= 0.0:
            raise ValueError(f"非法 Ratio: {self.ratio}。像素换算比必须严格大于零。")

        # 2. 矩阵空值防御
        if self.u_map is None or self.exx_map is None or self.mask is None:
            raise ValueError(f"Frame {self.frame_id} 拒绝构造：核心物理场矩阵不可为 None。")

        # 3. 空间拓扑一致性防御 (新增严格的 2D 校验)
        if self.u_map.ndim != 2:
            raise ValueError(f"Frame {self.frame_id} 空间拓扑异常：物理场必须是严格的二维矩阵 (当前为 {self.u_map.ndim}D)。")

        shape_u = self.u_map.shape
        shape_exx = self.exx_map.shape
        shape_mask = self.mask.shape

        if not (shape_u == shape_exx == shape_mask):
            logger.error(f"矩阵维度失配 -> u_map: {shape_u}, exx_map: {shape_exx}, mask: {shape_mask}")
            raise ValueError(
                f"Frame {self.frame_id} 空间拓扑异常：位移场、应变场和掩码的矩阵维度必须绝对对齐。"
            )

        # 4. 类型漂移矫正
        if self.mask.dtype != bool:
            # NOTE: 由于启用了 frozen=True，常规的 self.mask = xxx 会触发 FrozenInstanceError。
            # 必须调用底层的 object.__setattr__ 来进行强行内存重写。
            object.__setattr__(self, 'mask', self.mask.astype(bool))