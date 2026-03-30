import logging
import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import Generator, Any

from .models import FrameData

logger = logging.getLogger(__name__)


class PipelineIO:
    """
    流式 I/O 层：负责底层异构 MAT 数据矩阵的极速拆解与流式分发。
    """

    @staticmethod
    def stream_dic_frames(mat_path: Path, ratio: float) -> Generator[FrameData, None, None]:
        """
        双引擎流式加载 DIC 数据 (Dual-Engine Lazy Loading)。
        内置自适应结构体探测，兼容 MATLAB v7 (-v7) 与 v7.3 (-v7.3 HDF5) 格式。

        NOTE: 为防止调用方在迭代过程中引发的业务异常被底层的 try-except 错误捕获，
        本方法严格将数据读取/提取 与 yield (挂起产出) 的作用域分离。
        """
        if not mat_path.exists():
            raise FileNotFoundError(f"未找到 DIC 数据文件: {mat_path}")

        # ---------------------------------------------------------
        # 探测引擎类型
        # ---------------------------------------------------------
        use_hdf5 = False
        try:
            # 仅尝试轻量级加载，探测格式
            loadmat(str(mat_path), struct_as_record=False, squeeze_me=True, variable_names=['__ignore__'])
        except NotImplementedError:
            use_hdf5 = True
        except Exception as e:
            # 其他真实的文件损坏或格式错误
            logger.error(f"MAT 文件底层读取探测失败: {mat_path}")
            raise RuntimeError(f"MAT 文件损坏或格式不支持: {e}") from e

        # ---------------------------------------------------------
        # 分支流式调度
        # ---------------------------------------------------------
        if not use_hdf5:
            yield from PipelineIO._stream_scipy_engine(mat_path, ratio)
        else:
            yield from PipelineIO._stream_h5py_engine(mat_path, ratio)

    @staticmethod
    def _stream_scipy_engine(mat_path: Path, ratio: float) -> Generator[FrameData, None, None]:
        """Engine A: 原生 Scipy 引擎 (适用于 -v7 及更早格式)"""
        logger.debug(f"正在使用 Scipy (v7) 引擎解析: {mat_path.name}")
        mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)

        if 'data_dic_save' not in mat:
            raise KeyError("MAT 文件中缺少 'data_dic_save' 根节点")

        data = mat['data_dic_save']
        strains = getattr(data, 'strains', None)
        displacements = getattr(data, 'displacements', None)

        if strains is None or displacements is None:
            raise KeyError("MAT 结构体中缺失 'strains' 或 'displacements' 字段")

        # 统一处理为可迭代列表
        strains_list = strains if isinstance(strains, (list, tuple, np.ndarray)) else [strains]
        disp_list = displacements if isinstance(displacements, (list, tuple, np.ndarray)) else [displacements]

        for i, (s_item, d_item) in enumerate(zip(strains_list, disp_list)):
            # NOTE: 获取对象字段，兼容不同版本的 scipy.io.mat_struct
            s_keys = getattr(s_item, '_fieldnames', dir(s_item))
            d_keys = getattr(d_item, '_fieldnames', dir(d_item))

            exx_key = next((k for k in s_keys if 'exx' in k.lower()), 'plot_exx')
            u_key = next((k for k in d_keys if 'u' in k.lower()), 'plot_u')

            try:
                raw_u = getattr(d_item, u_key)
                raw_exx = getattr(s_item, exx_key)
            except AttributeError as e:
                raise KeyError(f"第 {i} 帧缺少指定的位移或应变矩阵 ({u_key}, {exx_key})") from e

            # 矩阵清洗与构建
            u = np.nan_to_num(raw_u)
            mask = ~np.isnan(raw_exx)
            exx = np.nan_to_num(raw_exx)

            # NOTE: yield 不在任何内部 try-except 中，避免错误捕获调用方的异常
            yield FrameData(frame_id=i, u_map=u, exx_map=exx, mask=mask, ratio=ratio, time_s=0.0)

    @staticmethod
    def _stream_h5py_engine(mat_path: Path, ratio: float) -> Generator[FrameData, None, None]:
        """Engine B: h5py 极速流式引擎 (适用于 v7.3 HDF5 格式)"""
        logger.debug(f"正在使用 H5py (v7.3) 引擎解析: {mat_path.name}")
        try:
            import h5py
        except ImportError as e:
            raise ImportError("检测到 MATLAB v7.3 格式，需要 h5py 库支持。请执行: pip install h5py") from e

        with h5py.File(str(mat_path), 'r') as f:
            def _deref(node: Any) -> Any:
                """辅助方法：递归剥除 HDF5 中的 Object Reference 伪装"""
                while isinstance(node, h5py.Dataset) and node.dtype.kind == 'O' and node.size >= 1:
                    ref = node[:].flatten()[0]
                    node = f[ref]
                return node

            if 'data_dic_save' not in f:
                raise KeyError("HDF5 中缺少 'data_dic_save' 根节点")

            data_dic = _deref(f['data_dic_save'])
            strains_node = _deref(data_dic['strains'])
            disp_node = _deref(data_dic['displacements'])

            # ---------------------------------------------------------
            # 提取阶段：将 HDF5 指针解析为内存矩阵，脱离 with h5py 锁
            # ---------------------------------------------------------
            extracted_frames = []

            # 形态 1：结构体属性按文件夹拆分存放 (Group)
            if isinstance(strains_node, h5py.Group):
                s_keys = list(strains_node.keys())
                d_keys = list(disp_node.keys())

                exx_key = next((k for k in s_keys if 'exx' in k.lower()), None)
                u_key = next((k for k in d_keys if 'u' in k.lower()), None)

                if not exx_key or not u_key:
                    raise KeyError(f"HDF5 字段匹配失败。strains 可用: {s_keys}, disp 可用: {d_keys}")

                exx_item = strains_node[exx_key]
                u_item = disp_node[u_key]

                if isinstance(exx_item, h5py.Dataset) and exx_item.dtype.kind == 'O':
                    exx_refs = exx_item[:].flatten()
                    u_refs = u_item[:].flatten()

                    for i in range(len(exx_refs)):
                        # MATLAB 矩阵存入 HDF5 默认是 Fortran 序，需转置 (.T) 回 C 序
                        raw_exx = f[exx_refs[i]][:].T
                        raw_u = f[u_refs[i]][:].T
                        extracted_frames.append((i, raw_u, raw_exx))
                else:
                    raw_exx = exx_item[:].T
                    raw_u = u_item[:].T
                    extracted_frames.append((0, raw_u, raw_exx))

            # 形态 2：结构体按帧被存成指针列表 (Dataset of References)
            elif isinstance(strains_node, h5py.Dataset) and strains_node.dtype.kind == 'O':
                s_refs = strains_node[:].flatten()
                d_refs = disp_node[:].flatten()

                for i in range(len(s_refs)):
                    s_grp = _deref(f[s_refs[i]])
                    d_grp = _deref(f[d_refs[i]])

                    s_keys = list(s_grp.keys())
                    d_keys = list(d_grp.keys())

                    exx_key = next((k for k in s_keys if 'exx' in k.lower()), None)
                    u_key = next((k for k in d_keys if 'u' in k.lower()), None)

                    if not exx_key or not u_key:
                        raise KeyError(f"HDF5 单帧内匹配字段失败。strains: {s_keys}, disp: {d_keys}")

                    raw_exx = _deref(s_grp[exx_key])[:].T
                    raw_u = _deref(d_grp[u_key])[:].T
                    extracted_frames.append((i, raw_u, raw_exx))
            else:
                raise ValueError(f"无法识别 HDF5 结构体形态: {type(strains_node)}")

        # ---------------------------------------------------------
        # 分发阶段：脱离了 h5py.File 上下文，安全地流式产出
        # ---------------------------------------------------------
        for i, raw_u, raw_exx in extracted_frames:
            u = np.nan_to_num(raw_u)
            mask = ~np.isnan(raw_exx)
            exx = np.nan_to_num(raw_exx)

            yield FrameData(frame_id=i, u_map=u, exx_map=exx, mask=mask, ratio=ratio, time_s=0.0)