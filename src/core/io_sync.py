import logging
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import numpy as np
from scipy.io import loadmat

from .models import FrameData

logger = logging.getLogger(__name__)


class PipelineIO:
    """Streaming DIC frame loader for MATLAB v7 and v7.3 files."""

    @staticmethod
    def stream_dic_frames(mat_path: Path, ratio: float) -> Generator[FrameData, None, None]:
        mat_path = Path(mat_path)
        if not mat_path.exists():
            raise FileNotFoundError(f"DIC data file not found: {mat_path}")

        use_hdf5 = False
        try:
            loadmat(str(mat_path), struct_as_record=False, squeeze_me=True, variable_names=["__ignore__"])
        except NotImplementedError:
            use_hdf5 = True
        except Exception as exc:
            logger.error("Failed to probe MAT file: %s", mat_path)
            raise RuntimeError(f"MAT file is damaged or unsupported: {exc}") from exc

        if use_hdf5:
            yield from PipelineIO._stream_h5py_engine(mat_path, ratio)
        else:
            yield from PipelineIO._stream_scipy_engine(mat_path, ratio)

    @staticmethod
    def _field_names(item: Any) -> list[str]:
        names = getattr(item, "_fieldnames", None)
        if names:
            return [str(name) for name in names]
        return [name for name in dir(item) if not name.startswith("_")]

    @staticmethod
    def _pick_field(keys: Iterable[str], preferred: tuple[str, ...], contains: tuple[str, ...] = ()) -> Optional[str]:
        key_list = list(keys)
        lower_map = {k.lower(): k for k in key_list}
        for name in preferred:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        for key in key_list:
            key_l = key.lower()
            if any(token in key_l for token in contains):
                return key
        return None

    @staticmethod
    def _stream_scipy_engine(mat_path: Path, ratio: float) -> Generator[FrameData, None, None]:
        logger.debug("Parsing MAT with scipy engine: %s", mat_path.name)
        mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        if "data_dic_save" not in mat:
            raise KeyError("MAT file does not contain 'data_dic_save'.")

        data = mat["data_dic_save"]
        strains = getattr(data, "strains", None)
        displacements = getattr(data, "displacements", None)
        if strains is None or displacements is None:
            raise KeyError("MAT structure is missing 'strains' or 'displacements'.")

        strains_list = strains if isinstance(strains, (list, tuple, np.ndarray)) else [strains]
        disp_list = displacements if isinstance(displacements, (list, tuple, np.ndarray)) else [displacements]

        for i, (s_item, d_item) in enumerate(zip(strains_list, disp_list)):
            s_keys = PipelineIO._field_names(s_item)
            d_keys = PipelineIO._field_names(d_item)
            exx_key = PipelineIO._pick_field(s_keys, ("plot_exx", "exx"), ("exx",))
            u_key = PipelineIO._pick_field(d_keys, ("plot_u", "u", "disp_u"))
            v_key = PipelineIO._pick_field(d_keys, ("plot_v", "v", "disp_v"))
            if not exx_key or not u_key:
                raise KeyError(f"Frame {i} is missing exx or u data. fields={d_keys}/{s_keys}")

            raw_exx = np.asarray(getattr(s_item, exx_key), dtype=np.float64)
            raw_u = np.asarray(getattr(d_item, u_key), dtype=np.float64)
            raw_v = np.asarray(getattr(d_item, v_key), dtype=np.float64) if v_key else None

            mask = np.isfinite(raw_exx)
            exx = np.nan_to_num(raw_exx, nan=0.0)
            u = np.nan_to_num(raw_u, nan=np.nan)
            v = np.nan_to_num(raw_v, nan=np.nan) if raw_v is not None else None

            yield FrameData(i, u, exx, mask, ratio, 0.0, v_map=v)

    @staticmethod
    def _stream_h5py_engine(mat_path: Path, ratio: float) -> Generator[FrameData, None, None]:
        logger.debug("Parsing MAT with h5py engine: %s", mat_path.name)
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("MATLAB v7.3 files require h5py. Please install h5py.") from exc

        extracted_frames: list[tuple[int, np.ndarray, Optional[np.ndarray], np.ndarray]] = []
        with h5py.File(str(mat_path), "r") as f:
            def deref(node: Any) -> Any:
                while isinstance(node, h5py.Dataset) and node.dtype.kind == "O" and node.size >= 1:
                    node = f[node[:].flatten()[0]]
                return node

            def read_matrix(node: Any) -> np.ndarray:
                return np.asarray(deref(node)[:], dtype=np.float64).T

            if "data_dic_save" not in f:
                raise KeyError("HDF5 file does not contain 'data_dic_save'.")

            data_dic = deref(f["data_dic_save"])
            strains_node = deref(data_dic["strains"])
            disp_node = deref(data_dic["displacements"])

            if isinstance(strains_node, h5py.Group):
                s_keys = list(strains_node.keys())
                d_keys = list(disp_node.keys())
                exx_key = PipelineIO._pick_field(s_keys, ("plot_exx", "exx"), ("exx",))
                u_key = PipelineIO._pick_field(d_keys, ("plot_u", "u", "disp_u"))
                v_key = PipelineIO._pick_field(d_keys, ("plot_v", "v", "disp_v"))
                if not exx_key or not u_key:
                    raise KeyError(f"HDF5 field matching failed. strains={s_keys}, disp={d_keys}")

                exx_item = strains_node[exx_key]
                u_item = disp_node[u_key]
                v_item = disp_node[v_key] if v_key else None

                if isinstance(exx_item, h5py.Dataset) and exx_item.dtype.kind == "O":
                    exx_refs = exx_item[:].flatten()
                    u_refs = u_item[:].flatten()
                    v_refs = v_item[:].flatten() if v_item is not None else [None] * len(exx_refs)
                    for i in range(len(exx_refs)):
                        raw_exx = read_matrix(f[exx_refs[i]])
                        raw_u = read_matrix(f[u_refs[i]])
                        raw_v = read_matrix(f[v_refs[i]]) if v_refs[i] is not None else None
                        extracted_frames.append((i, raw_u, raw_v, raw_exx))
                else:
                    raw_v = read_matrix(v_item) if v_item is not None else None
                    extracted_frames.append((0, read_matrix(u_item), raw_v, read_matrix(exx_item)))

            elif isinstance(strains_node, h5py.Dataset) and strains_node.dtype.kind == "O":
                s_refs = strains_node[:].flatten()
                d_refs = disp_node[:].flatten()
                for i in range(len(s_refs)):
                    s_grp = deref(f[s_refs[i]])
                    d_grp = deref(f[d_refs[i]])
                    s_keys = list(s_grp.keys())
                    d_keys = list(d_grp.keys())
                    exx_key = PipelineIO._pick_field(s_keys, ("plot_exx", "exx"), ("exx",))
                    u_key = PipelineIO._pick_field(d_keys, ("plot_u", "u", "disp_u"))
                    v_key = PipelineIO._pick_field(d_keys, ("plot_v", "v", "disp_v"))
                    if not exx_key or not u_key:
                        raise KeyError(f"HDF5 frame field matching failed. strains={s_keys}, disp={d_keys}")
                    raw_v = read_matrix(d_grp[v_key]) if v_key else None
                    extracted_frames.append((i, read_matrix(d_grp[u_key]), raw_v, read_matrix(s_grp[exx_key])))
            else:
                raise ValueError(f"Unsupported HDF5 DIC structure: {type(strains_node)}")

        for i, raw_u, raw_v, raw_exx in extracted_frames:
            mask = np.isfinite(raw_exx)
            exx = np.nan_to_num(raw_exx, nan=0.0)
            u = np.nan_to_num(raw_u, nan=np.nan)
            v = np.nan_to_num(raw_v, nan=np.nan) if raw_v is not None else None
            yield FrameData(i, u, exx, mask, ratio, 0.0, v_map=v)
