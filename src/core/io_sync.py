import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import numpy as np
from scipy.io import loadmat

from .models import FrameData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DicMetadata:
    pixel_size_mm: float
    subset_spacing_px: float
    dic_point_spacing_mm: float
    source: str


class PipelineIO:
    """Streaming DIC frame loader for MATLAB v7 and v7.3 files."""

    SCALE_KEYS = ("pixtounits", "pixel_size_mm", "mm_per_pixel", "scale", "calibration")
    SPACING_KEYS = (
        "subsetspacing",
        "subset_spacing",
        "subset_spacing_px",
        "spacing",
        "step",
        "step_size",
    )
    QUALITY_KEYS = (
        "plot_corrcoef",
        "corrcoef",
        "correlation",
        "plot_sigma",
        "sigma",
        "quality",
        "validity",
    )

    @staticmethod
    def stream_dic_frames(
        mat_path: Path, fallback_ratio: float, config: Optional[dict] = None
    ) -> Generator[FrameData, None, None]:
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
            yield from PipelineIO._stream_h5py_engine(mat_path, fallback_ratio, config)
        else:
            yield from PipelineIO._stream_scipy_engine(mat_path, fallback_ratio, config)

    @staticmethod
    def read_metadata(mat_path: Path, fallback_ratio: float, config: Optional[dict] = None) -> DicMetadata:
        try:
            loadmat(str(mat_path), struct_as_record=False, squeeze_me=True, variable_names=["__ignore__"])
            return PipelineIO._read_scipy_metadata(mat_path, fallback_ratio, config)
        except NotImplementedError:
            return PipelineIO._read_hdf5_metadata(mat_path, fallback_ratio, config)
        except Exception:
            return PipelineIO._fallback_metadata(fallback_ratio, config)

    @staticmethod
    def _fallback_metadata(fallback_ratio: float, config: Optional[dict]) -> DicMetadata:
        experiment = (config or {}).get("experiment", {})
        spacing = float(experiment.get("dic_subset_spacing_px", 1.0))
        ratio = float(fallback_ratio)
        return DicMetadata(ratio, spacing, ratio * spacing, "config_fallback")

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            arr = np.asarray(value, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            val = float(arr.flat[0])
            return val if val > 0 else None
        except Exception:
            return None

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
    def _read_obj_number(obj: Any, names: tuple[str, ...]) -> Optional[float]:
        if obj is None:
            return None
        for name in names:
            if hasattr(obj, name):
                val = PipelineIO._as_float(getattr(obj, name))
                if val is not None:
                    return val
        for field in PipelineIO._field_names(obj):
            field_l = field.lower()
            if any(name.lower() in field_l for name in names):
                val = PipelineIO._as_float(getattr(obj, field))
                if val is not None:
                    return val
        return None

    @staticmethod
    def _read_scipy_metadata(mat_path: Path, fallback_ratio: float, config: Optional[dict]) -> DicMetadata:
        mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        data = mat.get("data_dic_save")
        if data is None:
            return PipelineIO._fallback_metadata(fallback_ratio, config)

        dispinfo = getattr(data, "dispinfo", None)
        straininfo = getattr(data, "straininfo", None)
        ratio = (
            PipelineIO._read_obj_number(dispinfo, PipelineIO.SCALE_KEYS)
            or PipelineIO._read_obj_number(straininfo, PipelineIO.SCALE_KEYS)
            or float(fallback_ratio)
        )
        spacing = (
            PipelineIO._read_obj_number(dispinfo, PipelineIO.SPACING_KEYS)
            or PipelineIO._read_obj_number(straininfo, PipelineIO.SPACING_KEYS)
            or float((config or {}).get("experiment", {}).get("dic_subset_spacing_px", 1.0))
        )
        source = "mat_metadata" if ratio != fallback_ratio or spacing != 1.0 else "config_fallback"
        return DicMetadata(float(ratio), float(spacing), float(ratio) * float(spacing), source)

    @staticmethod
    def _read_hdf5_metadata(mat_path: Path, fallback_ratio: float, config: Optional[dict]) -> DicMetadata:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("MATLAB v7.3 files require h5py. Please install h5py.") from exc

        def read_child_number(group: Any, names: tuple[str, ...]) -> Optional[float]:
            if group is None:
                return None
            keys = list(group.keys()) if hasattr(group, "keys") else []
            for name in names:
                for key in keys:
                    if key.lower() == name.lower() or name.lower() in key.lower():
                        val = PipelineIO._as_float(group[key][:])
                        if val is not None:
                            return val
            return None

        with h5py.File(str(mat_path), "r") as f:
            if "data_dic_save" not in f:
                return PipelineIO._fallback_metadata(fallback_ratio, config)
            data_dic = f["data_dic_save"]
            dispinfo = data_dic["dispinfo"] if "dispinfo" in data_dic else None
            straininfo = data_dic["straininfo"] if "straininfo" in data_dic else None
            ratio = (
                read_child_number(dispinfo, PipelineIO.SCALE_KEYS)
                or read_child_number(straininfo, PipelineIO.SCALE_KEYS)
                or float(fallback_ratio)
            )
            spacing = (
                read_child_number(dispinfo, PipelineIO.SPACING_KEYS)
                or read_child_number(straininfo, PipelineIO.SPACING_KEYS)
                or float((config or {}).get("experiment", {}).get("dic_subset_spacing_px", 1.0))
            )
        source = "mat_metadata" if ratio != fallback_ratio or spacing != 1.0 else "config_fallback"
        return DicMetadata(float(ratio), float(spacing), float(ratio) * float(spacing), source)

    @staticmethod
    def _stream_scipy_engine(
        mat_path: Path, fallback_ratio: float, config: Optional[dict]
    ) -> Generator[FrameData, None, None]:
        logger.debug("Parsing MAT with scipy engine: %s", mat_path.name)
        metadata = PipelineIO._read_scipy_metadata(mat_path, fallback_ratio, config)
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
            q_key = PipelineIO._pick_field(d_keys + s_keys, PipelineIO.QUALITY_KEYS, PipelineIO.QUALITY_KEYS)
            if not exx_key or not u_key:
                raise KeyError(f"Frame {i} is missing exx or u data. fields={d_keys}/{s_keys}")

            raw_exx = np.asarray(getattr(s_item, exx_key), dtype=np.float64)
            raw_u = np.asarray(getattr(d_item, u_key), dtype=np.float64)
            raw_v = np.asarray(getattr(d_item, v_key), dtype=np.float64) if v_key else None
            raw_q = None
            if q_key:
                q_src = d_item if hasattr(d_item, q_key) else s_item
                raw_q = np.asarray(getattr(q_src, q_key), dtype=np.float64)

            mask = np.isfinite(raw_exx) & np.isfinite(raw_u)
            if raw_v is not None:
                mask &= np.isfinite(raw_v)
            yield FrameData(
                i,
                np.nan_to_num(raw_u, nan=np.nan),
                np.nan_to_num(raw_exx, nan=0.0),
                mask,
                metadata.pixel_size_mm,
                0.0,
                v_map=np.nan_to_num(raw_v, nan=np.nan) if raw_v is not None else None,
                quality_map=np.nan_to_num(raw_q, nan=np.nan) if raw_q is not None else None,
                subset_spacing_px=metadata.subset_spacing_px,
                dic_point_spacing_mm=metadata.dic_point_spacing_mm,
                metadata_source=metadata.source,
            )

    @staticmethod
    def _stream_h5py_engine(
        mat_path: Path, fallback_ratio: float, config: Optional[dict]
    ) -> Generator[FrameData, None, None]:
        logger.debug("Parsing MAT with h5py engine: %s", mat_path.name)
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("MATLAB v7.3 files require h5py. Please install h5py.") from exc

        metadata = PipelineIO._read_hdf5_metadata(mat_path, fallback_ratio, config)
        extracted_frames: list[tuple[int, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]] = []
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

            def read_frame_group(i: int, s_grp: Any, d_grp: Any) -> None:
                s_keys = list(s_grp.keys())
                d_keys = list(d_grp.keys())
                exx_key = PipelineIO._pick_field(s_keys, ("plot_exx", "exx"), ("exx",))
                u_key = PipelineIO._pick_field(d_keys, ("plot_u", "u", "disp_u"))
                v_key = PipelineIO._pick_field(d_keys, ("plot_v", "v", "disp_v"))
                q_key = PipelineIO._pick_field(d_keys + s_keys, PipelineIO.QUALITY_KEYS, PipelineIO.QUALITY_KEYS)
                if not exx_key or not u_key:
                    raise KeyError(f"HDF5 frame field matching failed. strains={s_keys}, disp={d_keys}")
                raw_q = None
                if q_key:
                    raw_q = read_matrix(d_grp[q_key] if q_key in d_grp else s_grp[q_key])
                raw_v = read_matrix(d_grp[v_key]) if v_key else None
                extracted_frames.append((i, read_matrix(d_grp[u_key]), raw_v, read_matrix(s_grp[exx_key]), raw_q))

            if isinstance(strains_node, h5py.Group):
                s_keys = list(strains_node.keys())
                d_keys = list(disp_node.keys())
                exx_key = PipelineIO._pick_field(s_keys, ("plot_exx", "exx"), ("exx",))
                u_key = PipelineIO._pick_field(d_keys, ("plot_u", "u", "disp_u"))
                v_key = PipelineIO._pick_field(d_keys, ("plot_v", "v", "disp_v"))
                q_key = PipelineIO._pick_field(d_keys + s_keys, PipelineIO.QUALITY_KEYS, PipelineIO.QUALITY_KEYS)
                if not exx_key or not u_key:
                    raise KeyError(f"HDF5 field matching failed. strains={s_keys}, disp={d_keys}")
                exx_item = strains_node[exx_key]
                u_item = disp_node[u_key]
                v_item = disp_node[v_key] if v_key else None
                q_item = disp_node[q_key] if q_key and q_key in disp_node else strains_node[q_key] if q_key else None
                if isinstance(exx_item, h5py.Dataset) and exx_item.dtype.kind == "O":
                    exx_refs = exx_item[:].flatten()
                    u_refs = u_item[:].flatten()
                    v_refs = v_item[:].flatten() if v_item is not None else [None] * len(exx_refs)
                    q_refs = q_item[:].flatten() if q_item is not None and q_item.dtype.kind == "O" else [None] * len(exx_refs)
                    for i in range(len(exx_refs)):
                        raw_q = read_matrix(f[q_refs[i]]) if q_refs[i] is not None else None
                        raw_v = read_matrix(f[v_refs[i]]) if v_refs[i] is not None else None
                        extracted_frames.append((i, read_matrix(f[u_refs[i]]), raw_v, read_matrix(f[exx_refs[i]]), raw_q))
                else:
                    raw_q = read_matrix(q_item) if q_item is not None else None
                    raw_v = read_matrix(v_item) if v_item is not None else None
                    extracted_frames.append((0, read_matrix(u_item), raw_v, read_matrix(exx_item), raw_q))
            elif isinstance(strains_node, h5py.Dataset) and strains_node.dtype.kind == "O":
                s_refs = strains_node[:].flatten()
                d_refs = disp_node[:].flatten()
                for i in range(len(s_refs)):
                    read_frame_group(i, deref(f[s_refs[i]]), deref(f[d_refs[i]]))
            else:
                raise ValueError(f"Unsupported HDF5 DIC structure: {type(strains_node)}")

        for i, raw_u, raw_v, raw_exx, raw_q in extracted_frames:
            mask = np.isfinite(raw_exx) & np.isfinite(raw_u)
            if raw_v is not None:
                mask &= np.isfinite(raw_v)
            yield FrameData(
                i,
                np.nan_to_num(raw_u, nan=np.nan),
                np.nan_to_num(raw_exx, nan=0.0),
                mask,
                metadata.pixel_size_mm,
                0.0,
                v_map=np.nan_to_num(raw_v, nan=np.nan) if raw_v is not None else None,
                quality_map=np.nan_to_num(raw_q, nan=np.nan) if raw_q is not None else None,
                subset_spacing_px=metadata.subset_spacing_px,
                dic_point_spacing_mm=metadata.dic_point_spacing_mm,
                metadata_source=metadata.source,
            )
