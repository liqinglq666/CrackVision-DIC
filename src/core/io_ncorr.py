from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import numpy as np
from scipy.io import loadmat

from .models import FrameData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NcorrMetadata:
    pixel_size_mm: float
    dic_step_px: float
    dic_point_spacing_mm: float
    source: str
    ncorr_spacing_raw: Optional[float] = None


class NcorrLoader:
    """Load Ncorr fields into the internal reference-coordinate pixel-displacement contract."""

    SCALE_KEYS = ("pixtounits", "pixel_size_mm", "mm_per_pixel", "scale", "calibration")
    SPACING_KEYS = ("subsetspacing", "subset_spacing", "spacing", "step", "step_size")
    TIME_KEYS = ("time", "time_s", "seconds", "sec", "timestamp", "frame_time", "t")

    # Reference-formatted fields are preferred because crack geometry and strains are
    # evaluated in the reference configuration. Formatted U/V are physical-unit
    # displacements after Ncorr unit conversion and are normalized back to pixels here.
    U_KEYS = ("plot_u_ref_formatted", "plot_u_dic", "plot_u", "disp_u", "u")
    V_KEYS = ("plot_v_ref_formatted", "plot_v_dic", "plot_v", "disp_v", "v")
    EXX_KEYS = ("plot_exx_ref_formatted", "plot_exx_dic", "plot_exx", "exx")
    EYY_KEYS = ("plot_eyy_ref_formatted", "plot_eyy_dic", "plot_eyy", "eyy")
    EXY_KEYS = ("plot_exy_ref_formatted", "plot_exy_dic", "plot_exy", "exy")

    @staticmethod
    def _as_float(value: Any, *, allow_zero: bool = False) -> Optional[float]:
        try:
            arr = np.asarray(value, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            val = float(arr.flat[0])
            if not np.isfinite(val):
                return None
            if allow_zero:
                return val if val >= 0 else None
            return val if val > 0 else None
        except Exception:
            return None

    @staticmethod
    def _field_names(item: Any) -> list[str]:
        names = getattr(item, "_fieldnames", None)
        if names:
            return [str(name) for name in names]
        if hasattr(item, "keys"):
            return [str(name) for name in item.keys()]
        return [name for name in dir(item) if not name.startswith("_")]

    @classmethod
    def _pick_field(
        cls,
        keys: Iterable[str],
        preferred: tuple[str, ...],
        contains: tuple[str, ...],
    ) -> Optional[str]:
        key_list = list(keys)
        lower_map = {k.lower(): k for k in key_list}
        for name in preferred:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        for key in key_list:
            key_l = key.lower()
            if "cur_formatted" in key_l:
                continue
            if any(token.lower() in key_l for token in contains):
                return key
        return None

    @classmethod
    def _read_obj_number(
        cls,
        obj: Any,
        names: tuple[str, ...],
        *,
        allow_zero: bool = False,
    ) -> Optional[float]:
        if obj is None:
            return None
        for name in names:
            if hasattr(obj, name):
                value = cls._as_float(getattr(obj, name), allow_zero=allow_zero)
                if value is not None:
                    return value
        for field in cls._field_names(obj):
            field_l = field.lower()
            if any(name.lower() == field_l or name.lower() in field_l for name in names):
                try:
                    value = cls._as_float(getattr(obj, field), allow_zero=allow_zero)
                except Exception:
                    continue
                if value is not None:
                    return value
        return None

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if isinstance(value, np.ndarray):
            return list(value.flat)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def spacing_to_step_px(
        raw_spacing: float,
        *,
        ncorr_spacing_is_gap_count: bool = True,
    ) -> float:
        raw_spacing = float(raw_spacing)
        if raw_spacing < 0 or not np.isfinite(raw_spacing):
            raise ValueError("Ncorr spacing must be finite and >= 0")
        return raw_spacing + 1.0 if ncorr_spacing_is_gap_count else raw_spacing

    @classmethod
    def _metadata_from_values(
        cls,
        ratio_from_mat: Optional[float],
        spacing_from_mat: Optional[float],
        fallback_ratio: float,
        config: Optional[dict],
    ) -> NcorrMetadata:
        exp = (config or {}).get("experiment", {})
        pixel_size = float(ratio_from_mat if ratio_from_mat is not None else fallback_ratio)
        if pixel_size <= 0:
            raise ValueError("mm_per_pixel fallback must be > 0")

        if spacing_from_mat is not None:
            native = bool(exp.get("ncorr_spacing_is_gap_count", True))
            step_px = cls.spacing_to_step_px(
                spacing_from_mat,
                ncorr_spacing_is_gap_count=native,
            )
            spacing_source = "mat_ncorr_spacing_plus_one" if native else "mat_spacing_direct"
            raw_spacing = float(spacing_from_mat)
        else:
            step_px = float(exp.get("dic_step_px", 1.0))
            if step_px <= 0:
                raise ValueError("experiment.dic_step_px must be > 0")
            spacing_source = "config_dic_step_px"
            raw_spacing = None

        ratio_source = "mat_pixtounits" if ratio_from_mat is not None else "config_mm_per_pixel"
        return NcorrMetadata(
            pixel_size_mm=pixel_size,
            dic_step_px=step_px,
            dic_point_spacing_mm=pixel_size * step_px,
            source=f"{ratio_source};{spacing_source}",
            ncorr_spacing_raw=raw_spacing,
        )

    @staticmethod
    def _field_is_formatted(name: str) -> bool:
        return "_formatted" in str(name).lower()

    @classmethod
    def _normalize_displacements(
        cls,
        u: np.ndarray,
        v: np.ndarray,
        u_field: str,
        v_field: str,
        meta: NcorrMetadata,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        u_formatted = cls._field_is_formatted(u_field)
        v_formatted = cls._field_is_formatted(v_field)
        if u_formatted != v_formatted:
            raise ValueError(
                "Ncorr U/V displacement fields use inconsistent units/configurations: "
                f"u={u_field}, v={v_field}"
            )

        if u_formatted:
            return (
                np.asarray(u, dtype=np.float64) / meta.pixel_size_mm,
                np.asarray(v, dtype=np.float64) / meta.pixel_size_mm,
                "ref_formatted_mm_to_pixel",
            )
        return (
            np.asarray(u, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
            "raw_pixel_displacement",
        )

    @classmethod
    def stream_frames(
        cls,
        mat_path: Path,
        fallback_ratio: float,
        config: Optional[dict] = None,
    ) -> Generator[FrameData, None, None]:
        mat_path = Path(mat_path)
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)

        try:
            loadmat(
                str(mat_path),
                struct_as_record=False,
                squeeze_me=True,
                variable_names=["__probe__"],
            )
            yield from cls._stream_scipy(mat_path, fallback_ratio, config)
        except NotImplementedError:
            yield from cls._stream_hdf5(mat_path, fallback_ratio, config)
        except ValueError:
            try:
                import h5py
            except ImportError:
                raise
            if h5py.is_hdf5(str(mat_path)):
                yield from cls._stream_hdf5(mat_path, fallback_ratio, config)
            else:
                raise

    @classmethod
    def _stream_scipy(
        cls,
        mat_path: Path,
        fallback_ratio: float,
        config: Optional[dict],
    ) -> Generator[FrameData, None, None]:
        mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        data = mat.get("data_dic_save")
        if data is None:
            raise KeyError("MAT file does not contain data_dic_save")

        strains = getattr(data, "strains", None)
        displacements = getattr(data, "displacements", None)
        if strains is None or displacements is None:
            raise KeyError("data_dic_save is missing strains or displacements")

        dispinfo = getattr(data, "dispinfo", None)
        straininfo = getattr(data, "straininfo", None)
        ratio = cls._read_obj_number(dispinfo, cls.SCALE_KEYS) or cls._read_obj_number(
            straininfo, cls.SCALE_KEYS
        )
        spacing = cls._read_obj_number(dispinfo, cls.SPACING_KEYS, allow_zero=True)
        if spacing is None:
            spacing = cls._read_obj_number(straininfo, cls.SPACING_KEYS, allow_zero=True)
        meta = cls._metadata_from_values(ratio, spacing, fallback_ratio, config)

        strain_frames = cls._to_list(strains)
        disp_frames = cls._to_list(displacements)
        if len(strain_frames) != len(disp_frames):
            raise ValueError(
                f"DIC frame count mismatch: strains={len(strain_frames)}, "
                f"displacements={len(disp_frames)}"
            )

        for frame_id, (s_item, d_item) in enumerate(zip(strain_frames, disp_frames)):
            s_keys = cls._field_names(s_item)
            d_keys = cls._field_names(d_item)
            exx_key = cls._pick_field(s_keys, cls.EXX_KEYS, ("exx",))
            eyy_key = cls._pick_field(s_keys, cls.EYY_KEYS, ("eyy",))
            exy_key = cls._pick_field(s_keys, cls.EXY_KEYS, ("exy",))
            u_key = cls._pick_field(d_keys, cls.U_KEYS, ("disp_u",))
            v_key = cls._pick_field(d_keys, cls.V_KEYS, ("disp_v",))
            missing = [
                name
                for name, key in (
                    ("u", u_key),
                    ("v", v_key),
                    ("exx", exx_key),
                    ("eyy", eyy_key),
                    ("exy", exy_key),
                )
                if not key
            ]
            if missing:
                raise KeyError(
                    f"Frame {frame_id} missing required Ncorr fields: {', '.join(missing)}"
                )

            u_raw = np.asarray(getattr(d_item, u_key), dtype=np.float64)
            v_raw = np.asarray(getattr(d_item, v_key), dtype=np.float64)
            exx = np.asarray(getattr(s_item, exx_key), dtype=np.float64)
            eyy = np.asarray(getattr(s_item, eyy_key), dtype=np.float64)
            exy = np.asarray(getattr(s_item, exy_key), dtype=np.float64)
            u, v, displacement_source = cls._normalize_displacements(
                u_raw, v_raw, str(u_key), str(v_key), meta
            )
            cls._validate_shapes(frame_id, u, v, exx, eyy, exy)
            mask = (
                np.isfinite(u)
                & np.isfinite(v)
                & np.isfinite(exx)
                & np.isfinite(eyy)
                & np.isfinite(exy)
            )

            time_s = float("nan")
            for item in (d_item, s_item):
                time_s = cls._read_obj_time(item)
                if np.isfinite(time_s):
                    break

            yield FrameData(
                frame_id=frame_id,
                u_map=u,
                v_map=v,
                exx_map=exx,
                eyy_map=eyy,
                exy_map=exy,
                mask=mask,
                pixel_size_mm=meta.pixel_size_mm,
                dic_point_spacing_mm=meta.dic_point_spacing_mm,
                time_s=time_s,
                metadata_source=f"{meta.source};{displacement_source}",
                ncorr_spacing_raw=meta.ncorr_spacing_raw,
                dic_step_px=meta.dic_step_px,
            )

    @classmethod
    def _read_obj_time(cls, obj: Any) -> float:
        if obj is None:
            return float("nan")
        for name in cls.TIME_KEYS:
            if hasattr(obj, name):
                try:
                    arr = np.asarray(getattr(obj, name), dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        return float(arr.flat[0])
                except Exception:
                    pass
        return float("nan")

    @staticmethod
    def _validate_shapes(frame_id: int, *arrays: np.ndarray) -> None:
        shapes = [a.shape for a in arrays]
        if any(a.ndim != 2 for a in arrays) or len(set(shapes)) != 1:
            raise ValueError(
                f"Frame {frame_id} DIC fields must be same-shape 2D matrices; "
                f"shapes={shapes}"
            )

    @classmethod
    def _stream_hdf5(
        cls,
        mat_path: Path,
        fallback_ratio: float,
        config: Optional[dict],
    ) -> Generator[FrameData, None, None]:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("MATLAB v7.3 files require h5py") from exc

        with h5py.File(str(mat_path), "r") as f:
            if "data_dic_save" not in f:
                raise KeyError("HDF5 MAT file does not contain data_dic_save")

            def deref(node: Any) -> Any:
                while (
                    isinstance(node, h5py.Dataset)
                    and node.dtype.kind == "O"
                    and node.size == 1
                ):
                    node = f[node[:].flat[0]]
                return node

            def read_matrix(node: Any) -> np.ndarray:
                return np.asarray(deref(node)[:], dtype=np.float64).T

            def read_group_number(
                group: Any,
                names: tuple[str, ...],
                *,
                allow_zero: bool = False,
            ) -> Optional[float]:
                if group is None or not hasattr(group, "keys"):
                    return None
                for key in group.keys():
                    key_l = key.lower()
                    if any(name.lower() == key_l or name.lower() in key_l for name in names):
                        try:
                            arr = np.asarray(deref(group[key])[:], dtype=float)
                            arr = arr[np.isfinite(arr)]
                            if arr.size:
                                val = float(arr.flat[0])
                                if (allow_zero and val >= 0) or (
                                    not allow_zero and val > 0
                                ):
                                    return val
                        except Exception:
                            pass
                return None

            data = deref(f["data_dic_save"])
            dispinfo = deref(data["dispinfo"]) if "dispinfo" in data else None
            straininfo = deref(data["straininfo"]) if "straininfo" in data else None
            ratio = read_group_number(dispinfo, cls.SCALE_KEYS) or read_group_number(
                straininfo, cls.SCALE_KEYS
            )
            spacing = read_group_number(dispinfo, cls.SPACING_KEYS, allow_zero=True)
            if spacing is None:
                spacing = read_group_number(
                    straininfo, cls.SPACING_KEYS, allow_zero=True
                )
            meta = cls._metadata_from_values(ratio, spacing, fallback_ratio, config)

            strains_node = deref(data["strains"])
            disp_node = deref(data["displacements"])

            if isinstance(strains_node, h5py.Group) and isinstance(
                disp_node, h5py.Group
            ):
                keys = cls._resolve_required_keys(
                    list(strains_node.keys()), list(disp_node.keys())
                )
                exx_ds, eyy_ds, exy_ds = (
                    strains_node[keys[k]] for k in ("exx", "eyy", "exy")
                )
                u_ds, v_ds = disp_node[keys["u"]], disp_node[keys["v"]]

                if all(
                    isinstance(ds, h5py.Dataset) and ds.dtype.kind == "O"
                    for ds in (exx_ds, eyy_ds, exy_ds, u_ds, v_ds)
                ):
                    refs = {
                        name: ds[:].flatten()
                        for name, ds in (
                            ("exx", exx_ds),
                            ("eyy", eyy_ds),
                            ("exy", exy_ds),
                            ("u", u_ds),
                            ("v", v_ds),
                        )
                    }
                    counts = {len(values) for values in refs.values()}
                    if len(counts) != 1:
                        raise ValueError(
                            f"HDF5 frame count mismatch: {[len(v) for v in refs.values()]}"
                        )
                    for frame_id in range(len(refs["u"])):
                        yield cls._make_frame(
                            frame_id,
                            read_matrix(f[refs["u"][frame_id]]),
                            read_matrix(f[refs["v"][frame_id]]),
                            read_matrix(f[refs["exx"][frame_id]]),
                            read_matrix(f[refs["eyy"][frame_id]]),
                            read_matrix(f[refs["exy"][frame_id]]),
                            meta,
                            keys["u"],
                            keys["v"],
                        )
                else:
                    yield cls._make_frame(
                        0,
                        read_matrix(u_ds),
                        read_matrix(v_ds),
                        read_matrix(exx_ds),
                        read_matrix(eyy_ds),
                        read_matrix(exy_ds),
                        meta,
                        keys["u"],
                        keys["v"],
                    )
                return

            if isinstance(strains_node, h5py.Dataset) and strains_node.dtype.kind == "O":
                s_refs = strains_node[:].flatten()
                d_refs = disp_node[:].flatten()
                if len(s_refs) != len(d_refs):
                    raise ValueError("HDF5 strains/displacements frame count mismatch")
                for frame_id, (s_ref, d_ref) in enumerate(zip(s_refs, d_refs)):
                    s_group = deref(f[s_ref])
                    d_group = deref(f[d_ref])
                    keys = cls._resolve_required_keys(
                        list(s_group.keys()), list(d_group.keys())
                    )
                    yield cls._make_frame(
                        frame_id,
                        read_matrix(d_group[keys["u"]]),
                        read_matrix(d_group[keys["v"]]),
                        read_matrix(s_group[keys["exx"]]),
                        read_matrix(s_group[keys["eyy"]]),
                        read_matrix(s_group[keys["exy"]]),
                        meta,
                        keys["u"],
                        keys["v"],
                    )
                return

            raise ValueError(
                f"Unsupported HDF5 Ncorr structure: strains={type(strains_node)}, "
                f"displacements={type(disp_node)}"
            )

    @classmethod
    def _resolve_required_keys(
        cls,
        strain_keys: list[str],
        disp_keys: list[str],
    ) -> dict[str, str]:
        out = {
            "u": cls._pick_field(disp_keys, cls.U_KEYS, ("disp_u",)),
            "v": cls._pick_field(disp_keys, cls.V_KEYS, ("disp_v",)),
            "exx": cls._pick_field(strain_keys, cls.EXX_KEYS, ("exx",)),
            "eyy": cls._pick_field(strain_keys, cls.EYY_KEYS, ("eyy",)),
            "exy": cls._pick_field(strain_keys, cls.EXY_KEYS, ("exy",)),
        }
        missing = [k for k, v in out.items() if v is None]
        if missing:
            raise KeyError(
                f"Missing required Ncorr fields: {', '.join(missing)}; "
                f"strain={strain_keys}; disp={disp_keys}"
            )
        return {k: str(v) for k, v in out.items()}

    @classmethod
    def _make_frame(
        cls,
        frame_id: int,
        u: np.ndarray,
        v: np.ndarray,
        exx: np.ndarray,
        eyy: np.ndarray,
        exy: np.ndarray,
        meta: NcorrMetadata,
        u_field: str,
        v_field: str,
    ) -> FrameData:
        u, v, displacement_source = cls._normalize_displacements(
            u, v, u_field, v_field, meta
        )
        cls._validate_shapes(frame_id, u, v, exx, eyy, exy)
        mask = (
            np.isfinite(u)
            & np.isfinite(v)
            & np.isfinite(exx)
            & np.isfinite(eyy)
            & np.isfinite(exy)
        )
        return FrameData(
            frame_id=frame_id,
            u_map=u,
            v_map=v,
            exx_map=exx,
            eyy_map=eyy,
            exy_map=exy,
            mask=mask,
            pixel_size_mm=meta.pixel_size_mm,
            dic_point_spacing_mm=meta.dic_point_spacing_mm,
            metadata_source=f"{meta.source};{displacement_source}",
            ncorr_spacing_raw=meta.ncorr_spacing_raw,
            dic_step_px=meta.dic_step_px,
        )
