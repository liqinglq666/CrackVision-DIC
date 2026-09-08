from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np

from .models import FrameData


class CrackVisionNcorrH5Loader:
    """Read the compact HDF5 bridge created by the MATLAB Ncorr exporter."""

    FORMAT = "CrackVision-Ncorr"
    FORMAT_VERSION = 1
    REQUIRED_FIELDS = ("u", "v", "exx", "eyy", "exy")

    @staticmethod
    def _decode_attr(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8", errors="replace")
        return str(value)

    @classmethod
    def is_bridge_file(cls, path: Path) -> bool:
        try:
            import h5py
        except ImportError:
            return False
        path = Path(path)
        if not path.exists() or not h5py.is_hdf5(str(path)):
            return False
        try:
            with h5py.File(str(path), "r") as f:
                return cls._decode_attr(f.attrs.get("format", "")) == cls.FORMAT
        except OSError:
            return False

    @classmethod
    def stream_frames(cls, path: Path) -> Generator[FrameData, None, None]:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("CrackVision-Ncorr HDF5 files require h5py") from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not h5py.is_hdf5(str(path)):
            raise ValueError("Input is not an HDF5 file")

        with h5py.File(str(path), "r") as f:
            fmt = cls._decode_attr(f.attrs.get("format", ""))
            version = int(f.attrs.get("format_version", 0))
            if fmt != cls.FORMAT:
                raise ValueError(
                    "HDF5 file is not a CrackVision-Ncorr bridge. "
                    "Create it with matlab/export_ncorr_to_crackvision.m."
                )
            if version != cls.FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported {cls.FORMAT} version {version}; "
                    f"expected {cls.FORMAT_VERSION}"
                )

            fields = f.get("fields")
            if fields is None:
                raise KeyError("Bridge file is missing /fields")

            missing = [name for name in cls.REQUIRED_FIELDS if name not in fields]
            if missing:
                raise KeyError(
                    f"Bridge file missing required fields: {', '.join(missing)}"
                )

            datasets = {name: fields[name] for name in cls.REQUIRED_FIELDS}
            shapes = {name: tuple(ds.shape) for name, ds in datasets.items()}
            if any(ds.ndim != 3 for ds in datasets.values()):
                raise ValueError(
                    f"Bridge fields must use [frame, y, x] layout; shapes={shapes}"
                )
            if len(set(shapes.values())) != 1:
                raise ValueError(f"Bridge field shape mismatch: {shapes}")

            n_frames, height, width = next(iter(shapes.values()))
            if n_frames <= 0 or height <= 1 or width <= 1:
                raise ValueError(
                    f"Invalid bridge field shape: {(n_frames, height, width)}"
                )

            pixel_size_mm = cls._positive_attr(f, "pixel_size_mm")
            dic_step_px = cls._positive_attr(f, "dic_step_px")
            dic_point_spacing_mm = float(
                f.attrs.get("dic_point_spacing_mm", np.nan)
            )
            if (
                not np.isfinite(dic_point_spacing_mm)
                or dic_point_spacing_mm <= 0
            ):
                dic_point_spacing_mm = pixel_size_mm * dic_step_px

            raw_spacing_value = float(
                f.attrs.get("ncorr_spacing_raw", np.nan)
            )
            ncorr_spacing_raw: Optional[float] = (
                raw_spacing_value if np.isfinite(raw_spacing_value) else None
            )

            coordinate = cls._decode_attr(
                f.attrs.get("coordinate_system", "reference")
            )
            strain_measure = cls._decode_attr(
                f.attrs.get("strain_measure", "Green-Lagrange")
            )
            precision = cls._decode_attr(
                f.attrs.get("numeric_precision", "unknown")
            )
            metadata_source = (
                f"crackvision_ncorr_h5;{coordinate};"
                f"{strain_measure};{precision}"
            )

            time_values = cls._time_values(f, n_frames)
            mask_ds = fields.get("mask")
            if mask_ds is not None and tuple(mask_ds.shape) != (
                n_frames,
                height,
                width,
            ):
                raise ValueError(
                    f"Bridge mask shape {tuple(mask_ds.shape)} does not match "
                    f"{(n_frames, height, width)}"
                )

            for frame_id in range(n_frames):
                u = np.asarray(datasets["u"][frame_id], dtype=np.float64)
                v = np.asarray(datasets["v"][frame_id], dtype=np.float64)
                exx = np.asarray(datasets["exx"][frame_id], dtype=np.float64)
                eyy = np.asarray(datasets["eyy"][frame_id], dtype=np.float64)
                exy = np.asarray(datasets["exy"][frame_id], dtype=np.float64)

                cls._validate_shapes(frame_id, u, v, exx, eyy, exy)
                finite = (
                    np.isfinite(u)
                    & np.isfinite(v)
                    & np.isfinite(exx)
                    & np.isfinite(eyy)
                    & np.isfinite(exy)
                )
                mask = (
                    finite
                    if mask_ds is None
                    else np.asarray(mask_ds[frame_id], dtype=bool) & finite
                )

                yield FrameData(
                    frame_id=frame_id,
                    u_map=u,
                    v_map=v,
                    exx_map=exx,
                    eyy_map=eyy,
                    exy_map=exy,
                    mask=mask,
                    pixel_size_mm=pixel_size_mm,
                    dic_point_spacing_mm=dic_point_spacing_mm,
                    time_s=float(time_values[frame_id]),
                    metadata_source=metadata_source,
                    ncorr_spacing_raw=ncorr_spacing_raw,
                    dic_step_px=dic_step_px,
                )

    @staticmethod
    def _positive_attr(h5_file: Any, name: str) -> float:
        value = float(h5_file.attrs.get(name, np.nan))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Bridge metadata {name} must be finite and > 0")
        return value

    @staticmethod
    def _time_values(h5_file: Any, n_frames: int) -> np.ndarray:
        if "time_s" in h5_file:
            values = np.asarray(h5_file["time_s"][:], dtype=float).reshape(-1)
            if len(values) != n_frames:
                raise ValueError(
                    f"/time_s length {len(values)} != frame count {n_frames}"
                )
            return values

        dt = float(h5_file.attrs.get("sampling_interval_s", np.nan))
        t0 = float(h5_file.attrs.get("start_time_s", 0.0))
        if np.isfinite(dt) and dt > 0:
            return t0 + np.arange(n_frames, dtype=float) * dt
        return np.full(n_frames, np.nan, dtype=float)

    @staticmethod
    def _validate_shapes(frame_id: int, *arrays: np.ndarray) -> None:
        shapes = [array.shape for array in arrays]
        if any(array.ndim != 2 for array in arrays) or len(set(shapes)) != 1:
            raise ValueError(
                f"Frame {frame_id} bridge fields must be same-shape 2D matrices; "
                f"shapes={shapes}"
            )
