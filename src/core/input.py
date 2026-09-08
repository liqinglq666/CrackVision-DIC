from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_bridge import CrackVisionNcorrH5Loader
from .io_ncorr import NcorrLoader
from .models import FrameData


H5_SUFFIXES = {".h5", ".hdf5"}
MAT_SUFFIXES = {".mat"}
FILE_DIALOG_FILTER = (
    "Ncorr data (*.h5 *.hdf5 *.mat);;"
    "CrackVision-Ncorr H5 (*.h5 *.hdf5);;"
    "Original Ncorr MAT (*.mat)"
)


@dataclass(frozen=True, slots=True)
class SelectedFrame:
    frame: FrameData
    dic_time_s: float
    time_source: str
    target_dic_time_s: float
    match_error_s: float


def input_kind(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in H5_SUFFIXES:
        return "CrackVision-Ncorr H5"
    if suffix in MAT_SUFFIXES:
        return "original Ncorr MAT"
    raise ValueError(f"Unsupported input type: {suffix or '<no extension>'}")


def select_nearest_frame(
    path: Path,
    config: dict[str, Any],
    target_dic_time_s: float,
) -> SelectedFrame:
    """Read only the DIC frame nearest to the requested DIC-relative time."""
    path = Path(path)
    if not np.isfinite(target_dic_time_s):
        raise ValueError("target_dic_time_s must be finite")

    suffix = path.suffix.lower()
    if suffix in H5_SUFFIXES:
        frame = CrackVisionNcorrH5Loader.read_nearest_frame(
            path, float(target_dic_time_s)
        )
        if not np.isfinite(frame.time_s):
            raise ValueError("Selected H5 frame has no usable timestamp.")
        return SelectedFrame(
            frame=frame,
            dic_time_s=float(frame.time_s),
            time_source="input_metadata",
            target_dic_time_s=float(target_dic_time_s),
            match_error_s=float(frame.time_s - target_dic_time_s),
        )

    if suffix in MAT_SUFFIXES:
        exp = config.get("experiment", {})
        fallback_ratio = float(exp.get("mm_per_pixel", 0.045))
        fallback_dt = float(exp.get("sampling_interval_s", 5.0))
        if fallback_ratio <= 0 or fallback_dt <= 0:
            raise ValueError(
                "experiment.mm_per_pixel and sampling_interval_s must be > 0"
            )

        best_frame: FrameData | None = None
        best_time = float("nan")
        best_source = ""
        best_error = float("inf")

        for frame in NcorrLoader.stream_frames(path, fallback_ratio, config):
            if np.isfinite(frame.time_s):
                dic_time = float(frame.time_s)
                source = "input_metadata"
            else:
                dic_time = float(frame.frame_id * fallback_dt)
                source = "frame_index_fallback"

            error = abs(dic_time - target_dic_time_s)
            if error < best_error:
                best_frame = frame
                best_time = dic_time
                best_source = source
                best_error = error

            if dic_time >= target_dic_time_s and error > best_error:
                break

        if best_frame is None:
            raise ValueError("No DIC frames were found in the input MAT file")

        return SelectedFrame(
            frame=best_frame,
            dic_time_s=best_time,
            time_source=best_source,
            target_dic_time_s=float(target_dic_time_s),
            match_error_s=float(best_time - target_dic_time_s),
        )

    raise ValueError(f"Unsupported input type: {suffix or '<no extension>'}")


def output_stem(path: Path) -> str:
    stem = Path(path).stem
    suffix = "_crackvision"
    return stem[: -len(suffix)] if stem.lower().endswith(suffix) else stem
