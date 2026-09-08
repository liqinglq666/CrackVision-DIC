from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .io_bridge import CrackVisionNcorrH5Loader
from .io_ncorr import NcorrLoader
from .models import FrameData


H5_SUFFIXES = {".h5", ".hdf5"}
MAT_SUFFIXES = {".mat"}
SUPPORTED_SUFFIXES = H5_SUFFIXES | MAT_SUFFIXES


def input_kind(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in H5_SUFFIXES:
        return "CrackVision-Ncorr H5"
    if suffix in MAT_SUFFIXES:
        return "original Ncorr MAT"
    raise ValueError(f"Unsupported input type: {suffix or '<no extension>'}")


def stream_frames(path: Path, config: dict[str, Any]) -> Iterator[FrameData]:
    """Dispatch one supported Ncorr data file to the appropriate reader."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in H5_SUFFIXES:
        if not CrackVisionNcorrH5Loader.is_bridge_file(path):
            raise ValueError(
                "HDF5 file is not a CrackVision-Ncorr bridge. "
                "Create it with matlab/export_ncorr_to_crackvision.m."
            )
        yield from CrackVisionNcorrH5Loader.stream_frames(path)
        return

    if suffix in MAT_SUFFIXES:
        exp = config.get("experiment", {})
        fallback_ratio = float(exp.get("mm_per_pixel", 0.045))
        if fallback_ratio <= 0:
            raise ValueError("experiment.mm_per_pixel must be > 0")
        yield from NcorrLoader.stream_frames(path, fallback_ratio, config)
        return

    raise ValueError(f"Unsupported input type: {suffix or '<no extension>'}")


def output_stem(path: Path) -> str:
    """Normalize the specimen stem so bridge files do not duplicate the suffix."""
    stem = Path(path).stem
    lower = stem.lower()
    suffix = "_crackvision"
    return stem[: -len(suffix)] if lower.endswith(suffix) else stem
