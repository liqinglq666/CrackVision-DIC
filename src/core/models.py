from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class FrameData:
    """One Ncorr frame in a strict, physically explicit representation.

    Displacements ``u_map`` and ``v_map`` are kept in Ncorr image-pixel units.
    ``pixel_size_mm`` converts displacement pixels to millimetres.
    ``dic_point_spacing_mm`` converts one DIC-grid index step to millimetres.
    """

    frame_id: int
    u_map: NDArray[np.float64]
    v_map: NDArray[np.float64]
    exx_map: NDArray[np.float64]
    eyy_map: NDArray[np.float64]
    exy_map: NDArray[np.float64]
    mask: NDArray[np.bool_]
    pixel_size_mm: float
    dic_point_spacing_mm: float
    time_s: float = float("nan")
    metadata_source: str = "unknown"
    ncorr_spacing_raw: Optional[float] = None
    dic_step_px: Optional[float] = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not np.isfinite(self.pixel_size_mm) or self.pixel_size_mm <= 0:
            raise ValueError("pixel_size_mm must be finite and > 0")
        if not np.isfinite(self.dic_point_spacing_mm) or self.dic_point_spacing_mm <= 0:
            raise ValueError("dic_point_spacing_mm must be finite and > 0")

        arrays = {
            "u_map": self.u_map,
            "v_map": self.v_map,
            "exx_map": self.exx_map,
            "eyy_map": self.eyy_map,
            "exy_map": self.exy_map,
            "mask": self.mask,
        }
        shape = self.u_map.shape
        if len(shape) != 2:
            raise ValueError("DIC fields must be 2D matrices")
        for name, arr in arrays.items():
            if arr is None:
                raise ValueError(f"{name} is required")
            if arr.ndim != 2 or arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} does not match u_map {shape}")

        if self.mask.dtype != bool:
            object.__setattr__(self, "mask", self.mask.astype(bool))
