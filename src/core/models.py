import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FrameData:
    """Validated DIC data and spatial metadata for one frame.

    ratio/pixel_size_mm converts Ncorr displacement values from image pixels to mm.
    dic_point_spacing_mm converts one DIC grid step to mm for lengths/search radii.
    These are intentionally separate because Ncorr fields are sampled on subset
    centers, not on every raw camera pixel.
    """

    frame_id: int
    u_map: NDArray[np.float64]
    exx_map: NDArray[np.float64]
    mask: NDArray[np.bool_]
    ratio: float
    time_s: float
    v_map: Optional[NDArray[np.float64]] = None
    quality_map: Optional[NDArray[np.float64]] = None
    subset_spacing_px: float = 1.0
    dic_point_spacing_mm: Optional[float] = None
    metadata_source: str = "fallback"
    load_n: Optional[float] = None
    stress_mpa: Optional[float] = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError(f"Invalid frame_id: {self.frame_id}.")
        if self.ratio <= 0.0:
            raise ValueError(f"Invalid ratio: {self.ratio}. It must be greater than zero.")
        if self.subset_spacing_px <= 0.0:
            raise ValueError(
                f"Invalid subset_spacing_px: {self.subset_spacing_px}. It must be greater than zero."
            )
        if self.u_map is None or self.exx_map is None or self.mask is None:
            raise ValueError(f"Frame {self.frame_id} is missing u_map, exx_map, or mask.")
        if self.u_map.ndim != 2 or self.exx_map.ndim != 2 or self.mask.ndim != 2:
            raise ValueError(f"Frame {self.frame_id} fields must all be 2D matrices.")

        shape_u = self.u_map.shape
        if not (shape_u == self.exx_map.shape == self.mask.shape):
            logger.error(
                "Matrix shape mismatch -> u_map: %s, exx_map: %s, mask: %s",
                shape_u,
                self.exx_map.shape,
                self.mask.shape,
            )
            raise ValueError(f"Frame {self.frame_id} matrices must have identical shapes.")

        if self.v_map is not None:
            self._validate_optional_map(self.v_map, shape_u, "v_map")
        if self.quality_map is not None:
            self._validate_optional_map(self.quality_map, shape_u, "quality_map")

        if self.mask.dtype != bool:
            object.__setattr__(self, "mask", self.mask.astype(bool))
        if self.dic_point_spacing_mm is None:
            object.__setattr__(
                self, "dic_point_spacing_mm", float(self.ratio) * float(self.subset_spacing_px)
            )
        elif self.dic_point_spacing_mm <= 0.0:
            raise ValueError(
                f"Invalid dic_point_spacing_mm: {self.dic_point_spacing_mm}. It must be greater than zero."
            )

    def _validate_optional_map(
        self, arr: NDArray[np.float64], expected_shape: tuple[int, int], name: str
    ) -> None:
        if arr.ndim != 2:
            raise ValueError(f"Frame {self.frame_id} {name} must be a 2D matrix.")
        if arr.shape != expected_shape:
            raise ValueError(
                f"Frame {self.frame_id} {name} shape {arr.shape} does not match u_map {expected_shape}."
            )
