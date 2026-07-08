import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FrameData:
    """Validated data for one DIC frame."""

    frame_id: int
    u_map: NDArray[np.float64]
    exx_map: NDArray[np.float64]
    mask: NDArray[np.bool_]
    ratio: float
    time_s: float
    v_map: Optional[NDArray[np.float64]] = None
    load_n: Optional[float] = None
    stress_mpa: Optional[float] = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError(f"Invalid frame_id: {self.frame_id}.")
        if self.ratio <= 0.0:
            raise ValueError(f"Invalid ratio: {self.ratio}. It must be greater than zero.")
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
            raise ValueError(
                f"Frame {self.frame_id} matrices must have identical shapes."
            )

        if self.v_map is not None:
            if self.v_map.ndim != 2:
                raise ValueError(f"Frame {self.frame_id} v_map must be a 2D matrix.")
            if self.v_map.shape != shape_u:
                raise ValueError(
                    f"Frame {self.frame_id} v_map shape {self.v_map.shape} "
                    f"does not match u_map {shape_u}."
                )

        if self.mask.dtype != bool:
            object.__setattr__(self, "mask", self.mask.astype(bool))
