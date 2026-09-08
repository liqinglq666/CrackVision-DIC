from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class MtsPeak:
    csv_path: Path
    peak_force_N: float
    peak_time_s: float
    tension_sign: int
    sample_count: int
    force_column: str
    time_column: str


_FORCE_ALIASES = {"力", "force", "load", "载荷"}
_TIME_ALIASES = {"时间", "time", "sec", "seconds"}


def _normalize_label(value: str) -> str:
    return "".join(str(value).strip().lower().split())


def _find_column(labels: list[str], aliases: set[str]) -> int | None:
    normalized = [_normalize_label(label) for label in labels]
    alias_norm = {_normalize_label(alias) for alias in aliases}
    for idx, label in enumerate(normalized):
        if label in alias_norm:
            return idx
    for idx, label in enumerate(normalized):
        if any(alias in label for alias in alias_norm):
            return idx
    return None


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to decode MTS CSV: {path}")


def read_mts_peak(csv_path: Path) -> MtsPeak:
    """Locate the peak tensile-force time in an MTS/DAQ CSV.

    For a constant specimen cross-section, peak tensile stress occurs at the
    same instant as peak tensile force, so cross-sectional area is not needed
    for DIC-frame selection.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = list(csv.reader(io.StringIO(_read_text(path))))
    header_index: int | None = None
    force_index: int | None = None
    time_index: int | None = None

    for row_index, row in enumerate(rows):
        f_idx = _find_column(row, _FORCE_ALIASES)
        t_idx = _find_column(row, _TIME_ALIASES)
        if f_idx is not None and t_idx is not None:
            header_index = row_index
            force_index = f_idx
            time_index = t_idx
            break

    if header_index is None or force_index is None or time_index is None:
        raise ValueError(
            "MTS CSV must contain force/load and time columns "
            "(for example '力' and '时间')."
        )

    force_values: list[float] = []
    time_values: list[float] = []
    max_index = max(force_index, time_index)
    for row in rows[header_index + 1 :]:
        if len(row) <= max_index:
            continue
        try:
            force = float(str(row[force_index]).strip())
            time_s = float(str(row[time_index]).strip())
        except (TypeError, ValueError):
            # Unit rows such as N/sec are intentionally skipped.
            continue
        if np.isfinite(force) and np.isfinite(time_s):
            force_values.append(force)
            time_values.append(time_s)

    if len(force_values) < 3:
        raise ValueError("MTS CSV contains fewer than 3 valid force/time samples.")

    force_arr = np.asarray(force_values, dtype=float)
    time_arr = np.asarray(time_values, dtype=float)
    if np.any(np.diff(time_arr) < 0):
        order = np.argsort(time_arr, kind="stable")
        time_arr = time_arr[order]
        force_arr = force_arr[order]

    # MTS installations may record tension as positive or negative. Infer the
    # sign from the larger excursion away from the initial force level.
    baseline = float(np.median(force_arr[: min(20, len(force_arr))]))
    positive_excursion = float(np.nanmax(force_arr) - baseline)
    negative_excursion = float(baseline - np.nanmin(force_arr))

    if positive_excursion >= negative_excursion:
        peak_idx = int(np.nanargmax(force_arr))
        sign = 1
    else:
        peak_idx = int(np.nanargmin(force_arr))
        sign = -1

    labels = rows[header_index]
    return MtsPeak(
        csv_path=path,
        peak_force_N=float(force_arr[peak_idx]),
        peak_time_s=float(time_arr[peak_idx]),
        tension_sign=sign,
        sample_count=int(len(force_arr)),
        force_column=str(labels[force_index]).strip(),
        time_column=str(labels[time_index]).strip(),
    )
