import csv
import io
import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class EvolutionAnalyzer:
    """Synchronize DIC frame metrics with MTS mechanical time-series data."""

    TIME_KEYS = ("time", "sec", "second", "时间", "秒")
    FORCE_KEYS = ("load", "force", "kn", "载荷", "负荷", "力")
    DISP_KEYS = ("disp", "displacement", "extension", "elongation", "位移", "伸长", "横梁")

    def __init__(self, config: dict, mts_path: Path) -> None:
        self.config = config
        self.mts_path = Path(mts_path)
        experiment = self.config.get("experiment", {})
        sync = self.config.get("sync", {})
        self.area_mm2 = float(experiment.get("cross_section_area_mm2", 100.0))
        self.gauge_length_mm = float(experiment.get("gauge_length_mm", 80.0))
        self.min_overlap_fraction = float(sync.get("min_overlap_fraction", 0.6))
        self.max_missing_fraction = float(sync.get("max_missing_fraction", 0.05))
        self.override_dic_strain = bool(sync.get("override_dic_strain_with_mts", False))
        if self.area_mm2 <= 0:
            raise ValueError("cross_section_area_mm2 must be greater than zero.")
        if self.gauge_length_mm <= 0:
            raise ValueError("gauge_length_mm must be greater than zero.")

    def _decode_file(self) -> str:
        raw = self.mts_path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "cp1252"):
            try:
                return raw.decode(encoding).replace('"', "").replace("'", "")
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode MTS file: {self.mts_path}")

    @staticmethod
    def _contains_any(text: str, keys: tuple[str, ...]) -> bool:
        text_l = text.lower()
        return any(key.lower() in text_l for key in keys)

    def _locate_header(self, lines: List[str]) -> int:
        for i, line in enumerate(lines):
            if self._contains_any(line, self.TIME_KEYS) and (
                self._contains_any(line, self.FORCE_KEYS) or self._contains_any(line, self.DISP_KEYS)
            ):
                return i
        raise ValueError("No valid MTS header row was found.")

    @staticmethod
    def _detect_separator(header_line: str) -> str:
        try:
            dialect = csv.Sniffer().sniff(header_line, delimiters=",\t;")
            return dialect.delimiter
        except csv.Error:
            return "\t" if "\t" in header_line else ","

    @staticmethod
    def _token_match(col: str, keys: tuple[str, ...], exact_units: tuple[str, ...] = ()) -> bool:
        col_l = col.lower().strip()
        if col_l in exact_units:
            return True
        tokens = [t for t in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", col_l) if t]
        return any(key.lower() in tokens or key.lower() in col_l for key in keys)

    @classmethod
    def _pick_column(
        cls, columns: list[str], keys: tuple[str, ...], exclude: tuple[str, ...] = (), exact_units: tuple[str, ...] = ()
    ) -> Optional[str]:
        candidates = []
        for col in columns:
            col_l = col.lower()
            if exclude and any(token.lower() in col_l for token in exclude):
                continue
            if cls._token_match(col, keys, exact_units):
                candidates.append(col)
        if not candidates:
            return None
        candidates.sort(key=lambda c: (len(c), c))
        return candidates[0]

    def _smart_read_mts(self) -> pd.DataFrame:
        content = self._decode_file()
        lines = [line for line in content.splitlines() if line.strip()]
        header_idx = self._locate_header(lines)
        sep = self._detect_separator(lines[header_idx])

        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), sep=sep, on_bad_lines="skip")
        if df.empty:
            raise ValueError("MTS file contains no data rows.")

        df.columns = [str(col).strip() for col in df.columns]
        if df.iloc[0].astype(str).str.contains(r"mm|sec|kn|mpa| n$|^n$", case=False, regex=True).any():
            df = df.drop(0).reset_index(drop=True)

        columns = list(df.columns)
        time_col = self._pick_column(columns, self.TIME_KEYS, exact_units=("s", "sec"))
        force_col = self._pick_column(
            columns, self.FORCE_KEYS, exclude=("time", "时间"), exact_units=("n", "kn")
        )
        disp_col = self._pick_column(
            columns, self.DISP_KEYS, exclude=("load", "force", "载荷", "负荷"), exact_units=("mm",)
        )
        if not time_col or not force_col:
            raise ValueError(f"Could not identify MTS time/force columns. columns={columns}")

        clean = pd.DataFrame()
        clean["Time_s"] = pd.to_numeric(df[time_col], errors="coerce")
        clean["Force_N"] = pd.to_numeric(df[force_col], errors="coerce")
        clean["Disp_mm"] = pd.to_numeric(df[disp_col], errors="coerce") if disp_col else np.nan

        clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=["Time_s", "Force_N"])
        if clean.empty:
            raise ValueError("MTS data contains no valid time/force samples.")

        force = np.abs(clean["Force_N"].to_numpy(dtype=float))
        if "kn" in force_col.lower() or np.nanmax(force) < 100.0:
            force = force * 1000.0
        clean["Force_N"] = force
        clean["Stress_MPa"] = force / self.area_mm2
        return clean.sort_values("Time_s").drop_duplicates("Time_s").reset_index(drop=True)

    def _validate_overlap(self, dic_times: np.ndarray, mts_times: np.ndarray) -> None:
        dic_min, dic_max = float(np.nanmin(dic_times)), float(np.nanmax(dic_times))
        mts_min, mts_max = float(np.nanmin(mts_times)), float(np.nanmax(mts_times))
        overlap = min(dic_max, mts_max) - max(dic_min, mts_min)
        dic_span = max(dic_max - dic_min, 1e-12)
        overlap_fraction = overlap / dic_span
        if overlap <= 0:
            raise ValueError(
                f"MTS/DIC time ranges do not overlap. DIC={dic_min:.3f}-{dic_max:.3f}s, "
                f"MTS={mts_min:.3f}-{mts_max:.3f}s."
            )
        if overlap_fraction < self.min_overlap_fraction:
            raise ValueError(
                f"MTS/DIC overlap is too small ({overlap_fraction:.1%}); "
                f"required >= {self.min_overlap_fraction:.1%}."
            )

    def synchronize(self, df_dic: pd.DataFrame) -> pd.DataFrame:
        if "Time_s" not in df_dic.columns:
            raise KeyError("DIC dataframe must contain Time_s.")

        df_mts = self._smart_read_mts()
        mts_times = df_mts["Time_s"].to_numpy(dtype=float)
        if mts_times.size < 2:
            raise ValueError("MTS data needs at least two unique time samples for interpolation.")

        dic_times = df_dic["Time_s"].to_numpy(dtype=float)
        self._validate_overlap(dic_times, mts_times)

        df_sync = df_dic.copy()
        for col in ("Stress_MPa", "Disp_mm", "Force_N"):
            if col in df_mts.columns:
                values = df_mts[col].to_numpy(dtype=float)
                valid = np.isfinite(values)
                if np.count_nonzero(valid) >= 2:
                    interp = interp1d(
                        mts_times[valid],
                        values[valid],
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    df_sync[col] = interp(dic_times)

        missing = float(df_sync["Stress_MPa"].isna().mean()) if "Stress_MPa" in df_sync else 1.0
        if missing > self.max_missing_fraction:
            raise ValueError(
                f"MTS interpolation produced too many missing samples ({missing:.1%}); "
                f"required <= {self.max_missing_fraction:.1%}."
            )

        if "Disp_mm" in df_sync.columns:
            df_sync["MTS_Strain"] = np.abs(df_sync["Disp_mm"]) / self.gauge_length_mm
            if self.override_dic_strain:
                df_sync["global_strain"] = df_sync["MTS_Strain"]
                df_sync["strain_source"] = "mts_displacement"
            else:
                df_sync["strain_source"] = df_sync.get("strain_source", "dic_virtual_extensometer")

        df_sync["sync_status"] = "synced"
        return df_sync
