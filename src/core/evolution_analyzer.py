import csv
import io
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class EvolutionAnalyzer:
    """Synchronize DIC frame metrics with MTS mechanical time-series data."""

    TIME_KEYS = ("time", "sec", "second", "s", "时间", "秒")
    FORCE_KEYS = ("load", "force", "kn", "n", "载荷", "负荷", "力")
    DISP_KEYS = ("disp", "displacement", "extension", "elongation", "mm", "位移", "伸长", "横梁")

    def __init__(self, config: dict, mts_path: Path) -> None:
        self.config = config
        self.mts_path = Path(mts_path)
        experiment = self.config.get("experiment", {})
        self.area_mm2 = float(experiment.get("cross_section_area_mm2", 100.0))
        self.gauge_length_mm = float(experiment.get("gauge_length_mm", 80.0))
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
    def _pick_column(columns: list[str], keys: tuple[str, ...], exclude: tuple[str, ...] = ()) -> Optional[str]:
        for col in columns:
            col_l = col.lower()
            if exclude and any(token.lower() in col_l for token in exclude):
                continue
            if any(key.lower() in col_l for key in keys):
                return col
        return None

    def _smart_read_mts(self) -> pd.DataFrame:
        content = self._decode_file()
        lines = [line for line in content.splitlines() if line.strip()]
        header_idx = self._locate_header(lines)
        sep = self._detect_separator(lines[header_idx])

        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), sep=sep, on_bad_lines="skip")
        if df.empty:
            raise ValueError("MTS file contains no data rows.")

        df.columns = [str(col).strip() for col in df.columns]
        if df.iloc[0].astype(str).str.contains(r"mm|sec|kn|mpa|n", case=False, regex=True).any():
            df = df.drop(0).reset_index(drop=True)

        columns = list(df.columns)
        time_col = self._pick_column(columns, self.TIME_KEYS)
        force_col = self._pick_column(columns, self.FORCE_KEYS, exclude=("time", "时间"))
        disp_col = self._pick_column(columns, self.DISP_KEYS, exclude=("load", "force", "载荷", "负荷"))
        if not time_col or not force_col:
            raise ValueError(f"Could not identify MTS time/force columns. columns={columns}")

        clean = pd.DataFrame()
        clean["Time_s"] = pd.to_numeric(df[time_col], errors="coerce")
        clean["Force_N"] = pd.to_numeric(df[force_col], errors="coerce")
        if disp_col:
            clean["Disp_mm"] = pd.to_numeric(df[disp_col], errors="coerce")
        else:
            clean["Disp_mm"] = np.nan

        clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=["Time_s", "Force_N"])
        if clean.empty:
            raise ValueError("MTS data contains no valid time/force samples.")

        force = np.abs(clean["Force_N"].to_numpy(dtype=float))
        if np.nanmax(force) < 100.0:
            force = force * 1000.0
        clean["Force_N"] = force
        clean["Stress_MPa"] = force / self.area_mm2
        return clean.sort_values("Time_s").reset_index(drop=True)

    def synchronize(self, df_dic: pd.DataFrame) -> pd.DataFrame:
        if "Time_s" not in df_dic.columns:
            raise KeyError("DIC dataframe must contain Time_s.")

        df_mts = self._smart_read_mts()
        mts_times = df_mts["Time_s"].to_numpy(dtype=float)
        unique_times, unique_idx = np.unique(mts_times, return_index=True)
        df_mts = df_mts.iloc[unique_idx].reset_index(drop=True)
        if unique_times.size < 2:
            raise ValueError("MTS data needs at least two unique time samples for interpolation.")

        dic_times = df_dic["Time_s"].to_numpy(dtype=float)
        df_sync = df_dic.copy()
        for col in ("Stress_MPa", "Disp_mm", "Force_N"):
            if col in df_mts.columns:
                values = df_mts[col].to_numpy(dtype=float)
                valid = np.isfinite(values)
                if np.count_nonzero(valid) >= 2:
                    interp = interp1d(
                        unique_times[valid],
                        values[valid],
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    df_sync[col] = interp(dic_times)

        if "Disp_mm" in df_sync.columns:
            df_sync["global_strain"] = np.abs(df_sync["Disp_mm"]) / self.gauge_length_mm

        return df_sync
