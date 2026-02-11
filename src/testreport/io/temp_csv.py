from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class TempParseReport:
    time_source: str
    numeric_columns: list[str]
    dropped_columns: list[str]
    warnings: list[str]


def _excel_days_to_datetime(excel_days: pd.Series, *, as_utc: bool) -> pd.Series:
    """
    Excel serial date (days since 1899-12-30).
    Returns tz-aware datetime if as_utc=True (UTC), else tz-naive.
    """
    s = pd.to_numeric(excel_days, errors="coerce")
    # utc=as_utc makes it tz-aware UTC if True, else tz-naive
    return pd.to_datetime(s, unit="D", origin="1899-12-30", utc=as_utc)


def _looks_like_excel_days(s: pd.Series) -> bool:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().mean() < 0.95:
        return False
    med = x.median()
    # Excel days for modern dates are ~40000-50000
    return 20000 < med < 90000


def _pick_time_column(df: pd.DataFrame) -> str:
    for c in ["time", "Time", "timestamp", "Timestamp", "DateTime", "Datetime"]:
        if c in df.columns:
            return c
    # fallback: first column
    return str(df.columns[0])


def _find_numeric_columns(df: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # keep if mostly numeric (or at least sometimes numeric)
        if s.notna().sum() > 0:
            cols.append(c)
    return cols


def parse_temp_rh_csv(
    path: str | Path,
    *,
    tz: str = "Europe/London",
    numeric_time_is_utc: bool = True,
    time_base: str = "auto",  # "auto" | "excel_days" | "datetime"
) -> Tuple[pd.DataFrame, TempParseReport]:
    """
    Parse temperature/RH CSV.

    Supports:
      - Excel-days numeric time (e.g. 45936.5417...)
      - Datetime strings (ISO-ish)

    Returns:
      df with tz-aware 'time' column in tz,
      report with parse details.
    """
    path = Path(path)
    df = pd.read_csv(path)

    warnings: list[str] = []
    dropped: list[str] = []

    time_col = _pick_time_column(df)
    raw_time = df[time_col]

    # Decide time parsing mode
    mode = time_base.lower().strip()
    if mode not in {"auto", "excel_days", "datetime"}:
        raise ValueError("time_base must be one of: auto, excel_days, datetime")

    use_excel = False
    if mode == "excel_days":
        use_excel = True
    elif mode == "datetime":
        use_excel = False
    else:
        use_excel = _looks_like_excel_days(raw_time)

    if use_excel:
        dt = _excel_days_to_datetime(raw_time, as_utc=numeric_time_is_utc)
        if numeric_time_is_utc:
            # dt is UTC tz-aware -> convert to local tz
            dt = dt.dt.tz_convert(tz)
            time_source = f"excel_days:{time_col}:utc->tz({tz})"
        else:
            # numeric is local clock (rare)
            dt = dt.dt.tz_localize(tz)
            time_source = f"excel_days:{time_col}:localize({tz})"
    else:
        # datetime string parse
        dt = pd.to_datetime(raw_time, errors="coerce")
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(tz)
            time_source = f"datetime_strings:{time_col}:localize({tz})"
        else:
            dt = dt.dt.tz_convert(tz)
            time_source = f"datetime_strings:{time_col}:convert({tz})"

    df = df.copy()
    df["time"] = dt

    # Drop rows without time
    before = len(df)
    df = df.dropna(subset=["time"]).copy()
    after = len(df)
    if after < before:
        warnings.append(f"Dropped {before - after} rows with invalid time parsing.")

    # Numeric column detection (but don’t coerce everything yet; keep original columns)
    numeric_cols = _find_numeric_columns(df, exclude={"time", time_col})

    # Ensure sort + unique time
    df = (
        df.sort_values("time")
        .drop_duplicates(subset=["time"], keep="first")
        .reset_index(drop=True)
    )

    rep = TempParseReport(
        time_source=time_source,
        numeric_columns=numeric_cols,
        dropped_columns=dropped,
        warnings=warnings,
    )
    return df, rep
