from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import pandas as pd


@dataclass(frozen=True)
class TempParseReport:
    time_source: str
    numeric_columns: list[str]
    dropped_columns: list[str]
    warnings: list[str]


def _pick_datetime_string_column(
    df: pd.DataFrame, candidates: Sequence[str]
) -> Optional[str]:
    best_col = None
    best_rate = 0.0
    for c in candidates:
        if c not in df.columns:
            continue
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        rate = parsed.notna().mean()
        if rate > best_rate:
            best_rate = rate
            best_col = c
    return best_col if best_rate >= 0.90 else None


def _pick_numeric_time_column(
    df: pd.DataFrame, candidates: Sequence[str]
) -> Optional[str]:
    best_col = None
    best_rate = 0.0
    for c in candidates:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        rate = s.notna().mean()
        if rate > best_rate:
            best_rate = rate
            best_col = c
    return best_col if best_rate >= 0.95 else None


def _convert_numeric_time(
    s: pd.Series,
    *,
    time_base: str,
    excel_origin: str = "1899-12-30",
) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")

    if time_base == "excel_days":
        return pd.to_datetime(v, unit="D", origin=excel_origin, errors="coerce")
    if time_base == "unix_seconds":
        return pd.to_datetime(v, unit="s", origin="unix", errors="coerce")
    if time_base == "unix_millis":
        return pd.to_datetime(v, unit="ms", origin="unix", errors="coerce")

    raise ValueError(f"Unsupported time_base={time_base!r}")


def _detect_numeric_columns(
    df: pd.DataFrame,
    exclude: Sequence[str],
    *,
    min_numeric_rate: float = 0.90,
) -> Tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    dropped_cols: list[str] = []

    for c in df.columns:
        if c in exclude:
            continue

        if df[c].isna().all():
            dropped_cols.append(c)
            continue

        if str(c).lower().startswith("unnamed:"):
            dropped_cols.append(c)
            continue

        series = pd.to_numeric(df[c], errors="coerce")
        rate = series.notna().mean()

        if rate >= min_numeric_rate:
            numeric_cols.append(c)
        else:
            dropped_cols.append(c)

    return numeric_cols, dropped_cols


def parse_temp_rh_csv(
    path: str | Path,
    *,
    # Timezone handling:
    tz: str | None = None,
    numeric_time_is_utc: bool = True,
    time_offset_seconds: int = 0,
    # CSV:
    encoding: str | None = None,
    min_numeric_rate: float = 0.90,
    # Numeric time handling:
    time_base: str = "excel_days",
    excel_origin: str = "1899-12-30",
    time_column: str | None = None,
) -> tuple[pd.DataFrame, TempParseReport]:
    """
    Parse a temperature/RH CSV with either:
      - datetime strings, or
      - numeric timestamps (Excel days / Unix)

    DST fix:
      - If numeric_time_is_utc=True and tz is provided, timestamps are treated as UTC then converted
        to the target tz (e.g. Europe/London), which correctly applies BST/GMT transitions.
      - If numeric_time_is_utc=False, numeric timestamps are treated as local wall time and localized directly.

    Returns:
      df: ['time'] + numeric sensor columns
      report: parse details/warnings
    """
    path = Path(path)

    sample = pd.read_csv(path, nrows=200, encoding=encoding)  # type: ignore[arg-type]

    preferred = []
    for c in sample.columns:
        cl = str(c).strip().lower()
        if cl in {"time", "timestamp", "datetime", "date_time", "date"}:
            preferred.append(c)
    candidates = preferred + [c for c in sample.columns if c not in preferred]

    if time_column is not None:
        if time_column not in sample.columns:
            raise ValueError(f"time_column={time_column!r} not found in CSV.")
        candidates = [time_column]

    dt_col = _pick_datetime_string_column(sample, candidates)
    numeric_time_col = None
    time_source = ""

    if dt_col is not None:
        time_source = f"datetime_strings:{dt_col}"
    else:
        numeric_time_col = _pick_numeric_time_column(sample, candidates)
        if numeric_time_col is None:
            raise ValueError(
                "Could not detect a time column as datetime strings or numeric time. "
                "Specify time_column=... explicitly."
            )
        time_source = (
            f"numeric_time:{numeric_time_col} ({time_base}, origin={excel_origin})"
        )

    df = pd.read_csv(path, encoding=encoding)  # type: ignore[arg-type]

    # Build time series
    if dt_col is not None:
        time = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
        exclude_time_cols = [dt_col]
    else:
        time = _convert_numeric_time(
            df[numeric_time_col], time_base=time_base, excel_origin=excel_origin
        )  # type: ignore[arg-type]
        exclude_time_cols = [numeric_time_col]  # type: ignore[list-item]

    if time.isna().mean() > 0.05:
        raise ValueError(
            f"Too many unparseable timestamps ({time.isna().mean() * 100:.1f}%). Detected: {time_source}"
        )

    # Apply timezone logic (DST-aware)
    if tz:
        if dt_col is None and numeric_time_is_utc:
            # Numeric time treated as UTC/GMT -> convert to local tz (DST handled)
            time = time.dt.tz_localize("UTC").dt.tz_convert(tz)
        else:
            # Treat as local wall-time in tz
            time = time.dt.tz_localize(tz)

    if time_offset_seconds:
        time = time + pd.to_timedelta(time_offset_seconds, unit="s")

    df_out = df.copy()
    df_out.insert(0, "time", time)
    for c in exclude_time_cols:
        if c in df_out.columns:
            df_out = df_out.drop(columns=[c])

    df_out = (
        df_out.sort_values("time")
        .drop_duplicates(subset=["time"], keep="first")
        .reset_index(drop=True)
    )

    numeric_cols, dropped_cols = _detect_numeric_columns(
        df_out, exclude=["time"], min_numeric_rate=min_numeric_rate
    )
    for c in numeric_cols:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    warnings: List[str] = []
    # Optional: sanity warning if time steps vary wildly
    dt = df_out["time"].diff().dropna()
    if len(dt):
        med = dt.median()
        if (dt.gt(med * 3).mean() > 0.01) or (dt.lt(med / 3).mean() > 0.01):
            warnings.append(
                "Timestamp spacing looks jittery/irregular; resampling will be important."
            )

    report = TempParseReport(
        time_source=time_source,
        numeric_columns=list(numeric_cols),
        dropped_columns=list(dropped_cols),
        warnings=warnings,
    )
    return df_out[["time"] + list(numeric_cols)], report
