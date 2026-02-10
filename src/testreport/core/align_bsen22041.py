# src/testreport/core/align_bsen22041.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd


@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive


@dataclass(frozen=True)
class AlignQC:
    tz: str
    test_start: pd.Timestamp
    resample_seconds: int

    windows: list[Window]
    expected_rows: dict[str, int]
    temp_rows_present: dict[str, int]
    power_rows_present: dict[str, int]
    temp_missing_frac: dict[str, float]
    power_missing_frac: dict[str, float]

    temp_range: tuple[pd.Timestamp, pd.Timestamp]
    power_range: tuple[pd.Timestamp, pd.Timestamp]

    warnings: list[str]


def _ensure_time_sorted_unique(
    df: pd.DataFrame, time_col: str = "time"
) -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"Missing required column '{time_col}'")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col])
    out = out.drop_duplicates(subset=[time_col], keep="first")
    out = out.sort_values(time_col).reset_index(drop=True)
    return out


def _build_windows(
    test_start: pd.Timestamp, *, stable_hours: int = 24, test_hours: int = 48
) -> list[Window]:
    stable = Window(
        "stable_24h", test_start - pd.Timedelta(hours=stable_hours), test_start
    )
    test_total = Window(
        "test_48h", test_start, test_start + pd.Timedelta(hours=test_hours)
    )
    test_first = Window(
        "test_first_24h", test_start, test_start + pd.Timedelta(hours=24)
    )
    test_last = Window(
        "test_last_24h",
        test_start + pd.Timedelta(hours=24),
        test_start + pd.Timedelta(hours=48),
    )
    return [stable, test_total, test_first, test_last]


def _clip(df: pd.DataFrame, w: Window, time_col: str = "time") -> pd.DataFrame:
    return df[(df[time_col] >= w.start) & (df[time_col] < w.end)].copy()


def _resample_to_grid(
    df: pd.DataFrame, *, freq: str, how: str = "mean", time_col: str = "time"
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({time_col: []})

    out = df.copy().sort_values(time_col).set_index(time_col)

    # Only aggregate numeric columns
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    out = out[numeric_cols]

    if how == "mean":
        rs = out.resample(freq).mean()
    elif how == "nearest":
        rs = out.resample(freq).nearest()
    else:
        raise ValueError(f"Unsupported resample method: {how}")

    return rs.reset_index()


def _make_grid(w: Window, freq: str) -> pd.DatetimeIndex:
    if w.end <= w.start:
        return pd.DatetimeIndex([])
    end_inclusive = w.end - pd.Timedelta(nanoseconds=1)
    return pd.date_range(start=w.start, end=end_inclusive, freq=freq)


def _reindex_to_grid(
    df_rs: pd.DataFrame, grid: pd.DatetimeIndex, *, time_col: str = "time"
) -> pd.DataFrame:
    if len(grid) == 0:
        return pd.DataFrame({time_col: []})

    if df_rs.empty:
        return pd.DataFrame({time_col: grid})

    return (
        df_rs.set_index(time_col)
        .reindex(grid)
        .reset_index()
        .rename(columns={"index": time_col})
    )


def _missing_frac_all_nan_rows(df: pd.DataFrame, time_col: str = "time") -> float:
    if df.empty or df.shape[1] <= 1:
        return 1.0
    data = df.drop(columns=[time_col])
    return float(data.isna().all(axis=1).mean())


def _present_rows(df: pd.DataFrame, time_col: str = "time") -> int:
    if df.empty or df.shape[1] <= 1:
        return 0
    data = df.drop(columns=[time_col])
    return int((~data.isna().all(axis=1)).sum())


def align_bsen22041_by_test_start(
    temp_df: pd.DataFrame,
    power_df: pd.DataFrame,
    *,
    test_start_time: str | pd.Timestamp,
    tz: str = "Europe/London",
    resample_seconds: int = 10,
    stable_hours: int = 24,
    test_hours: int = 48,
    temp_resample: str = "mean",
    power_resample: str = "mean",
    warn_if_missing_over: float = 0.01,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], AlignQC]:
    # Parse and localize test start time
    t0 = pd.to_datetime(test_start_time)
    if t0.tzinfo is None:
        t0 = t0.tz_localize(tz)
    else:
        t0 = t0.tz_convert(tz)

    temp_df = _ensure_time_sorted_unique(temp_df, "time")
    power_df = _ensure_time_sorted_unique(power_df, "time")

    if temp_df["time"].dt.tz is None:
        raise ValueError("temp_df['time'] must be tz-aware (expected Europe/London).")
    if power_df["time"].dt.tz is None:
        raise ValueError("power_df['time'] must be tz-aware (expected Europe/London).")

    temp_df["time"] = temp_df["time"].dt.tz_convert(tz)
    power_df["time"] = power_df["time"].dt.tz_convert(tz)

    temp_range = (temp_df["time"].iloc[0], temp_df["time"].iloc[-1])
    power_range = (power_df["time"].iloc[0], power_df["time"].iloc[-1])

    windows = _build_windows(t0, stable_hours=stable_hours, test_hours=test_hours)
    freq = f"{int(resample_seconds)}S"

    temp_out: dict[str, pd.DataFrame] = {}
    power_out: dict[str, pd.DataFrame] = {}

    expected_rows: dict[str, int] = {}
    temp_present: dict[str, int] = {}
    power_present: dict[str, int] = {}
    temp_miss: dict[str, float] = {}
    power_miss: dict[str, float] = {}
    warnings: list[str] = []

    # Coverage check
    for w in windows:
        if w.start < temp_range[0] or w.end > temp_range[1]:
            warnings.append(
                f"Temp does not fully cover '{w.name}' ({w.start}..{w.end}); "
                f"temp range is {temp_range[0]}..{temp_range[1]}."
            )
        if w.start < power_range[0] or w.end > power_range[1]:
            warnings.append(
                f"Power does not fully cover '{w.name}' ({w.start}..{w.end}); "
                f"power range is {power_range[0]}..{power_range[1]}."
            )

    for w in windows:
        grid = _make_grid(w, freq)
        expected_rows[w.name] = len(grid)

        t_clip = _clip(temp_df, w, "time")
        p_clip = _clip(power_df, w, "time")

        t_rs = _resample_to_grid(t_clip, freq=freq, how=temp_resample, time_col="time")
        p_rs = _resample_to_grid(p_clip, freq=freq, how=power_resample, time_col="time")

        t_al = _reindex_to_grid(t_rs, grid, time_col="time")
        p_al = _reindex_to_grid(p_rs, grid, time_col="time")

        temp_present[w.name] = _present_rows(t_al, "time")
        power_present[w.name] = _present_rows(p_al, "time")
        temp_miss[w.name] = (
            _missing_frac_all_nan_rows(t_al, "time") if len(grid) else 1.0
        )
        power_miss[w.name] = (
            _missing_frac_all_nan_rows(p_al, "time") if len(grid) else 1.0
        )

        if expected_rows[w.name] > 0:
            if temp_miss[w.name] > warn_if_missing_over:
                warnings.append(
                    f"Temp missing > {warn_if_missing_over * 100:.1f}% in '{w.name}' "
                    f"({temp_miss[w.name] * 100:.2f}% fully-missing rows)."
                )
            if power_miss[w.name] > warn_if_missing_over:
                warnings.append(
                    f"Power missing > {warn_if_missing_over * 100:.1f}% in '{w.name}' "
                    f"({power_miss[w.name] * 100:.2f}% fully-missing rows)."
                )

        temp_out[w.name] = t_al
        power_out[w.name] = p_al

    qc = AlignQC(
        tz=tz,
        test_start=t0,
        resample_seconds=resample_seconds,
        windows=windows,
        expected_rows=expected_rows,
        temp_rows_present=temp_present,
        power_rows_present=power_present,
        temp_missing_frac=temp_miss,
        power_missing_frac=power_miss,
        temp_range=temp_range,
        power_range=power_range,
        warnings=warnings,
    )

    return temp_out, power_out, qc


def export_aligned_windows(
    temp_by_window: dict[str, pd.DataFrame],
    power_by_window: dict[str, pd.DataFrame],
    out_dir: str | Path,
    *,
    prefix: str = "aligned",
    merge: bool = True,
) -> dict[str, dict[str, Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, dict[str, Path]] = {}

    for wname, tdf in temp_by_window.items():
        pdf = power_by_window.get(wname)
        if pdf is None:
            continue

        temp_path = out_dir / f"{prefix}_{wname}_temp.csv"
        power_path = out_dir / f"{prefix}_{wname}_power.csv"
        tdf.to_csv(temp_path, index=False)
        pdf.to_csv(power_path, index=False)

        entry: dict[str, Path] = {"temp": temp_path, "power": power_path}

        if merge:
            merged = tdf.merge(pdf, on="time", how="left", suffixes=("", "_power"))
            merged_path = out_dir / f"{prefix}_{wname}_merged.csv"
            merged.to_csv(merged_path, index=False)
            entry["merged"] = merged_path

        paths[wname] = entry

    return paths


def export_qc_report(qc: AlignQC, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "tz": qc.tz,
        "test_start": str(qc.test_start),
        "resample_seconds": qc.resample_seconds,
        "temp_range": (str(qc.temp_range[0]), str(qc.temp_range[1])),
        "power_range": (str(qc.power_range[0]), str(qc.power_range[1])),
        "windows": [
            {"name": w.name, "start": str(w.start), "end": str(w.end)}
            for w in qc.windows
        ],
        "expected_rows": qc.expected_rows,
        "temp_rows_present": qc.temp_rows_present,
        "power_rows_present": qc.power_rows_present,
        "temp_missing_frac": qc.temp_missing_frac,
        "power_missing_frac": qc.power_missing_frac,
        "warnings": qc.warnings,
    }

    import json

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
