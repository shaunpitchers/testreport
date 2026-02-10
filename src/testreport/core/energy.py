from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergyResults:
    window_name: str
    duration_hours: float
    coverage_frac: float

    kwh: float
    kwh_per_day: float

    compressor_on_threshold_w: float
    compressor_off_threshold_w: float

    runtime_frac: float
    runtime_percent: float

    mean_power_on_w: float
    mean_power_off_w: float


def _infer_dt_seconds_from_time(time: pd.Series) -> float:
    dt = pd.to_datetime(time, errors="coerce").diff().dropna()
    if len(dt) == 0:
        return float("nan")
    return float(dt.median().total_seconds())


def _compressor_state_hysteresis(
    power_w: pd.Series,
    on_threshold_w: float,
    off_threshold_w: float,
) -> pd.Series:
    """
    Boolean compressor state with hysteresis:
      - turns ON when power >= on_threshold_w
      - turns OFF when power <= off_threshold_w
    """
    p = pd.to_numeric(power_w, errors="coerce").to_numpy()
    state = np.zeros(len(p), dtype=bool)

    on = False
    for i, val in enumerate(p):
        if np.isnan(val):
            state[i] = on
            continue

        if not on and val >= on_threshold_w:
            on = True
        elif on and val <= off_threshold_w:
            on = False

        state[i] = on

    return pd.Series(state, index=power_w.index, name="compressor_on")


def compute_energy_results(
    power_aligned: pd.DataFrame,
    *,
    window_name: str,
    resample_seconds: int | None = None,
    compressor_on_threshold_w: float = 50.0,
    compressor_off_threshold_w: float | None = None,
) -> EnergyResults:
    """
    Compute energy + compressor ON/OFF stats for one aligned window.
    Expects columns: time, power_W
    """
    if "time" not in power_aligned.columns or "power_W" not in power_aligned.columns:
        raise ValueError("power_aligned must contain columns: 'time', 'power_W'")

    df = power_aligned.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["power_W"] = pd.to_numeric(df["power_W"], errors="coerce")

    if resample_seconds is None:
        dt_s = _infer_dt_seconds_from_time(df["time"])
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError(
                "Could not infer dt from timestamps; provide resample_seconds."
            )
    else:
        dt_s = float(resample_seconds)

    n_total = len(df)
    n_valid = int(df["power_W"].notna().sum())
    coverage_frac = (n_valid / n_total) if n_total else 0.0

    duration_hours = (n_total * dt_s) / 3600.0 if n_total else 0.0

    if compressor_off_threshold_w is None:
        compressor_off_threshold_w = compressor_on_threshold_w * 0.8

    comp_on = _compressor_state_hysteresis(
        df["power_W"],
        on_threshold_w=compressor_on_threshold_w,
        off_threshold_w=compressor_off_threshold_w,
    )

    valid_mask = df["power_W"].notna()
    power_valid = df.loc[valid_mask, "power_W"]
    comp_on_valid = comp_on.loc[valid_mask]

    if len(power_valid) == 0:
        raise ValueError("No valid power samples in window; cannot compute energy.")

    runtime_frac = float(comp_on_valid.mean())
    runtime_percent = runtime_frac * 100.0

    mean_power_on = (
        float(power_valid[comp_on_valid].mean())
        if comp_on_valid.any()
        else float("nan")
    )
    mean_power_off = (
        float(power_valid[~comp_on_valid].mean())
        if (~comp_on_valid).any()
        else float("nan")
    )

    energy_kwh = float((power_valid * dt_s).sum() / 3_600_000.0)

    valid_hours = (len(power_valid) * dt_s) / 3600.0
    kwh_per_day = energy_kwh * (24.0 / valid_hours) if valid_hours > 0 else float("nan")

    return EnergyResults(
        window_name=window_name,
        duration_hours=duration_hours,
        coverage_frac=coverage_frac,
        kwh=energy_kwh,
        kwh_per_day=kwh_per_day,
        compressor_on_threshold_w=float(compressor_on_threshold_w),
        compressor_off_threshold_w=float(compressor_off_threshold_w),
        runtime_frac=runtime_frac,
        runtime_percent=runtime_percent,
        mean_power_on_w=mean_power_on,
        mean_power_off_w=mean_power_off,
    )
