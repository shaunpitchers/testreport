from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import pandas as pd
import matplotlib.pyplot as plt


def _to_time(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["time"], errors="coerce")


def plot_power(
    power_aligned: pd.DataFrame,
    out_dir: str | Path,
    *,
    prefix: str = "test_last_24h",
) -> Path:
    """
    Save power plot (power_W vs time).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = power_aligned.copy()
    x = _to_time(df)

    path = out_dir / f"{prefix}_power.png"
    if "power_W" not in df.columns:
        return path

    y = pd.to_numeric(df["power_W"], errors="coerce")
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.title("Power (W)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_voltage_current(
    power_aligned: pd.DataFrame,
    out_dir: str | Path,
    *,
    prefix: str = "test_last_24h",
) -> dict[str, Path]:
    """
    Save voltage and current plots if those columns exist.

    Creates:
      - <prefix>_voltage.png  (if voltage_V exists)
      - <prefix>_current.png  (if current_A exists)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = power_aligned.copy()
    x = _to_time(df)

    created: dict[str, Path] = {}

    if "voltage_V" in df.columns:
        path = out_dir / f"{prefix}_voltage.png"
        plt.figure()
        plt.plot(x, pd.to_numeric(df["voltage_V"], errors="coerce"))
        plt.xlabel("Time")
        plt.ylabel("Voltage (V)")
        plt.title("Voltage (V)")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        created["voltage"] = path

    if "current_A" in df.columns:
        path = out_dir / f"{prefix}_current.png"
        plt.figure()
        plt.plot(x, pd.to_numeric(df["current_A"], errors="coerce"))
        plt.xlabel("Time")
        plt.ylabel("Current (A)")
        plt.title("Current (A)")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        created["current"] = path

    return created


def plot_foodstuff_lines(
    df: pd.DataFrame,
    food_cols: Sequence[str],
    out_path: str | Path,
    *,
    title: str,
) -> Path:
    """
    Plot all foodstuff probes (multiple lines).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = _to_time(df)

    plt.figure()
    for c in food_cols:
        if c not in df.columns:
            continue
        y = pd.to_numeric(df[c], errors="coerce")
        plt.plot(x, y, label=str(c))

    plt.xlabel("Time")
    plt.ylabel("Foodstuff temperature (°C)")
    plt.title(title)
    if len(food_cols) > 1:
        plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_foodstuff_min_max_mean(
    df: pd.DataFrame,
    food_cols: Sequence[str],
    out_path: str | Path,
    *,
    title: str,
) -> Path:
    """
    Plot min/max/mean across all food probes at each timestamp.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = _to_time(df)

    cols = [c for c in food_cols if c in df.columns]
    if not cols:
        return out_path

    values = df[cols].apply(pd.to_numeric, errors="coerce")
    y_min = values.min(axis=1, skipna=True)
    y_max = values.max(axis=1, skipna=True)
    y_mean = values.mean(axis=1, skipna=True)

    plt.figure()
    plt.plot(x, y_min, label="min")
    plt.plot(x, y_mean, label="mean")
    plt.plot(x, y_max, label="max")
    plt.xlabel("Time")
    plt.ylabel("Foodstuff temperature (°C)")
    plt.title(title)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_ambient_temps_and_rh(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    ta_col: str,
    ground_col: str,
    ceiling_col: str,
    rh_col: str,
    title: str,
) -> Path:
    """
    Twin-axis plot:
      - Left axis: Ta + Ground + Ceiling (°C)
      - Right axis: RH (%)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = _to_time(df)

    ta = pd.to_numeric(df[ta_col], errors="coerce")
    g = pd.to_numeric(df[ground_col], errors="coerce")
    c = pd.to_numeric(df[ceiling_col], errors="coerce")
    rh = pd.to_numeric(df[rh_col], errors="coerce")

    fig, ax1 = plt.subplots()
    ax1.plot(x, ta, label="Ta")
    ax1.plot(x, g, label="Ground")
    ax1.plot(x, c, label="Ceiling")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Ambient temperature (°C)")
    ax1.legend(fontsize="small", ncol=3)

    ax2 = ax1.twinx()
    ax2.plot(x, rh)
    ax2.set_ylabel("Relative humidity (%)")

    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
