from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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

    Returns dict of created plot paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = power_aligned.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    created: dict[str, Path] = {}

    if "voltage_V" in df.columns:
        path = out_dir / f"{prefix}_voltage.png"
        plt.figure()
        plt.plot(df["time"], pd.to_numeric(df["voltage_V"], errors="coerce"))
        plt.xlabel("Time")
        plt.ylabel("Voltage (V)")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        created["voltage"] = path

    if "current_A" in df.columns:
        path = out_dir / f"{prefix}_current.png"
        plt.figure()
        plt.plot(df["time"], pd.to_numeric(df["current_A"], errors="coerce"))
        plt.xlabel("Time")
        plt.ylabel("Current (A)")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        created["current"] = path

    return created
