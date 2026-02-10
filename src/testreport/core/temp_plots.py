from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import pandas as pd
import matplotlib.pyplot as plt


def plot_foodstuff_temps(
    df: pd.DataFrame,
    columns: Sequence[str],
    out_path: str | Path,
    *,
    title: str,
) -> Path:
    """
    Single plot with multiple foodstuff probes (1..8).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_datetime(df["time"], errors="coerce")

    plt.figure()
    for c in columns:
        if c not in df.columns:
            continue
        y = pd.to_numeric(df[c], errors="coerce")
        plt.plot(x, y, label=str(c))
    plt.xlabel("Time")
    plt.ylabel("Foodstuff temperature (°C)")
    plt.title(title)
    if len(columns) > 1:
        plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def plot_ambient_twin_axis(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    ambient_temp_col: str,
    ambient_rh_col: str,
    title: str,
) -> Path:
    """
    Twin-axis plot for ambient dry bulb and RH over time.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_datetime(df["time"], errors="coerce")
    t = pd.to_numeric(df[ambient_temp_col], errors="coerce")
    rh = pd.to_numeric(df[ambient_rh_col], errors="coerce")

    fig, ax1 = plt.subplots()
    ax1.plot(x, t)
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"Ambient dry bulb (°C) [{ambient_temp_col}]")

    ax2 = ax1.twinx()
    ax2.plot(x, rh)
    ax2.set_ylabel(f"Ambient RH (%) [{ambient_rh_col}]")

    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path
