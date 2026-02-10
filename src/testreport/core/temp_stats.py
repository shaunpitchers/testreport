from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import pandas as pd


@dataclass(frozen=True)
class ColumnStats:
    column: str
    n: int
    n_valid: int
    mean: float
    min: float
    max: float


def compute_column_stats(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Compute min/mean/max for each column (ignoring NaNs).
    Returns a tidy dataframe with one row per column.
    """
    rows: list[dict] = []
    for c in columns:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append(
            {
                "column": c,
                "n": int(len(s)),
                "n_valid": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)) if s.notna().any() else float("nan"),
                "min": float(s.min(skipna=True)) if s.notna().any() else float("nan"),
                "max": float(s.max(skipna=True)) if s.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def detect_foodstuff_columns(df: pd.DataFrame, *, expected: int = 8) -> list[str]:
    """
    Default expectation: foodstuff probes are string columns "1".."8".
    Returns only those that exist in df.
    """
    cols = []
    for i in range(1, expected + 1):
        c = str(i)
        if c in df.columns:
            cols.append(c)
    return cols


def detect_ambient_columns(
    df: pd.DataFrame,
    *,
    ambient_temp_hint: str | None = None,
    ambient_rh_hint: str | None = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Heuristic detection of ambient dry bulb and RH columns by name.

    - If ambient_temp_hint / ambient_rh_hint provided, use those if present.
    - Else pick the first column whose name contains "room" and "temp" for dry bulb,
      and first containing "humidity" or "rh" for RH.
    """
    cols = list(df.columns)

    if ambient_temp_hint and ambient_temp_hint in cols:
        amb_t = ambient_temp_hint
    else:
        amb_t = None
        for c in cols:
            name = str(c).lower()
            if "room" in name and "temp" in name:
                amb_t = c
                break

    if ambient_rh_hint and ambient_rh_hint in cols:
        amb_rh = ambient_rh_hint
    else:
        amb_rh = None
        for c in cols:
            name = str(c).lower()
            if "humidity" in name or name.startswith("rh") or " rh" in name:
                amb_rh = c
                break

    return amb_t, amb_rh
