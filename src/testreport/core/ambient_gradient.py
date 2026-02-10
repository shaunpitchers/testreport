from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class AmbientGradientResult:
    window_name: str
    ground_col: str
    ceiling_col: str
    distance_m: float

    ground_mean_c: float
    ceiling_mean_c: float

    gradient_c_per_m: float


def compute_ambient_gradient(
    df: pd.DataFrame,
    *,
    window_name: str,
    ground_col: str,
    ceiling_col: str,
    distance_m: float = 2.5,
) -> AmbientGradientResult:
    """
    Compute ambient vertical gradient:
      (mean(ceiling) - mean(ground)) / distance

    df is an aligned temp window dataframe (must include specified columns).
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")

    if ground_col not in df.columns:
        raise ValueError(f"ground_col '{ground_col}' not found in dataframe columns")
    if ceiling_col not in df.columns:
        raise ValueError(f"ceiling_col '{ceiling_col}' not found in dataframe columns")

    g = pd.to_numeric(df[ground_col], errors="coerce")
    c = pd.to_numeric(df[ceiling_col], errors="coerce")

    if g.notna().sum() == 0:
        raise ValueError(f"No valid numeric data in ground_col '{ground_col}'")
    if c.notna().sum() == 0:
        raise ValueError(f"No valid numeric data in ceiling_col '{ceiling_col}'")

    g_mean = float(g.mean(skipna=True))
    c_mean = float(c.mean(skipna=True))

    grad = (c_mean - g_mean) / float(distance_m)

    return AmbientGradientResult(
        window_name=window_name,
        ground_col=ground_col,
        ceiling_col=ceiling_col,
        distance_m=float(distance_m),
        ground_mean_c=g_mean,
        ceiling_mean_c=c_mean,
        gradient_c_per_m=float(grad),
    )
