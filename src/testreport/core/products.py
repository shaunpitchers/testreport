from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductSpec:
    name: str
    width_mm: int
    depth_mm: int
    height_mm: int
    net_volume_l: float  # litres


# Your products
PRODUCTS: dict[str, ProductSpec] = {
    "VCS": ProductSpec("VCS", width_mm=1100, depth_mm=700, height_mm=397, net_volume_l=86.0),
    "VCR": ProductSpec("VCR", width_mm=880, depth_mm=885, height_mm=397, net_volume_l=86.0),
    "VCC": ProductSpec("VCC", width_mm=450, depth_mm=800, height_mm=890, net_volume_l=69.0),
    "VLS": ProductSpec("VLS", width_mm=1100, depth_mm=700, height_mm=349, net_volume_l=65.0),
}


# Target ranges you specified (auditable)
FRIDGE_RANGE_C = (-1.0, 5.0)
FREEZER_RANGE_C = (-18.0, -15.0)

# Classification threshold you chose (simple + robust)
FREEZER_CLASSIFY_THRESHOLD_C = -5.0


def classify_cabinet_by_food_mean(food_mean_c: float | None) -> str | None:
    """
    Simple robust classification:
      - freezer if mean food temp < -5°C
      - fridge otherwise
    """
    if food_mean_c is None:
        return None
    try:
        v = float(food_mean_c)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return "freezer" if v < FREEZER_CLASSIFY_THRESHOLD_C else "fridge"


def saec_constants(cabinet_class: str) -> tuple[float, float]:
    """
    SAEC = M*volume + N
    """
    if cabinet_class == "freezer":
        return 5.840, 2380.0
    return 2.555, 1790.0


def compute_energy_label(eei_percent: float) -> str:
    """
    Bands (per your spec):
      A+  : < 15%
      A   : >=15 and <25
      B   : >=25 and <35
      C   : >=35 and <50
      D   : >=50
    """
    if eei_percent < 15.0:
        return "A+"
    if eei_percent < 25.0:
        return "A"
    if eei_percent < 35.0:
        return "B"
    if eei_percent < 50.0:
        return "C"
    return "D"


def _range_for_class(cabinet_class: str) -> tuple[float, float]:
    return FREEZER_RANGE_C if cabinet_class == "freezer" else FRIDGE_RANGE_C


def check_food_temps_against_target(
    *,
    cabinet_class: str,
    overall_min_c: float | None,
    overall_mean_c: float | None,
    overall_max_c: float | None,
) -> list[str]:
    """
    Produce warnings if food temps are outside the defined target range.
    Uses overall min/mean/max from whichever window you choose.
    """
    warnings: list[str] = []
    lo, hi = _range_for_class(cabinet_class)

    def _bad(v: float | None) -> bool:
        return v is None or (isinstance(v, float) and v != v)

    if _bad(overall_min_c) or _bad(overall_mean_c) or _bad(overall_max_c):
        return ["Food temperature stats unavailable (cannot check target range)."]

    mn = float(overall_min_c)
    me = float(overall_mean_c)
    mx = float(overall_max_c)

    if me < lo or me > hi:
        warnings.append(
            f"Mean food temperature {me:.2f}°C is outside the target range [{lo:.1f}, {hi:.1f}]°C for {cabinet_class}."
        )
    if mn < lo or mx > hi:
        warnings.append(
            f"Food temperature range [{mn:.2f}, {mx:.2f}]°C exceeds the target range [{lo:.1f}, {hi:.1f}]°C for {cabinet_class}."
        )

    return warnings


# -------------------------------
# Food / pack product classes
# -------------------------------

# Apply tolerance to avoid classification flipping due to tiny spikes/noise
CLASS_TOL_C = 0.5

# M1 class: -1°C to +5°C (with tolerance)
M1_MIN_C = -1.0
M1_MAX_C = 5.0

# L1 class: FREEZER-style constraints (as you described)
# - At no point should the maximum pack temp be warmer than -15°C
#   -> max_of_max <= -15
# - The warmest pack's minimum should be <= -18°C
#   -> max_of_min <= -18
L1_MAX_OF_MAX_C = -15.0
L1_MAX_OF_MIN_C = -18.0


def classify_food_class(
    *,
    overall_min_c: float | None,
    overall_mean_c: float | None,
    overall_max_c: float | None,
    per_probe_max_c: list[float] | None,
    per_probe_min_c: list[float] | None = None,
) -> tuple[str | None, list[str]]:
    """
    Returns (food_class, warnings).

    - "M1" (fridge pack class): min/mean/max within [-1, +5] with tolerance.
    - "L1" (freezer pack class): max_of_max <= -15 and max_of_min <= -18 with tolerance.
      (Colder is always fine.)

    Note: per_probe_min_c is optional, but recommended for the L1 "max_of_min" check.
    """
    warnings: list[str] = []

    def _bad(v: float | None) -> bool:
        return v is None or (isinstance(v, float) and v != v)

    if _bad(overall_min_c) or _bad(overall_mean_c) or _bad(overall_max_c):
        return None, ["Food temperature stats unavailable (cannot classify M1/L1)."]

    mn = float(overall_min_c)
    me = float(overall_mean_c)
    mx = float(overall_max_c)

    # ---- M1 check (with tolerance) ----
    m1_lo = M1_MIN_C - CLASS_TOL_C
    m1_hi = M1_MAX_C + CLASS_TOL_C
    m1_ok = (mn >= m1_lo) and (mx <= m1_hi) and (m1_lo <= me <= m1_hi)

    # ---- L1 check (freezer-style, with tolerance) ----
    # max_of_max uses per-probe maxima if available; else overall max
    if per_probe_max_c and len(per_probe_max_c) > 0:
        max_of_max = max(float(v) for v in per_probe_max_c)
    else:
        max_of_max = mx

    # max_of_min needs per-probe minima ideally; else overall min is a weak proxy
    if per_probe_min_c and len(per_probe_min_c) > 0:
        max_of_min = max(float(v) for v in per_probe_min_c)
    else:
        max_of_min = mn
        warnings.append(
            "L1: per-probe minima not provided; using overall min as proxy for max_of_min."
        )

    l1_ok = (max_of_max <= (L1_MAX_OF_MAX_C + CLASS_TOL_C)) and (
        max_of_min <= (L1_MAX_OF_MIN_C + CLASS_TOL_C)
    )

    # Prefer M1 if it passes; else L1; else None
    if m1_ok:
        food_class: str | None = "M1"
    elif l1_ok:
        food_class = "L1"
    else:
        food_class = None

    # Warnings / diagnostics
    if food_class == "M1":
        if mn < m1_lo:
            warnings.append(
                f"M1: Food min {mn:.2f}°C is below {m1_lo:.2f}°C (limit -1°C with tol)."
            )
        if mx > m1_hi:
            warnings.append(
                f"M1: Food max {mx:.2f}°C is above {m1_hi:.2f}°C (limit +5°C with tol)."
            )
        if not (m1_lo <= me <= m1_hi):
            warnings.append(f"M1: Food mean {me:.2f}°C is outside [{m1_lo:.2f}, {m1_hi:.2f}]°C.")
    elif food_class == "L1":
        if max_of_max > (L1_MAX_OF_MAX_C + CLASS_TOL_C):
            warnings.append(
                f"L1: max_of_max {max_of_max:.2f}°C is warmer than {L1_MAX_OF_MAX_C + CLASS_TOL_C:.2f}°C."
            )
        if max_of_min > (L1_MAX_OF_MIN_C + CLASS_TOL_C):
            warnings.append(
                f"L1: max_of_min {max_of_min:.2f}°C is warmer than {L1_MAX_OF_MIN_C + CLASS_TOL_C:.2f}°C."
            )
    else:
        # Provide a helpful hint as to why neither class passed
        warnings.append(
            f"Food class not met. M1 limits (with tol): [{m1_lo:.2f},{m1_hi:.2f}]°C; "
            f"L1 limits (with tol): max_of_max <= {L1_MAX_OF_MAX_C + CLASS_TOL_C:.2f}°C and "
            f"max_of_min <= {L1_MAX_OF_MIN_C + CLASS_TOL_C:.2f}°C."
        )

    return food_class, warnings


# -------------------------------
# Climate classes
# -------------------------------
CC4_T_C = 30.0
CC4_T_TOL = 1.0
CC4_RH = 55.0
CC4_RH_TOL = 5.0

CC5_T_C = 40.0
CC5_T_TOL = 1.0
CC5_RH = 40.0
CC5_RH_TOL = 5.0


def classify_climate_class(
    *,
    ambient_mean_c: float | None,
    rh_mean_percent: float | None,
) -> tuple[str | None, list[str]]:
    """
    Returns (climate_class, warnings) where class is "CC4" or "CC5" or None.
    Uses mean ambient temp + mean RH.
    """
    if ambient_mean_c is None or rh_mean_percent is None:
        return None, ["Ambient mean temp/RH unavailable (cannot classify CC4/CC5)."]

    t = float(ambient_mean_c)
    rh = float(rh_mean_percent)

    def in_band(target: float, tol: float, val: float) -> bool:
        return (target - tol) <= val <= (target + tol)

    cc4 = in_band(CC4_T_C, CC4_T_TOL, t) and in_band(CC4_RH, CC4_RH_TOL, rh)
    cc5 = in_band(CC5_T_C, CC5_T_TOL, t) and in_band(CC5_RH, CC5_RH_TOL, rh)

    if cc4:
        return "CC4", []
    if cc5:
        return "CC5", []

    return None, [
        f"Ambient means not within CC4 (30±1°C, 55±5%RH) or CC5 (40±1°C, 40±5%RH). "
        f"Got {t:.2f}°C, {rh:.2f}%RH."
    ]
