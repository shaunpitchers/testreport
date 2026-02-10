from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd


_TS_RE = re.compile(r"(?P<ts>\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2})\s*$")

# Capture numeric + unit tokens anywhere in the line
_PWR_TOKEN_RE = re.compile(r"(?P<val>[+-]?\d+(?:\.\d+)?)\s+(?P<unit>mW|W|kW)\b")
_CUR_TOKEN_RE = re.compile(r"(?P<val>[+-]?\d+(?:\.\d+)?)\s+(?P<unit>mA|A|kA)\b")

# Optional: voltage too (already SI), handy for debugging / QA
_VOLT_TOKEN_RE = re.compile(r"(?P<val>[+-]?\d+(?:\.\d+)?)\s+(?P<unit>mV|V|kV)\b")


_UNIT_SCALE: Dict[str, float] = {
    # Power
    "mW": 1e-3,
    "W": 1.0,
    "kW": 1e3,
    # Current
    "mA": 1e-3,
    "A": 1.0,
    "kA": 1e3,
    # Voltage (optional)
    "mV": 1e-3,
    "V": 1.0,
    "kV": 1e3,
}


def _to_si(value: float, unit: str, *, target: str) -> float:
    """
    Convert a value+unit to SI. target is used only for clearer errors.
    """
    if unit not in _UNIT_SCALE:
        raise ValueError(f"Unsupported unit '{unit}' for {target}")
    return value * _UNIT_SCALE[unit]


def _parse_line_si(
    line: str,
) -> Optional[Tuple[pd.Timestamp, float, float, Optional[float]]]:
    """
    Parse one analyser export line into:
      (timestamp, power_W, current_A, voltage_V)

    Returns None if the line doesn't contain the required tokens.
    """
    line = line.strip()
    if not line:
        return None

    ts_m = _TS_RE.search(line)
    if not ts_m:
        return None

    ts = pd.to_datetime(ts_m.group("ts"), format="%Y/%m/%d-%H:%M:%S", errors="coerce")
    if pd.isna(ts):
        return None

    # Find first power token (active power in your format)
    pwr_matches = list(_PWR_TOKEN_RE.finditer(line))
    if not pwr_matches:
        return None
    pwr_val = float(pwr_matches[0].group("val"))
    pwr_unit = pwr_matches[0].group("unit")
    power_w = _to_si(pwr_val, pwr_unit, target="power")

    # Find first current token (your format: ... V <current> <A-unit> <power> <W-unit> ...)
    cur_matches = list(_CUR_TOKEN_RE.finditer(line))
    if not cur_matches:
        return None
    cur_val = float(cur_matches[0].group("val"))
    cur_unit = cur_matches[0].group("unit")
    current_a = _to_si(cur_val, cur_unit, target="current")

    # Voltage is optional (useful for QA / debugging)
    volt_v: Optional[float] = None
    volt_matches = list(_VOLT_TOKEN_RE.finditer(line))
    if volt_matches:
        vv = float(volt_matches[0].group("val"))
        vu = volt_matches[0].group("unit")
        volt_v = _to_si(vv, vu, target="voltage")

    return ts, power_w, current_a, volt_v


def parse_power_txt_si(
    path: str | Path,
    *,
    tz: str | None = None,
    time_offset_seconds: int = 0,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Parse the power analyser .txt into SI units.

    Output columns:
      - time
      - power_W      (SI)
      - current_A    (SI)
      - voltage_V    (SI, optional; may be NaN if not parsed)

    Notes:
      - This assumes the *first* mW/W/kW token is active power and the *first*
        mA/A/kA token is current (matches your file ordering).
      - If you later find lines with multiple W-like tokens where the first isn't
        active power, we can refine selection rules (e.g., choose the power token
        that occurs immediately before a VA token).
    """
    path = Path(path)

    rows = []
    skipped = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                parsed = _parse_line_si(line)
            except Exception:
                parsed = None

            if parsed is None:
                skipped += 1
                continue

            rows.append(parsed)

    if not rows:
        raise ValueError(
            f"No valid rows parsed from {path.name}. "
            "Expected lines ending with 'YYYY/MM/DD-HH:MM:SS' and containing power (mW/W/kW) and current (mA/A/kA)."
        )

    df = pd.DataFrame(rows, columns=["time", "power_W", "current_A", "voltage_V"])

    if tz:
        df["time"] = df["time"].dt.tz_localize(tz)

    if time_offset_seconds:
        df["time"] = df["time"] + pd.to_timedelta(time_offset_seconds, unit="s")

    df = df.sort_values("time").reset_index(drop=True)

    if drop_duplicates:
        df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

    # Helpful metadata (you can log this instead)
    df.attrs["skipped_lines"] = skipped
    df.attrs["parsed_lines"] = len(df)

    return df


if __name__ == "__main__":
    power_path = "/mnt/data/VCC_L1_CC4_ListMeas-20260119095655-001.txt"
    df = parse_power_txt_si(power_path, tz=None, time_offset_seconds=0)

    print(df.head())
    print(df.tail())
    print(
        "Parsed:",
        df.attrs.get("parsed_lines"),
        "Skipped:",
        df.attrs.get("skipped_lines"),
    )

    dt = df["time"].diff().dropna()
    print("Median dt (s):", dt.median().total_seconds())
    print(
        "Power (W) min/mean/max:",
        df["power_W"].min(),
        df["power_W"].mean(),
        df["power_W"].max(),
    )
    print(
        "Current (A) min/mean/max:",
        df["current_A"].min(),
        df["current_A"].mean(),
        df["current_A"].max(),
    )
