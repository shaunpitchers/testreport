from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd


def _to_float(x: str) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _u(u: str) -> str:
    return (u or "").strip().lower()


def _convert_power_to_w(value: float, unit: str) -> float | None:
    u = _u(unit)
    if u == "w":
        return value
    if u == "mw":
        return value * 1e-3
    if u == "kw":
        return value * 1e3
    return None


def _convert_current_to_a(value: float, unit: str) -> float | None:
    u = _u(unit)
    if u == "a":
        return value
    if u == "ma":
        return value * 1e-3
    if u == "ka":
        return value * 1e3
    return None


def _convert_va_to_va(value: float, unit: str) -> float | None:
    u = _u(unit)
    if u == "va":
        return value
    if u == "kva":
        return value * 1e3
    return None


def _convert_var_to_var(value: float, unit: str) -> float | None:
    u = _u(unit)
    if u == "var":
        return value
    if u == "kvar":
        return value * 1e3
    return None


def _parse_timestamp(token: str) -> pd.Timestamp | None:
    # token format: YYYY/MM/DD-HH:MM:SS
    ts = pd.to_datetime(token, format="%Y/%m/%d-%H:%M:%S", errors="coerce")
    if ts is pd.NaT:
        return None
    return ts


def _parse_header_layout(header_line: str) -> List[str]:
    """
    Header looks like:
      Record_No  V_1 Unit  I_2 Unit  P_3 Unit  ...  Time

    We return the ordered list of measurement labels:
      ["V_1", "I_2", "P_3", ... "PF_7", "VAR_7", "IHz_8"]
    (excluding the literal "Unit" tokens and excluding "Record_No" and "Time")
    """
    toks = header_line.split()
    labels: List[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "Record_No":
            i += 1
            continue
        if t == "Time":
            break
        if t == "Unit":
            i += 1
            continue
        # a label like V_1, PF_7, Vrms_1, Irms_2, etc.
        labels.append(t)
        i += 1
    return labels


def parse_power_txt_si(
    path: str | Path,
    *,
    tz: str = "Europe/London",
    time_offset_seconds: int = 0,
) -> pd.DataFrame:
    """
    Robust power analyser TXT parser.

    Uses header layout to parse rows:
      Record_No  (value unit)*  Time

    Supports:
      - V_1 or Vrms_1
      - I_2 or Irms_2
      - P_3 (active power)
      - PF_* (PF_6 or PF_7)
      - VA_* (apparent power)
      - VAR_* (reactive power)

    Returns df with:
      time, voltage_V, current_A, power_W,
      power_factor, apparent_power_VA, reactive_power_var
    """
    path = Path(path)
    skipped = 0
    rows: list[dict] = []

    in_table = False
    labels: List[str] = []

    # Which label names will we use (resolved once we know header)
    key_v: str | None = None
    key_i: str | None = None
    key_p: str | None = None
    key_pf: str | None = None
    key_va: str | None = None
    key_var: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("Record_No") and "Time" in line:
                in_table = True
                labels = _parse_header_layout(line)

                # Resolve keys from labels
                # Voltage
                if "Vrms_1" in labels:
                    key_v = "Vrms_1"
                elif "V_1" in labels:
                    key_v = "V_1"

                # Current
                if "Irms_2" in labels:
                    key_i = "Irms_2"
                elif "I_2" in labels:
                    key_i = "I_2"

                # Active power
                key_p = "P_3" if "P_3" in labels else None

                # PF (PF_6 or PF_7 or other PF_n)
                for lab in labels:
                    if lab.startswith("PF_"):
                        key_pf = lab
                        break

                # VA and VAR
                for lab in labels:
                    if lab.startswith("VA_"):
                        key_va = lab
                        break
                for lab in labels:
                    if lab.startswith("VAR_"):
                        key_var = lab
                        break

                continue

            if not in_table:
                continue

            parts = line.split()
            if not parts or not parts[0].isdigit():
                skipped += 1
                continue

            # timestamp: find last token that matches the expected format
            ts_token = None
            for tok in reversed(parts):
                if _parse_timestamp(tok) is not None:
                    ts_token = tok
                    break
            if ts_token is None:
                skipped += 1
                continue

            ts = _parse_timestamp(ts_token)
            if ts is None:
                skipped += 1
                continue

            # localize as local clock time
            ts = ts.tz_localize(tz)
            if time_offset_seconds:
                ts = ts + pd.Timedelta(seconds=int(time_offset_seconds))

            # Now parse the row based on header label count.
            # After Record_No, we expect for each label: value unit
            # and then final token is timestamp (already found).
            # So the tokens before timestamp should be: rec + 2*len(labels)
            # But sometimes extra whitespace or oddities happen; we'll parse safely.

            # Find index of timestamp token in this row:
            try:
                t_idx = parts.index(ts_token)
            except ValueError:
                t_idx = len(parts) - 1

            payload = parts[1:t_idx]  # everything after Record_No and before timestamp

            # Walk value/unit pairs in order of labels
            # Build map label -> (value_str, unit_str)
            data_map: Dict[str, Tuple[str | None, str | None]] = {}
            j = 0
            for lab in labels:
                if j >= len(payload):
                    data_map[lab] = (None, None)
                    continue
                val = payload[j]
                unit = payload[j + 1] if (j + 1) < len(payload) else ""
                data_map[lab] = (val, unit)
                j += 2

            # Extract what we care about
            voltage_V = None
            if key_v:
                v_str, v_unit = data_map.get(key_v, (None, None))
                v_val = _to_float(v_str) if v_str is not None else None
                if v_val is not None:
                    voltage_V = v_val  # units expected V

            current_A = None
            if key_i:
                i_str, i_unit = data_map.get(key_i, (None, None))
                i_val = _to_float(i_str) if i_str is not None else None
                if i_val is not None:
                    current_A = _convert_current_to_a(i_val, i_unit or "")

            power_W = None
            if key_p:
                p_str, p_unit = data_map.get(key_p, (None, None))
                p_val = _to_float(p_str) if p_str is not None else None
                if p_val is not None:
                    power_W = _convert_power_to_w(p_val, p_unit or "")

            apparent_power_VA = None
            if key_va:
                va_str, va_unit = data_map.get(key_va, (None, None))
                va_val = _to_float(va_str) if va_str is not None else None
                if va_val is not None:
                    apparent_power_VA = _convert_va_to_va(va_val, va_unit or "")

            reactive_power_var = None
            if key_var:
                var_str, var_unit = data_map.get(key_var, (None, None))
                var_val = _to_float(var_str) if var_str is not None else None
                if var_val is not None:
                    reactive_power_var = _convert_var_to_var(var_val, var_unit or "")

            power_factor = None
            if key_pf:
                pf_str, _pf_unit = data_map.get(key_pf, (None, None))
                pf_val = _to_float(pf_str) if pf_str is not None else None
                if pf_val is not None:
                    power_factor = pf_val  # dimensionless; unit may be (null)

            # If nothing useful, skip
            if power_W is None and current_A is None and voltage_V is None:
                skipped += 1
                continue

            rows.append(
                {
                    "time": ts,
                    "voltage_V": voltage_V,
                    "current_A": current_A,
                    "power_W": power_W,
                    "power_factor": power_factor,
                    "apparent_power_VA": apparent_power_VA,
                    "reactive_power_var": reactive_power_var,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "time",
                "voltage_V",
                "current_A",
                "power_W",
                "power_factor",
                "apparent_power_VA",
                "reactive_power_var",
            ]
        )

    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    df.attrs["skipped_lines"] = skipped
    return df
