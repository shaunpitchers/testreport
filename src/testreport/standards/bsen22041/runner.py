from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import datetime as dt
import json
from typing import Any

import pandas as pd

from testreport.io.temp_csv import parse_temp_rh_csv
from testreport.io.power_txt import parse_power_txt_si
from testreport.core.align_bsen22041 import (
    align_bsen22041_by_test_start,
    export_aligned_windows,
    export_qc_report,
)
from testreport.core.energy import compute_energy_results
from testreport.core.ambient_gradient import compute_ambient_gradient
from testreport.core.plots import (
    plot_power,
    plot_voltage_current,
    plot_foodstuff_lines,
    plot_foodstuff_min_max_mean,
    plot_ambient_temps_and_rh,
)
from testreport.core.temp_stats import compute_column_stats


@dataclass(frozen=True)
class Bsen22041RunResult:
    run_dir: Path
    results_dir: Path
    qc_path: Path
    summary_path: Path
    passed_coverage_gate: bool
    failed_reasons: list[str]
    warnings: list[str]
    plots: dict[str, str | None]
    summary: dict[str, Any]


def _detect_food_cols(df: pd.DataFrame, *, n: int = 8) -> list[str]:
    cols = []
    for i in range(1, n + 1):
        c = str(i)
        if c in df.columns:
            cols.append(c)
    return cols


def _find_first_matching(df: pd.DataFrame, contains_any: list[str]) -> str | None:
    for c in df.columns:
        name = str(c).strip().lower()
        if any(k in name for k in contains_any):
            return c
    return None


def _detect_ambient_columns(
    df: pd.DataFrame,
    *,
    ta_col: str | None,
    ground_col: str | None,
    ceiling_col: str | None,
    rh_col: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    cols = set(df.columns)

    ta = ta_col if (ta_col and ta_col in cols) else None
    g = ground_col if (ground_col and ground_col in cols) else None
    c = ceiling_col if (ceiling_col and ceiling_col in cols) else None
    rh = rh_col if (rh_col and rh_col in cols) else None

    # Ta (room probe)
    if ta is None:
        ta = _find_first_matching(df, ["ta"])
    if ta is None and "ROOM TEMP 1" in cols:
        ta = "ROOM TEMP 1"
    if ta is None:
        ta = _find_first_matching(df, ["room temp", "ambient temp", "air temp"])

    # Ground/Ceiling
    if g is None:
        g = "Ground" if "Ground" in cols else _find_first_matching(df, ["ground"])
    if c is None:
        c = "Ceiling" if "Ceiling" in cols else _find_first_matching(df, ["ceiling"])

    # RH
    if rh is None and "ROOM HUMIDITY 1" in cols:
        rh = "ROOM HUMIDITY 1"
    if rh is None:
        rh = _find_first_matching(df, ["humidity", " rh", "rh"])

    return ta, g, c, rh


def _temp_summary(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    """
    Returns overall + per-probe min/mean/max.
    """
    if not cols:
        return {"overall": None, "per_probe": {}}

    data = df[cols].apply(pd.to_numeric, errors="coerce")

    overall_min = float(data.min(axis=None, skipna=True))
    overall_max = float(data.max(axis=None, skipna=True))
    overall_mean = float(data.stack().mean(skipna=True))

    per = {}
    for c in cols:
        s = data[c]
        per[str(c)] = {
            "min": float(s.min(skipna=True)),
            "mean": float(s.mean(skipna=True)),
            "max": float(s.max(skipna=True)),
        }

    return {
        "overall": {"min": overall_min, "mean": overall_mean, "max": overall_max},
        "per_probe": per,
    }


def run_bsen22041(
    *,
    temp_file: Path,
    power_file: Path,
    test_start: str,
    out_dir: Path,
    tz: str = "Europe/London",
    resample_seconds: int = 10,
    prefix: str = "aligned",
    compressor_on_threshold_w: float = 50.0,
    coverage_max_missing_percent: float = 0.5,
    ta_col: str | None = None,
    ground_col: str | None = None,
    ceiling_col: str | None = None,
    rh_col: str | None = None,
    probe_distance_m: float = 2.5,
    stamp: str | None = None,
) -> Bsen22041RunResult:
    stamp = stamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []

    # Parse temp (Excel days, numeric UTC)
    temp_df, temp_rep = parse_temp_rh_csv(
        temp_file,
        tz=tz,
        numeric_time_is_utc=True,
        time_base="excel_days",
    )
    warnings.extend(temp_rep.warnings)

    # Parse power (local clock)
    power_df = parse_power_txt_si(power_file, tz=tz)

    # dt sanity
    temp_dt = pd.to_datetime(temp_df["time"], errors="coerce").diff().dropna()
    power_dt = pd.to_datetime(power_df["time"], errors="coerce").diff().dropna()
    temp_dt_med = temp_dt.median().total_seconds() if len(temp_dt) else float("nan")
    power_dt_med = power_dt.median().total_seconds() if len(power_dt) else float("nan")

    # Align
    temp_w, power_w, qc = align_bsen22041_by_test_start(
        temp_df,
        power_df,
        test_start_time=test_start,
        tz=tz,
        resample_seconds=resample_seconds,
    )

    export_aligned_windows(temp_w, power_w, out_dir=run_dir, prefix=prefix, merge=True)
    qc_path = export_qc_report(qc, run_dir / f"{prefix}_qc.json")

    # Coverage gate policy
    max_missing = float(coverage_max_missing_percent) / 100.0
    required_temp_windows = {
        "stable_24h",
        "test_48h",
        "test_first_24h",
        "test_last_24h",
    }
    required_power_windows = {"test_last_24h"}  # stable_24h not required for power
    warn_power_windows = {"test_48h", "test_first_24h"}

    failed: list[str] = []

    for wname in required_temp_windows:
        miss = qc.temp_missing_frac.get(wname, 1.0)
        if miss > max_missing:
            failed.append(
                f"Temp window {wname}: missing={miss * 100:.2f}% (limit {coverage_max_missing_percent:.2f}%)"
            )

    for wname in required_power_windows:
        miss = qc.power_missing_frac.get(wname, 1.0)
        if miss > max_missing:
            failed.append(
                f"Power window {wname}: missing={miss * 100:.2f}% (limit {coverage_max_missing_percent:.2f}%)"
            )

    for wname in sorted(warn_power_windows):
        miss = qc.power_missing_frac.get(wname, None)
        if miss is not None and miss > max_missing:
            warnings.append(
                f"Power window {wname} missing={miss * 100:.2f}% (limit {coverage_max_missing_percent:.2f}%)"
            )

    # Power results + plots
    p_last = power_w.get("test_last_24h")
    if p_last is None:
        raise RuntimeError("Missing power window: test_last_24h")

    power_results = compute_energy_results(
        p_last,
        window_name="test_last_24h",
        resample_seconds=resample_seconds,
        compressor_on_threshold_w=compressor_on_threshold_w,
    )

    (results_dir / "power_results.json").write_text(
        json.dumps(power_results.__dict__, indent=2), encoding="utf-8"
    )
    pd.DataFrame([power_results.__dict__]).to_csv(
        results_dir / "power_results.csv", index=False
    )

    power_plot = plot_power(p_last, results_dir, prefix="test_last_24h")
    vc_paths = plot_voltage_current(p_last, results_dir, prefix="test_last_24h")

    # Temp plots + stats
    t_stable = temp_w.get("stable_24h")
    t_last = temp_w.get("test_last_24h")
    t_test48 = temp_w.get("test_48h")
    if t_stable is None or t_last is None or t_test48 is None:
        raise RuntimeError(
            "Missing one or more temp windows: stable_24h, test_last_24h, test_48h"
        )

    food_cols = _detect_food_cols(t_stable, n=8)

    if food_cols:
        plot_foodstuff_lines(
            t_stable,
            food_cols,
            results_dir / "foodstuff_stable_24h.png",
            title="Foodstuff temperatures (Stable 24h)",
        )
        plot_foodstuff_lines(
            t_last,
            food_cols,
            results_dir / "foodstuff_test_last_24h.png",
            title="Foodstuff temperatures (Test last 24h)",
        )
        plot_foodstuff_min_max_mean(
            t_stable,
            food_cols,
            results_dir / "foodstuff_stable_24h_min_max_mean.png",
            title="Foodstuff min / mean / max (Stable 24h)",
        )
        plot_foodstuff_min_max_mean(
            t_last,
            food_cols,
            results_dir / "foodstuff_test_last_24h_min_max_mean.png",
            title="Foodstuff min / mean / max (Test last 24h)",
        )

        compute_column_stats(t_stable, food_cols).to_csv(
            results_dir / "foodstuff_stats_stable_24h.csv", index=False
        )
        compute_column_stats(t_last, food_cols).to_csv(
            results_dir / "foodstuff_stats_test_last_24h.csv", index=False
        )
    else:
        warnings.append(
            "Could not detect foodstuff columns '1'..'8' in aligned temp data."
        )

    # Ambient plot + gradient
    ta, g, c, rh = _detect_ambient_columns(
        t_test48,
        ta_col=ta_col,
        ground_col=ground_col,
        ceiling_col=ceiling_col,
        rh_col=rh_col,
    )

    ambient_plot_path: str | None = None
    if ta and g and c and rh:
        ambient_plot = plot_ambient_temps_and_rh(
            t_test48,
            results_dir / "ambient_test_48h_ta_ground_ceiling_rh.png",
            ta_col=ta,
            ground_col=g,
            ceiling_col=c,
            rh_col=rh,
            title="Ambient Ta/Ground/Ceiling and RH (Test 48h)",
        )
        ambient_plot_path = str(ambient_plot)
    else:
        warnings.append(
            "Ambient plot not generated (could not detect required columns). Use overrides."
        )

    gradient_payload = None
    if g and c:
        try:
            grad = compute_ambient_gradient(
                t_test48,
                window_name="test_48h",
                ground_col=g,
                ceiling_col=c,
                distance_m=probe_distance_m,
            )
            gradient_payload = grad.__dict__
            (results_dir / "ambient_gradient.json").write_text(
                json.dumps(gradient_payload, indent=2), encoding="utf-8"
            )
        except Exception as e:
            warnings.append(f"Ambient gradient failed: {e}")

    # Temperature summary for GUI
    temp_summary = {
        "stable_24h": _temp_summary(t_stable, food_cols),
        "test_last_24h": _temp_summary(t_last, food_cols),
    }

    # Summary for GUI
    summary: dict[str, Any] = {
        "software_name": "ADE Insight",
        "test_start": str(qc.test_start),
        "tz": tz,
        "resample_seconds": resample_seconds,
        "raw_dt_median_seconds": {"temp": temp_dt_med, "power": power_dt_med},
        "power_results": power_results.__dict__,
        "temp_summary": temp_summary,
        "ambient_gradient": gradient_payload,
        "coverage_missing_frac": {
            "temp": qc.temp_missing_frac,
            "power": qc.power_missing_frac,
        },
        "warnings": {"temp_parse": temp_rep.warnings, "qc": qc.warnings},
        "detected_columns": {
            "Ta": ta,
            "Ground": g,
            "Ceiling": c,
            "RH": rh,
            "food": food_cols,
        },
    }

    plots: dict[str, str | None] = {
        "power": str(power_plot),
        "voltage": str(vc_paths.get("voltage")) if vc_paths.get("voltage") else None,
        "current": str(vc_paths.get("current")) if vc_paths.get("current") else None,
        "ambient": ambient_plot_path,
        "food_stable": str(results_dir / "foodstuff_stable_24h.png")
        if food_cols
        else None,
        "food_last": str(results_dir / "foodstuff_test_last_24h.png")
        if food_cols
        else None,
        "food_stable_mmm": str(results_dir / "foodstuff_stable_24h_min_max_mean.png")
        if food_cols
        else None,
        "food_last_mmm": str(results_dir / "foodstuff_test_last_24h_min_max_mean.png")
        if food_cols
        else None,
    }

    summary_path = results_dir / "summary.json"
    summary_path.write_text(
        json.dumps({**summary, "plots": plots}, indent=2), encoding="utf-8"
    )

    return Bsen22041RunResult(
        run_dir=run_dir,
        results_dir=results_dir,
        qc_path=qc_path,
        summary_path=summary_path,
        passed_coverage_gate=(len(failed) == 0),
        failed_reasons=failed,
        warnings=warnings,
        plots=plots,
        summary=summary,
    )
