from __future__ import annotations

from pathlib import Path
import datetime as dt
import json

import pandas as pd
import typer

from testreport.io.temp_csv import parse_temp_rh_csv
from testreport.io.power_txt import parse_power_txt_si
from testreport.core.align_bsen22041 import (
    align_bsen22041_by_test_start,
    export_aligned_windows,
    export_qc_report,
)
from testreport.core.energy import compute_energy_results
from testreport.core.plots import (
    plot_power,
    plot_voltage_current,
    plot_foodstuff_lines,
    plot_foodstuff_min_max_mean,
    plot_ambient_temps_and_rh,
)
from testreport.core.temp_stats import compute_column_stats

app = typer.Typer(help="BS EN 22041 tools")


def _detect_food_cols(df: pd.DataFrame, *, n: int = 8) -> list[str]:
    # Food probes are usually "1".."8"
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

    # Prefer explicit overrides if provided and present
    ta = ta_col if (ta_col and ta_col in cols) else None
    g = ground_col if (ground_col and ground_col in cols) else None
    c = ceiling_col if (ceiling_col and ceiling_col in cols) else None
    rh = rh_col if (rh_col and rh_col in cols) else None

    # Auto-detect Ta (room probe): prefer "ta" then "room temp 1" then first "room temp"
    if ta is None:
        ta = _find_first_matching(df, ["ta"])
    if ta is None and "ROOM TEMP 1" in cols:
        ta = "ROOM TEMP 1"
    if ta is None:
        ta = _find_first_matching(df, ["room temp", "ambient temp", "air temp"])

    # Ground/Ceiling
    if g is None:
        # prefer exact "Ground"
        if "Ground" in cols:
            g = "Ground"
        else:
            g = _find_first_matching(df, ["ground"])
    if c is None:
        if "Ceiling" in cols:
            c = "Ceiling"
        else:
            c = _find_first_matching(df, ["ceiling"])

    # RH: prefer "ROOM HUMIDITY 1" then first RH/humidity column
    if rh is None and "ROOM HUMIDITY 1" in cols:
        rh = "ROOM HUMIDITY 1"
    if rh is None:
        rh = _find_first_matching(df, ["humidity", " rh", "rh"])

    return ta, g, c, rh


@app.command("align")
def align(
    temp_file: Path = typer.Argument(..., exists=True, readable=True),
    power_file: Path = typer.Argument(..., exists=True, readable=True),
    test_start: str = typer.Option(
        ..., "--test-start", help="Door-opening start time (local clock)"
    ),
    out_dir: Path = typer.Option(Path("out/inspect")),
    tz: str = typer.Option("Europe/London"),
    resample_seconds: int = typer.Option(10),
    prefix: str = typer.Option("aligned"),
    compressor_on_threshold_w: float = typer.Option(
        50.0, "--compressor-on-threshold-w"
    ),
    coverage_max_missing_percent: float = typer.Option(
        0.5, "--coverage-max-missing-percent"
    ),
    # Ambient plot column overrides
    ta_col: str | None = typer.Option(
        None, "--ta-col", help="Ambient room probe column (Ta) override"
    ),
    ground_col: str | None = typer.Option(
        None, "--ground-col", help="Ambient ground probe column override"
    ),
    ceiling_col: str | None = typer.Option(
        None, "--ceiling-col", help="Ambient ceiling probe column override"
    ),
    rh_col: str | None = typer.Option(
        None, "--rh-col", help="Ambient RH column override (usually first)"
    ),
):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Parse temp: Excel-days numeric time (UTC) -> Europe/London
    temp_df, temp_rep = parse_temp_rh_csv(
        temp_file,
        tz=tz,
        numeric_time_is_utc=True,
        time_base="excel_days",
    )

    # Parse power: timestamps are local clock time
    power_df = parse_power_txt_si(
        power_file,
        tz=tz,
    )

    # Raw dt sanity (median)
    temp_dt = pd.to_datetime(temp_df["time"], errors="coerce").diff().dropna()
    power_dt = pd.to_datetime(power_df["time"], errors="coerce").diff().dropna()
    temp_dt_med = temp_dt.median().total_seconds() if len(temp_dt) else float("nan")
    power_dt_med = power_dt.median().total_seconds() if len(power_dt) else float("nan")
    typer.echo(f"Median raw dt: temp={temp_dt_med:.1f}s, power={power_dt_med:.1f}s")

    # Align windows
    temp_w, power_w, qc = align_bsen22041_by_test_start(
        temp_df,
        power_df,
        test_start_time=test_start,
        tz=tz,
        resample_seconds=resample_seconds,
    )

    # Export aligned datasets + QC JSON
    export_aligned_windows(temp_w, power_w, out_dir=run_dir, prefix=prefix, merge=True)
    qc_path = export_qc_report(qc, run_dir / f"{prefix}_qc.json")

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Coverage gate policy (BS EN 22041 as you described)
    # ----------------------------
    max_missing = float(coverage_max_missing_percent) / 100.0

    required_temp_windows = {
        "stable_24h",
        "test_48h",
        "test_first_24h",
        "test_last_24h",
    }

    # Power: stable_24h is irrelevant; only last 24h must be clean for energy.
    required_power_windows = {"test_last_24h"}
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
            typer.echo(
                f"Warning: Power window {wname} missing={miss * 100:.2f}% (limit {coverage_max_missing_percent:.2f}%)"
            )

    # ----------------------------
    # Power results + plots (test_last_24h)
    # ----------------------------
    p_last = power_w.get("test_last_24h")
    if p_last is None:
        typer.echo("Missing power window: test_last_24h")
        raise typer.Exit(code=2)

    power_results = compute_energy_results(
        p_last,
        window_name="test_last_24h",
        resample_seconds=resample_seconds,
        compressor_on_threshold_w=compressor_on_threshold_w,
    )

    (results_dir / "power_results.json").write_text(
        json.dumps(power_results.__dict__, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([power_results.__dict__]).to_csv(
        results_dir / "power_results.csv", index=False
    )

    # Plots: power + voltage + current
    p_power_path = plot_power(p_last, results_dir, prefix="test_last_24h")
    vc_paths = plot_voltage_current(p_last, results_dir, prefix="test_last_24h")

    # ----------------------------
    # Temperature plots + stats
    # ----------------------------
    t_stable = temp_w.get("stable_24h")
    t_last = temp_w.get("test_last_24h")
    t_test48 = temp_w.get("test_48h")

    if t_stable is None or t_last is None or t_test48 is None:
        typer.echo(
            "Missing one or more temp windows: stable_24h, test_last_24h, test_48h"
        )
        raise typer.Exit(code=2)

    food_cols = _detect_food_cols(t_stable, n=8)

    if not food_cols:
        typer.echo("Could not detect foodstuff columns '1'..'8' in aligned temp data.")
    else:
        # Multi-line plots
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

        # Min/Max/Mean envelope plots
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

        # Stats tables for each probe
        stable_stats = compute_column_stats(t_stable, food_cols)
        last_stats = compute_column_stats(t_last, food_cols)

        stable_stats.to_csv(results_dir / "foodstuff_stats_stable_24h.csv", index=False)
        last_stats.to_csv(
            results_dir / "foodstuff_stats_test_last_24h.csv", index=False
        )

    # ----------------------------
    # Ambient plot: Ta + Ground + Ceiling (left axis), RH (right axis)
    # ----------------------------
    ta, g, c, rh = _detect_ambient_columns(
        t_test48,
        ta_col=ta_col,
        ground_col=ground_col,
        ceiling_col=ceiling_col,
        rh_col=rh_col,
    )

    if ta and g and c and rh:
        plot_ambient_temps_and_rh(
            t_test48,
            results_dir / "ambient_test_48h_ta_ground_ceiling_rh.png",
            ta_col=ta,
            ground_col=g,
            ceiling_col=c,
            rh_col=rh,
            title="Ambient Ta/Ground/Ceiling and RH (Test 48h)",
        )
    else:
        typer.echo(
            "Ambient plot not generated (could not detect required columns). "
            "Use --ta-col, --ground-col, --ceiling-col, --rh-col to specify."
        )
        typer.echo(f"Detected: Ta={ta}, Ground={g}, Ceiling={c}, RH={rh}")

    # ----------------------------
    # Console summary
    # ----------------------------
    typer.echo(f"Alignment complete. Output: {run_dir}")
    typer.echo(f"QC: {qc_path}")
    typer.echo(f"Test start: {qc.test_start}")
    typer.echo(
        f"Power (test_last_24h): kWh/day={power_results.kwh_per_day:.3f}, "
        f"Mean ON={power_results.mean_power_on_w:.1f} W, Mean OFF={power_results.mean_power_off_w:.1f} W, "
        f"Runtime={power_results.runtime_percent:.1f}%"
    )

    typer.echo(f"Power plot: {p_power_path}")
    if vc_paths:
        typer.echo("Voltage/Current plots:")
        for k, p in vc_paths.items():
            typer.echo(f"- {k}: {p}")

    if temp_rep.warnings:
        typer.echo("Temp parse warnings:")
        for w in temp_rep.warnings:
            typer.echo(f"- {w}")

    if qc.warnings:
        typer.echo("QC warnings:")
        for w in qc.warnings:
            # Ignore power stable_24h warnings (not required for your workflow)
            if "Power" in w and "stable_24h" in w:
                continue
            typer.echo(f"- {w}")

    # Fail after writing outputs (so user can inspect)
    if failed:
        typer.echo(
            "FAILED coverage gate (must be <= "
            f"{coverage_max_missing_percent:.2f}% missing rows per required window):"
        )
        for f in failed:
            typer.echo(f"- {f}")
        raise typer.Exit(code=1)
