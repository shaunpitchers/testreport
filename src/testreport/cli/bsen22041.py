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
from testreport.core.plots import plot_voltage_current
from testreport.core.temp_stats import (
    compute_column_stats,
    detect_foodstuff_columns,
    detect_ambient_columns,
)
from testreport.core.temp_plots import (
    plot_foodstuff_temps,
    plot_ambient_twin_axis,
)
from testreport.core.ambient_gradient import compute_ambient_gradient

app = typer.Typer(help="BS EN 22041 tools")


@app.command("align")
def align(
    temp_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Temperature/RH CSV (numeric time, UTC-based)",
    ),
    power_file: Path = typer.Argument(
        ..., exists=True, readable=True, help="Power analyser TXT (local clock time)"
    ),
    test_start: str = typer.Option(
        ..., "--test-start", help="Door-opening start time (local clock, Europe/London)"
    ),
    out_dir: Path = typer.Option(Path("out/inspect"), help="Output directory"),
    tz: str = typer.Option("Europe/London", help="Timezone"),
    resample_seconds: int = typer.Option(10, help="Alignment grid step in seconds"),
    prefix: str = typer.Option("aligned", help="Output file prefix"),
    compressor_on_threshold_w: float = typer.Option(
        50.0,
        "--compressor-on-threshold-w",
        help="Power threshold for compressor ON (W)",
    ),
    coverage_max_missing_percent: float = typer.Option(
        0.5,
        "--coverage-max-missing-percent",
        help="Max allowed missing rows per window (%)",
    ),
    # Ambient detection overrides (optional)
    ambient_temp_col: str | None = typer.Option(
        None, "--ambient-temp-col", help="Override ambient dry bulb column name"
    ),
    ambient_rh_col: str | None = typer.Option(
        None, "--ambient-rh-col", help="Override ambient RH column name"
    ),
    # Ambient gradient selection (required for gradient)
    ground_col: str | None = typer.Option(
        None, "--ground-col", help="Ambient ground probe column name (for gradient)"
    ),
    ceiling_col: str | None = typer.Option(
        None, "--ceiling-col", help="Ambient ceiling probe column name (for gradient)"
    ),
    probe_distance_m: float = typer.Option(
        2.5, "--probe-distance-m", help="Distance between ceiling and ground probes (m)"
    ),
):
    # Timestamped run folder to avoid overwriting
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Parse temperature: numeric time is UTC -> convert to Europe/London (DST handled)
    temp_df, temp_rep = parse_temp_rh_csv(
        temp_file,
        tz=tz,
        numeric_time_is_utc=True,
        time_base="excel_days",
    )

    # Parse power: timestamps are local clock time -> localize directly
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

    # Align by test start and create windows
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

    # Coverage gate (<= 0.5% missing for now, configurable)
    max_missing = float(coverage_max_missing_percent) / 100.0
    failed: list[str] = []
    for wname, miss in qc.temp_missing_frac.items():
        if miss > max_missing:
            failed.append(f"Temp window {wname}: missing={miss * 100:.2f}%")
    for wname, miss in qc.power_missing_frac.items():
        if miss > max_missing:
            failed.append(f"Power window {wname}: missing={miss * 100:.2f}%")

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Power results: test_last_24h
    # ----------------------------
    if "test_last_24h" not in power_w:
        typer.echo("Missing power window: test_last_24h")
        raise typer.Exit(code=2)

    p_last = power_w["test_last_24h"]
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

    plot_paths = plot_voltage_current(p_last, results_dir, prefix="test_last_24h")

    # ----------------------------
    # Temperature plots + stats
    # ----------------------------
    if (
        "stable_24h" not in temp_w
        or "test_last_24h" not in temp_w
        or "test_48h" not in temp_w
    ):
        typer.echo(
            "Missing one or more temp windows: stable_24h, test_last_24h, test_48h"
        )
        raise typer.Exit(code=2)

    t_stable = temp_w["stable_24h"]
    t_last = temp_w["test_last_24h"]
    t_test48 = temp_w["test_48h"]

    food_cols = detect_foodstuff_columns(t_stable, expected=8)
    if not food_cols:
        typer.echo("Could not detect foodstuff columns '1'..'8' in aligned temp data.")
    else:
        plot_foodstuff_temps(
            t_stable,
            food_cols,
            results_dir / "foodstuff_stable_24h.png",
            title="Foodstuff temperatures (Stable 24h)",
        )
        plot_foodstuff_temps(
            t_last,
            food_cols,
            results_dir / "foodstuff_test_last_24h.png",
            title="Foodstuff temperatures (Test last 24h)",
        )

        stable_stats = compute_column_stats(t_stable, food_cols)
        last_stats = compute_column_stats(t_last, food_cols)

        stable_stats.to_csv(results_dir / "foodstuff_stats_stable_24h.csv", index=False)
        last_stats.to_csv(
            results_dir / "foodstuff_stats_test_last_24h.csv", index=False
        )

        (results_dir / "foodstuff_stats_stable_24h.json").write_text(
            stable_stats.to_json(orient="records", indent=2),
            encoding="utf-8",
        )
        (results_dir / "foodstuff_stats_test_last_24h.json").write_text(
            last_stats.to_json(orient="records", indent=2),
            encoding="utf-8",
        )

    # Ambient twin-axis plot over full test (48h)
    amb_t, amb_rh = detect_ambient_columns(
        t_test48,
        ambient_temp_hint=ambient_temp_col,
        ambient_rh_hint=ambient_rh_col,
    )

    if amb_t and amb_rh:
        plot_ambient_twin_axis(
            t_test48,
            results_dir / "ambient_test_48h.png",
            ambient_temp_col=amb_t,
            ambient_rh_col=amb_rh,
            title="Ambient dry bulb and RH (Test 48h)",
        )
    else:
        typer.echo(
            "Ambient columns not detected for plot. Use --ambient-temp-col and --ambient-rh-col."
        )

    # ----------------------------
    # Ambient gradient (Test 48h)
    # ----------------------------
    gradient_payload = None
    if ground_col and ceiling_col:
        try:
            grad = compute_ambient_gradient(
                t_test48,
                window_name="test_48h",
                ground_col=ground_col,
                ceiling_col=ceiling_col,
                distance_m=probe_distance_m,
            )
            gradient_payload = grad.__dict__
            (results_dir / "ambient_gradient.json").write_text(
                json.dumps(gradient_payload, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            typer.echo(f"Ambient gradient failed: {e}")
    else:
        typer.echo(
            "Ambient gradient not computed. Provide --ground-col and --ceiling-col (and optional --probe-distance-m)."
        )

    # ----------------------------
    # Summary JSON (single place to read later)
    # ----------------------------
    summary = {
        "test_start": str(qc.test_start),
        "tz": tz,
        "resample_seconds": resample_seconds,
        "raw_dt_median_seconds": {"temp": temp_dt_med, "power": power_dt_med},
        "power_results": power_results.__dict__,
        "ambient_gradient": gradient_payload,
        "coverage_missing_frac": {
            "temp": qc.temp_missing_frac,
            "power": qc.power_missing_frac,
        },
        "warnings": {"temp_parse": temp_rep.warnings, "qc": qc.warnings},
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

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

    if gradient_payload:
        typer.echo(
            f"Ambient gradient (test_48h): {gradient_payload['gradient_c_per_m']:.3f} °C/m "
            f"(ceiling_mean={gradient_payload['ceiling_mean_c']:.2f}°C, "
            f"ground_mean={gradient_payload['ground_mean_c']:.2f}°C, "
            f"distance={gradient_payload['distance_m']:.2f}m)"
        )

    if plot_paths:
        typer.echo("Power plots:")
        for k, p in plot_paths.items():
            typer.echo(f"- {k}: {p}")

    if temp_rep.warnings:
        typer.echo("Temp parse warnings:")
        for w in temp_rep.warnings:
            typer.echo(f"- {w}")

    if qc.warnings:
        typer.echo("QC warnings:")
        for w in qc.warnings:
            typer.echo(f"- {w}")

    # Fail after writing outputs (so user can inspect)
    if failed:
        typer.echo(
            "FAILED coverage gate (must be <= "
            f"{coverage_max_missing_percent:.2f}% missing rows per window):"
        )
        for f in failed:
            typer.echo(f"- {f}")
        raise typer.Exit(code=1)
