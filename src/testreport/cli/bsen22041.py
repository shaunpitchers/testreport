from __future__ import annotations

from pathlib import Path
import typer

from testreport.standards.bsen22041.runner import run_bsen22041

app = typer.Typer(help="BS EN 22041 tools")


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
    ta_col: str | None = typer.Option(None, "--ta-col"),
    ground_col: str | None = typer.Option(None, "--ground-col"),
    ceiling_col: str | None = typer.Option(None, "--ceiling-col"),
    rh_col: str | None = typer.Option(None, "--rh-col"),
    probe_distance_m: float = typer.Option(2.5, "--probe-distance-m"),
):
    result = run_bsen22041(
        temp_file=temp_file,
        power_file=power_file,
        test_start=test_start,
        out_dir=out_dir,
        tz=tz,
        resample_seconds=resample_seconds,
        prefix=prefix,
        compressor_on_threshold_w=compressor_on_threshold_w,
        coverage_max_missing_percent=coverage_max_missing_percent,
        ta_col=ta_col,
        ground_col=ground_col,
        ceiling_col=ceiling_col,
        rh_col=rh_col,
        probe_distance_m=probe_distance_m,
    )

    typer.echo(f"Output: {result.run_dir}")
    typer.echo(f"QC: {result.qc_path}")
    typer.echo(f"Summary: {result.summary_path}")

    pr = result.summary["power_results"]
    typer.echo(
        f"Power (test_last_24h): kWh/day={pr['kwh_per_day']:.3f}, "
        f"Mean ON={pr['mean_power_on_w']:.1f} W, Mean OFF={pr['mean_power_off_w']:.1f} W, "
        f"Runtime={pr['runtime_percent']:.1f}%"
    )
    typer.echo(
        f"Current (A): mean={pr['mean_current_a']:.3f}, "
        f"ON={pr['mean_current_on_a']:.3f}, OFF={pr['mean_current_off_a']:.3f}"
    )
    typer.echo(
        f"Power factor: mean={pr['mean_power_factor']:.3f}, "
        f"ON={pr['mean_power_factor_on']:.3f}, OFF={pr['mean_power_factor_off']:.3f}"
    )

    if result.warnings:
        typer.echo("Warnings:")
        for w in result.warnings:
            typer.echo(f"- {w}")

    if not result.passed_coverage_gate:
        typer.echo("FAILED coverage gate:")
        for f in result.failed_reasons:
            typer.echo(f"- {f}")
        raise typer.Exit(code=1)
