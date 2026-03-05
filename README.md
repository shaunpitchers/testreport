# ADE Insight

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-Proprietary-red)
![GitHub release](https://img.shields.io/github/v/release/shaunpitchers/ade-insight)

ADE Insight is a professional test results analyser for **BS EN 22041**.

It processes environmental and electrical measurement logs and produces
**aligned datasets, validated statistics, and report-ready figures**
suitable for engineering analysis and compliance documentation.

------------------------------------------------------------------------

## Features

ADE Insight ingests:

-   Temperature / RH **CSV logs**
-   Electrical power analyser **TXT logs**

And produces:

-   Aligned datasets
-   QC validation checks
-   Summary statistics tables
-   Electrical energy metrics
-   Temperature performance statistics
-   Report-ready **PNG plots**

------------------------------------------------------------------------

## Installation

See the installation guide:

➡️ **[INSTALL.md](INSTALL.md)**

This document covers:

-   Linux installation (pip + venv)
-   Windows installation (installer / MSI)
-   Building Windows binaries

------------------------------------------------------------------------

## Usage

### CLI

Display help:

``` bash
adeinsight --help
```

Typical workflow:

``` bash
adeinsight run <temperature_csv> <power_log>
```

Example:

``` bash
adeinsight run data/temperature_log.csv data/power_log.txt
```

Outputs are written to:

    out/gui/<timestamp>/

including:

-   aligned datasets
-   QC validation files
-   statistics tables
-   plots

------------------------------------------------------------------------

### GUI

Launch the graphical interface:

``` bash
adeinsight-gui
```

The GUI allows:

-   selecting input datasets
-   running BS EN 22041 analysis
-   exporting results and plots

------------------------------------------------------------------------

## Example Output

Typical outputs include:

**Temperature statistics plots**

-   Mean / min / max profiles
-   Stability comparisons
-   24‑hour test summaries

**Electrical analysis**

-   Voltage / current / power plots
-   Energy consumption tables
-   Derived performance metrics

Example output structure:

    out/gui/20260305_155141/
    ├── aligned_stable_24h_merged.csv
    ├── aligned_test_48h_merged.csv
    └── results/
        ├── foodstuff_stable_24h.png
        ├── foodstuff_test_last_24h.png
        ├── power_results.csv
        └── summary.json

These outputs are suitable for:

-   engineering analysis
-   validation checks
-   report figures

------------------------------------------------------------------------

## CLI Workflow Example (BS EN 22041)

A typical BS EN 22041 workflow:

1.  Export temperature and power logs from the test rig
2.  Run ADE Insight analysis
3.  Review generated statistics and plots
4.  Include figures and tables in test reports

Example:

``` bash
adeinsight run \
    tests/data/Tc_50_60Hz_CC4_M1_temp.csv \
    tests/data/Tec_CC4_L1_energy_ListMeas.txt
```

Results will include:

-   aligned datasets
-   energy metrics
-   stability plots
-   summary statistics

------------------------------------------------------------------------

## Development

Repository structure (simplified):

    src/ade_insight/     main application code
    build/               packaging and installer configuration
    scripts/             build and install scripts
    tests/               example test data

Windows builds are produced with:

``` powershell
scripts/build_windows.ps1
```

This generates:

    artifacts/
    dist/
    dist-installer/

including installers and checksums.

------------------------------------------------------------------------

## Documentation

Additional documentation:

-   Installation: **INSTALL.md**
-   Release workflow: **release-workflow.md**
-   Changelog: **CHANGELOG.md**

------------------------------------------------------------------------

## Author

**Shaun Mark Charles Pitchers**\
Adande Refrigeration Ltd.

------------------------------------------------------------------------

## License

Proprietary software © ADE.
