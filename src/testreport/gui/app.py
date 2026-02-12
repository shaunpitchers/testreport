from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QGridLayout,
    QDialog,
)

# parsing temp only for "load columns" helper
from testreport.io.temp_csv import parse_temp_rh_csv

# runner
from testreport.standards.bsen22041.runner import run_bsen22041

# Matplotlib viewer (robust)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.image as mpimg


def _set_item(table: QTableWidget, r: int, c: int, text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
    table.setItem(r, c, item)
    return item


class MatplotlibImageDialog(QDialog):
    """Simple interactive viewer using matplotlib toolbar (pan/zoom/save)."""

    def __init__(self, title: str, image_path: Path) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 800)

        layout = QVBoxLayout(self)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        ax = self.fig.add_subplot(111)
        ax.set_axis_off()

        if image_path.exists():
            img = mpimg.imread(str(image_path))
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
        else:
            ax.text(0.5, 0.5, "Missing image", ha="center", va="center")

        self.fig.tight_layout()
        self.canvas.draw()


class PlotThumb(QWidget):
    def __init__(
        self, title: str, image_path: Path, on_open: Callable[[str, Path], None]
    ) -> None:
        super().__init__()
        self._title = title
        self._path = image_path
        self._on_open = on_open

        self.setObjectName("plotThumb")
        self.setStyleSheet(
            """
            QWidget#plotThumb {
                background: #ffffff;
                border: 1px solid #e4e6ef;
                border-radius: 10px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 900; color: #111;")
        self.title_label.setWordWrap(True)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumHeight(160)
        self.img_label.setStyleSheet(
            "border: 1px solid #eef1f7; border-radius: 8px; background:#fafbff;"
        )

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.open_btn = QPushButton("Open")
        self.open_btn.setObjectName("secondary")
        self.open_btn.clicked.connect(self.open)
        btn_row.addWidget(self.open_btn)

        layout.addWidget(self.title_label)
        layout.addWidget(self.img_label, 1)
        layout.addLayout(btn_row)

        self._load_thumb()

    def _load_thumb(self) -> None:
        pix = QPixmap(str(self._path))
        if pix.isNull():
            self.img_label.setText("Missing image")
            return
        thumb = pix.scaled(560, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(thumb)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.open()

    def open(self) -> None:
        self._on_open(self._title, self._path)


class PlotBrowser(QWidget):
    """Scrollable auto-reflow thumbnail grid."""

    def __init__(self) -> None:
        super().__init__()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)

        self.inner = QWidget()
        self.grid = QGridLayout(self.inner)
        self.grid.setContentsMargins(12, 12, 12, 12)
        self.grid.setSpacing(12)
        self.scroll.setWidget(self.inner)

        self._plots: list[tuple[str, Path]] = []
        self._cards: list[PlotThumb] = []
        self._on_open: Callable[[str, Path], None] | None = None

    def clear(self) -> None:
        for w in self._cards:
            self.grid.removeWidget(w)
            w.setParent(None)
        self._cards = []
        self._plots = []

    def set_plots(
        self, plots: list[tuple[str, Path]], on_open: Callable[[str, Path], None]
    ) -> None:
        self.clear()
        self._plots = plots
        self._on_open = on_open
        self._rebuild()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rebuild()

    def _calc_cols(self) -> int:
        viewport_w = self.scroll.viewport().width()
        target = 520
        cols = max(1, viewport_w // target)
        return min(cols, 4)

    def _rebuild(self) -> None:
        for w in self._cards:
            self.grid.removeWidget(w)
            w.setParent(None)
        self._cards = []

        if not self._plots:
            lbl = QLabel("No plots available for this run.")
            lbl.setAlignment(Qt.AlignCenter)
            self.grid.addWidget(lbl, 0, 0)
            return

        cols = self._calc_cols()
        r = 0
        c = 0
        for title, path in self._plots:
            card = PlotThumb(title, path, self._on_open or (lambda *_: None))
            self._cards.append(card)
            self.grid.addWidget(card, r, c)
            c += 1
            if c >= cols:
                c = 0
                r += 1


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ADE Insight — BS EN 22041")
        self.resize(1400, 900)

        self._apply_stylesheet()

        # Inputs
        self.temp_path = QLineEdit()
        self.power_path = QLineEdit()
        self.out_dir = QLineEdit("out/gui")
        self.tz = QLineEdit("Europe/London")
        self.test_start = QLineEdit("2025-10-07 14:00:00")

        # Settings
        self.resample_seconds = QSpinBox()
        self.resample_seconds.setRange(1, 3600)
        self.resample_seconds.setValue(10)

        self.compressor_threshold = QDoubleSpinBox()
        self.compressor_threshold.setRange(0.0, 10000.0)
        self.compressor_threshold.setDecimals(1)
        self.compressor_threshold.setValue(50.0)

        self.coverage_max_missing_percent = QDoubleSpinBox()
        self.coverage_max_missing_percent.setRange(0.0, 100.0)
        self.coverage_max_missing_percent.setDecimals(2)
        self.coverage_max_missing_percent.setValue(0.50)

        self.probe_distance_m = QDoubleSpinBox()
        self.probe_distance_m.setRange(0.1, 100.0)
        self.probe_distance_m.setDecimals(2)
        self.probe_distance_m.setValue(2.50)

        # Column overrides
        self.combo_ta = QComboBox()
        self.combo_ground = QComboBox()
        self.combo_ceiling = QComboBox()
        self.combo_rh = QComboBox()

        # Temp stats selector (will remain empty unless summary.json contains temp_summary)
        self.temp_stats_window = QComboBox()
        self.temp_stats_window.addItem("Stable 24h", "stable_24h")
        self.temp_stats_window.addItem("Test last 24h", "test_last_24h")
        self.temp_stats_window.currentIndexChanged.connect(
            self._refresh_temp_stats_table
        )

        # Buttons
        btn_temp = QPushButton("Browse…")
        btn_power = QPushButton("Browse…")
        btn_out = QPushButton("Browse…")
        btn_load_cols = QPushButton("Load columns")
        btn_run = QPushButton("Run")
        btn_open = QPushButton("Open output folder")
        btn_exit = QPushButton("Exit")

        btn_open.setObjectName("secondary")
        btn_load_cols.setObjectName("secondary")
        btn_exit.setObjectName("danger")

        btn_temp.clicked.connect(self.pick_temp)
        btn_power.clicked.connect(self.pick_power)
        btn_out.clicked.connect(self.pick_out_dir)
        btn_load_cols.clicked.connect(self.load_columns_from_temp)
        btn_run.clicked.connect(self.run_pipeline)
        btn_open.clicked.connect(self.open_output_folder)
        btn_exit.clicked.connect(self.close)

        # Branding header
        self.logo = QLabel()
        self.logo.setFixedHeight(88)
        self.logo.setFixedWidth(320)
        self.logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.brand_title = QLabel("ADE Insight")
        self.brand_title.setObjectName("brandTitle")
        self.brand_subtitle = QLabel(
            "Test results analyser for report-ready outputs (BS EN 22041)"
        )
        self.brand_subtitle.setObjectName("brandSubtitle")

        brand_text = QVBoxLayout()
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(2)
        brand_text.addWidget(self.brand_title, 0, Qt.AlignLeft)
        brand_text.addWidget(self.brand_subtitle, 0, Qt.AlignLeft)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(6)
        brand_row.addWidget(self.logo, 0, Qt.AlignLeft)
        brand_row.addLayout(brand_text, 0)
        brand_row.addStretch(1)

        brand_box = QWidget()
        brand_box.setLayout(brand_row)

        self._try_load_logo()

        # Tabs
        self.tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)

        # QC table (readable + coloured)
        self.qc_table = QTableWidget(0, 6)
        self.qc_table.setHorizontalHeaderLabels(
            ["Window", "Dataset", "Missing %", "Gate %", "Status", "Notes"]
        )
        self.qc_table.horizontalHeader().setStretchLastSection(True)
        self.qc_table.setAlternatingRowColors(True)

        # Temp stats table (requires summary["temp_summary"])
        self.temp_stats_table = QTableWidget(0, 4)
        self.temp_stats_table.setHorizontalHeaderLabels(
            ["Probe", "Min (°C)", "Mean (°C)", "Max (°C)"]
        )
        self.temp_stats_table.horizontalHeader().setStretchLastSection(True)

        temp_stats_page = QWidget()
        temp_stats_layout = QVBoxLayout(temp_stats_page)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Window:"))
        selector_row.addWidget(self.temp_stats_window)
        selector_row.addStretch(1)
        temp_stats_layout.addLayout(selector_row)
        temp_stats_layout.addWidget(self.temp_stats_table)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # Plot tabs
        self.plots_tabs = QTabWidget()
        self.plots_electrical = PlotBrowser()
        self.plots_food = PlotBrowser()
        self.plots_ambient = PlotBrowser()
        self.plots_tabs.addTab(self.plots_electrical, "Electrical")
        self.plots_tabs.addTab(self.plots_food, "Food temperature")
        self.plots_tabs.addTab(self.plots_ambient, "Ambient")

        self.tabs.addTab(self.summary_text, "Summary")
        self.tabs.addTab(self.qc_table, "QC")
        self.tabs.addTab(temp_stats_page, "Temp stats")
        self.tabs.addTab(self.plots_tabs, "Plots")
        self.tabs.addTab(self.log, "Logs")

        # Layout
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        left = QVBoxLayout()
        main.addLayout(left, 1)
        left.addWidget(brand_box)

        file_box = QGroupBox("Inputs")
        file_form = QFormLayout(file_box)

        temp_row = QHBoxLayout()
        temp_row.addWidget(self.temp_path, 1)
        temp_row.addWidget(btn_temp)
        file_form.addRow("Temp/RH CSV:", temp_row)

        power_row = QHBoxLayout()
        power_row.addWidget(self.power_path, 1)
        power_row.addWidget(btn_power)
        file_form.addRow("Power TXT:", power_row)

        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(btn_out)
        file_form.addRow("Output dir:", out_row)

        file_form.addRow("Test start (local):", self.test_start)
        file_form.addRow("Timezone:", self.tz)
        left.addWidget(file_box)

        settings_box = QGroupBox("Settings")
        settings_form = QFormLayout(settings_box)
        settings_form.addRow("Resample seconds:", self.resample_seconds)
        settings_form.addRow("Compressor ON threshold (W):", self.compressor_threshold)
        settings_form.addRow("Max missing % (gate):", self.coverage_max_missing_percent)
        settings_form.addRow("Probe distance (m):", self.probe_distance_m)
        left.addWidget(settings_box)

        col_box = QGroupBox("Ambient column selection (auto-detect + override)")
        col_form = QFormLayout(col_box)
        col_form.addRow("Ta (room):", self.combo_ta)
        col_form.addRow("Ground:", self.combo_ground)
        col_form.addRow("Ceiling:", self.combo_ceiling)
        col_form.addRow("RH:", self.combo_rh)

        col_btn_row = QHBoxLayout()
        col_btn_row.addWidget(btn_load_cols)
        col_form.addRow(col_btn_row)
        left.addWidget(col_box)

        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_run)
        btn_row.addWidget(btn_open)
        btn_row.addWidget(btn_exit)
        left.addLayout(btn_row)

        main.addWidget(self.tabs, 2)

        self._last_run_dir: Path | None = None
        self._last_summary: dict[str, Any] | None = None
        self._last_qc: dict[str, Any] | None = None

    # ---------------------------
    # Styling / logo
    # ---------------------------
    def _apply_stylesheet(self) -> None:
        qss_path = Path(__file__).parent / "style.qss"
        if qss_path.exists():
            self.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    def _try_load_logo(self) -> None:
        candidate = Path(__file__).parent / "assets" / "adande_logo.png"
        if candidate.exists():
            pix = QPixmap(str(candidate))
            if not pix.isNull():
                scaled = pix.scaled(
                    self.logo.width(),
                    self.logo.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.logo.setPixmap(scaled)
                return
        self.logo.setText("ADE")
        self.logo.setStyleSheet("color:#333; font-weight:900; font-size:18pt;")

    # ---------------------------
    # Helpers
    # ---------------------------
    def log_line(self, s: str) -> None:
        self.log.append(s)

    def pick_temp(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select temperature/RH CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if fn:
            self.temp_path.setText(fn)

    def pick_power(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select power TXT", "", "Text files (*.txt);;All files (*)"
        )
        if fn:
            self.power_path.setText(fn)

    def pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.out_dir.setText(d)

    def open_output_folder(self) -> None:
        if not self._last_run_dir:
            QMessageBox.information(
                self, "No output", "Run ADE Insight first to generate output."
            )
            return
        import os
        import subprocess

        p = str(self._last_run_dir)
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", p], check=False)
        else:
            subprocess.run(["xdg-open", p], check=False)

    def _combo_set(
        self, combo: QComboBox, items: list[str], preferred: str | None
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(auto)", "")
        for it in items:
            combo.addItem(it, it)
        if preferred:
            idx = combo.findData(preferred)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def load_columns_from_temp(self) -> None:
        temp = self.temp_path.text().strip()
        tz = self.tz.text().strip() or "Europe/London"
        if not temp or not Path(temp).exists():
            QMessageBox.critical(
                self, "Missing input", "Select a valid Temp/RH CSV first."
            )
            return

        self.log_line("Loading columns from temp CSV (quick parse)…")
        try:
            temp_df, rep = parse_temp_rh_csv(
                Path(temp),
                tz=tz,
                numeric_time_is_utc=True,
                time_base="excel_days",
            )
            cols = [c for c in temp_df.columns if c != "time"]

            def find_pref(keys: list[str]) -> str | None:
                for c in cols:
                    name = str(c).strip().lower()
                    if any(k in name for k in keys):
                        return c
                return None

            pref_ta = find_pref(["ta"]) or (
                "ROOM TEMP 1"
                if "ROOM TEMP 1" in cols
                else find_pref(["room temp", "ambient temp", "air temp"])
            )
            pref_g = "Ground" if "Ground" in cols else find_pref(["ground"])
            pref_c = "Ceiling" if "Ceiling" in cols else find_pref(["ceiling"])
            pref_rh = (
                "ROOM HUMIDITY 1"
                if "ROOM HUMIDITY 1" in cols
                else find_pref(["humidity", " rh", "rh"])
            )

            self._combo_set(self.combo_ta, cols, pref_ta)
            self._combo_set(self.combo_ground, cols, pref_g)
            self._combo_set(self.combo_ceiling, cols, pref_c)
            self._combo_set(self.combo_rh, cols, pref_rh)

            if rep.warnings:
                self.log_line("Temp parse warnings:")
                for w in rep.warnings:
                    self.log_line(f"- {w}")

            QMessageBox.information(
                self, "Columns loaded", "Column dropdowns populated."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load columns:\n{e}")
            self.log_line(f"ERROR loading columns: {e}")

    def _open_plot_viewer(self, title: str, path: Path) -> None:
        try:
            dlg = MatplotlibImageDialog(title, path)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(
                self, "Plot viewer error", f"Failed to open plot:\n{e}"
            )
            self.log_line(f"Plot viewer ERROR ({path}): {e}")

    # ---------------------------
    # QC helpers (colour + rows)
    # ---------------------------
    def _set_row_color(self, row: int, status: str) -> None:
        # subtle but obvious
        if status == "PASS":
            bg = Qt.green
            fg = Qt.black
        elif status == "WARN":
            bg = Qt.yellow
            fg = Qt.black
        else:
            bg = Qt.red
            fg = Qt.white

        for c in range(self.qc_table.columnCount()):
            it = self.qc_table.item(row, c)
            if it:
                it.setBackground(bg)
                it.setForeground(fg)

    def _qc_add_row(
        self,
        window: str,
        dataset: str,
        missing_frac: float | None,
        gate_percent: float,
        status: str,
        notes: str,
    ) -> None:
        r = self.qc_table.rowCount()
        self.qc_table.insertRow(r)

        miss_str = "" if missing_frac is None else f"{missing_frac * 100:.2f}"
        _set_item(self.qc_table, r, 0, window)
        _set_item(self.qc_table, r, 1, dataset)
        _set_item(self.qc_table, r, 2, miss_str)
        _set_item(self.qc_table, r, 3, f"{gate_percent:.2f}")
        _set_item(self.qc_table, r, 4, status)
        _set_item(self.qc_table, r, 5, notes)

        self._set_row_color(r, status)

    # ---------------------------
    # Loading outputs into UI
    # ---------------------------
    def _read_json(self, p: Path) -> dict[str, Any] | None:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _load_run_outputs(
        self, run_dir: Path
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        summary_path = run_dir / "results" / "summary.json"
        summary = self._read_json(summary_path) if summary_path.exists() else None

        qc = None
        qc_candidates = (
            list(run_dir.glob("*qc.json"))
            + list(run_dir.glob("*_qc.json"))
            + list(run_dir.glob("**/*qc.json"))
        )
        qc_candidates.sort(key=lambda x: (0 if "aligned" in x.name else 1, len(str(x))))
        for c in qc_candidates:
            q = self._read_json(c)
            if isinstance(q, dict) and q:
                qc = q
                break

        return summary, qc

    def _format_summary_text(self, summary: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("ADE Insight — BS EN 22041")
        lines.append("")
        lines.append(f"Test start: {summary.get('test_start', '')}")
        lines.append(f"Timezone: {summary.get('tz', '')}")
        lines.append(f"Resample: {summary.get('resample_seconds', '')} s")
        lines.append("")

        pr = summary.get("power_results", {}) or {}
        if pr:
            lines.append("Electrical (test_last_24h)")
            lines.append(f"- kWh/day: {pr.get('kwh_per_day', '')}")
            lines.append(f"- Mean power ON (W): {pr.get('mean_power_on_w', '')}")
            lines.append(f"- Mean power OFF (W): {pr.get('mean_power_off_w', '')}")
            lines.append(f"- Runtime (%): {pr.get('runtime_percent', '')}")
            lines.append(f"- Mean current (A): {pr.get('mean_current_a', '')}")
            lines.append(f"- Mean current ON (A): {pr.get('mean_current_on_a', '')}")
            lines.append(f"- Mean current OFF (A): {pr.get('mean_current_off_a', '')}")
            lines.append(f"- Mean PF: {pr.get('mean_power_factor', '')}")
            lines.append(f"- Mean PF ON: {pr.get('mean_power_factor_on', '')}")
            lines.append(f"- Mean PF OFF: {pr.get('mean_power_factor_off', '')}")
            lines.append("")

        ag = summary.get("ambient_gradient", None)
        if isinstance(ag, dict) and ag:
            lines.append("Ambient gradient (test_48h)")
            lines.append(f"- Gradient (°C/m): {ag.get('gradient_c_per_m', '')}")
            lines.append(f"- Ceiling mean (°C): {ag.get('ceiling_mean_c', '')}")
            lines.append(f"- Ground mean (°C): {ag.get('ground_mean_c', '')}")
            lines.append(f"- Distance (m): {ag.get('distance_m', '')}")
            lines.append("")

        warns = summary.get("warnings", {}) or {}
        qc_warns = []
        if isinstance(warns, dict):
            qc_warns = warns.get("qc", []) or []
        if qc_warns:
            lines.append("QC warnings:")
            for w in qc_warns:
                lines.append(f"- {w}")
            lines.append("")

        return "\n".join(lines)

    def _populate_qc_table(
        self, summary: dict[str, Any] | None, qc: dict[str, Any] | None
    ) -> None:
        self.qc_table.setRowCount(0)

        # Prefer summary coverage; fallback to qc json
        temp_missing: dict[str, float] = {}
        power_missing: dict[str, float] = {}

        if summary and isinstance(summary.get("coverage_missing_frac"), dict):
            cov = summary["coverage_missing_frac"]
            if isinstance(cov.get("temp"), dict):
                temp_missing = {k: float(v) for k, v in cov["temp"].items()}
            if isinstance(cov.get("power"), dict):
                power_missing = {k: float(v) for k, v in cov["power"].items()}

        if qc and not temp_missing and isinstance(qc.get("temp_missing_frac"), dict):
            temp_missing = {k: float(v) for k, v in qc["temp_missing_frac"].items()}
        if qc and not power_missing and isinstance(qc.get("power_missing_frac"), dict):
            power_missing = {k: float(v) for k, v in qc["power_missing_frac"].items()}

        gate_percent = float(self.coverage_max_missing_percent.value())
        gate_frac = gate_percent / 100.0

        # Policy
        required_temp = {"stable_24h", "test_48h", "test_first_24h", "test_last_24h"}
        required_power = {"test_last_24h"}
        warn_power = {"test_48h", "test_first_24h"}  # warn-only if present/over gate

        windows = sorted(
            set(temp_missing.keys())
            | set(power_missing.keys())
            | required_temp
            | required_power
        )

        for w in windows:
            # TEMP
            t = temp_missing.get(w, None)
            if w in required_temp:
                if t is None:
                    self._qc_add_row(
                        w,
                        "Temp",
                        None,
                        gate_percent,
                        "FAIL",
                        "Missing window or no data",
                    )
                elif t <= gate_frac:
                    self._qc_add_row(w, "Temp", t, gate_percent, "PASS", "")
                else:
                    self._qc_add_row(
                        w, "Temp", t, gate_percent, "FAIL", "Over coverage gate"
                    )
            else:
                if t is not None:
                    self._qc_add_row(w, "Temp", t, gate_percent, "PASS", "Info")

            # POWER
            p = power_missing.get(w, None)
            if w in required_power:
                if p is None:
                    self._qc_add_row(
                        w,
                        "Power",
                        None,
                        gate_percent,
                        "FAIL",
                        "Missing window or no data",
                    )
                elif p <= gate_frac:
                    self._qc_add_row(w, "Power", p, gate_percent, "PASS", "")
                else:
                    self._qc_add_row(
                        w, "Power", p, gate_percent, "FAIL", "Over coverage gate"
                    )
            elif w in warn_power:
                if p is None:
                    self._qc_add_row(
                        w,
                        "Power",
                        None,
                        gate_percent,
                        "WARN",
                        "No data for this window",
                    )
                elif p <= gate_frac:
                    self._qc_add_row(w, "Power", p, gate_percent, "PASS", "")
                else:
                    self._qc_add_row(
                        w, "Power", p, gate_percent, "WARN", "Over gate (warn-only)"
                    )
            else:
                # stable_24h power is irrelevant; show only if present
                if p is not None:
                    status = "PASS" if p <= gate_frac else "WARN"
                    note = "Info" if status == "PASS" else "Over gate (info)"
                    self._qc_add_row(w, "Power", p, gate_percent, status, note)

        self.qc_table.resizeColumnsToContents()
        self.qc_table.horizontalHeader().setStretchLastSection(True)

    def _refresh_temp_stats_table(self) -> None:
        # Uses summary payload if runner includes it; otherwise stays blank.
        if not self._last_summary:
            self.temp_stats_table.setRowCount(0)
            return

        win = self.temp_stats_window.currentData()
        ts = self._last_summary.get("temp_summary", {})
        win_payload = ts.get(win, {})
        per_probe = (
            win_payload.get("per_probe", {}) if isinstance(win_payload, dict) else {}
        )

        probes = sorted(per_probe.keys(), key=lambda x: (len(x), x))
        self.temp_stats_table.setRowCount(len(probes))
        for r, probe in enumerate(probes):
            stats = per_probe.get(probe, {})
            _set_item(self.temp_stats_table, r, 0, str(probe))
            _set_item(
                self.temp_stats_table,
                r,
                1,
                f"{float(stats.get('min', float('nan'))):.2f}",
            )
            _set_item(
                self.temp_stats_table,
                r,
                2,
                f"{float(stats.get('mean', float('nan'))):.2f}",
            )
            _set_item(
                self.temp_stats_table,
                r,
                3,
                f"{float(stats.get('max', float('nan'))):.2f}",
            )
        self.temp_stats_table.resizeColumnsToContents()

    def _update_plot_tabs(self, plots: dict[str, str | None]) -> None:
        electrical_keys = {"power", "voltage", "current"}
        food_keys = {"food_stable", "food_last", "food_stable_mmm", "food_last_mmm"}
        ambient_keys = {"ambient"}

        def build(keys: set[str]) -> list[tuple[str, Path]]:
            out: list[tuple[str, Path]] = []
            for k in sorted(keys):
                p = plots.get(k)
                if p:
                    pp = Path(p)
                    if pp.exists():
                        out.append((k, pp))
            return out

        self.plots_electrical.set_plots(build(electrical_keys), self._open_plot_viewer)
        self.plots_food.set_plots(build(food_keys), self._open_plot_viewer)
        self.plots_ambient.set_plots(build(ambient_keys), self._open_plot_viewer)

    # ---------------------------
    # Run
    # ---------------------------
    def run_pipeline(self) -> None:
        temp = self.temp_path.text().strip()
        power = self.power_path.text().strip()
        out_dir = self.out_dir.text().strip()
        test_start = self.test_start.text().strip()
        tz = self.tz.text().strip()

        if not temp or not Path(temp).exists():
            QMessageBox.critical(
                self, "Missing input", "Please select a valid Temp/RH CSV file."
            )
            return
        if not power or not Path(power).exists():
            QMessageBox.critical(
                self, "Missing input", "Please select a valid Power TXT file."
            )
            return
        if not out_dir:
            QMessageBox.critical(
                self, "Missing output", "Please specify an output directory."
            )
            return
        if not test_start:
            QMessageBox.critical(
                self, "Missing test start", "Please enter test start time."
            )
            return
        if not tz:
            QMessageBox.critical(
                self,
                "Missing timezone",
                "Please enter a timezone (e.g. Europe/London).",
            )
            return

        # Reset UI
        self.log.clear()
        self.summary_text.setPlainText("Running ADE Insight…")
        self.qc_table.setRowCount(0)
        self.temp_stats_table.setRowCount(0)
        self._last_summary = None
        self._last_qc = None
        self.plots_electrical.clear()
        self.plots_food.clear()
        self.plots_ambient.clear()

        self.tabs.setCurrentIndex(4)

        overrides = {
            "ta_col": self.combo_ta.currentData() or None,
            "ground_col": self.combo_ground.currentData() or None,
            "ceiling_col": self.combo_ceiling.currentData() or None,
            "rh_col": self.combo_rh.currentData() or None,
        }

        try:
            self.log_line("Running BS EN 22041 pipeline…")
            result = run_bsen22041(
                temp_file=Path(temp),
                power_file=Path(power),
                test_start=test_start,
                out_dir=Path(out_dir),
                tz=tz,
                resample_seconds=int(self.resample_seconds.value()),
                compressor_on_threshold_w=float(self.compressor_threshold.value()),
                coverage_max_missing_percent=float(
                    self.coverage_max_missing_percent.value()
                ),
                probe_distance_m=float(self.probe_distance_m.value()),
                **overrides,
            )

            run_dir = Path(result.run_dir)
            self._last_run_dir = run_dir

            # Load outputs from disk (robust)
            summary, qc = self._load_run_outputs(run_dir)
            self._last_summary = (
                summary if summary else (getattr(result, "summary", None) or None)
            )
            self._last_qc = qc

            # Plots: prefer result.plots, fallback summary["plots"]
            plots = getattr(result, "plots", None)
            if (
                not plots
                and self._last_summary
                and isinstance(self._last_summary.get("plots"), dict)
            ):
                plots = self._last_summary["plots"]
            if isinstance(plots, dict):
                self._update_plot_tabs(plots)

            # Summary
            if self._last_summary and isinstance(self._last_summary, dict):
                self.summary_text.setPlainText(
                    self._format_summary_text(self._last_summary)
                )
            else:
                self.summary_text.setPlainText(
                    f"Run completed. Output: {run_dir}\n(No summary.json found)"
                )

            # QC
            self._populate_qc_table(
                self._last_summary if isinstance(self._last_summary, dict) else None,
                qc,
            )

            # Logs
            self.log_line(f"Output directory: {run_dir}")
            if self._last_summary and isinstance(
                self._last_summary.get("warnings"), dict
            ):
                warns = self._last_summary["warnings"]
                tpw = warns.get("temp_parse", []) or []
                qcw = warns.get("qc", []) or []
                if tpw:
                    self.log_line("Temp parse warnings:")
                    for w in tpw:
                        self.log_line(f"- {w}")
                if qcw:
                    self.log_line("QC warnings:")
                    for w in qcw:
                        self.log_line(f"- {w}")

            # Temp stats (only if temp_summary exists in summary.json)
            self._refresh_temp_stats_table()

            self.tabs.setCurrentIndex(0)
            QMessageBox.information(self, "Run finished", "Run finished successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Run failed:\n{e}")
            self.log_line(f"ERROR: {e}")
            self.tabs.setCurrentIndex(4)


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
