#!/usr/bin/env bash
set -euo pipefail

APP_NAME="ade-insight"
INSTALL_DIR="${HOME}/.local/${APP_NAME}"
VENV_DIR="${INSTALL_DIR}/venv"
BIN_DIR="${HOME}/.local/bin"

# You can install from:
#  1) a wheel file path (recommended for releases), or
#  2) the current repo directory (dev install)
WHEEL_PATH="${1:-}"

msg() { echo -e "\n==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 not found. Install Python 3.10+ and retry."

PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python3 - <<'PY' || die "Python 3.10+ required."
import sys
assert sys.version_info >= (3,10)
PY

msg "Creating install directories"
mkdir -p "${INSTALL_DIR}" "${BIN_DIR}"

msg "Creating virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

msg "Upgrading pip"
python -m pip install --upgrade pip

if [[ -n "${WHEEL_PATH}" ]]; then
  [[ -f "${WHEEL_PATH}" ]] || die "Wheel not found: ${WHEEL_PATH}"
  msg "Installing from wheel: ${WHEEL_PATH}"
  pip install "${WHEEL_PATH}"
else
  msg "Installing from current directory (editable)"
  # assumes this script is run from the repo root or you cd there first
  pip install -e ".[gui]"
fi

msg "Creating launchers in ${BIN_DIR}"
cat > "${BIN_DIR}/adeinsight" <<EOF
#!/usr/bin/env bash
exec "${VENV_DIR}/bin/adeinsight" "\$@"
EOF

cat > "${BIN_DIR}/adeinsight-gui" <<EOF
#!/usr/bin/env bash
exec "${VENV_DIR}/bin/adeinsight-gui" "\$@"
EOF

chmod +x "${BIN_DIR}/adeinsight" "${BIN_DIR}/adeinsight-gui"

msg "Done."
echo "If ${BIN_DIR} is not on your PATH, add this to your shell profile:"
echo "  export PATH=\"${BIN_DIR}:\$PATH\""
echo ""
echo "Try:"
echo "  adeinsight --help"
