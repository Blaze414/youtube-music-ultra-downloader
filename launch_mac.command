#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

APP_MAIN="ultra_downloader_qt_modern.py"
VENV_DIR=".venv"

print_header() {
  echo
  echo "============================================================"
  echo "  YouTube Music Ultra Downloader - macOS Launcher"
  echo "============================================================"
  echo
}

die() {
  echo
  echo "ERROR: $1"
  echo
  read -r -p "Press Enter to close..."
  exit 1
}

find_python() {
  if [ -x "$VENV_DIR/bin/python3" ]; then
    echo "$VENV_DIR/bin/python3"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  return 1
}

ensure_venv() {
  local system_python="$1"
  if [ ! -x "$VENV_DIR/bin/python3" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$system_python" -m venv "$VENV_DIR" || die "Could not create the virtual environment."
  fi
}

ensure_requirements() {
  local venv_python="$1"
  "$venv_python" - <<'PY' >/dev/null 2>&1
import importlib
mods = ["yt_dlp", "PyQt6", "PIL", "mutagen"]
for mod in mods:
    importlib.import_module(mod)
PY

  if [ $? -ne 0 ]; then
    echo "Installing required Python packages ..."
    "$venv_python" -m pip install --upgrade pip || die "Could not upgrade pip."
    "$venv_python" -m pip install -r requirements.txt || die "Could not install requirements."
  fi
}

warn_ffmpeg() {
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Warning: ffmpeg was not found in PATH."
    echo "The app may launch, but downloads and cover-art embedding can fail."
    echo "Install ffmpeg with: brew install ffmpeg"
    echo
  fi
}

print_header

SYSTEM_PYTHON="$(find_python)" || die "Python 3 is not installed."
ensure_venv "$SYSTEM_PYTHON"
VENV_PYTHON="$VENV_DIR/bin/python3"
ensure_requirements "$VENV_PYTHON"
warn_ffmpeg

TARGET_APP="$APP_MAIN"

if [ ! -f "$TARGET_APP" ]; then
  die "Could not find $TARGET_APP."
fi

echo "Launching $TARGET_APP ..."
echo
"$VENV_PYTHON" "$TARGET_APP"

EXIT_CODE=$?
echo
echo "Application exited with code $EXIT_CODE."
read -r -p "Press Enter to close..."
exit "$EXIT_CODE"
