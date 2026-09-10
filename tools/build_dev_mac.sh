#!/usr/bin/env bash
# Personal dev helper: build a Release macOS build of Handwriting, drop it in
# ~/Downloads, kill any running instance, and relaunch the fresh build.
set -euo pipefail

APP_NAME="Handwriting"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILT_APP="$PROJECT_DIR/dist/$APP_NAME.app"
DEST_APP="$HOME/Downloads/$APP_NAME.app"

cd "$PROJECT_DIR"

if [ ! -x .venv/bin/pyinstaller ]; then
  command -v uv >/dev/null || { echo "uv is not installed: brew install uv"; exit 1; }
  echo "==> Creating .venv (numpy, pillow, pyinstaller)..."
  uv venv --python 3.12 .venv
  VIRTUAL_ENV=.venv uv pip install numpy pillow pyinstaller
fi

echo "==> Closing running instances of $APP_NAME..."
osascript -e "tell application \"$APP_NAME\" to quit" >/dev/null 2>&1 || true
pkill -x "$APP_NAME" >/dev/null 2>&1 || true

echo "==> Building Release macOS app..."
./tools/build_app.sh

echo "==> Copying build to $DEST_APP..."
rm -rf "$DEST_APP"
ditto "$BUILT_APP" "$DEST_APP"

echo "==> Launching $DEST_APP..."
open "$DEST_APP"

echo "Done."
