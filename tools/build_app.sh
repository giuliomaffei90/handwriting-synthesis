#!/bin/bash
# Build dist/Handwriting.app (Apple Silicon). Needs: uv, and once:
#   uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install numpy pillow pyinstaller
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${VERSION:-1.0.0}"

test -f hw/weights.npz || { echo "hw/weights.npz missing - run tools/tf_export.py first"; exit 1; }

.venv/bin/pyinstaller --noconfirm --clean --windowed --name Handwriting \
  --icon assets/AppIcon.icns \
  --osx-bundle-identifier com.giuliomaffei.handwriting \
  --add-data "hw/weights.npz:hw" \
  --add-data "styles:styles" \
  --exclude-module matplotlib --exclude-module scipy --exclude-module tensorflow \
  --exclude-module pandas --exclude-module sklearn \
  app.py

PLIST=dist/Handwriting.app/Contents/Info.plist
for key in CFBundleShortVersionString CFBundleVersion; do
  plutil -replace "$key" -string "$VERSION" "$PLIST" 2>/dev/null ||
    plutil -insert "$key" -string "$VERSION" "$PLIST"
done
plutil -replace LSMinimumSystemVersion -string "11.0" "$PLIST" 2>/dev/null ||
  plutil -insert LSMinimumSystemVersion -string "11.0" "$PLIST"

# ad-hoc signature must come last: it seals the Info.plist above
codesign --force --deep --sign - dist/Handwriting.app

dist/Handwriting.app/Contents/MacOS/Handwriting --selftest

rm -rf dist/Handwriting
du -sh dist/Handwriting.app
echo "built dist/Handwriting.app"
