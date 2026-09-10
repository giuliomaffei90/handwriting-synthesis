#!/bin/bash
# Build dist/Handwriting.app - the native Swift app (Apple Silicon).
# Needs Xcode's toolchain, and uv for the one Python step that packs the model.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${VERSION:-2.0.0}"
APP="dist/Handwriting.app"

test -f hw/weights.npz || { echo "hw/weights.npz missing - run tools/tf_export.py first"; exit 1; }

needs_packing=false
for source in hw/weights.npz hw/previews.npz styles; do
  [ -e "$source" ] || continue
  if [ ! -f swift/Resources/styles.bin ] || [ "$source" -nt swift/Resources/styles.bin ]; then
    needs_packing=true
  fi
done
if [ ! -f swift/Resources/model.bin ] || [ "$needs_packing" = true ]; then
  echo "==> Packing the model and styles for Swift..."
  if [ ! -x .venv/bin/python ]; then
    command -v uv >/dev/null || { echo "uv is not installed: brew install uv"; exit 1; }
    uv venv --python 3.12 .venv
    VIRTUAL_ENV=.venv uv pip install numpy pillow
  fi
  .venv/bin/python tools/make_swift_resources.py
fi

echo "==> Building the app..."
swift build -c release --package-path swift

echo "==> Assembling $APP..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp swift/.build/release/Handwriting "$APP/Contents/MacOS/Handwriting"
cp swift/Resources/*.bin "$APP/Contents/Resources/"
cp assets/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Handwriting</string>
  <key>CFBundleDisplayName</key><string>Handwriting</string>
  <key>CFBundleExecutable</key><string>Handwriting</string>
  <key>CFBundleIdentifier</key><string>com.giuliomaffei.handwriting</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSSupportsAutomaticTermination</key><true/>
</dict>
</plist>
PLIST

# ad-hoc signature: required for arm64 binaries to run at all
codesign --force --deep --sign - "$APP"

"$APP/Contents/MacOS/Handwriting" --selftest

du -sh "$APP"
echo "built $APP"
