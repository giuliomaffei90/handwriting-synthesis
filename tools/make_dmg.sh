#!/bin/bash
# Package dist/Handwriting.app as a disk image for release.
#
# The image holds the app and an alias to Applications, so it is installed by
# dragging it across before it is ever opened - which is also the only way to
# avoid moving a running app out from under itself. Finder is asked to lay the
# window out; if it refuses (it needs permission to be driven), the image is
# still built, just with default icon placement.
set -euo pipefail
cd "$(dirname "$0")/.."

APP="dist/Handwriting.app"
test -d "$APP" || { echo "$APP missing - run tools/build_app.sh first"; exit 1; }

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")"
DMG="dist/Handwriting-$VERSION.dmg"
VOLUME="Handwriting"
STAGE="$(mktemp -d)"
WRITABLE="$(mktemp -d)/rw.dmg"
MOUNT="/Volumes/$VOLUME"
trap 'hdiutil detach "$MOUNT" -quiet 2>/dev/null || true; rm -rf "$STAGE" "$(dirname "$WRITABLE")"' EXIT

echo "==> Staging $VERSION..."
ditto "$APP" "$STAGE/Handwriting.app"
ln -s /Applications "$STAGE/Applications"

echo "==> Building the image..."
hdiutil create -volname "$VOLUME" -srcfolder "$STAGE" -fs HFS+ -format UDRW -quiet -ov "$WRITABLE"
hdiutil attach "$WRITABLE" -quiet

echo "==> Laying out the window..."
osascript <<APPLESCRIPT || echo "   (Finder would not cooperate - default layout kept)"
tell application "Finder"
    tell disk "$VOLUME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {240, 160, 840, 560}
        set options to the icon view options of container window
        set arrangement of options to not arranged
        set icon size of options to 128
        set text size of options to 13
        set position of item "Handwriting.app" of container window to {150, 190}
        set position of item "Applications" of container window to {450, 190}
        update without registering applications
        delay 1
        close
    end tell
end tell
APPLESCRIPT

sync
hdiutil detach "$MOUNT" -quiet

echo "==> Compressing..."
rm -f "$DMG"
hdiutil convert "$WRITABLE" -format UDZO -imagekey zlib-level=9 -quiet -o "$DMG"

echo "==> Checking it mounts and runs..."
hdiutil attach "$DMG" -quiet -readonly
"$MOUNT/Handwriting.app/Contents/MacOS/Handwriting" --selftest | tail -1
test -L "$MOUNT/Applications" && echo "Applications alias present"
hdiutil detach "$MOUNT" -quiet

ls -lh "$DMG" | awk '{print "built", $NF, $5}'
