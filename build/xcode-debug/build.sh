#!/bin/bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR"
OUTFILE="$BUILD_DIR/latest-build-out.txt"
DEFAULT_TARGET="runTests"
TARGET="${1:-$DEFAULT_TARGET}"

run_build() {
    local target="$1"

    {
        echo "=================================================="
        echo "Camellia build"
        echo "Directory: $BUILD_DIR"
        echo "Target: $target"
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=================================================="
    } >> "$OUTFILE"

    cd "$BUILD_DIR" || return 1

    xcodebuild build \
      -project Camellia.xcodeproj \
      -target "$target" \
      -configuration Debug \
      >> "$OUTFILE" 2>&1

    return $?
}

: > "$OUTFILE"
run_build "$TARGET"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "✅ Build succeeded"
    echo "Target: $TARGET"
    echo "Log: $OUTFILE"
    grep -E '\*\* BUILD SUCCEEDED \*\*' "$OUTFILE" || true
    exit 0
fi

echo "❌ Build failed"
echo "Target: $TARGET"
echo "Log: $OUTFILE"
echo

echo "Last 120 lines of output:"
tail -n 120 "$OUTFILE"

exit $STATUS
