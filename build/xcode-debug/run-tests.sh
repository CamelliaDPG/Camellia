#!/bin/bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR"
OUTFILE="$BUILD_DIR/latest-test-out.txt"
TEST_BINARY="$BUILD_DIR/unit_tests/Debug/runTests"

{
    echo "=================================================="
    echo "Camellia tests"
    echo "Directory: $BUILD_DIR"
    echo "Binary: $TEST_BINARY"
    echo "Arguments: $*"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
} > "$OUTFILE"

if [ ! -x "$TEST_BINARY" ]; then
    echo "❌ Test binary not found or not executable"
    echo "Binary: $TEST_BINARY"
    echo "Log: $OUTFILE"
    echo >> "$OUTFILE"
    echo "ERROR: Test binary not found or not executable: $TEST_BINARY" >> "$OUTFILE"
    exit 1
fi

cd "$BUILD_DIR" || exit 1
"$TEST_BINARY" "$@" >> "$OUTFILE" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "✅ Tests succeeded"
    echo "Binary: $TEST_BINARY"
    echo "Log: $OUTFILE"
else
    echo "❌ Tests failed"
    echo "Binary: $TEST_BINARY"
    echo "Log: $OUTFILE"
    echo
    echo "Last 120 lines of output:"
    tail -n 120 "$OUTFILE"
fi

exit $STATUS
