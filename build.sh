#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/cpp/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
# Build the extension for the Python that will import it (the active venv, if
# any).  Override with PYTHON=/path/to/python.  Letting CMake choose on its own
# can produce a module for a different interpreter, which then fails to import
# while a stale .so from an earlier build keeps getting picked up instead.
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

usage() {
    echo "Usage: $0 [--clean] [--debug] [--test]"
    echo ""
    echo "  --clean   Remove build directory before building"
    echo "  --debug   Build with Debug configuration"
    echo "  --test    Run test_cpp.py after a successful build"
    echo ""
    echo "  PYTHON=<path>  Build against a specific interpreter"
    exit 1
}

CLEAN=0
RUN_TESTS=0

for arg in "$@"; do
    case "$arg" in
        --clean)  CLEAN=1 ;;
        --debug)  BUILD_TYPE=Debug ;;
        --test)   RUN_TESTS=1 ;;
        --help|-h) usage ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

if [[ $CLEAN -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "Removing $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

echo "Configuring ($BUILD_TYPE) for $PYTHON..."
cmake -B "$BUILD_DIR" -S "$SCRIPT_DIR/cpp" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DPython_EXECUTABLE="$PYTHON" -DPYTHON_EXECUTABLE="$PYTHON"

echo "Building..."
cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

echo "Installing..."
EXT_SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
# Remove extension modules built for other interpreters — a leftover .so with a
# different ABI tag silently shadows the fresh build for whichever Python can
# still import it.
find "$SCRIPT_DIR" -maxdepth 1 -name 'occupancy_map_cpp*.so' \
     ! -name "occupancy_map_cpp${EXT_SUFFIX}" -print -delete
cmake --install "$BUILD_DIR"

echo "Build complete — occupancy_map_cpp installed to $SCRIPT_DIR"

if [[ $RUN_TESTS -eq 1 ]]; then
    echo ""
    echo "Running tests..."
    cd "$SCRIPT_DIR"
    "$PYTHON" test_cpp.py
fi
