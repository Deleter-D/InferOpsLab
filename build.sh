#!/bin/bash
# Build script for InferOpsLab core C++ library
# Produces inferopslab_core/ with include/ and lib/ directories

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build directory
BUILD_DIR="build"
OUTPUT_DIR="inferopslab_core"

# Options
BUILD_EXAMPLES="${BUILD_EXAMPLES:-OFF}"
CUDA_ARCHS="${CUDA_ARCHS:-80;86;89;90}"

# Clean previous build
rm -rf "$BUILD_DIR" "$OUTPUT_DIR"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR/include"
mkdir -p "$OUTPUT_DIR/lib"

echo "Building InferOpsLab Core Library..."
echo "CUDA Architectures: $CUDA_ARCHS"
echo "Build Examples: $BUILD_EXAMPLES"
echo ""

# Run CMake configure
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES"

# Build
cmake --build "$BUILD_DIR" -j$(nproc)

# Copy headers
cp -r include/inferopslab "$OUTPUT_DIR/include/"

# Copy library
cp "$BUILD_DIR/libinferopslab_core.so" "$OUTPUT_DIR/lib/"

echo ""
echo "=========================================="
echo "Build complete! Output in $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/lib/"
if [ "$BUILD_EXAMPLES" = "ON" ]; then
    echo ""
    echo "Examples:"
    find "$BUILD_DIR/examples" -type f -executable -name "*_example" 2>/dev/null | while read f; do
        ls -la "$f"
    done
fi
echo "=========================================="
