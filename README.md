# InferOpsLab

A laboratory for building high-performance LLM inference operators from scratch.

## Installation

### PyTorch Extension Mode

Install as a Python package with CUDA extensions:

```bash
pip install -e . --no-build-isolation
```

Requirements:
- Python 3.8+
- PyTorch 2.0+
- CUDA Toolkit 11.8+

Usage:

```python
import inferopslab

# SGEMM: C = alpha * A @ B + beta * C
C = inferopslab.sgemm_naive(A, B, C, alpha=1.0, beta=0.0)
```

### C++ Library Mode

Build the standalone C++/CUDA library:

```bash
# Build core library only
bash build_core.sh

# Build with examples
BUILD_EXAMPLES=ON bash build_core.sh

# Specify CUDA architectures
CUDA_ARCHS=90 bash build_core.sh
```

Output structure:

```
inferopslab_core/
├── include/
│   └── inferopslab/
│       ├── inferopslab.h
│       └── gemm/
│           └── gemm.h
└── lib/
    └── libinferopslab_core.so
```

Usage in C++:

```cpp
#include "inferopslab/gemm/gemm.h"

// C = alpha * A * B + beta * C
inferopslab::gemm::SGemmNaive(M, N, K, alpha, d_A, d_B, beta, d_C);
```

## Examples

C++ examples are located in `examples/`:

```bash
# Build and run
BUILD_EXAMPLES=ON bash build_core.sh
./build/examples/sgemm_example
```

## Project Structure

```
InferOpsLab/
├── include/           # Public headers
├── src/               # CUDA kernel implementations
├── python/            # Python bindings
├── examples/          # C++ examples
├── tests/             # Python tests
├── CMakeLists.txt     # CMake for C++ library
├── setup.py           # Python extension build
└── build_core.sh      # C++ library build script
```

## Development

Run tests:

```bash
pytest tests/ -v
```
