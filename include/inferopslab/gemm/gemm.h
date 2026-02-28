#pragma once

namespace inferopslab {
namespace gemm {

void SGemmNaive(int M,
                int N,
                int K,
                float alpha,
                const float *A,
                const float *B,
                float beta,
                float *C);
}  // namespace gemm
}  // namespace inferopslab
