#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "common/cuda_utils.cuh"
#include "inferopslab/gemm/gemm.h"

namespace inferopslab {
namespace gemm {

__global__ void sgemm_naive_kernel(int M,
                                   int N,
                                   int K,
                                   float alpha,
                                   const float *A,
                                   const float *B,
                                   float beta,
                                   float *C);

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
