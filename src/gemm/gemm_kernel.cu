#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "gemm_launch.cuh"

namespace inferopslab {
namespace gemm {

__global__ void sgemm_naive_kernel(int M,
                                   int N,
                                   int K,
                                   float alpha,
                                   const float *A,
                                   const float *B,
                                   float beta,
                                   float *C) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void SGemmNaive(int M,
                int N,
                int K,
                float alpha,
                const float *A,
                const float *B,
                float beta,
                float *C) {
  // Configure kernel launch parameters (32x32 thread block)
  const int BLOCK_SIZE = 32;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Launch kernel
  sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm
}  // namespace inferopslab
