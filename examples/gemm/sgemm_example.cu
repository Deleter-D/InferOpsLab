/**
 * @file sgemm_example.cu
 * @brief Example demonstrating usage of inferopslab_core SGEMM kernel
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && make
 *
 * Run:
 *   ./sgemm_example
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "inferopslab/gemm/gemm.h"

void check_cuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
    exit(1);
  }
}

int main() {
  // Matrix dimensions: C = alpha * A * B + beta * C
  const int M = 128;
  const int N = 128;
  const int K = 64;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  printf("SGEMM Example: C(%dx%d) = %.1f * A(%dx%d) * B(%dx%d) + %.1f * C\n",
         M,
         N,
         alpha,
         M,
         K,
         K,
         N,
         beta);

  // Allocate host memory
  float *h_A = (float *)malloc(M * K * sizeof(float));
  float *h_B = (float *)malloc(K * N * sizeof(float));
  float *h_C = (float *)malloc(M * N * sizeof(float));

  // Initialize matrices with random values
  for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
  for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
  for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  check_cuda(cudaMalloc(&d_A, M * K * sizeof(float)), "cudaMalloc A");
  check_cuda(cudaMalloc(&d_B, K * N * sizeof(float)), "cudaMalloc B");
  check_cuda(cudaMalloc(&d_C, M * N * sizeof(float)), "cudaMalloc C");

  // Copy to device
  check_cuda(
      cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice),
      "cudaMemcpy A");
  check_cuda(
      cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice),
      "cudaMemcpy B");
  check_cuda(
      cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice),
      "cudaMemcpy C");

  // Call inferopslab SGEMM kernel
  inferopslab::gemm::SGemmNaive(M, N, K, alpha, d_A, d_B, beta, d_C);

  // Check for kernel errors
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "kernel execution");

  // Copy result back
  check_cuda(
      cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost),
      "cudaMemcpy C");

  printf("SGEMM completed successfully!\n");
  printf("Sample output: C[0] = %.4f\n", h_C[0]);

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
