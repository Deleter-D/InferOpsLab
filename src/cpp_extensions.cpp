#include <torch/extension.h>
#include "inferopslab/gemm/gemm.h"

namespace inferopslab {
namespace pybind {

torch::Tensor sgemm_naive_torch(torch::Tensor A,
                                torch::Tensor B,
                                torch::Tensor C,
                                float alpha,
                                float beta) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
  TORCH_CHECK(C.is_cuda(), "C must be CUDA tensor");

  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");

  float* dA = A.data_ptr<float>();
  float* dB = B.data_ptr<float>();
  float* dC = C.data_ptr<float>();

  inferopslab::gemm::SGemmNaive(M, N, K, alpha, dA, dB, beta, dC);

  return C;
}

}  // namespace pybind
}  // namespace inferopslab

PYBIND11_MODULE(_C, m) {
  m.def("sgemm_naive",
        &inferopslab::pybind::sgemm_naive_torch,
        "Naive SGEMM (CUDA)",
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f);
}
