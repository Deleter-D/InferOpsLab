import inferopslab
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@pytest.mark.parametrize(
    "M,N,K",
    [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 64),
        (256, 128, 64),
    ],
)
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (1.0, 1.0)])
def test_sgemm_naive(M, N, K, alpha, beta):
    torch.manual_seed(42)

    device = "cuda"
    dtype = torch.float32

    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    C = torch.randn(M, N, device=device, dtype=dtype)

    C_ref = C.clone()

    # 调用你的 CUDA kernel
    C_out = inferopslab.sgemm_naive(A, B, C, alpha, beta)

    # PyTorch 参考
    C_ref = alpha * torch.matmul(A, B) + beta * C_ref

    max_abs_error = (C_out - C_ref).abs().max().item()

    print(f"\n[M={M},N={N},K={K}] " f"max_abs_error={max_abs_error:.6e}")

    assert max_abs_error < 1e-4
