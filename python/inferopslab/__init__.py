"""InferOpsLab: A laboratory for building high-performance LLM inference operators."""

from ._C import sgemm_naive

__version__ = "0.1.0"
__all__ = ["sgemm_naive"]
