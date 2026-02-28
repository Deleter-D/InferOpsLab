import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")


setup(
    name="inferopslab",
    version="0.1.0",
    author="GoldPancake",
    description="InferOpsLab: A laboratory for building high-performance LLM inference operators",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="inferopslab._C",
            sources=glob.glob("src/**/*.cpp", recursive=True) + glob.glob("src/**/*.cu", recursive=True),
            include_dirs=[
                os.path.join(this_dir, "include"),
                os.path.join(this_dir, "src"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ],
            },
            extra_link_args=[f"-Wl,-rpath,{torch_lib_path}", "-Wl,-rpath,$ORIGIN"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
