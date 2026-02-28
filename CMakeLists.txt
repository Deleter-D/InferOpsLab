cmake_minimum_required(VERSION 3.21)
project(InferOpsLab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect CUDA architecture from environment or use default
if(DEFINED ENV{CUDA_ARCHS})
  set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCHS})
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90)
endif()

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# =====================================================
# Core C++ / CUDA Library
# =====================================================
add_library(inferopslab_core SHARED src/gemm/gemm_kernel.cu)

target_include_directories(
  inferopslab_core
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)

set_target_properties(inferopslab_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(
  inferopslab_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>
                           $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

# =====================================================
# Examples (optional submodule)
# =====================================================
option(BUILD_EXAMPLES "Build example programs" OFF)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
