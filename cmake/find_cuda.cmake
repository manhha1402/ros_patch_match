find_package(CUDA REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --default-stream per-thread
    --use_fast_math
    -Xcudafe "--diag_suppress=integer_sign_change"
    -Xcudafe "--diag_suppress=partial_override"
    -Xcudafe "--diag_suppress=virtual_function_decl_hidden")
