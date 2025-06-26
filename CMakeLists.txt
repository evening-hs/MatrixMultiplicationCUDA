cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 89)

project(MatrixMultiplicationCUDA CUDA)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add executable
add_executable(matrix_multiplication main.cu)

# Set CUDA specific properties
set_target_properties(matrix_multiplication PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif() 