cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(test_coarse
        VERSION 0.1
        LANGUAGES CXX CUDA)

# ---------- General build settings -------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------- GPU architecture detection --------------------------------------------
# If the caller already set -DCMAKE_CUDA_ARCHITECTURES we respect it.
if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
          "Auto-detected architectures with CMake >=3.24" FORCE)
    else()
        execute_process(
          COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
          OUTPUT_VARIABLE _cc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_cc MATCHES "^9")
            set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "" FORCE)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "" FORCE)
        endif()
    endif()
endif()

# ---------- Target -----------------------------------------------------------------
add_executable(test_coarse src/main.cu)

# Definitions & compile flags
target_compile_definitions(test_coarse PRIVATE NDEBUG)
target_compile_options(test_coarse PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extra-device-vectorization -lineinfo>)

# Properties
set_target_properties(test_coarse PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# Linking
target_link_libraries(test_coarse PRIVATE cudadevrt)

# ---- NVTX3 ----
message(STATUS "Fetching nvtx3 from GitHub…")
include(FetchContent)
FetchContent_Declare(
    nvtx3
    GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
    GIT_TAG        v3.2.0-c-cpp
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(nvtx3)
target_link_libraries(test_coarse PRIVATE nvtx3-cpp)

# Final info
message(STATUS "CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES} If you want to change it, use -DCMAKE_CUDA_ARCHITECTURES=XX")
