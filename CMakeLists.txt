# Tsugi — 継ぎ — GPU ベンダーを接合する移植検証層
# Apache-2.0
#
# 要: LLVM/MLIR (NVPTX + AMDGPU backend有効). CMake>=3.20, Ninja推奨.
#   cmake -B build -G Ninja -DMLIR_DIR=<llvm>/lib/cmake/mlir \
#         -DTsugi_ENABLE_NVIDIA=ON -DTsugi_ENABLE_AMD=ON

cmake_minimum_required(VERSION 3.20)
project(Tsugi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(Tsugi_ENABLE_NVIDIA "Enable NVIDIA (NVPTX→PTX) backend" ON)
option(Tsugi_ENABLE_AMD    "Enable AMD (AMDGPU→AMDGCN) backend" ON)
option(Tsugi_ENABLE_SPIRV  "Enable SPIR-V fallback (Intel/Vulkan)" OFF)
option(Tsugi_BUILD_TESTS   "Build correctness/perf tests" ON)

# --- MLIR / LLVM ---------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)
add_definitions(${LLVM_DEFINITIONS})

# --- 必須バックエンドの存在確認（fail fast） ------------------------------
if(Tsugi_ENABLE_NVIDIA AND NOT "NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(FATAL_ERROR "Tsugi_ENABLE_NVIDIA=ON but LLVM was built without NVPTX target")
endif()
if(Tsugi_ENABLE_AMD AND NOT "AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(FATAL_ERROR "Tsugi_ENABLE_AMD=ON but LLVM was built without AMDGPU target")
endif()

# --- サブディレクトリ ----------------------------------------------------
add_subdirectory(src/tsugi/ir)        # tsugi.tile / tsugi.gpu dialect
add_subdirectory(src/tsugi/backend)   # vendor lowering passes
add_subdirectory(src/tsugi/runtime)   # Layer B dispatch

if(Tsugi_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "Tsugi config: NVIDIA=${Tsugi_ENABLE_NVIDIA} AMD=${Tsugi_ENABLE_AMD} SPIRV=${Tsugi_ENABLE_SPIRV}")
