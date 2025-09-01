cmake_minimum_required(VERSION 3.15)
project(gcfl_fast LANGUAGES CXX)

# Require pybind11
find_package(pybind11 CONFIG REQUIRED)

# Build options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -DNDEBUG)

# Module
pybind11_add_module(gcfl_fast
  pybind_module.cpp
  fast_kernels.cpp
)

# If you want to install into the package, you can adjust the output directory,
# or build via `pip`/scikit-build-core later.
