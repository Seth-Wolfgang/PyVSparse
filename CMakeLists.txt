include(FetchContent)

cmake_minimum_required(VERSION 3.15...3.26)
add_compile_options(-O2 -fopenmp)

project(PyVSparse)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}")
set(SOURCE_DIR "${PROJECT_ROOT}/src")


FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
  SOURCE_DIR ${SOURCE_DIR}/pybind11)


set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

FetchContent_MakeAvailable(pybind11)

include_directories(${IVSPARSE} ${PROJECT_ROOT} ${PROJECT_ROOT}/IVSparse-dev ${PROJECT_ROOT}/IVSparse-dev/IVSparse )
link_directories(${IVSPARSE} ${PROJECT_ROOT} ${PROJECT_ROOT}/IVSparse-dev ${PROJECT_ROOT}/IVSparse-dev/IVSparse)
add_library(IVSparse STATIC ${PROJECT_ROOT}/IVSparse-dev/IVSparse/SparseMatrix )
set_target_properties(IVSparse PROPERTIES LINKER_LANGUAGE CXX)

pybind11_add_module(PyVSparse ${SOURCE_DIR}/PyVSparse.cpp ${SOURCE_DIR}/IVCSC/IVCSC_Wrapper.cpp ${SOURCE_DIR}/VCSC/VCSC_Wrapper.cpp)
target_include_directories(PyVSparse PRIVATE ${SOURCE_DIR}/VCSC/VCSC_Wrapper.hpp ${SOURCE_DIR}/IVCSC/IVCSC_Wrapper.hpp)

target_link_libraries(PyVSparse PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

pybind11_extension(PyVSparse)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(PyVSparse)
endif()



find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(PyVSparse  PUBLIC OpenMP::OpenMP_CXX)
    set_target_properties(PyVSparse  PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif()

set_target_properties(PyVSparse PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden")
target_compile_definitions(PyVSparse PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})


