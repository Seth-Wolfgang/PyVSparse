include(FetchContent)

cmake_minimum_required(VERSION 3.15...3.26)
add_compile_options(-O2 -fopenmp)

project(PyVSparse)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(WRAPPER_DIR "${SOURCE_DIR}/IVSparse")
set(IVSPARSE_DEV_DIR "${PROJECT_ROOT}/src/IVSparse/IVSparse-dev")
set(IVSPARSE_DIR "${IVSPARSE_DEV_DIR}/IVSparse")

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
    SOURCE_DIR ${SOURCE_DIR}/pybind11)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    SOURCE_DIR ${SOURCE_DIR}/eigen)


set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

FetchContent_MakeAvailable(pybind11)

include_directories(${IVSPARSE_DEV_DIR} ${IVSPARSE_DIR} ${SOURCE_DIR}/eigen)
link_directories(${IVSPARSE_DIR} ${IVSPARSE_DEV_DIR})
add_library(IVSparse STATIC ${IVSPARSE_DIR}/SparseMatrix )
set_target_properties(IVSparse PROPERTIES LINKER_LANGUAGE CXX)

pybind11_add_module(_PyVSparse ${WRAPPER_DIR}/PyVSparse.cpp ${WRAPPER_DIR}/IVCSC/IVCSC_Wrapper.cpp ${WRAPPER_DIR}/VCSC/VCSC_Wrapper.cpp)
target_include_directories(_PyVSparse PRIVATE ${WRAPPER_DIR})

target_link_libraries(_PyVSparse PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

pybind11_extension(_PyVSparse)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(_PyVSparse)
endif()



find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_PyVSparse  PUBLIC OpenMP::OpenMP_CXX)
    set_target_properties(_PyVSparse  PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif()

set_target_properties(_PyVSparse PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden")
target_compile_definitions(_PyVSparse PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
install(TARGETS _PyVSparse DESTINATION PyVSparse)

