include(FetchContent)

cmake_minimum_required(VERSION 3.15...3.26)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}")
set(SOURCE_DIR "${PROJECT_ROOT}/src")
set(IVSPARSE "IVSparse-dev/IVSparse")
set(SPARSE_MATRIX "${IVSPARSE}/SparseMatrix")

if(NOT SKBUILD_STATE STREQUAL "sdist"
   AND EXISTS "${SOURCE_DIR}/pybind11/CMakeLists.txt")
  message(STATUS "Using integrated pybind11")
  set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
  SOURCE_DIR ${SOURCE_DIR}/pybind11)

set(PYBIND11_FINDPYTHON ON)
FetchContent_MakeAvailable(pybind11)


add_custom_target(
    run ALL 
    COMMAND ${PYTHON_EXECUTABLE} ${SOURCE_DIR}/setupParallelBuild.py
    WORKING_DIRECTORY ${SOURCE_DIR}
    COMMENT "Generating extra files for parallel build"
)

pybind11_add_module(PyVSparseExec  ${SOURCE_DIR}/${SKBUILD_PROJECT_NAME}.cpp)

add_dependencies(PyVSparseExec run)

add_compile_options(-O2 -fvisibility=hidden)

target_link_libraries(PyVSparseExec PUBLIC pybind11::module pybind11::lto pybind11::windows_extras )
target_include_directories(PyVSparseExec PUBLIC ${IVSPARSE} ${SPARSE_MATRIX} ${PROJECT_ROOT} ${PROJECT_ROOT}/IVSparse-dev ${PROJECT_ROOT}/IVSparse-dev/IVSparse ${PROJECT_ROOT}/IVSparse-dev/IVSparse/SparseMatrix) # check to see if SPARSE_MATRIX is needed

pybind11_extension(PyVSparseExec )

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(PyVSparseExec )
endif()


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(PyVSparseExec  PUBLIC OpenMP::OpenMP_CXX)
    set_target_properties(PyVSparseExec  PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif()


# install(${SKBUILD_PROJECT_NAME} PyVSparseExec LIBRARY DESTINATION ${SKBUILD_DATA_DIR})
