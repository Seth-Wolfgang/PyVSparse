cmake_minimum_required(VERSION 3.15...3.26)
project(PyVSparse LANGUAGES CXX)

find_package(pybind11)

set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/..")
set(IVSPARSE "${PROJECT_ROOT}/IVSparse-dev/IVSparse")
set(SPARSE_MATRIX "${IVSPARSE}/SparseMatrix")

add_custom_target(
    run ALL 
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/setupParallelBuild.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating extra files for parallel build"
)


link_directories(${PROJECT_ROOT})

pybind11_add_module(PyVSparse ${CMAKE_SOURCE_DIR}/PyVSparse.cpp)

add_dependencies(PyVSparse run)


add_library(IVSparse STATIC ${IVSPARSE} ${SPARSE_MATRIX} )
set_target_properties(IVSparse PROPERTIES LINKER_LANGUAGE CXX)
add_compile_options(-O2 -fvisibility=hidden)

target_link_libraries(PyVSparse PRIVATE pybind11::module pybind11::lto pybind11::windows_extras )
target_include_directories(PyVSparse PRIVATE ${IVSPARSE} ${SPARSE_MATRIX} ${PROJECT_ROOT} ${PROJECT_ROOT}/IVSparse-dev ${PROJECT_ROOT}/IVSparse-dev/IVSparse ${PROJECT_ROOT}/IVSparse-dev/IVSparse/SparseMatrix)

pybind11_extension(PyVSparse)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(PyVSparse)
endif()


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(PyVSparse PUBLIC OpenMP::OpenMP_CXX)
    set_target_properties(PyVSparse PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif()


set_target_properties(PyVSparse PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden")
target_compile_definitions(PyVSparse PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})