#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "IVSparse/SparseMatrix"
#include <omp.h>

#pragma once

namespace py = pybind11;

template <typename T, int compLevel>
void generateForEachIndexType(py::module& m);

template <typename T, typename indexT, int compressionLevel, bool isColMajor>
void declareForOtherTypes(py::class_<IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>>& mat);

template <typename T> 
constexpr const char* returnTypeName();

template <typename T, typename indexT, int compressionLevel, bool isColMajor>
py::class_<IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>> declareSelfFunc(py::module& m);




/**************************Functions and includes to help create a parallel build time -> These are all generated by setupParallelBuild.py ****************************/

#include "classFiles/int8_t_2.cpp"
#include "classFiles/int8_t_3.cpp"
#include "classFiles/uint8_t_2.cpp"
#include "classFiles/uint8_t_3.cpp"
#include "classFiles/int16_t_2.cpp"
#include "classFiles/int16_t_3.cpp"
#include "classFiles/uint16_t_2.cpp"
#include "classFiles/uint16_t_3.cpp"
#include "classFiles/int32_t_2.cpp"
#include "classFiles/int32_t_3.cpp"
#include "classFiles/uint32_t_2.cpp"
#include "classFiles/uint32_t_3.cpp"
#include "classFiles/int64_t_2.cpp"
#include "classFiles/int64_t_3.cpp"
#include "classFiles/uint64_t_2.cpp"
#include "classFiles/uint64_t_3.cpp"
#include "classFiles/float_2.cpp"
#include "classFiles/float_3.cpp"
#include "classFiles/double_2.cpp"
#include "classFiles/double_3.cpp"

void init_int8_t_2(py::module& m);
void init_int8_t_3(py::module& m);
void init_uint8_t_2(py::module& m);
void init_uint8_t_3(py::module& m);
void init_int16_t_2(py::module& m);
void init_int16_t_3(py::module& m);
void init_uint16_t_2(py::module& m);
void init_uint16_t_3(py::module& m);
void init_int32_t_2(py::module& m);
void init_int32_t_3(py::module& m);
void init_uint32_t_2(py::module& m);
void init_uint32_t_3(py::module& m);
void init_int64_t_2(py::module& m);
void init_int64_t_3(py::module& m);
void init_uint64_t_2(py::module& m);
void init_uint64_t_3(py::module& m);
void init_float_2(py::module& m);
void init_float_3(py::module& m);
void init_double_2(py::module& m);
void init_double_3(py::module& m);


void init_int8_t_2(py::module& m);
void init_int8_t_3(py::module& m);
void init_uint8_t_2(py::module& m);
void init_uint8_t_3(py::module& m);
void init_int16_t_2(py::module& m);
void init_int16_t_3(py::module& m);
void init_uint16_t_2(py::module& m);
void init_uint16_t_3(py::module& m);
void init_int32_t_2(py::module& m);
void init_int32_t_3(py::module& m);
void init_uint32_t_2(py::module& m);
void init_uint32_t_3(py::module& m);
void init_int64_t_2(py::module& m);
void init_int64_t_3(py::module& m);
void init_uint64_t_2(py::module& m);
void init_uint64_t_3(py::module& m);
void init_float_2(py::module& m);
void init_float_3(py::module& m);
void init_double_2(py::module& m);
void init_double_3(py::module& m);

/*

Make typedefs for all of these

// [[int8_t], [uint8_t], [int16_t], [uint16_t], [int32_t], [uint32_t], [int64_t], [uint64_t], [float], [double]]

// [uint8_t], [uint16_t], [uint32_t], [uint64_t]]

// 2 3

// [[true],[false]]


*/

