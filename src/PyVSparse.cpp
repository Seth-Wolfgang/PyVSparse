#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "IVSparse/SparseMatrix"
#include <omp.h>

namespace py = pybind11;

template <typename T, int compLevel>
void generateForEachIndexType(py::module& m);

template <typename T, typename indexT, int compressionLevel, bool isColMajor>
void declareForOtherTypes(py::class_<IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>>& mat);

template <typename T> 
constexpr const char* returnTypeName();

template <typename T, typename indexT, int compressionLevel, bool isColMajor>
py::class_<IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>> declareSelfFunc(py::module& m);



PYBIND11_MODULE(PyVSparse, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    generateForEachIndexType<int8_t, 2>(m);
    generateForEachIndexType<int8_t, 3>(m);
    generateForEachIndexType<uint8_t, 2>(m);
    generateForEachIndexType<uint8_t, 3>(m);
    generateForEachIndexType<int16_t, 2>(m);
    generateForEachIndexType<int16_t, 3>(m);
    generateForEachIndexType<uint16_t, 2>(m);
    generateForEachIndexType<uint16_t, 3>(m);
    generateForEachIndexType<int32_t, 2>(m);
    generateForEachIndexType<int32_t, 3>(m);
    generateForEachIndexType<uint32_t, 2>(m);
    generateForEachIndexType<uint32_t, 3>(m);
    generateForEachIndexType<int64_t, 2>(m);
    generateForEachIndexType<int64_t, 3>(m);
    generateForEachIndexType<uint64_t, 2>(m);
    generateForEachIndexType<uint64_t, 3>(m);
    generateForEachIndexType<float, 2>(m);
    generateForEachIndexType<float, 3>(m);
    generateForEachIndexType<double, 2>(m);
    generateForEachIndexType<double, 3>(m);

};

