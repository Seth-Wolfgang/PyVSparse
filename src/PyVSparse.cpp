#include "PyVSparse.hpp"


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

