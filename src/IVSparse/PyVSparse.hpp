#pragma once 

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "IVSparse/SparseMatrix"
#include <omp.h>

namespace py = pybind11;

template <typename T>
constexpr const char* returnTypeName() {

    if constexpr (std::is_same<T, std::int8_t>::value) return "int8";
    else if constexpr (std::is_same<T, std::uint8_t>::value) return "uint8";
    else if constexpr (std::is_same<T, std::int16_t>::value) return "int16";
    else if constexpr (std::is_same<T, std::uint16_t>::value) return "uint16";
    else if constexpr (std::is_same<T, std::int32_t>::value) return "int32";
    else if constexpr (std::is_same<T, std::uint32_t>::value) return "uint32";
    else if constexpr (std::is_same<T, std::int64_t>::value) return "int64";
    else if constexpr (std::is_same<T, std::uint64_t>::value) return "uint64";
    else if constexpr (std::is_same<T, float>::value) return "float32";
    else if constexpr (std::is_same<T, double>::value) return "float64";
    else {
        static_assert(std::is_same<T, std::int8_t>::value, "Unknown type");
        return "unknown";
    }
}
