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


template <typename T, typename indexT, int compressionLevel, bool isColMajor>
void defineIterator(IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> mat, const char* name, py::module& m);


// self as in self types T = T 
template <typename T, typename indexT, int compressionLevel, bool isColMajor>
py::class_<IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>> declareSelfFunc(py::module& m) {

    const char* format = (compressionLevel == 2) ? "VCSC" : "IVCSC";
    const char* isCol = (isColMajor) ? "Col" : "Row";

    std::string uniqueName = format;
    uniqueName += "_";
    uniqueName += returnTypeName<T>();
    uniqueName += "_";
    uniqueName += returnTypeName<indexT>();
    uniqueName += "_";
    uniqueName += isCol;

    py::class_<IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>> mat(m, uniqueName.c_str());
    mat.def(py::init<Eigen::SparseMatrix<T>& >());
    mat.def(py::init<Eigen::SparseMatrix<T, Eigen::RowMajor>& >());
    mat.def(py::init<T*, indexT*, indexT*, uint32_t, uint32_t, uint32_t>());
    mat.def(py::init<std::vector<std::tuple<indexT, indexT, T>>&, uint32_t, uint32_t, uint32_t>());
    mat.def(py::init<std::unordered_map<T, std::vector<indexT>>*, uint32_t, uint32_t>()); //<std::unordered_map<T, std::vector<indexT>>[], uint32_t, uint32_t> ;
    mat.def(py::init<const char*>());
    mat.def(py::init<>());
    mat.def("rows", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::rows, py::return_value_policy::copy);
    mat.def("cols", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::cols, py::return_value_policy::copy);
    mat.def("innerSize", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::innerSize, py::return_value_policy::copy);
    mat.def("outerSize", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::outerSize, py::return_value_policy::copy);
    mat.def("nonZeros", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::nonZeros, py::return_value_policy::copy);
    mat.def("byteSize", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::byteSize, py::return_value_policy::copy);
    mat.def("write", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::write, py::arg("filename"));
    mat.def("print", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::print);
    mat.def("isColumnMajor", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::isColumnMajor, py::return_value_policy::copy);
    mat.def("coeff", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::coeff, py::return_value_policy::copy, "Sets value at run of coefficient", py::arg("row").none(false), py::arg("col").none(false));
    mat.def("isColumnMajor", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::isColumnMajor, py::return_value_policy::copy);
    mat.def("outerSum", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::outerSum, py::return_value_policy::copy);
    mat.def("innerSum", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::innerSum, py::return_value_policy::copy);
    mat.def("maxColCoeff", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::maxColCoeff, py::return_value_policy::copy);
    mat.def("maxRowCoeff", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::maxRowCoeff, py::return_value_policy::copy);
    mat.def("minRowCoeff", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::minRowCoeff, py::return_value_policy::copy);
    mat.def("minColCoeff", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::minColCoeff, py::return_value_policy::copy);
    mat.def("trace", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::trace, py::return_value_policy::copy);
    mat.def("sum", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self) { return self.sum(); }, py::return_value_policy::copy);
    mat.def("norm", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::norm, py::return_value_policy::copy);
    mat.def("vectorLength", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::vectorLength, py::return_value_policy::copy);
    mat.def("toCSC", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::toCSC, py::return_value_policy::copy);
    mat.def("toEigen", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::toEigen, py::return_value_policy::copy);
    mat.def("transpose", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::transpose, py::return_value_policy::copy);
    mat.def("inPlaceTranspose", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::inPlaceTranspose);
    mat.def("append", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self, IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& other) {
        self.append(other);
            }, py::arg("other"), py::keep_alive<1, 2>());
    mat.def("append", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self, Eigen::SparseMatrix<T>& other) {
        self.append(other);
            }, py::arg("other"), py::keep_alive<1, 2>());

    mat.def("slice", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::slice, py::arg("startCol"), py::arg("endCol"), py::return_value_policy::move);
    mat.def(py::self *= int8_t());
    mat.def(py::self *= uint8_t());
    mat.def(py::self *= int16_t());
    mat.def(py::self *= uint16_t());
    mat.def(py::self *= int32_t());
    mat.def(py::self *= uint32_t());
    mat.def(py::self *= int64_t());
    mat.def(py::self *= uint64_t());
    mat.def("__repr__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self) {

        std::stringstream ss;
        ss << self;
        return ss.str();

            }, py::is_operator(), py::return_value_policy::move);
    mat.def("__str__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self) {

        std::stringstream ss;
        ss << self;
        return ss.str();

            }, py::is_operator(), py::return_value_policy::move);
    mat.def("__copy__", [](const IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self) {
        return IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>(self);
            });
    mat.def("__deepcopy__", [](const IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self, py::dict) {                  //TODO: NEED TO CHECK;
        return IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>(self);
            });
    mat.def("__eq__", [](const IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self, const IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& other) {
        return self == other;
            }, py::is_operator());
    mat.def("__ne__", [](const IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self, const IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& other) {
        return !(self == other);
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, int8_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, uint8_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, int16_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, uint16_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, int32_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, uint32_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, int64_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, uint64_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, double a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, float a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> self, Eigen::Matrix<T, Eigen::Dynamic, 1>& a) {
        return self * a;
            }, py::is_operator());
    mat.def("__getitem__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self, std::tuple<indexT, indexT> index) {
        return self(std::get<0>(index), std::get<1>(index));
            }, py::return_value_policy::copy);

    if constexpr (compressionLevel == 2) {
        mat.def("getNumUniqueVals", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::getNumUniqueVals, py::return_value_policy::copy);
        mat.def("getValues", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::getValues, py::return_value_policy::copy);
        mat.def("getCounts", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::getCounts, py::return_value_policy::copy);
        mat.def("getNumIndices", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::getNumIndices, py::return_value_policy::copy);
        mat.def("getIndices", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::getIndices, py::return_value_policy::copy);
    }
    else if constexpr (compressionLevel == 3) {
        mat.def("vectorPointer", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::vectorPointer, py::return_value_policy::reference);
        mat.def("getVectorSize", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::getVectorSize, py::return_value_policy::copy);
    }

    // uniqueName += "_";
    // uniqueName += "InnerIterator";

    // py::class_<typename IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator> it(m, uniqueName.c_str());

    // it.def(py::init<IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>&, uint32_t>(), py::arg("mat"), py::arg("outerIndex"));
    // it.def("getIndex", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator::getIndex, py::return_value_policy::copy);
    // it.def("outerDim", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator::outerDim, py::return_value_policy::copy);
    // it.def("row", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator::row, py::return_value_policy::copy);
    // it.def("col", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator::col, py::return_value_policy::copy);
    // it.def("value", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator::value, py::return_value_policy::copy);
    // it.def("coeff", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator::coeff, py::return_value_policy::copy);
    // it.def("__next__", [](typename IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator& self) {
    //     ++self;
    //     if (self) {
    //         return self;
    //     }
    //     else {
    //         throw py::stop_iteration();
    //     }
    // },
    // py::return_value_policy::move,
    // py::keep_alive<0, 1>(),
    // py::is_operator());
    // it.def("__iter__", [](typename IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator& self) {
    //     return self;
    //        }, py::keep_alive<0, 1>());
    // it.def("__bool__", [](typename IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator& self) {
    //     if(self) {
    //         return true;
    //     }
    //     else {
    //         return false;
    //     }
    //        }, py::is_operator());

    // if constexpr(compressionLevel == 3) {
    //     it.def("isNewRun", &IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator::isNewRun, py::return_value_policy::copy);
    // }

    // mat.def("__iter__", [](IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>& self, uint32_t outerIndex) {
    //     return typename IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor>::InnerIterator(self, outerIndex);
    //         }, py::keep_alive<0, 1>());

    return mat;
}

template <typename T, int compLevel>
void generateForEachIndexType(py::module& m) {


    if constexpr (compLevel == 3) {
        // py::class_<IVSparse::SparseMatrix<T, uint8_t,  3, false>> mat8  = declareSelfFunc<T, uint8_t , 3, false>(m);
        // py::class_<IVSparse::SparseMatrix<T, uint8_t,  3, true >> mat9  = declareSelfFunc<T, uint8_t , 3, true>(m);
        // py::class_<IVSparse::SparseMatrix<T, uint16_t, 3, false>> mat10 = declareSelfFunc<T, uint16_t, 3, false>(m);
        // py::class_<IVSparse::SparseMatrix<T, uint16_t, 3, true >> mat11 = declareSelfFunc<T, uint16_t, 3, true>(m);
        // py::class_<IVSparse::SparseMatrix<T, uint32_t, 3, false>> mat12 = declareSelfFunc<T, uint32_t, 3, false>(m);
        // py::class_<IVSparse::SparseMatrix<T, uint32_t, 3, true >> mat13 = declareSelfFunc<T, uint32_t, 3, true>(m);
        declareSelfFunc<T, uint64_t, 3, false>(m);
        declareSelfFunc<T, uint64_t, 3, true>(m);
    }
    else {
        // declareSelfFunc<T, uint8_t, 2, false>(m);
        // declareSelfFunc<T, uint8_t, 2, true>(m);
        // declareSelfFunc<T, uint16_t, 2, false>(m);
        // declareSelfFunc<T, uint16_t, 2, true>(m);
        declareSelfFunc<T, uint32_t, 2, false>(m);
        declareSelfFunc<T, uint32_t, 2, true>(m);
        // declareSelfFunc<T, uint64_t, 2, false>(m);
        // declareSelfFunc<T, uint64_t, 2, true>(m);
    }
}

// template <typename T, typename indexT, int compressionLevel, bool isColMajor>
// void defineIterator(IVSparse::SparseMatrix<T, indexT, compressionLevel, isColMajor> mat, const char* name, py::module& m) {

// }

template <typename T>
constexpr const char* returnTypeName() {

    if constexpr (std::is_same<T, std::int8_t>::value) return "int8_t";
    else if constexpr (std::is_same<T, std::uint8_t>::value) return "uint8_t";
    else if constexpr (std::is_same<T, std::int16_t>::value) return "int16_t";
    else if constexpr (std::is_same<T, std::uint16_t>::value) return "uint16_t";
    else if constexpr (std::is_same<T, std::int32_t>::value) return "int32_t";
    else if constexpr (std::is_same<T, std::uint32_t>::value) return "uint32_t";
    else if constexpr (std::is_same<T, std::int64_t>::value) return "int64_t";
    else if constexpr (std::is_same<T, std::uint64_t>::value) return "uint64_t";
    else if constexpr (std::is_same<T, float>::value) return "float";
    else if constexpr (std::is_same<T, double>::value) return "double";
    else return "unknown";
}
