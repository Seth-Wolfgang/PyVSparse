
#include "VCSC_Wrapper.hpp"


template <typename T>
void generateVCSCForEachIndexType(py::module& m) {

    py::class_<IVSparse::VCSC<T, uint8_t, false>> mat1 = declareVCSC<T, uint8_t, false>(m);
    py::class_<IVSparse::VCSC<T, uint8_t, true>> mat2 = declareVCSC<T, uint8_t, true>(m);
    py::class_<IVSparse::VCSC<T, uint16_t, false>> mat3 = declareVCSC<T, uint16_t, false>(m);
    py::class_<IVSparse::VCSC<T, uint16_t, true>> mat4 = declareVCSC<T, uint16_t, true>(m);
    py::class_<IVSparse::VCSC<T, uint32_t, false>> mat5 = declareVCSC<T, uint32_t, false>(m);
    py::class_<IVSparse::VCSC<T, uint32_t, true>> mat6 = declareVCSC<T, uint32_t, true>(m);
    py::class_<IVSparse::VCSC<T, uint64_t, false>> mat7 = declareVCSC<T, uint64_t, false>(m);
    py::class_<IVSparse::VCSC<T, uint64_t, true>> mat8 = declareVCSC<T, uint64_t, true>(m);

    declareVCSCFuncs<T, uint8_t, false>(m, mat1);
    declareVCSCFuncs<T, uint8_t, true>(m, mat2);
    declareVCSCFuncs<T, uint16_t, false>(m, mat3);
    declareVCSCFuncs<T, uint16_t, true>(m, mat4);
    declareVCSCFuncs<T, uint32_t, false>(m, mat5);
    declareVCSCFuncs<T, uint32_t, true>(m, mat6);
    declareVCSCFuncs<T, uint64_t, false>(m, mat7);
    declareVCSCFuncs<T, uint64_t, true>(m, mat8);

    declareVCSCOperators<T, uint8_t, false>(m, mat1);
    declareVCSCOperators<T, uint8_t, true>(m, mat2);
    declareVCSCOperators<T, uint16_t, false>(m, mat3);
    declareVCSCOperators<T, uint16_t, true>(m, mat4);
    declareVCSCOperators<T, uint32_t, false>(m, mat5);
    declareVCSCOperators<T, uint32_t, true>(m, mat6);
    declareVCSCOperators<T, uint64_t, false>(m, mat7);
    declareVCSCOperators<T, uint64_t, true>(m, mat8);
}


template <typename T, typename indexT, bool isColMajor>
py::class_<IVSparse::VCSC<T, indexT, isColMajor>> declareVCSC(py::module& m) {

    std::string uniqueName = "_";
    uniqueName += returnTypeName<T>();
    uniqueName += "_";
    uniqueName += returnTypeName<indexT>();
    uniqueName += "_";
    uniqueName += (isColMajor) ? "Col" : "Row";;

    py::class_<IVSparse::VCSC<T, indexT, isColMajor>> mat(m, uniqueName.c_str());
    mat.def(py::init<Eigen::SparseMatrix<T, !isColMajor>& >(), py::arg("mat"), py::keep_alive<1, 2>());
    mat.def(py::init<Eigen::SparseMatrix<T>& >(), py::arg("mat"), py::keep_alive<1, 2>());
    mat.def(py::init<T*, indexT*, indexT*, uint32_t, uint32_t, uint32_t>());
    mat.def(py::init<std::vector<std::tuple<indexT, indexT, T>>&, uint32_t, uint32_t, uint32_t>());
    mat.def(py::init<std::unordered_map<T, std::vector<indexT>>*, uint32_t, uint32_t>()); //<std::unordered_map<T, std::vector<indexT>>[], uint32_t, uint32_t> ;
    mat.def(py::init<char*>());
    mat.def(py::init<>());

    return mat;
}

template <typename T, typename indexT, bool isColMajor>
void declareVCSCOperators(py::module& m, py::class_<IVSparse::VCSC<T, indexT, isColMajor>>& mat) {

    mat.def(py::self *= int8_t());
    mat.def(py::self *= uint8_t());
    mat.def(py::self *= int16_t());
    mat.def(py::self *= uint16_t());
    mat.def(py::self *= int32_t());
    mat.def(py::self *= uint32_t());
    mat.def(py::self *= int64_t());
    mat.def(py::self *= uint64_t());
    mat.def("__repr__", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {

        std::stringstream ss;
        ss << self;
        return ss.str();

            }, py::is_operator(), py::return_value_policy::move);
    mat.def("__str__", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {

        std::stringstream ss;
        ss << self;
        return ss.str();

            }, py::is_operator(), py::return_value_policy::move);
    mat.def("__copy__", [](const IVSparse::VCSC<T, indexT, isColMajor>& self) {
        return IVSparse::VCSC<T, indexT, isColMajor>(self);
            }, py::return_value_policy::copy);
    mat.def("__deepcopy__", [](const IVSparse::VCSC<T, indexT, isColMajor>& self) {
        return IVSparse::VCSC<T, indexT, isColMajor>(self);
            }, py::return_value_policy::copy);
    mat.def("copy", [](const IVSparse::VCSC<T, indexT, isColMajor>& self) {
        return IVSparse::VCSC<T, indexT, isColMajor>(self);
            }, py::return_value_policy::copy);
    mat.def("__eq__", [](const IVSparse::VCSC<T, indexT, isColMajor>& self, const IVSparse::VCSC<T, indexT, isColMajor>& other) {
        return self == other;
            }, py::is_operator());
    mat.def("__ne__", [](const IVSparse::VCSC<T, indexT, isColMajor>& self, const IVSparse::VCSC<T, indexT, isColMajor>& other) {
        return !(self == other);
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, int8_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, uint8_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, int16_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, uint16_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, int32_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, uint32_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, int64_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, uint64_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, double a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, float a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, py::EigenDRef<Eigen::Matrix<T, -1, 1>> vec) {

        #ifdef IVSPARSE_DEBUG
        // check that the vector is the correct size
        assert(vec.rows() == self.cols() &&
               "The vector must be the same size as the number of columns in the "
               "matrix!");
        #endif

        Eigen::Matrix<T, -1, -1> eigenTemp = Eigen::Matrix<T, -1, -1>::Zero(self.rows(), 1);

        // iterate over the vector and multiply the corresponding row of the matrix by the vecIter value
        for (uint32_t i = 0; i < self.outerSize(); i++) {
            for (typename IVSparse::VCSC<T, indexT, isColMajor>::InnerIterator matIter(self, i); matIter; ++matIter) {
                eigenTemp(matIter.row()) += vec(matIter.col()) * matIter.value();
            }
        }
        return eigenTemp;
            }, py::is_operator(),
                py::keep_alive<1, 2>(),
                py::return_value_policy::copy);


    mat.def("__mul__", [](IVSparse::VCSC<T, indexT, isColMajor> self, py::EigenDRef<Eigen::Matrix<T, -1, -1, !isColMajor>> mat) {
        #ifdef IVSPARSE_DEBUG
        // check that the matrix is the correct size
        if (mat.rows() != self.cols())
            throw std::invalid_argument(
                "The left matrix must have the same # of rows as columns in the right "
                "matrix!");
        #endif

        Eigen::Matrix<T, -1, -1> newMatrix = Eigen::Matrix<T, -1, -1>::Zero(mat.cols(), self.rows());
        Eigen::Matrix<T, -1, -1> matTranspose = mat.transpose();
        // Fix Parallelism issue (race condition because of partial sums and
        // orientation of Sparse * Dense)

        for (uint32_t col = 0; col < self.outerSize(); col++) {
            for (typename IVSparse::VCSC<T, indexT, isColMajor>::InnerIterator matIter(self, col); matIter; ++matIter) {
                if constexpr (isColMajor) {
                    newMatrix.col(matIter.getIndex()) += matTranspose.col(col) * matIter.value();
                }
                else {
                    newMatrix.col(col) += matTranspose.col(matIter.getIndex()) * matIter.value();
                }
            }
        }
        newMatrix.transposeInPlace();
        return newMatrix;
            }, py::is_operator(),
                py::keep_alive<1, 2>(),
                py::return_value_policy::copy);
    // mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a) {
    //     return self * a;
    //         }, py::is_operator(),
    //             py::keep_alive<1, 2>(),
    //             py::return_value_policy::move); 
    mat.def("__getitem__", [](IVSparse::VCSC<T, indexT, isColMajor>& self, std::tuple<size_t, size_t> index) {
        return self(std::get<0>(index), std::get<1>(index));
            }, py::return_value_policy::copy);

}

template <typename T, typename indexT, bool isColMajor>
void declareVCSCFuncs(py::module& m, py::class_<IVSparse::VCSC<T, indexT, isColMajor>>& mat) {

    mat.def("rows", &IVSparse::VCSC<T, indexT, isColMajor>::rows, py::return_value_policy::copy);
    mat.def("cols", &IVSparse::VCSC<T, indexT, isColMajor>::cols, py::return_value_policy::copy);
    mat.def("innerSize", &IVSparse::VCSC<T, indexT, isColMajor>::innerSize, py::return_value_policy::copy);
    mat.def("outerSize", &IVSparse::VCSC<T, indexT, isColMajor>::outerSize, py::return_value_policy::copy);
    mat.def("shape", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {
        return std::make_tuple(self.rows(), self.cols());
            }, py::return_value_policy::copy);
    mat.def("nonZeros", &IVSparse::VCSC<T, indexT, isColMajor>::nonZeros, py::return_value_policy::copy);
    mat.def("byteSize", &IVSparse::VCSC<T, indexT, isColMajor>::byteSize, py::return_value_policy::copy);
    mat.def("write", &IVSparse::VCSC<T, indexT, isColMajor>::write, py::arg("filename"));
    mat.def("read", &IVSparse::VCSC<T, indexT, isColMajor>::read, py::arg("filename"));
    mat.def("print", &IVSparse::VCSC<T, indexT, isColMajor>::print);
    mat.def("coeff", &IVSparse::VCSC<T, indexT, isColMajor>::coeff, py::return_value_policy::copy, "Sets value at run of coefficient", py::arg("row").none(false), py::arg("col").none(false));
    mat.def("isColumnMajor", &IVSparse::VCSC<T, indexT, isColMajor>::isColumnMajor, py::return_value_policy::copy);
    // mat.def("outerSum", &IVSparse::VCSC<T, indexT, isColMajor>::outerSum, py::return_value_policy::copy);

    mat.def("colSum", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {

        Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = self.colSum().transpose();

        return eigenTemp;
            }, py::return_value_policy::move);

    // mat.def("innerSum", &IVSparse::VCSC<T, indexT, isColMajor>::innerSum, py::return_value_policy::copy);
    mat.def("rowSum", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {
        Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = self.rowSum();

        return eigenTemp;

            }, py::return_value_policy::move);


    mat.def("max", [](IVSparse::VCSC<T, indexT, isColMajor>& self, int axis) {
        Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = self.max(axis);

        return eigenTemp;

            }, py::arg("axis"), py::return_value_policy::move);


    mat.def("min", [](IVSparse::VCSC<T, indexT, isColMajor>& self, int axis) {
        Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = self.min(axis);

        return eigenTemp;

            }, py::arg("axis"), py::return_value_policy::move);

    mat.def("max", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {
        return self.max();

            }, py::return_value_policy::move);

    mat.def("min", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {
        return self.min();

            }, py::return_value_policy::move);




    mat.def("trace", [](IVSparse::VCSC<T, indexT, isColMajor>& self) { return self.trace(); }, py::return_value_policy::copy);
    mat.def("sum", [](IVSparse::VCSC<T, indexT, isColMajor>& self) { return self.sum(); }, py::return_value_policy::copy);
    mat.def("norm", &IVSparse::VCSC<T, indexT, isColMajor>::norm, py::return_value_policy::copy);
    // mat.def("vectorLength", &IVSparse::VCSC<T, indexT, isColMajor>::vectorLength, py::return_value_policy::copy);
    mat.def("toEigen", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {
        return self.toEigen();
            }, py::return_value_policy::move);
    mat.def("transpose", &IVSparse::VCSC<T, indexT, isColMajor>::transpose, py::return_value_policy::copy);
    mat.def("inPlaceTranspose", &IVSparse::VCSC<T, indexT, isColMajor>::inPlaceTranspose);
    mat.def("append", [](IVSparse::VCSC<T, indexT, isColMajor>& self, IVSparse::VCSC<T, indexT, isColMajor>& other) {self.append(other); }, py::arg("other"), py::keep_alive<1, 2>());
    mat.def("append", [](IVSparse::VCSC<T, indexT, isColMajor>& self, Eigen::SparseMatrix<T, !isColMajor>& other) {self.append(other); }, py::arg("other"), py::keep_alive<1, 2>());
    // mat.def("slice", &IVSparse::VCSC<T, indexT, isColMajor>::slice, py::arg("startCol"), py::arg("endCol"), py::return_value_policy::move);

    mat.def("slice", [](IVSparse::VCSC<T, indexT, isColMajor>& self, size_t start, size_t end) {
        IVSparse::VCSC<T, indexT, isColMajor> temp = self.slice(start, end);
        return temp;

            }, py::arg("start"), py::arg("end"), py::return_value_policy::copy, py::keep_alive<1, 2>());
    mat.def("getNumUniqueVals", &IVSparse::VCSC<T, indexT, isColMajor>::getNumUniqueVals, py::arg("col"), py::return_value_policy::copy);
    // mat.def("getValues", &IVSparse::VCSC<T, indexT, isColMajor>::getValues, py::arg("col"), py::return_value_policy::copy);
    mat.def("getValues", [](IVSparse::VCSC<T, indexT, isColMajor>& self, indexT col) {
        std::vector<T> values(self.getNumUniqueVals(col));
        memcpy(values.data(), self.getValues(col), self.getNumUniqueVals(col) * sizeof(T));
        return values;
            }, py::arg("col"), py::return_value_policy::copy);

    // mat.def("getCounts", &IVSparse::VCSC<T, indexT, isColMajor>::getCounts, py::arg("col"), py::return_value_policy::copy);
    mat.def("getCounts", [](IVSparse::VCSC<T, indexT, isColMajor>& self, indexT col) {
        std::vector<indexT> counts(self.getNumUniqueVals(col));
        memcpy(counts.data(), self.getCounts(col), self.getNumUniqueVals(col) * sizeof(indexT));
        return counts;
            }, py::arg("col"), py::return_value_policy::copy);
    mat.def("getNumIndices", &IVSparse::VCSC<T, indexT, isColMajor>::getNumIndices, py::arg("col"), py::return_value_policy::copy);
    // mat.def("getIndices", &IVSparse::VCSC<T, indexT, isColMajor>::getIndices, py::arg("col"), py::return_value_policy::copy);
    mat.def("getIndices", [](IVSparse::VCSC<T, indexT, isColMajor>& self, indexT col) {
        std::vector<indexT> indices(self.getNumIndices(col));
        memcpy(indices.data(), self.getIndices(col), self.getNumIndices(col) * sizeof(indexT));
        return indices;
            }, py::arg("col"), py::return_value_policy::copy);

    mat.def("dtype", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {
        return returnTypeName<T>();
            }, py::return_value_policy::copy);

    mat.def("indexType", [](IVSparse::VCSC<T, indexT, isColMajor>& self) {
        return returnTypeName<indexT>();
            }, py::return_value_policy::copy);
}
