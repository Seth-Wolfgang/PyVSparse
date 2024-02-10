#include "IVCSC_Wrapper.hpp"


template <typename T>
void generateIVCSCForEachType(py::module& m) {

    py::class_<IVSparse::IVCSC<T, false>> mat1 = declareIVCSC<T, false>(m);
    py::class_<IVSparse::IVCSC<T, true>> mat2 = declareIVCSC<T, true>(m);

    declareIVCSCFuncs<T, false>(m, mat1);
    declareIVCSCFuncs<T, true>(m, mat2);

    declareIVCSCOperators<T, false>(m, mat1);
    declareIVCSCOperators<T, true>(m, mat2);
}


template <typename T, bool isColMajor>
py::class_<IVSparse::IVCSC<T, isColMajor>> declareIVCSC(py::module& m) {

    std::string uniqueName = "_";
    uniqueName += returnTypeName<T>();
    uniqueName += "_";
    uniqueName += (isColMajor) ? "Col" : "Row";

    py::class_<IVSparse::IVCSC<T, isColMajor>> mat(m, uniqueName.c_str());
    mat.def(py::init<Eigen::SparseMatrix<T, !isColMajor>& >(), py::arg("mat"), py::keep_alive<1, 2>());
    // mat.def(py::init<Eigen::SparseMatrix<T, Eigen::RowMajor>& >(), py::arg("mat"), py::keep_alive<1, 2>());
    mat.def(py::init<T*, size_t*, size_t*, uint32_t, uint32_t, uint32_t>());
    mat.def(py::init<std::vector<std::tuple<size_t, size_t, T>>&, uint32_t, uint32_t, uint32_t>());
    mat.def(py::init<std::unordered_map<T, std::vector<size_t>>*, uint32_t, uint32_t>()); //<std::unordered_map<T, std::vector<size_t>>[], uint32_t, uint32_t> ;
    mat.def(py::init<char*>());
    mat.def(py::init<>());

    return mat;
}

template <typename T, bool isColMajor>
void declareIVCSCFuncs(py::module& m, py::class_<IVSparse::IVCSC<T, isColMajor>>& mat) {
    mat.def("rows", &IVSparse::IVCSC<T, isColMajor>::rows, py::return_value_policy::copy);
    mat.def("cols", &IVSparse::IVCSC<T, isColMajor>::cols, py::return_value_policy::copy);
    mat.def("innerSize", &IVSparse::IVCSC<T, isColMajor>::innerSize, py::return_value_policy::copy);
    mat.def("outerSize", &IVSparse::IVCSC<T, isColMajor>::outerSize, py::return_value_policy::copy);
    mat.def("shape", [](IVSparse::IVCSC<T, isColMajor>& self) {
        return std::make_tuple(self.rows(), self.cols());
            }, py::return_value_policy::copy);
    mat.def("nonZeros", &IVSparse::IVCSC<T, isColMajor>::nonZeros, py::return_value_policy::copy);
    mat.def("byteSize", &IVSparse::IVCSC<T, isColMajor>::byteSize, py::return_value_policy::copy);
    mat.def("write", &IVSparse::IVCSC<T, isColMajor>::write, py::arg("filename"));
    mat.def("read", &IVSparse::IVCSC<T, isColMajor>::read, py::arg("filename"));
    mat.def("print", &IVSparse::IVCSC<T, isColMajor>::print);
    mat.def("isColumnMajor", &IVSparse::IVCSC<T, isColMajor>::isColumnMajor, py::return_value_policy::copy);
    mat.def("coeff", &IVSparse::IVCSC<T, isColMajor>::coeff, py::return_value_policy::copy, "Sets value at run of coefficient", py::arg("row").none(false), py::arg("col").none(false));
    
    mat.def("dtype", [](IVSparse::IVCSC<T, isColMajor>& self) {
        return returnTypeName<T>();
            }, py::return_value_policy::copy);

    
    // mat.def("outerSum", [](IVSparse::IVCSC<T, isColMajor>& self){

    //     Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = Eigen::Matrix<T, -1, -1>::Zero(self.outerSize(), 1);   
    //     Eigen::Map<Eigen::Matrix<T, -1, -1, Eigen::RowMajor>> eigenMap(eigenTemp.data(), eigenTemp.rows(), eigenTemp.cols());

    //     memcpy(eigenMap.data(), self.outerSum().data(), self.outerSize() * sizeof(T));

    //     eigenTemp.transposeInPlace();
    //     return eigenTemp;
    //         }, py::return_value_policy::move);

    // mat.def("innerSum", &IVSparse::IVCSC<T, isColMajor>::innerSum, py::return_value_policy::copy);
    // mat.def("innerSum", [](IVSparse::IVCSC<T, isColMajor>& self){

    //     Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = Eigen::Matrix<T, -1, -1>::Zero(self.innerSize(), 1);   
    //     Eigen::Map<Eigen::Matrix<T, -1, -1, Eigen::RowMajor>> eigenMap(eigenTemp.data(), eigenTemp.rows(), eigenTemp.cols());

    //     memcpy(eigenMap.data(), self.innerSum().data(), self.innerSize() * sizeof(T));

    //     eigenTemp.transposeInPlace();
    //     return eigenTemp;
    //         }, py::return_value_policy::move);

    // mat.def("maxColCoeff", &IVSparse::IVCSC<T, isColMajor>::maxColCoeff, py::return_value_policy::copy);
    // mat.def("maxRowCoeff", &IVSparse::IVCSC<T, isColMajor>::maxRowCoeff, py::return_value_policy::copy);
    // mat.def("minRowCoeff", &IVSparse::IVCSC<T, isColMajor>::minRowCoeff, py::return_value_policy::copy);
    // mat.def("minColCoeff", &IVSparse::IVCSC<T, isColMajor>::minColCoeff, py::return_value_policy::copy);
    mat.def("colSum", [](IVSparse::IVCSC<T, isColMajor>& self) {

        Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = self.colSum();

        return eigenTemp;
            }, py::return_value_policy::move);

    // mat.def("innerSum", &IVSparse::IVCSC<T, isColMajor>::innerSum, py::return_value_policy::copy);
    mat.def("rowSum", [](IVSparse::IVCSC<T, isColMajor>& self) {

        Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = self.rowSum();
        return eigenTemp;

            }, py::return_value_policy::move);

    mat.def("max", [](IVSparse::IVCSC<T, isColMajor>& self, int axis) {
        Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = self.max(axis);

        return eigenTemp;

            }, py::arg("axis"), py::return_value_policy::move);


    mat.def("min", [](IVSparse::IVCSC<T, isColMajor>& self, int axis) {
        Eigen::Matrix<T, -1, -1, Eigen::RowMajor> eigenTemp = self.min(axis);

        return eigenTemp;

            }, py::arg("axis"), py::return_value_policy::move);

    mat.def("max", [](IVSparse::IVCSC<T, isColMajor>& self) {
        return self.max();

            }, py::return_value_policy::move);

    mat.def("min", [](IVSparse::IVCSC<T, isColMajor>& self) {
        return self.min();;

            }, py::return_value_policy::move);





    mat.def("trace", [](IVSparse::IVCSC<T, isColMajor>& self) { return self.trace(); }, py::return_value_policy::copy);
    mat.def("sum", [](IVSparse::IVCSC<T, isColMajor>& self) { return self.sum(); }, py::return_value_policy::copy);
    mat.def("norm", &IVSparse::IVCSC<T, isColMajor>::norm, py::return_value_policy::copy);
    // mat.def("vectorLength", &IVSparse::IVCSC<T, isColMajor>::vectorLength, py::return_value_policy::copy);
    mat.def("toEigen", &IVSparse::IVCSC<T, isColMajor>::toEigen, py::return_value_policy::copy);
    mat.def("transpose", &IVSparse::IVCSC<T, isColMajor>::transpose, py::return_value_policy::copy);
    mat.def("inPlaceTranspose", &IVSparse::IVCSC<T, isColMajor>::inPlaceTranspose);
    mat.def("append", [](IVSparse::IVCSC<T, isColMajor>& self, IVSparse::IVCSC<T, isColMajor>& other) {self.append(other); }, py::arg("other"), py::keep_alive<1, 2>());
    mat.def("append", [](IVSparse::IVCSC<T, isColMajor>& self, Eigen::SparseMatrix<T, !isColMajor> other) {self.append(other); });
    // mat.def("slice", &IVSparse::IVCSC<T, isColMajor>::slice, py::arg("startCol"), py::arg("endCol"), py::return_value_policy::copy, py::keep_alive<1, 2>());

    mat.def("slice", [](IVSparse::IVCSC<T, isColMajor>& self, size_t start, size_t end) {
        IVSparse::IVCSC<T, isColMajor> temp = self.slice(start, end);
        return temp;

            }, py::arg("start"), py::arg("end"), py::return_value_policy::copy, py::keep_alive<1, 2>());
    // mat.def("vectorPointer", &IVSparse::IVCSC<T, isColMajor>::vectorPointer, py::return_value_policy::reference);
    // mat.def("getVectorByteSize", &IVSparse::IVCSC<T, isColMajor>::getVectorByteSize, py::return_value_policy::copy);
}

template <typename T, bool isColMajor>
void declareIVCSCOperators(py::module& m, py::class_<IVSparse::IVCSC<T, isColMajor>>& mat) {

    mat.def(py::self *= int8_t());
    mat.def(py::self *= uint8_t());
    mat.def(py::self *= int16_t());
    mat.def(py::self *= uint16_t());
    mat.def(py::self *= int32_t());
    mat.def(py::self *= uint32_t());
    mat.def(py::self *= int64_t());
    mat.def(py::self *= uint64_t());
    mat.def("__repr__", [](IVSparse::IVCSC<T, isColMajor>& self) {

        std::stringstream ss;
        ss << self;
        return ss.str();

            }, py::is_operator(), py::return_value_policy::move);
    mat.def("__str__", [](IVSparse::IVCSC<T, isColMajor>& self) {

        std::stringstream ss;
        ss << self;
        return ss.str();

            }, py::is_operator(), py::return_value_policy::move);
    mat.def("__copy__", [](const IVSparse::IVCSC<T, isColMajor>& self) {
        return IVSparse::IVCSC<T, isColMajor>(self);
            }, py::return_value_policy::copy);
    mat.def("__deepcopy__", [](const IVSparse::IVCSC<T, isColMajor>& self) {                  //TODO: NEED TO CHECK;
        return IVSparse::IVCSC<T, isColMajor>(self);
            }, py::return_value_policy::copy);
    mat.def("copy", [](const IVSparse::IVCSC<T, isColMajor>& self) {
        return IVSparse::IVCSC<T, isColMajor>(self);
            }, py::return_value_policy::copy);
    mat.def("__eq__", [](const IVSparse::IVCSC<T, isColMajor>& self, const IVSparse::IVCSC<T, isColMajor>& other) {
        return self == other;
            }, py::is_operator());
    mat.def("__ne__", [](const IVSparse::IVCSC<T, isColMajor>& self, const IVSparse::IVCSC<T, isColMajor>& other) {
        return !(self == other);
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, int8_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, uint8_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, int16_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, uint16_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, int32_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, uint32_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, int64_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, uint64_t a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, double a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, float a) {
        return self * a;
            }, py::is_operator());
    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, py::EigenDRef<Eigen::Matrix<T, -1, 1>> vec) {

        #ifdef IVSPARSE_DEBUG
        // check that the vector is the correct size
        assert(vec.rows() == self.cols() &&
               "The vector must be the same size as the number of columns in the "
               "matrix!");
        #endif

        Eigen::Matrix<T, -1, -1> eigenTemp = Eigen::Matrix<T, -1, -1>::Zero(self.rows(), 1);

        // iterate over the vector and multiply the corresponding row of the matrix by the vecIter value
        for (uint32_t i = 0; i < self.outerSize(); i++) {
            for (typename IVSparse::IVCSC<T, isColMajor>::InnerIterator matIter(self, i); matIter; ++matIter) {
                eigenTemp(matIter.row()) += vec(matIter.col()) * matIter.value();
            }
        }
        return eigenTemp;
            }, py::is_operator(),
                py::keep_alive<1, 2>(),
                py::return_value_policy::copy);


    mat.def("__mul__", [](IVSparse::IVCSC<T, isColMajor> self, py::EigenDRef<Eigen::Matrix<T, -1, -1, !isColMajor>> mat) {
        // const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b = a;
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
            for (typename IVSparse::IVCSC<T, isColMajor>::InnerIterator matIter(self, col); matIter; ++matIter) {
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
    mat.def("__getitem__", [](IVSparse::IVCSC<T, isColMajor>& self, std::tuple<size_t, size_t> index) {
        return self(std::get<0>(index), std::get<1>(index));
            }, py::return_value_policy::copy);

}