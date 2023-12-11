
#include "../PyVSparse.hpp"

template <typename T>
void generateVCSCForEachIndexType(py::module& m);

template <typename T, typename indexT, bool isColMajor>
py::class_<IVSparse::VCSC<T, indexT, isColMajor>> declareVCSC(py::module& m);

template <typename T, typename indexT, bool isColMajor>
void declareVCSCFuncs(py::module& m, py::class_<IVSparse::VCSC<T, indexT, isColMajor>>& mat);

template <typename T, typename indexT, bool isColMajor>
void declareVCSCOperators(py::module& m, py::class_<IVSparse::VCSC<T, indexT, isColMajor>>& mat);
    