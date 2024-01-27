#include "../PyVSparse.hpp"

template <typename T>
void generateIVCSCForEachType(py::module& m);

template <typename T, bool isColMajor>
py::class_<IVSparse::IVCSC<T, isColMajor>> declareIVCSC(py::module& m);

template <typename T, bool isColMajor>
void declareIVCSCFuncs(py::module& m, py::class_<IVSparse::IVCSC<T, isColMajor>>& mat);

template <typename T, bool isColMajor>
void declareIVCSCOperators(py::module& m, py::class_<IVSparse::IVCSC<T, isColMajor>>& mat);
    
