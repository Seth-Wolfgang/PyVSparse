#include "../PyVSparse.hpp"

void init_double_3(py::module& m) {
    generateForEachIndexType<double, 3>(m);
}
