#include "../PyVSparse.hpp"

void init_double_2(py::module& m) {
    generateForEachIndexType<double, 2>(m);
}
