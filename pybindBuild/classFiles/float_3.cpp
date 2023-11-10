#include "../PyVSparse.hpp"

void init_float_3(py::module& m) {
    generateForEachIndexType<float, 3>(m);
}
