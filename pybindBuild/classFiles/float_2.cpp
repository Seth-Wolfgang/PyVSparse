#include "../PyVSparse.hpp"

void init_float_2(py::module& m) {
    generateForEachIndexType<float, 2>(m);
}
