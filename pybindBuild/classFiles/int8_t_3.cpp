#include "../PyVSparse.hpp"

void init_int8_t_3(py::module& m) {
    generateForEachIndexType<int8_t, 3>(m);
}
