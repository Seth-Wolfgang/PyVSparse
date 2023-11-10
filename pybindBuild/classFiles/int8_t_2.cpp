#include "../PyVSparse.hpp"

void init_int8_t_2(py::module& m) {
    generateForEachIndexType<int8_t, 2>(m);
}
