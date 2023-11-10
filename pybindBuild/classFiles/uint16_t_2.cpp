#include "../PyVSparse.hpp"

void init_uint16_t_2(py::module& m) {
    generateForEachIndexType<uint16_t, 2>(m);
}
