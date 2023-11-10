#include "../PyVSparse.hpp"

void init_uint16_t_3(py::module& m) {
    generateForEachIndexType<uint16_t, 3>(m);
}
