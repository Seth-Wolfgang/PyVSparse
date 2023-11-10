#include "../PyVSparse.hpp"

void init_uint8_t_3(py::module& m) {
    generateForEachIndexType<uint8_t, 3>(m);
}
