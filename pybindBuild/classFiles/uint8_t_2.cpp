#include "../PyVSparse.hpp"

void init_uint8_t_2(py::module& m) {
    generateForEachIndexType<uint8_t, 2>(m);
}
