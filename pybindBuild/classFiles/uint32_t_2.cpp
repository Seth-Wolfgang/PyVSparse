#include "../PyVSparse.hpp"

void init_uint32_t_2(py::module& m) {
    generateForEachIndexType<uint32_t, 2>(m);
}
