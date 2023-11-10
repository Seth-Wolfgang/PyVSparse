#include "../PyVSparse.hpp"

void init_uint32_t_3(py::module& m) {
    generateForEachIndexType<uint32_t, 3>(m);
}
