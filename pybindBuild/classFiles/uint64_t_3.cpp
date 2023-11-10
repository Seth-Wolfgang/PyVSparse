#include "../PyVSparse.hpp"

void init_uint64_t_3(py::module& m) {
    generateForEachIndexType<uint64_t, 3>(m);
}
