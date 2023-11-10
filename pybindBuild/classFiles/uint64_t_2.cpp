#include "../PyVSparse.hpp"

void init_uint64_t_2(py::module& m) {
    generateForEachIndexType<uint64_t, 2>(m);
}
