#include "../PyVSparse.hpp"

void init_int32_t_2(py::module& m) {
    generateForEachIndexType<int32_t, 2>(m);
}
