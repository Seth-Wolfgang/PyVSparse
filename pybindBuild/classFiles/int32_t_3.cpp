#include "../PyVSparse.hpp"

void init_int32_t_3(py::module& m) {
    generateForEachIndexType<int32_t, 3>(m);
}
