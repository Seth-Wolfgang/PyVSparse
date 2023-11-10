#include "../PyVSparse.hpp"

void init_int64_t_3(py::module& m) {
    generateForEachIndexType<int64_t, 3>(m);
}
