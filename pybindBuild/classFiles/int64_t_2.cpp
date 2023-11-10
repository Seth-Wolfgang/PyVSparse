#include "../PyVSparse.hpp"

void init_int64_t_2(py::module& m) {
    generateForEachIndexType<int64_t, 2>(m);
}
