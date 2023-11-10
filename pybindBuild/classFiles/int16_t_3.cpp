#include "../PyVSparse.hpp"

void init_int16_t_3(py::module& m) {
    generateForEachIndexType<int16_t, 3>(m);
}
