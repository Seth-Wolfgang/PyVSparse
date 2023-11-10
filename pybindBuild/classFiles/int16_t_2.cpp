#include "../PyVSparse.hpp"

void init_int16_t_2(py::module& m) {
    generateForEachIndexType<int16_t, 2>(m);
}
