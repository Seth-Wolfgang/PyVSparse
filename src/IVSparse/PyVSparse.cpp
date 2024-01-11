#include "PyVSparse.hpp"
#include "IVCSC/IVCSC_Wrapper.cpp"
#include "VCSC/VCSC_Wrapper.cpp"

PYBIND11_MODULE(_PyVSparse, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring


    py::module_ ivcsc = m.def_submodule("_IVCSC", "IVCSC submodule");
    py::module_ vcsc = m.def_submodule("_VCSC", "VCSC submodule");

    generateVCSCForEachIndexType<int8_t>(vcsc);
    generateVCSCForEachIndexType<uint8_t>(vcsc);
    generateVCSCForEachIndexType<int16_t>(vcsc);
    generateVCSCForEachIndexType<uint16_t>(vcsc);
    generateVCSCForEachIndexType<int32_t>(vcsc);
    generateVCSCForEachIndexType<uint32_t>(vcsc);
    generateVCSCForEachIndexType<int64_t>(vcsc);
    generateVCSCForEachIndexType<uint64_t>(vcsc);
    generateVCSCForEachIndexType<float>(vcsc);
    generateVCSCForEachIndexType<double>(vcsc);

    generateIVCSCForEachType<int8_t>(ivcsc);
    generateIVCSCForEachType<uint8_t>(ivcsc);
    generateIVCSCForEachType<int16_t>(ivcsc);
    generateIVCSCForEachType<uint16_t>(ivcsc);
    generateIVCSCForEachType<int32_t>(ivcsc);
    generateIVCSCForEachType<uint32_t>(ivcsc);
    generateIVCSCForEachType<int64_t>(ivcsc);
    generateIVCSCForEachType<uint64_t>(ivcsc);
    generateIVCSCForEachType<float>(ivcsc);
    generateIVCSCForEachType<double>(ivcsc);

};
