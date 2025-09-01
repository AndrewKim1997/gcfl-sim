#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "fast_kernels.cpp"  // or use a header if you split declarations

namespace py = pybind11;
using gcfl_fast::trimmed_mean;
using gcfl_fast::sorted_weighted_mean;

static std::vector<double> as_vector(const py::array_t<double>& a) {
    py::buffer_info info = a.request();
    if (info.ndim == 0) return {};
    // flatten
    const std::size_t n = static_cast<std::size_t>(info.size);
    const double* data = static_cast<const double*>(info.ptr);
    return std::vector<double>(data, data + n);
}

PYBIND11_MODULE(gcfl_fast, m) {
    m.doc() = "gcfl-sim fast kernels (pybind11)";

    m.def("trimmed_mean",
          [](py::array_t<double> values, double trim_ratio, bool assume_sorted) {
              auto v = as_vector(values);
              return trimmed_mean(std::move(v), trim_ratio, assume_sorted);
          },
          py::arg("values"), py::arg("trim_ratio") = 0.10, py::arg("assume_sorted") = false,
          R"pbdoc(
              Trimmed mean of `values` with symmetric ratio `trim_ratio` in [0, 0.5].
              Non-finite values are ignored. If `assume_sorted` is false, values are sorted.
          )pbdoc");

    m.def("sorted_weighted",
          [](py::array_t<double> values, py::array_t<double> weights, bool assume_sorted) {
              auto v = as_vector(values);
              auto w = as_vector(weights);
              return sorted_weighted_mean(std::move(v), std::move(w), assume_sorted);
          },
          py::arg("values"), py::arg("weights"), py::arg("assume_sorted") = false,
          R"pbdoc(
              Weighted mean of sorted `values` with normalized nonnegative `weights`.
              Requires len(weights) == len(values). Non-finite values are ignored.
          )pbdoc");
}
