#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gcfl_fast {

static inline void filter_finite(std::vector<double>& v) {
    v.erase(std::remove_if(v.begin(), v.end(),
           [](double x){ return !std::isfinite(x); }), v.end());
}

static inline double mean(const std::vector<double>& v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / static_cast<double>(v.size());
}

double trimmed_mean(std::vector<double> v, double trim_ratio, bool assume_sorted) {
    filter_finite(v);
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();

    if (!assume_sorted) std::sort(v.begin(), v.end());

    double r = std::isfinite(trim_ratio) ? trim_ratio : 0.0;
    if (r < 0.0) r = 0.0;
    if (r > 0.5) r = 0.5;

    const std::size_t n = v.size();
    const std::size_t k = static_cast<std::size_t>(std::llround(r * static_cast<double>(n)));
    if (2 * k >= n) return mean(v);

    double s = 0.0;
    std::size_t cnt = 0;
    for (std::size_t i = k; i < n - k; ++i) {
        s += v[i];
        ++cnt;
    }
    return s / static_cast<double>(cnt > 0 ? cnt : 1);
}

double sorted_weighted_mean(std::vector<double> v, std::vector<double> w, bool assume_sorted) {
    filter_finite(v);
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();

    if (!assume_sorted) std::sort(v.begin(), v.end());

    // clip negatives and non-finites to zero
    for (auto& x : w) {
        if (!std::isfinite(x) || x < 0.0) x = 0.0;
    }
    if (w.size() != v.size()) {
        // Require same length; interpolation is handled in Python path.
        throw std::invalid_argument("weights length must match values length");
    }
    double s = std::accumulate(w.begin(), w.end(), 0.0);
    if (s <= 0.0) return mean(v);
    for (auto& x : w) x /= s;

    double dot = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        dot += v[i] * w[i];
    }
    return dot;
}

} // namespace gcfl_fast
