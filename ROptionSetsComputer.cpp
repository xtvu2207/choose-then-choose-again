// reversible option-sets module
// author: Xuan Tung VU

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "helpers/model_interface.hpp"
#include "helpers/nanoflann.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;
using Vec = viability::Vec;

namespace r_option_sets_internal {

// inclusive 1d bounds for each state dimension
struct Bounds {
    double lo{};
    double hi{};
};

inline void print_progress(int current, int total) {
    if (total <= 0) {
        return;
    }
    const int width = 40;
    const double ratio = std::clamp(
        static_cast<double>(current) / static_cast<double>(total),
        0.0,
        1.0
    );
    const int filled = static_cast<int>(ratio * width);
    std::cout << '\r' << '[';
    for (int i = 0; i < width; ++i) {
        std::cout << (i < filled ? '#' : '-');
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << "%" << std::flush;
}

struct PointCloud {
    // flat storage: point i, dimension k => pts[i * d + k]
    std::vector<double> pts;
    size_t N{0};
    size_t d{0};

    inline size_t kdtree_get_point_count() const { return N; }
    inline double kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx * d + dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    -1,
    uint32_t
>;

class ROptionSetsComputer {
public:
    // uniform resolution per state dimension
    int grid_points_{};
    // viability fixed-point horizon
    int T_viability_{};
    // capture fixed-point horizon
    int T_capture_{};

    // build reusable grid/kd-tree resources for reversible option-set computation
    ROptionSetsComputer(const std::vector<std::pair<double, double>>& K_bounds_in,
                        const std::vector<Vec>& U_values_in,
                        int grid_points,
                        int T_viability,
                        int T_capture,
                        bool verbose = true)
        : grid_points_(grid_points)
        , T_viability_(T_viability)
        , T_capture_(T_capture)
        , verbose_enabled_(verbose)
    {
        if (grid_points_ <= 1) {
            throw std::runtime_error("grid_points must be >= 2");
        }
        if (K_bounds_in.empty()) {
            throw std::runtime_error("K_bounds must be non-empty");
        }
        if (T_viability_ < 0) {
            throw std::runtime_error("T_viability must be >= 0");
        }
        if (T_capture_ < 0) {
            throw std::runtime_error("T_capture must be >= 0");
        }

        // normalize and store k bounds
        K_bounds_ = parse_bounds(K_bounds_in, "K_bounds");
        d_ = K_bounds_.size();

        // validate and store discrete controls
        U_values_ = U_values_in;
        if (U_values_.empty()) {
            throw std::runtime_error("U_values must be non-empty");
        }
        u_dim_ = U_values_.front().size();
        if (u_dim_ == 0) {
            throw std::runtime_error("Each control must be non-empty");
        }
        for (const auto& u : U_values_) {
            if (u.size() != u_dim_) {
                throw std::runtime_error("Each control must have the same dimension");
            }
        }

        // precompute state grid and nearest-neighbor structure
        build_grid_and_points();
        build_kdtree();
    }

    // attach dynamics model used by successor generation
    void set_dynamics(std::shared_ptr<viability::IDynamics> dyn) {
        if (!dyn) {
            dyn_.reset();
            has_successors_ = false;
            successors_ = py::array_t<int32_t>();
            return;
        }

        if (dyn->state_dim() != d_) {
            throw std::runtime_error(
                "Dynamics state dimension mismatch: expected " + std::to_string(d_) +
                ", got " + std::to_string(dyn->state_dim())
            );
        }
        if (dyn->control_dim() != u_dim_) {
            throw std::runtime_error(
                "Dynamics control dimension mismatch: expected " + std::to_string(u_dim_) +
                ", got " + std::to_string(dyn->control_dim())
            );
        }

        dyn_ = std::move(dyn);
        has_successors_ = false;
        successors_ = py::array_t<int32_t>();
    }

    py::object get_successors() const {
        if (!has_successors_) {
            return py::none();
        }
        return successors_;
    }

    bool converged() const { return converged_; }
    int n_iter() const { return n_iter_; }

    // compute r(d1,d2) and r(d2,d1) by alternating viability/capture updates
    py::tuple run(const std::vector<std::pair<double, double>>& D1_bounds_in,
                  const std::vector<std::pair<double, double>>& D2_bounds_in,
                  int max_iter,
                  py::object successors_obj = py::none(),
                  bool parallel = true,
                  py::object n_cores = py::none(),
                  bool verbose = false)
    {
        py::gil_scoped_acquire gil;

        if (max_iter < 0) {
            throw std::runtime_error("max_iter must be >= 0");
        }

        // parse and validate domain bounds
        const auto D1_bounds = parse_bounds(D1_bounds_in, "D1_bounds");
        const auto D2_bounds = parse_bounds(D2_bounds_in, "D2_bounds");

        if (!successors_obj.is_none()) {
            auto successors_in = successors_obj.cast<
                py::array_t<int32_t, py::array::c_style | py::array::forcecast>
            >();
            set_successors(successors_in);
        } else if (!has_successors_) {
            compute_successors(parallel, n_cores);
        }

        const auto& successors = successors_;

        // initialize viability from rectangular domains
        const std::vector<uint8_t> init_D1 = build_rect_mask(D1_bounds);
        const std::vector<uint8_t> init_D2 = build_rect_mask(D2_bounds);

        std::vector<uint8_t> viab_D1;
        std::vector<uint8_t> viab_D2;
        iterate_viability(init_D1, successors, viab_D1, parallel, n_cores, verbose);
        iterate_viability(init_D2, successors, viab_D2, parallel, n_cores, verbose);

        // start from viability sets and alternate capture/intersection updates
        std::vector<uint8_t> E2 = viab_D1;
        std::vector<uint8_t> EB = viab_D2;
        std::vector<uint8_t> capture_from_E2;
        std::vector<uint8_t> capture_from_EB;
        std::vector<uint8_t> EB_new(N_, 0u);
        std::vector<uint8_t> E2_new(N_, 0u);

        converged_ = false;
        n_iter_ = 0;

        for (int it = 0; it < max_iter; ++it) {
            // eb_{k+1} = viab(d2) ∩ capt(e2_k)
            iterate_capture(E2, successors, capture_from_E2, parallel, n_cores, verbose);
            intersect_masks(viab_D2, capture_from_E2, EB_new);

            // e2_{k+1} = viab(d1) ∩ capt(eb_k)
            iterate_capture(EB, successors, capture_from_EB, parallel, n_cores, verbose);
            intersect_masks(viab_D1, capture_from_EB, E2_new);

            const bool stable = std::equal(EB_new.begin(), EB_new.end(), EB.begin()) &&
                                std::equal(E2_new.begin(), E2_new.end(), E2.begin());

            EB.swap(EB_new);
            E2.swap(E2_new);
            n_iter_ = it + 1;

            if (stable) {
                converged_ = true;
                break;
            }
        }

        return py::make_tuple(mask_to_numpy(E2), mask_to_numpy(EB));
    }

private:
    // dynamics used to evolve each state under candidate controls
    std::shared_ptr<viability::IDynamics> dyn_;

    // domain bounds and discrete controls
    std::vector<Bounds> K_bounds_;
    std::vector<Vec> U_values_;
    // grid metadata
    size_t d_{0};
    size_t u_dim_{0};
    size_t N_{0};

    // grid point cloud and nn index
    PointCloud points_;
    std::unique_ptr<KDTree> kdtree_;

    // optional cached successor table
    py::array_t<int32_t> successors_{};
    bool has_successors_{false};
    bool converged_{false};
    int n_iter_{0};
    bool verbose_enabled_{true};

    std::vector<Bounds> parse_bounds(const std::vector<std::pair<double, double>>& bounds_in,
                                     const std::string& name) const
    {
        if (bounds_in.size() != d_ && name != "K_bounds") {
            throw std::runtime_error(
                name + " dimension mismatch: expected " + std::to_string(d_) +
                ", got " + std::to_string(bounds_in.size())
            );
        }

        std::vector<Bounds> out(bounds_in.size());
        for (size_t k = 0; k < bounds_in.size(); ++k) {
            double a = bounds_in[k].first;
            double b = bounds_in[k].second;
            if (b < a) {
                std::swap(a, b);
            }
            out[k].lo = a;
            out[k].hi = b;
        }
        return out;
    }

    void set_successors(py::array_t<int32_t, py::array::c_style | py::array::forcecast> successors) {
        if (successors.ndim() != 2) {
            throw std::runtime_error("successors must be a 2D int32 array of shape (N, m)");
        }
        if ((size_t)successors.shape(0) != N_) {
            throw std::runtime_error("successors rows must match N");
        }
        if (successors.shape(1) <= 0) {
            throw std::runtime_error("successors must have at least one control column");
        }

        auto S = successors.unchecked<2>();
        for (py::ssize_t i = 0; i < successors.shape(0); ++i) {
            for (py::ssize_t j = 0; j < successors.shape(1); ++j) {
                const int32_t s = S(i, j);
                if (s < -1 || s >= static_cast<int32_t>(N_)) {
                    throw std::runtime_error("successors contain out-of-range indices");
                }
            }
        }

        successors_ = successors;
        has_successors_ = true;
    }

    // compute successor table from dynamics and controls
    py::array_t<int32_t> compute_successors(bool parallel, const py::object& n_cores) {
        py::gil_scoped_acquire gil;
        if (!dyn_) {
            throw std::runtime_error("No dynamics set. Call set_dynamics(...) before run()");
        }

        if (dyn_->state_dim() != d_) {
            throw std::runtime_error(
                "Dynamics state dimension mismatch: expected " + std::to_string(d_) +
                ", got " + std::to_string(dyn_->state_dim())
            );
        }
        if (dyn_->control_dim() != u_dim_) {
            throw std::runtime_error(
                "Dynamics control dimension mismatch: expected " + std::to_string(u_dim_) +
                ", got " + std::to_string(dyn_->control_dim())
            );
        }

        const size_t m = U_values_.size();
        py::array_t<int32_t> succ({(py::ssize_t)N_, (py::ssize_t)m});
        int32_t* succ_ptr = succ.mutable_data();
        const size_t succ_stride = m;
        std::fill(succ_ptr, succ_ptr + N_ * m, int32_t(-1));

#ifdef _OPENMP
        int threads = decide_threads(n_cores);
        if (parallel && threads > 0) {
            omp_set_num_threads(threads);
        }
#endif

        const auto dyn = dyn_;
        const double* points_ptr = points_.pts.data();
        const size_t d = d_;

        py::gil_scoped_release no_gil;

        // worker over contiguous state blocks for openmp or serial execution
        auto worker = [&](size_t i0, size_t i1) {
            std::vector<double> query(d, 0.0);
            Vec x(d, 0.0);
            Vec xnext(d, 0.0);
            nanoflann::KNNResultSet<double> result_set(1);

            for (size_t i = i0; i < i1; ++i) {
                const double* point = points_ptr + i * d;
                for (size_t k = 0; k < d; ++k) {
                    x[k] = point[k];
                }

                for (size_t j = 0; j < m; ++j) {
                    dyn->step_inplace(x, U_values_[j], xnext);

                    // skip transitions leaving k: successor remains -1
                    bool inside_k = true;
                    for (size_t k = 0; k < d; ++k) {
                        const double xn = xnext[k];
                        if (xn < K_bounds_[k].lo || xn > K_bounds_[k].hi) {
                            inside_k = false;
                            break;
                        }
                        query[k] = xn;
                    }
                    if (!inside_k) {
                        continue;
                    }

                    // snap continuous successor to nearest grid node
                    size_t ret_index = static_cast<size_t>(-1);
                    double out_dist_sqr = 0.0;
                    result_set.init(&ret_index, &out_dist_sqr);
                    kdtree_->findNeighbors(result_set, query.data());
                    succ_ptr[i * succ_stride + j] = static_cast<int32_t>(ret_index);
                }
            }
        };

#ifdef _OPENMP
        if (parallel && N_ >= 1024) {
            const size_t chunk = (N_ < (1u << 14))
                ? std::max<size_t>(1, N_ / static_cast<size_t>(omp_get_max_threads()))
                : (1u << 14);
            #pragma omp parallel for schedule(dynamic)
            for (int64_t t = 0; t < static_cast<int64_t>(N_); t += static_cast<int64_t>(chunk)) {
                const size_t i0 = static_cast<size_t>(t);
                const size_t i1 = std::min(N_, i0 + chunk);
                worker(i0, i1);
            }
        } else {
            worker(0, N_);
        }
#else
        worker(0, N_);
#endif

        py::gil_scoped_acquire gil2;
        successors_ = succ;
        has_successors_ = true;
        return succ;
    }

    // build a rectangular domain mask over the internal grid
    std::vector<uint8_t> build_rect_mask(const std::vector<Bounds>& rect_bounds) const {
        std::vector<uint8_t> mask(N_, 0u);
        for (size_t i = 0; i < N_; ++i) {
            const double* point = points_.pts.data() + i * d_;
            bool inside = true;
            for (size_t k = 0; k < d_; ++k) {
                const double v = point[k];
                if (v < rect_bounds[k].lo || v > rect_bounds[k].hi) {
                    inside = false;
                    break;
                }
            }
            mask[i] = inside ? 1u : 0u;
        }
        return mask;
    }

    // fixed-point viability iteration on the given initial mask
    void iterate_viability(const std::vector<uint8_t>& initial_mask,
                           const py::array_t<int32_t>& successors,
                           std::vector<uint8_t>& out_mask,
                           bool parallel,
                           const py::object& n_cores,
                           bool verbose) const
    {
        if (initial_mask.size() != N_) {
            throw std::runtime_error("initial viability mask size mismatch");
        }
        if (successors.ndim() != 2 || (size_t)successors.shape(0) != N_) {
            throw std::runtime_error("successors must be of shape (N, m)");
        }
        const size_t m = (size_t)successors.shape(1);
        const int32_t* succ_ptr = successors.data();
        const size_t succ_stride = m;

#ifdef _OPENMP
        int threads = decide_threads(n_cores);
        if (parallel && threads > 0) {
            omp_set_num_threads(threads);
        }
#endif

        std::vector<uint8_t> viable = initial_mask;
        std::vector<uint8_t> new_viable = viable;
        bool changed = true;
        int iteration = 0;
        const int progress_total = std::max(1, T_viability_);
        const int progress_stride = std::max(1, progress_total / 100);

        py::gil_scoped_release no_gil;

        while (changed && iteration < T_viability_) {
            ++iteration;
            std::copy(viable.begin(), viable.end(), new_viable.begin());
            int changed_flag = 0;

#ifdef _OPENMP
            if (parallel && N_ >= 1024) {
                #pragma omp parallel for schedule(static) reduction(|:changed_flag)
                for (int64_t idx = 0; idx < static_cast<int64_t>(N_); ++idx) {
                    if (!viable[(size_t)idx]) {
                        continue;
                    }
                    const int32_t* succ_row = succ_ptr + (size_t)idx * succ_stride;
                    bool still_viable = false;
                    for (size_t j = 0; j < m; ++j) {
                        const int32_t s = succ_row[j];
                        if (s >= 0 && viable[(size_t)s]) {
                            still_viable = true;
                            break;
                        }
                    }
                    if (!still_viable) {
                        // viability operator: remove states with no viable successor
                        new_viable[(size_t)idx] = 0u;
                        changed_flag |= 1;
                    }
                }
            } else
#endif
            {
                for (size_t i = 0; i < N_; ++i) {
                    if (!viable[i]) {
                        continue;
                    }
                    const int32_t* succ_row = succ_ptr + i * succ_stride;
                    bool still_viable = false;
                    for (size_t j = 0; j < m; ++j) {
                        const int32_t s = succ_row[j];
                        if (s >= 0 && viable[(size_t)s]) {
                            still_viable = true;
                            break;
                        }
                    }
                    if (!still_viable) {
                        // viability operator: remove states with no viable successor
                        new_viable[i] = 0u;
                        changed_flag |= 1;
                    }
                }
            }

            changed = (changed_flag != 0);
            viable.swap(new_viable);

            if (verbose && verbose_enabled_ &&
                (iteration == 1 || iteration == progress_total || (iteration % progress_stride) == 0)) {
                print_progress(iteration, progress_total);
            }
        }

        if (verbose && verbose_enabled_) {
            print_progress(progress_total, progress_total);
            std::cout << std::endl;
        }

        out_mask.swap(viable);
    }

    // fixed-point capture iteration for a target mask
    void iterate_capture(const std::vector<uint8_t>& target_mask,
                         const py::array_t<int32_t>& successors,
                         std::vector<uint8_t>& out_mask,
                         bool parallel,
                         const py::object& n_cores,
                         bool verbose) const
    {
        if (target_mask.size() != N_) {
            throw std::runtime_error("target mask size mismatch");
        }
        if (successors.ndim() != 2 || (size_t)successors.shape(0) != N_) {
            throw std::runtime_error("successors must be of shape (N, m)");
        }
        const size_t m = (size_t)successors.shape(1);
        const int32_t* succ_ptr = successors.data();
        const size_t succ_stride = m;

#ifdef _OPENMP
        int threads = decide_threads(n_cores);
        if (parallel && threads > 0) {
            omp_set_num_threads(threads);
        }
#endif

        std::vector<uint8_t> capture = target_mask;
        std::vector<uint8_t> next_capture = capture;
        bool changed = true;
        int iteration = 0;
        const int progress_total = std::max(1, T_capture_);
        const int progress_stride = std::max(1, progress_total / 100);

        py::gil_scoped_release no_gil;

        while (changed && iteration < T_capture_) {
            ++iteration;
            std::copy(capture.begin(), capture.end(), next_capture.begin());
            int changed_flag = 0;

#ifdef _OPENMP
            if (parallel && N_ >= 1024) {
                #pragma omp parallel for schedule(static) reduction(|:changed_flag)
                for (int64_t idx = 0; idx < static_cast<int64_t>(N_); ++idx) {
                    if (capture[(size_t)idx]) {
                        continue;
                    }
                    const int32_t* succ_row = succ_ptr + (size_t)idx * succ_stride;
                    bool reaches_target = false;
                    for (size_t j = 0; j < m; ++j) {
                        const int32_t s = succ_row[j];
                        if (s >= 0 && capture[(size_t)s]) {
                            reaches_target = true;
                            break;
                        }
                    }
                    if (reaches_target) {
                        // capture operator: add states that can reach current capture
                        next_capture[(size_t)idx] = 1u;
                        changed_flag |= 1;
                    }
                }
            } else
#endif
            {
                for (size_t i = 0; i < N_; ++i) {
                    if (capture[i]) {
                        continue;
                    }
                    const int32_t* succ_row = succ_ptr + i * succ_stride;
                    bool reaches_target = false;
                    for (size_t j = 0; j < m; ++j) {
                        const int32_t s = succ_row[j];
                        if (s >= 0 && capture[(size_t)s]) {
                            reaches_target = true;
                            break;
                        }
                    }
                    if (reaches_target) {
                        // capture operator: add states that can reach current capture
                        next_capture[i] = 1u;
                        changed_flag |= 1;
                    }
                }
            }

            changed = (changed_flag != 0);
            capture.swap(next_capture);

            if (verbose && verbose_enabled_ &&
                (iteration == 1 || iteration == progress_total || (iteration % progress_stride) == 0)) {
                print_progress(iteration, progress_total);
            }
        }

        if (verbose && verbose_enabled_) {
            print_progress(progress_total, progress_total);
            std::cout << std::endl;
        }

        out_mask.swap(capture);
    }

    static void intersect_masks(const std::vector<uint8_t>& a,
                                const std::vector<uint8_t>& b,
                                std::vector<uint8_t>& out)
    {
        if (a.size() != b.size()) {
            throw std::runtime_error("mask dimension mismatch for intersection");
        }
        out.resize(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            out[i] = (a[i] && b[i]) ? 1u : 0u;
        }
    }

    py::array_t<uint8_t> mask_to_numpy(const std::vector<uint8_t>& mask) const {
        py::array_t<uint8_t> out((py::ssize_t)N_);
        auto r = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < (py::ssize_t)N_; ++i) {
            r(i) = mask[(size_t)i];
        }
        return out;
    }

    // build a cartesian product grid over k and flatten into point cloud storage
    void build_grid_and_points() {
        std::vector<Vec> axes(d_);
        for (size_t k = 0; k < d_; ++k) {
            axes[k].resize((size_t)grid_points_);
            const double a = K_bounds_[k].lo;
            const double b = K_bounds_[k].hi;
            const double step = (grid_points_ == 1) ? 0.0 : (b - a) / double(grid_points_ - 1);
            for (int i = 0; i < grid_points_; ++i) {
                axes[k][(size_t)i] = a + step * i;
            }
        }

        N_ = 1;
        for (size_t k = 0; k < d_; ++k) {
            N_ *= (size_t)grid_points_;
        }

        points_.d = d_;
        points_.N = N_;
        points_.pts.resize(N_ * d_);

        std::vector<size_t> idx(d_, 0);
        for (size_t n = 0; n < N_; ++n) {
            for (size_t k = 0; k < d_; ++k) {
                points_.pts[n * d_ + k] = axes[k][idx[k]];
            }
            for (size_t k = d_; k-- > 0;) {
                idx[k]++;
                if (idx[k] < (size_t)grid_points_) {
                    break;
                }
                idx[k] = 0;
            }
        }
    }

    // build kd-tree for nearest-neighbor projection back to the grid
    void build_kdtree() {
        kdtree_ = std::make_unique<KDTree>(
            (int)d_,
            points_,
            nanoflann::KDTreeSingleIndexAdaptorParams(10)
        );
        kdtree_->buildIndex();
    }

#ifdef _OPENMP
    // resolve user thread request with sane fallback rules
    int decide_threads(const py::object& n_cores) const {
        if (n_cores.is_none()) {
            return omp_get_max_threads();
        }
        try {
            int v = n_cores.cast<int>();
            return (v <= 0) ? omp_get_max_threads() : v;
        } catch (...) {
            return omp_get_max_threads();
        }
    }
#else
    int decide_threads(const py::object&) const { return 1; }
#endif
};

}  // namespace r_option_sets_internal

void register_r_option_sets(py::module_& m) {
    using namespace r_option_sets_internal;

    // python bindings for reversible option-set computations
    py::class_<ROptionSetsComputer>(m, "ROptionSetsComputer")
        .def(py::init<const std::vector<std::pair<double, double>>&,
                      const std::vector<Vec>&,
                      int,
                      int,
                      int,
                      bool>(),
             py::arg("K_bounds"),
             py::arg("U_values"),
             py::arg("grid_points"),
             py::arg("T_viability"),
             py::arg("T_capture"),
             py::arg("verbose") = true)
        .def("set_dynamics", &ROptionSetsComputer::set_dynamics,
             py::arg("model") = std::shared_ptr<viability::IDynamics>{})
        .def("run", &ROptionSetsComputer::run,
             py::arg("D1_bounds"),
             py::arg("D2_bounds"),
             py::arg("max_iter"),
             py::arg("successors") = py::none(),
             py::arg("parallel") = true,
             py::arg("n_cores") = py::none(),
             py::arg("verbose") = false)
        .def("get_successors", &ROptionSetsComputer::get_successors)
        .def_property_readonly("converged", &ROptionSetsComputer::converged)
        .def_property_readonly("n_iter", &ROptionSetsComputer::n_iter)
        .def_readonly("grid_points", &ROptionSetsComputer::grid_points_)
        .def_readonly("T_viability", &ROptionSetsComputer::T_viability_)
        .def_readonly("T_capture", &ROptionSetsComputer::T_capture_);
}
