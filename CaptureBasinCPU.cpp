// capture basin module
// author: Xuan Tung VU
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "helpers/model_interface.hpp"
#include "helpers/nanoflann.hpp"        
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <iostream>
#include <iomanip>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;
using Vec = viability::Vec;

namespace capture_basin_internal {

// inclusive 1d bounds for each state dimension
struct Bounds { double lo{}, hi{}; };

inline void print_progress(int current, int total) {
    if (total <= 0) return;
    const int width = 40;
    double ratio = std::clamp(static_cast<double>(current) / static_cast<double>(total), 0.0, 1.0);
    int filled = static_cast<int>(ratio * width);
    std::cout << '\r' << '[';
    for (int x = 0; x < width; ++x) {
        std::cout << (x < filled ? '#' : '-');
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << "%" << std::flush;
}

struct PointCloud {
    // flat storage: point i, dimension k => pts[i * d + k]
    std::vector<double> pts;
    size_t N{0}, d{0};
    inline size_t kdtree_get_point_count() const { return N; }
    inline double kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx*d + dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    -1,
    uint32_t
>;

class CaptureBasinCPU {
public:
    // uniform resolution per state dimension
    int grid_points_{};
    // maximum number of fixed-point iterations for capture expansion
    int T_{};
    bool exclude_target_{false};  // whether to remove c from the final capture basin

    // build regular grid and data structures for capture computations
    CaptureBasinCPU(const std::vector<std::pair<double,double>>& K_bounds_in,
                    const std::vector<Vec>& U_values_in,
                    int grid_points,
                    int T,
                    bool exclude_target = false,
                    bool verbose = true)
    : grid_points_(grid_points)
    , T_(T)
    , exclude_target_(exclude_target)
    , verbose_enabled_(verbose)
    {
        if (grid_points_ <= 1) throw std::runtime_error("grid_points must be >= 2");
        if (K_bounds_in.empty()) throw std::runtime_error("K_bounds must be non-empty");

        // normalize and store k bounds
        K_bounds_.resize(K_bounds_in.size());
        for (size_t i=0;i<K_bounds_in.size();++i){
            double a = K_bounds_in[i].first;
            double b = K_bounds_in[i].second;
            if (b < a) std::swap(a,b);
            K_bounds_[i].lo = a; K_bounds_[i].hi = b;
        }
        d_ = K_bounds_.size();

        // validate and store discrete controls
        U_values_ = U_values_in;
        if (U_values_.empty()) throw std::runtime_error("U_values must be non-empty");
        u_dim_ = U_values_.front().size();
        if (u_dim_ == 0) throw std::runtime_error("Each control must be non-empty");
        for (const auto& u : U_values_) {
            if (u.size() != u_dim_) {
                throw std::runtime_error("Each control must have the same dimension");
            }
        }

        // precompute state grid and nearest-neighbor structure
        build_grid_and_points();
        build_kdtree();

        // target and capture masks start empty until set_target()
        is_target_.assign(N_, 0);
        is_capture_.assign(N_, 0);
    }

    // attach dynamics model used for successor generation
    void set_dynamics(std::shared_ptr<viability::IDynamics> dyn) {
        if (!dyn) {
            dyn_.reset();
            return;
        }

        if (dyn->state_dim() != d_) {
            throw std::runtime_error(
                "Dynamics state dimension mismatch: expected " + std::to_string(d_) +
                ", got " + std::to_string(dyn->state_dim()));
        }
        if (dyn->control_dim() != u_dim_) {
            throw std::runtime_error(
                "Dynamics control dimension mismatch: expected " + std::to_string(u_dim_) +
                ", got " + std::to_string(dyn->control_dim()));
        }

        dyn_ = std::move(dyn);
    }

    // configure whether to exclude the target from the final capture basin
    void set_exclude_target(bool exclude) {
        exclude_target_ = exclude;
    }

    // set target mask c, with optional remapping from an external grid
    void set_target(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> target_mask,
                    py::object target_bounds_obj = py::none(),
                    py::object target_grid_points_obj = py::none()) {
        
        if (target_mask.ndim()!=1) throw std::runtime_error("target_mask must be 1D");
        
        py::gil_scoped_acquire gil;
        
        // case 1: no target bounds -> direct compatibility with legacy behavior
        if (target_bounds_obj.is_none()) {
            if ((size_t)target_mask.shape(0) != N_) {
                throw std::runtime_error("target_mask has wrong length (should match basin grid)");
            }
            auto r = target_mask.unchecked<1>();
            for (size_t i=0;i<N_;++i) {
                is_target_[i]  = (r((py::ssize_t)i)!=0) ? 1u : 0u;
                is_capture_[i] = is_target_[i]; // initialize c0 = c
            }
            has_target_ = true;
            return;
        }
        // case 2: target bounds provided -> map with nearest neighbor
        
        // read target bounds
        std::vector<std::pair<double,double>> target_bounds_vec = 
            target_bounds_obj.cast<std::vector<std::pair<double,double>>>();
        
        if (target_bounds_vec.size() != d_) {
            throw std::runtime_error("target_bounds dimension mismatch");
        }
        
        // read target grid resolution
        int target_gp = target_grid_points_obj.is_none() ? grid_points_ : target_grid_points_obj.cast<int>();
        
        // build target bounds in kernel format
        std::vector<Bounds> target_bounds(d_);
        for (size_t k=0; k<d_; ++k) {
            double a = target_bounds_vec[k].first;
            double b = target_bounds_vec[k].second;
            if (b < a) std::swap(a,b);
            target_bounds[k].lo = a;
            target_bounds[k].hi = b;
        }
        
        // build target-grid axes
        std::vector<Vec> target_axes(d_);
        for (size_t k=0; k<d_; ++k) {
            target_axes[k].resize((size_t)target_gp);
            double a = target_bounds[k].lo;
            double b = target_bounds[k].hi;
            double step = (target_gp==1) ? 0.0 : (b-a)/double(target_gp-1);
            for (int i=0; i<target_gp; ++i) {
                target_axes[k][(size_t)i] = a + step*i;
            }
        }
        
        // compute target-grid point count
        size_t N_target = 1;
        for (size_t k=0; k<d_; ++k) N_target *= (size_t)target_gp;
        
        if ((size_t)target_mask.shape(0) != N_target) {
            throw std::runtime_error("target_mask length doesn't match target grid size");
        }

        // fast-path: target grid exactly matches the basin grid -> direct copy
        bool same_grid = (target_gp == grid_points_);
        if (same_grid) {
            for (size_t k = 0; k < d_; ++k) {
                if (target_bounds[k].lo != K_bounds_[k].lo || target_bounds[k].hi != K_bounds_[k].hi) {
                    same_grid = false;
                    break;
                }
            }
        }
        if (same_grid) {
            const uint8_t* target_data = target_mask.data();
            if (has_forbid_mask_) {
                for (size_t i = 0; i < N_; ++i) {
                    if (forbid_mask_[i]) {
                        is_target_[i] = 0u;
                        is_capture_[i] = 0u;
                    } else {
                        const uint8_t mark = target_data[i] ? 1u : 0u;
                        is_target_[i] = mark;
                        is_capture_[i] = mark;
                    }
                }
            } else {
                for (size_t i = 0; i < N_; ++i) {
                    const uint8_t mark = target_data[i] ? 1u : 0u;
                    is_target_[i] = mark;
                    is_capture_[i] = mark;
                }
            }
            has_target_ = true;
            return;
        }
        
        // build target-grid points
        PointCloud target_cloud;
        target_cloud.d = d_;
        target_cloud.N = N_target;
        target_cloud.pts.resize(N_target * d_);
        
        std::vector<size_t> idx(d_, 0);
        for (size_t n=0; n<N_target; ++n) {
            for (size_t k=0; k<d_; ++k) {
                target_cloud.pts[n*d_ + k] = target_axes[k][idx[k]];
            }
            // increment multi-dimensional indices
            for (size_t k = d_; k-- > 0; ) {
                idx[k]++;
                if (idx[k] < (size_t)target_gp) break;
                idx[k] = 0;
            }
        }
        
        // build a kd-tree over the target grid
        auto target_kdtree = std::make_unique<KDTree>(
            (int)d_, target_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)
        );
        target_kdtree->buildIndex();
        
        // map external target mask to this grid by nearest neighbor
        const uint8_t* target_data = target_mask.data();
        const double* points_ptr = points_.pts.data();
        const size_t stride = d_;

        py::gil_scoped_release no_gil;

#ifdef _OPENMP
        if (N_ >= 1024) {
            #pragma omp parallel
            {
                std::vector<double> query(d_);
                #pragma omp for schedule(static)
                for (int64_t idx = 0; idx < static_cast<int64_t>(N_); ++idx) {
                    if (has_forbid_mask_ && forbid_mask_[(size_t)idx]) {
                        is_target_[(size_t)idx]  = 0u;
                        is_capture_[(size_t)idx] = 0u;
                        continue;
                    }

                    const double* point = points_ptr + (size_t)idx * stride;
                    bool in_target_bounds = true;
                    for (size_t k=0; k<d_; ++k) {
                        double val = point[k];
                        if (val < target_bounds[k].lo || val > target_bounds[k].hi) {
                            in_target_bounds = false;
                            break;
                        }
                        query[k] = val;
                    }

                    uint8_t mark = 0u;
                    if (in_target_bounds) {
                        size_t ret_index = static_cast<size_t>(-1);
                        double out_dist_sqr = 0.0;
                        nanoflann::KNNResultSet<double> resultSet(1);
                        resultSet.init(&ret_index, &out_dist_sqr);
                        target_kdtree->findNeighbors(resultSet, query.data());
                        mark = target_data[ret_index] ? 1u : 0u;
                    }

                    is_target_[(size_t)idx]  = mark;
                    is_capture_[(size_t)idx] = mark;
                }

            }
        } else
#endif
        {
            std::vector<double> query(d_);
            for (size_t idx = 0; idx < N_; ++idx) {
                if (has_forbid_mask_ && forbid_mask_[idx]) {
                    is_target_[idx]  = 0u;
                    is_capture_[idx] = 0u;
                    continue;
                }

                const double* point = points_ptr + idx * stride;
                bool in_target_bounds = true;
                for (size_t k=0; k<d_; ++k) {
                    double val = point[k];
                    if (val < target_bounds[k].lo || val > target_bounds[k].hi) {
                        in_target_bounds = false;
                        break;
                    }
                    query[k] = val;
                }

                uint8_t mark = 0u;
                if (in_target_bounds) {
                    size_t ret_index = static_cast<size_t>(-1);
                    double out_dist_sqr = 0.0;
                    nanoflann::KNNResultSet<double> resultSet(1);
                    resultSet.init(&ret_index, &out_dist_sqr);
                    target_kdtree->findNeighbors(resultSet, query.data());
                    mark = target_data[ret_index] ? 1u : 0u;
                }

                is_target_[idx]  = mark;
                is_capture_[idx] = mark;
            }
        }
        
        has_target_ = true;
    }

 
    // register externally computed successor table
    void set_successors(py::array_t<int32_t, py::array::c_style | py::array::forcecast> successors) {
        if (successors.ndim()!=2) throw std::runtime_error("successors must be 2D (N x m)");
        if ((size_t)successors.shape(0) != N_) throw std::runtime_error("successors rows != N");
        m_controls_ = (size_t)successors.shape(1);
        successors_ = successors; // keep a local reference copy
        has_successors_ = true;
    }

 
    // compute successor table from dynamics and controls
    py::array_t<int32_t> compute_successors(bool parallel=true,
                                            py::object n_cores=py::none(),
                                            py::object chunk_size=py::none())
    {
        py::gil_scoped_acquire gil;
        (void)chunk_size;
        if (!dyn_) {
            throw std::runtime_error("No dynamics set. Call set_dynamics(...) before compute_successors/run.");
        }
        const size_t m = U_values_.size();
        if (dyn_->state_dim() != d_) {
            throw std::runtime_error(
                "Dynamics state dimension mismatch: expected " + std::to_string(d_) +
                ", got " + std::to_string(dyn_->state_dim()));
        }
        if (dyn_->control_dim() != u_dim_) {
            throw std::runtime_error(
                "Dynamics control dimension mismatch: expected " + std::to_string(u_dim_) +
                ", got " + std::to_string(dyn_->control_dim()));
        }

        // one-shot runtime check to detect malformed dynamics output early
        if (N_ > 0 && m > 0) {
            Vec x0(d_);
            for (size_t k = 0; k < d_; ++k) x0[k] = points_.pts[k];
            Vec x1;
            dyn_->step_inplace(x0, U_values_[0], x1);
            if (x1.size() != d_) {
                throw std::runtime_error(
                    "Dynamics returned invalid next-state dimension: expected " + std::to_string(d_) +
                    ", got " + std::to_string(x1.size()));
            }
        }

        py::array_t<int32_t> succ({(py::ssize_t)N_, (py::ssize_t)m});
        int32_t* succ_ptr = succ.mutable_data();
        const size_t succ_stride = m;

#ifdef _OPENMP
        int threads = decide_threads(n_cores);
        if (parallel && threads>0) omp_set_num_threads(threads);
#endif

        std::fill(succ_ptr, succ_ptr + N_ * m, int32_t(-1));

        py::gil_scoped_release no_gil;
        const auto dyn = dyn_;

        const double* points_ptr = points_.pts.data();
        const size_t stride = d_;

#ifdef _OPENMP
        if (parallel && N_ >= 1024) {
            #pragma omp parallel
            {
                std::vector<double> query(d_, 0.0);
                Vec x(d_);
                Vec xnext(d_);
                nanoflann::KNNResultSet<double> resultSet(1);
                #pragma omp for schedule(dynamic)
                for (int64_t idx = 0; idx < static_cast<int64_t>(N_); ++idx) {
                    const double* point = points_ptr + (size_t)idx * stride;
                    for (size_t k=0; k<d_; ++k) x[k] = point[k];

                    for (size_t j=0; j<m; ++j) {
                        const Vec& u = U_values_[j];
                        dyn->step_inplace(x, u, xnext);

                        // snap continuous successor to nearest grid node
                        size_t ret_index = static_cast<size_t>(-1);
                        double out_dist_sqr = 0.0;
                        resultSet.init(&ret_index, &out_dist_sqr);
                        for (size_t k=0; k<d_; ++k) query[k] = xnext[k];
                        kdtree_->findNeighbors(resultSet, query.data());

                        succ_ptr[(size_t)idx * succ_stride + j] = (int32_t)ret_index;
                    }
                }
            }
        } else
#endif
        {
            std::vector<double> query(d_, 0.0);
            Vec x(d_);
            Vec xnext(d_);
            nanoflann::KNNResultSet<double> resultSet(1);
            for (size_t idx = 0; idx < N_; ++idx) {
                const double* point = points_ptr + idx * stride;
                for (size_t k=0; k<d_; ++k) x[k] = point[k];

                for (size_t j=0; j<m; ++j) {
                    const Vec& u = U_values_[j];
                    dyn->step_inplace(x, u, xnext);

                    // snap continuous successor to nearest grid node
                    size_t ret_index = static_cast<size_t>(-1);
                    double out_dist_sqr = 0.0;
                    resultSet.init(&ret_index, &out_dist_sqr);
                    for (size_t k=0; k<d_; ++k) query[k] = xnext[k];
                    kdtree_->findNeighbors(resultSet, query.data());

                    succ_ptr[idx * succ_stride + j] = (int32_t)ret_index;
                }
            }
        }

        py::gil_scoped_acquire gil2;
        m_controls_ = m;
        successors_ = succ;       
        has_successors_ = true;
        return succ;
    }

 
    // set an additional mask that is removed from the final capture set
    void set_exclude_mask(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
                          py::object bounds_obj,
                          py::object grid_points_obj)
    {
        if (mask.ndim() != 1)
            throw std::runtime_error("exclude_mask must be 1D");

        py::gil_scoped_acquire gil;

        std::vector<std::pair<double,double>> bounds =
            bounds_obj.cast<std::vector<std::pair<double,double>>>();
        int gp = grid_points_obj.cast<int>();
        const size_t d = bounds.size();

        if (d != d_)
            throw std::runtime_error("exclude_mask dimension mismatch");

        // build the grid for the exclusion mask
        std::vector<Vec> axes(d);
        for (size_t k=0; k<d; ++k){
            double a = bounds[k].first;
            double b = bounds[k].second;
            if (b < a) std::swap(a, b);
            axes[k].resize((size_t)gp);
            double step = (gp==1) ? 0.0 : (b-a)/double(gp-1);
            for (int i=0; i<gp; ++i)
                axes[k][(size_t)i] = a + step*i;
        }

        size_t N_excl = 1;
        for (size_t k=0; k<d; ++k)
            N_excl *= (size_t)gp;

        if ((size_t)mask.shape(0) != N_excl)
            throw std::runtime_error("exclude_mask size mismatch");

 
        PointCloud excl_cloud;
        excl_cloud.d = d;
        excl_cloud.N = N_excl;
        excl_cloud.pts.resize(N_excl * d);

        std::vector<size_t> idx(d, 0);
        for (size_t n=0; n<N_excl; ++n){
            for (size_t k=0; k<d; ++k)
                excl_cloud.pts[n*d + k] = axes[k][idx[k]];
 
            for (size_t k = d; k-- > 0; ){
                idx[k]++;
                if (idx[k] < (size_t)gp) break;
                idx[k] = 0;
            }
        }
 
        auto excl_kdtree = std::make_unique<KDTree>(
            (int)d, excl_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)
        );
        excl_kdtree->buildIndex();
 
        const uint8_t* mask_data = mask.data();
        
 
        std::vector<Bounds> excl_bounds(d);
        for (size_t k=0; k<d; ++k) {
            double a = bounds[k].first;
            double b = bounds[k].second;
            if (b < a) std::swap(a, b);
            excl_bounds[k].lo = a;
            excl_bounds[k].hi = b;
        }

        py::gil_scoped_release no_gil;

        exclude_mask_.assign(N_, 0);
        const double* points_ptr = points_.pts.data();
        const size_t stride = d;

#ifdef _OPENMP
        if (N_ >= 1024) {
            #pragma omp parallel
            {
                std::vector<double> query(d);
                #pragma omp for schedule(static)
                for (int64_t idx = 0; idx < static_cast<int64_t>(N_); ++idx) {
                    const double* point = points_ptr + (size_t)idx * stride;
                    bool in_excl_bounds = true;
                    for (size_t k=0; k<d; ++k) {
                        double val = point[k];
                        if (val < excl_bounds[k].lo || val > excl_bounds[k].hi) {
                            in_excl_bounds = false;
                            break;
                        }
                        query[k] = val;
                    }

                    uint8_t mark = 0u;
                    if (in_excl_bounds) {
                        size_t ret_index = static_cast<size_t>(-1);
                        double out_dist_sqr = 0.0;
                        nanoflann::KNNResultSet<double> resultSet(1);
                        resultSet.init(&ret_index, &out_dist_sqr);
                        excl_kdtree->findNeighbors(resultSet, query.data());
                        mark = mask_data[ret_index] ? 1u : 0u;
                    }

                    exclude_mask_[(size_t)idx] = mark;
                }
            }
        } else
#endif
        {
            std::vector<double> query(d);
            for (size_t idx = 0; idx < N_; ++idx) {
                const double* point = points_ptr + idx * stride;
                bool in_excl_bounds = true;
                for (size_t k=0; k<d; ++k) {
                    double val = point[k];
                    if (val < excl_bounds[k].lo || val > excl_bounds[k].hi) {
                        in_excl_bounds = false;
                        break;
                    }
                    query[k] = val;
                }

                uint8_t mark = 0u;
                if (in_excl_bounds) {
                    size_t ret_index = static_cast<size_t>(-1);
                    double out_dist_sqr = 0.0;
                    nanoflann::KNNResultSet<double> resultSet(1);
                    resultSet.init(&ret_index, &out_dist_sqr);
                    excl_kdtree->findNeighbors(resultSet, query.data());
                    mark = mask_data[ret_index] ? 1u : 0u;
                }

                exclude_mask_[idx] = mark;
            }
        }

        has_exclude_mask_ = true;
    }


    // set a hard forbidden mask that is excluded during expansion
    void set_forbid_mask(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
                     py::object bounds_obj,
                     py::object grid_points_obj)
{
    if (mask.ndim() != 1)
        throw std::runtime_error("forbid_mask must be 1D");

    py::gil_scoped_acquire gil;

    std::vector<std::pair<double,double>> bounds =
        bounds_obj.cast<std::vector<std::pair<double,double>>>();
    int gp = grid_points_obj.cast<int>();
    const size_t d = bounds.size();

    if (d != d_)
        throw std::runtime_error("forbid_mask dimension mismatch");

    // build the grid for the forbidden mask
    std::vector<Vec> axes(d);
    for (size_t k=0; k<d; ++k){
        double a = bounds[k].first;
        double b = bounds[k].second;
        if (b < a) std::swap(a, b);
        axes[k].resize((size_t)gp);
        double step = (gp==1) ? 0.0 : (b-a)/double(gp-1);
        for (int i=0; i<gp; ++i)
            axes[k][(size_t)i] = a + step*i;
    }

    size_t N_mask = 1;
    for (size_t k=0; k<d; ++k)
        N_mask *= (size_t)gp;

    if ((size_t)mask.shape(0) != N_mask)
        throw std::runtime_error("forbid_mask size mismatch");

    PointCloud cloud;
    cloud.d = d;
    cloud.N = N_mask;
    cloud.pts.resize(N_mask * d);

    std::vector<size_t> idx(d, 0);
    for (size_t n=0; n<N_mask; ++n){
        for (size_t k=0; k<d; ++k)
            cloud.pts[n*d + k] = axes[k][idx[k]];

        for (size_t k = d; k-- > 0; ){
            idx[k]++;
            if (idx[k] < (size_t)gp) break;
            idx[k] = 0;
        }
    }

    auto kdtree = std::make_unique<KDTree>(
        (int)d, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)
    );
    kdtree->buildIndex();

    const uint8_t* mask_data = mask.data();

    std::vector<Bounds> mask_bounds(d);
    for (size_t k=0; k<d; ++k) {
        double a = bounds[k].first;
        double b = bounds[k].second;
        if (b < a) std::swap(a, b);
        mask_bounds[k].lo = a;
        mask_bounds[k].hi = b;
    }

    py::gil_scoped_release no_gil;

    forbid_mask_.assign(N_, 0);
    const double* points_ptr = points_.pts.data();
    const size_t stride = d;

#ifdef _OPENMP
    if (N_ >= 1024) {
        #pragma omp parallel
        {
            std::vector<double> query(d);
            #pragma omp for schedule(static)
            for (int64_t ii = 0; ii < static_cast<int64_t>(N_); ++ii) {
                const double* point = points_ptr + (size_t)ii * stride;

                bool in_bounds = true;
                for (size_t k=0; k<d; ++k) {
                    double val = point[k];
                    if (val < mask_bounds[k].lo || val > mask_bounds[k].hi) {
                        in_bounds = false;
                        break;
                    }
                    query[k] = val;
                }

                uint8_t mark = 0u;
                if (in_bounds) {
                    size_t ret_index = static_cast<size_t>(-1);
                    double out_dist_sqr = 0.0;
                    nanoflann::KNNResultSet<double> resultSet(1);
                    resultSet.init(&ret_index, &out_dist_sqr);
                    kdtree->findNeighbors(resultSet, query.data());
                    mark = mask_data[ret_index] ? 1u : 0u;
                }

                forbid_mask_[(size_t)ii] = mark;
            }
        }
    } else
#endif
    {
        std::vector<double> query(d);
        for (size_t ii = 0; ii < N_; ++ii) {
            const double* point = points_ptr + ii * stride;

            bool in_bounds = true;
            for (size_t k=0; k<d; ++k) {
                double val = point[k];
                if (val < mask_bounds[k].lo || val > mask_bounds[k].hi) {
                    in_bounds = false;
                    break;
                }
                query[k] = val;
            }

            uint8_t mark = 0u;
            if (in_bounds) {
                size_t ret_index = static_cast<size_t>(-1);
                double out_dist_sqr = 0.0;
                nanoflann::KNNResultSet<double> resultSet(1);
                resultSet.init(&ret_index, &out_dist_sqr);
                kdtree->findNeighbors(resultSet, query.data());
                mark = mask_data[ret_index] ? 1u : 0u;
            }

            forbid_mask_[ii] = mark;
        }
    }

    has_forbid_mask_ = true;
}


    // capture-basin fixed-point iteration:
    // c_{t+1} = c_t u pre(c_t), with optional forbidden-state filtering
    py::array_t<uint8_t> iterate_capture(bool parallel=true,
                                         py::object n_cores=py::none(),
                                         py::object chunk_size=py::none(),
                                         bool verbose=true)
    {
        if (!has_target_)      throw std::runtime_error("Target not set. Call set_target(mask) first");
        if (!has_successors_)  throw std::runtime_error("Successors not set. Call set_successors(...) or compute_successors()");
        (void)chunk_size;

        py::gil_scoped_acquire gil;
        const int32_t* succ_ptr = successors_.data();
        const size_t m = (size_t)successors_.shape(1);
        const size_t succ_stride = m;

#ifdef _OPENMP
        int threads = decide_threads(n_cores);
        if (parallel && threads>0) omp_set_num_threads(threads);
#endif

        std::vector<uint8_t> capture = is_capture_;     // c_t
        std::vector<uint8_t> next_capture = capture;    // c_{t+1}
        if (has_forbid_mask_) {
            for (size_t i=0; i<N_; ++i) {
                if (forbid_mask_[i]) {
                    capture[i] = 0u;
                    is_target_[i] = 0u;
                }
            }
            next_capture = capture;
        }

        bool changed = true;
        int iteration = 0;
        const int progress_total = std::max(1, T_);
        const int progress_stride = std::max(1, progress_total / 100);

        py::gil_scoped_release no_gil;

        while (changed && iteration < T_) {
            iteration += 1;

            // start each iteration from current c_t
            next_capture = capture;
            bool iteration_changed = false;
            const uint8_t* capture_ptr = capture.data();

#ifdef _OPENMP
        if (parallel && N_ >= 1024) {
            int changed_flag = 0;
            #pragma omp parallel for schedule(static) reduction(|:changed_flag)
            for (int64_t idx = 0; idx < static_cast<int64_t>(N_); ++idx) {

                if (has_forbid_mask_ && forbid_mask_[(size_t)idx]) continue;
                if (capture_ptr[(size_t)idx]) continue;

                const int32_t* succ_row = succ_ptr + (size_t)idx * succ_stride;
                for (size_t j = 0; j < m; ++j) {
                    int32_t s = succ_row[j];
                    if (s < 0) continue;
                    if (has_forbid_mask_ && forbid_mask_[(size_t)s]) continue;

                    // state enters capture set if any control reaches current capture
                    if (capture_ptr[(size_t)s]) {
                        next_capture[(size_t)idx] = 1u;
                        changed_flag |= 1;
                        break;
                    }
                }
            }
            iteration_changed = (changed_flag != 0);
        } else
#endif
            {
                for (size_t idx = 0; idx < N_; ++idx) {
                    if (has_forbid_mask_ && forbid_mask_[idx]) continue;
                    if (capture_ptr[idx]) continue;

                    const int32_t* succ_row = succ_ptr + idx * succ_stride;
                    for (size_t j = 0; j < m; ++j) {
                        int32_t s = succ_row[j];
                        if (s < 0) continue;
                        if (has_forbid_mask_ && forbid_mask_[(size_t)s]) continue;

                        // state enters capture set if any control reaches current capture
                        if (capture_ptr[(size_t)s]) {
                            next_capture[idx] = 1u;
                            iteration_changed = true;
                            break;
                        }
                    }
                }
            }

            changed = iteration_changed;
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

        is_capture_ = capture;

        // apply configured post-filters on the converged capture mask
        if (exclude_target_) {
            for (size_t i=0; i<N_; ++i) {
                if (is_target_[i]) {
                    is_capture_[i] = 0;  // remove target points
                }
            }
        }

        if (has_exclude_mask_) {
            for (size_t i=0; i<N_; ++i) {
                if (exclude_mask_[i]) {
                    is_capture_[i] = 0;
                }
            }
        }

        py::gil_scoped_acquire gil3;
        py::array_t<uint8_t> out((py::ssize_t)N_);
        auto r = out.mutable_unchecked<1>();
        for (py::ssize_t i=0;i<(py::ssize_t)N_;++i) r(i)=is_capture_[(size_t)i];
        return out;
    }

    // convenience entrypoint: ensure successors, iterate capture, return points
    py::array_t<double> run(bool parallel=true,
                            py::object n_cores=py::none(),
                            py::object chunk_size=py::none(),
                            bool verbose=true)
    {
        if (!dyn_)              throw std::runtime_error("No dynamics set. Call set_dynamics(...)");
        if (!has_target_)       throw std::runtime_error("Target not set. Call set_target(mask)");

        if (!has_successors_) {
            (void)compute_successors(parallel, n_cores, chunk_size);
        }

        auto cmask = iterate_capture(parallel, n_cores, chunk_size, verbose);
        (void)cmask;

        // return capture-basin points
        size_t Ccount=0; for(uint8_t v: is_capture_) if (v) ++Ccount;
        py::gil_scoped_acquire gil3;
        py::array_t<double> out({(py::ssize_t)Ccount, (py::ssize_t)d_});
        auto R = out.mutable_unchecked<2>();
        size_t row=0;
        for (size_t i=0;i<N_;++i){
            if (!is_capture_[i]) continue;
            for (size_t k=0;k<d_;++k) R((py::ssize_t)row,(py::ssize_t)k)=points_.pts[i*d_+k];
            ++row;
        }
        return out;
    }

 
    py::array_t<double> points() const {
        py::gil_scoped_acquire gil;
        py::array_t<double> out({(py::ssize_t)N_, (py::ssize_t)d_});
        auto R = out.mutable_unchecked<2>();
        for (py::ssize_t i=0;i<(py::ssize_t)N_;++i)
            for (py::ssize_t k=0;k<(py::ssize_t)d_;++k)
                R(i,k)=points_.pts[(size_t)i*d_+(size_t)k];
        return out;
    }

    py::array_t<uint8_t> is_capture() const {
        py::gil_scoped_acquire gil;
        py::array_t<uint8_t> out((py::ssize_t)N_);
        auto r = out.mutable_unchecked<1>();
        for (py::ssize_t i=0;i<(py::ssize_t)N_;++i) r(i)=is_capture_[(size_t)i];
        return out;
    }

    py::array_t<double> K_bounds() const {
        py::gil_scoped_acquire gil;
        py::array_t<double> out({(py::ssize_t)d_, (py::ssize_t)2});
        auto R = out.mutable_unchecked<2>();
        for (py::ssize_t k=0;k<(py::ssize_t)d_;++k){
            R(k,0)=K_bounds_[(size_t)k].lo;
            R(k,1)=K_bounds_[(size_t)k].hi;
        }
        return out;
    }

    py::array_t<double> U_values() const {
        py::gil_scoped_acquire gil;
        const size_t m = U_values_.size();
        py::array_t<double> out({(py::ssize_t)m, (py::ssize_t)u_dim_});
        auto R = out.mutable_unchecked<2>();
        for (py::ssize_t j=0;j<(py::ssize_t)m;++j)
            for (py::ssize_t k=0;k<(py::ssize_t)u_dim_;++k)
                R(j,k)=U_values_[(size_t)j][(size_t)k];
        return out;
    }

private:
    // dynamics used to evolve each state under candidate controls
    std::shared_ptr<viability::IDynamics> dyn_;

    // domain bounds and discrete controls
    std::vector<Bounds> K_bounds_;
    std::vector<Vec>    U_values_;
    // grid metadata
    size_t d_{};
    size_t u_dim_{};
    size_t N_{};
    size_t m_controls_{0};

    // grid point cloud and nn index
    PointCloud points_;
    std::unique_ptr<KDTree> kdtree_;

    // target and capture masks
    std::vector<uint8_t> is_target_;   // c
    std::vector<uint8_t> is_capture_;  // capt(k, c)
    bool has_target_{false};

    // successor table
    py::array_t<int32_t> successors_{};
    bool has_successors_{false};

    // additional exclusion mask
    std::vector<uint8_t> exclude_mask_;
    bool has_exclude_mask_{false};
    bool verbose_enabled_{true};

    // hard exclusion applied during expansion
    std::vector<uint8_t> forbid_mask_;
    bool has_forbid_mask_{false};


    // build a cartesian product grid over k and flatten into point cloud storage
    void build_grid_and_points(){
        std::vector<Vec> axes(d_);
        for (size_t k=0;k<d_;++k){
            axes[k].resize((size_t)grid_points_);
            double a=K_bounds_[k].lo, b=K_bounds_[k].hi;
            double step = (grid_points_==1)? 0.0 : (b-a)/double(grid_points_-1);
            for (int i=0;i<grid_points_;++i) axes[k][(size_t)i]=a+step*i;
        }

        N_ = 1;
        for (size_t k=0;k<d_;++k) N_ *= (size_t)grid_points_;

        points_.d=d_;
        points_.N=N_;
        points_.pts.resize(N_*d_);

        std::vector<size_t> idx(d_,0);
        for (size_t n=0;n<N_;++n){
            for (size_t k=0;k<d_;++k) points_.pts[n*d_+k]=axes[k][idx[k]];
            for (size_t k = d_; k-- > 0; ) {
                idx[k]++;
                if (idx[k] < (size_t)grid_points_) break;
                idx[k] = 0;
            }
        }
    }

    // build kd-tree for nearest-neighbor projection back to the grid
    void build_kdtree(){
        kdtree_ = std::make_unique<KDTree>((int)d_, points_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kdtree_->buildIndex();
    }

#ifdef _OPENMP
    // resolve user thread request with sane fallback rules
    int decide_threads(const py::object& n_cores) const {
        if (n_cores.is_none()) return omp_get_max_threads();
        try {
            int v = n_cores.cast<int>();
            return (v<=0 ? omp_get_max_threads() : v);
        } catch(...) {
            return omp_get_max_threads();
        }
    }
#else
    int decide_threads(const py::object&) const { return 1; }
#endif
};

}  

void register_capture_basin(py::module_& m) {
    using namespace capture_basin_internal;
    // python bindings for the capture-basin kernel
    py::class_<CaptureBasinCPU>(m, "CaptureBasinCPU")
        .def(py::init<const std::vector<std::pair<double,double>>&,
                      const std::vector<Vec>&,
                      int,int,bool,bool>(),
             py::arg("K_bounds"),
             py::arg("U_values"),
             py::arg("grid_points"),
             py::arg("T") = 100,
             py::arg("exclude_target") = false,
             py::arg("verbose") = true)

        .def("set_dynamics", &CaptureBasinCPU::set_dynamics,
             py::arg("model") = std::shared_ptr<viability::IDynamics>{})

        .def("set_exclude_target", &CaptureBasinCPU::set_exclude_target,
             py::arg("exclude"))

        .def("set_target", &CaptureBasinCPU::set_target,
             py::arg("target_mask"),
             py::arg("target_bounds") = py::none(),
             py::arg("target_grid_points") = py::none())

        .def("set_exclude_mask", &CaptureBasinCPU::set_exclude_mask,
             py::arg("mask"),
             py::arg("bounds"),
             py::arg("grid_points"))

        .def("set_forbid_mask", &CaptureBasinCPU::set_forbid_mask,
             py::arg("mask"),
             py::arg("bounds"),
             py::arg("grid_points"))

        .def("set_successors", &CaptureBasinCPU::set_successors,
             py::arg("successors"))

        .def("compute_successors", &CaptureBasinCPU::compute_successors,
             py::arg("parallel")=true, py::arg("n_cores")=py::none(), py::arg("chunk_size")=py::none())

        .def("iterate_capture", &CaptureBasinCPU::iterate_capture,
             py::arg("parallel")=true, py::arg("n_cores")=py::none(), py::arg("chunk_size")=py::none(), py::arg("verbose")=true)

        .def("run", &CaptureBasinCPU::run,
             py::arg("parallel")=true, py::arg("n_cores")=py::none(), py::arg("chunk_size")=py::none(), py::arg("verbose")=true)

        .def_property_readonly("points", &CaptureBasinCPU::points)
        .def_property_readonly("is_capture", &CaptureBasinCPU::is_capture)
        .def_property_readonly("K_bounds", &CaptureBasinCPU::K_bounds)
        .def_property_readonly("U_values", &CaptureBasinCPU::U_values)
        .def_readonly("grid_points", &CaptureBasinCPU::grid_points_)
        .def_readonly("T", &CaptureBasinCPU::T_);
}
