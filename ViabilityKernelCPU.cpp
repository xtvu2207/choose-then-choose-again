// viability kernel module
// author: Xuan Tung VU
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "helpers/model_interface.hpp"
#include "models/model_bindings.hpp"
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

#include "helpers/nanoflann.hpp"

namespace py = pybind11;
using Vec = viability::Vec;

// trampoline class so python subclasses can implement idynamics
class PyDynamics : public viability::IDynamics {
public:
    using viability::IDynamics::IDynamics;

    std::size_t state_dim() const override {
        PYBIND11_OVERRIDE_PURE(std::size_t, viability::IDynamics, state_dim);
    }

    std::size_t control_dim() const override {
        PYBIND11_OVERRIDE_PURE(std::size_t, viability::IDynamics, control_dim);
    }

    Vec step(const Vec& x, const Vec& u) const override {
        PYBIND11_OVERRIDE_PURE(Vec, viability::IDynamics, step, x, u);
    }

    void step_inplace(const Vec& x, const Vec& u, Vec& xnext) const override {
        PYBIND11_OVERRIDE(void, viability::IDynamics, step_inplace, x, u, xnext);
    }
};

// inclusive 1d bounds for each state dimension
struct Bounds { double lo{}, hi{}; };

static inline void print_progress(int current, int total) {
    if (total <= 0) return;
    const int width = 40;
    double ratio = static_cast<double>(current) / static_cast<double>(total);
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;
    int filled = static_cast<int>(ratio * width);
    std::cout << '\r' << '[';
    for (int i = 0; i < width; ++i) {
        std::cout << (i < filled ? '#' : '-');
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << "%" << std::flush;
}

static inline bool in_K_vec(const Vec& x, const std::vector<Bounds>& K_bounds) {
    for (size_t k=0;k<K_bounds.size();++k){
        if (x[k] < K_bounds[k].lo || x[k] > K_bounds[k].hi) return false;
    }
    return true;
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

class ViabilityKernelCPU {
public:
    // uniform resolution per state dimension
    int grid_points_{};
    // maximum number of fixed-point iterations for viability pruning
    int T_{};
    // reserved for compatibility/tuning hooks
    double alpha_{0.25};  


    // build the regular grid, kd-tree, and default viability mask over k
    ViabilityKernelCPU(const std::vector<std::pair<double,double>>& K_bounds_in,
                       const std::vector<Vec>& U_values_in,
                       int grid_points,
                       int T,
                       double alpha = 0.25,
                       bool verbose = true)
    : grid_points_(grid_points)
    , T_(T)
    , alpha_(alpha)
    , verbose_enabled_(verbose)
    {
        if (grid_points_ <= 1) throw std::runtime_error("grid_points must be >= 2");
        if (K_bounds_in.empty()) throw std::runtime_error("K_bounds must be non-empty");

        // normalize and store domain bounds
        K_bounds_.resize(K_bounds_in.size());
        for (size_t i=0;i<K_bounds_in.size();++i){
            double a = K_bounds_in[i].first;
            double b = K_bounds_in[i].second;
            if (b < a) std::swap(a,b);
            K_bounds_[i].lo = a; K_bounds_[i].hi = b;
        }
        d_ = K_bounds_.size();

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

        // default initialization: all grid points inside k are marked viable
        is_viable_.assign(N_, 0);
        Vec x(d_);
        for (size_t i=0;i<N_;++i){
            for (size_t k=0;k<d_;++k) x[k]=points_.pts[i*d_+k];
            is_viable_[i] = in_K_vec(x, K_bounds_) ? 1 : 0;
        }
    }

    // attach dynamics model used by successor computation
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

    // set initial viability mask, optionally remapping from another grid
    void set_initial_viable(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
                           py::object bounds_obj = py::none(),
                           py::object grid_points_obj = py::none())
    {
        if (mask.ndim() != 1) throw std::runtime_error("mask must be 1D");
        
        py::gil_scoped_acquire gil;
        
 
        // fast path: incoming mask already matches the kernel grid
        if (bounds_obj.is_none()) {
            if ((size_t)mask.shape(0) != N_) {
                throw std::runtime_error("mask length doesn't match grid size");
            }
            auto r = mask.unchecked<1>();
            for (size_t i=0; i<N_; ++i) {
                is_viable_[i] = r((py::ssize_t)i);
            }
            return;
        }
 
        // general path: remap a mask defined on another grid/bounds
        std::vector<std::pair<double,double>> mask_bounds_vec = 
            bounds_obj.cast<std::vector<std::pair<double,double>>>();
        
        if (mask_bounds_vec.size() != d_) {
            throw std::runtime_error("mask bounds dimension mismatch");
        }
  
        int mask_gp = grid_points_obj.is_none() ? grid_points_ : grid_points_obj.cast<int>();
 
        std::vector<Bounds> mask_bounds(d_);
        for (size_t k=0; k<d_; ++k) {
            double a = mask_bounds_vec[k].first;
            double b = mask_bounds_vec[k].second;
            if (b < a) std::swap(a, b);
            mask_bounds[k].lo = a;
            mask_bounds[k].hi = b;
        }
        
        // build source-grid axes
        std::vector<Vec> mask_axes(d_);
        for (size_t k=0; k<d_; ++k) {
            mask_axes[k].resize((size_t)mask_gp);
            double a = mask_bounds[k].lo;
            double b = mask_bounds[k].hi;
            double step = (mask_gp==1) ? 0.0 : (b-a)/double(mask_gp-1);
            for (int i=0; i<mask_gp; ++i) {
                mask_axes[k][(size_t)i] = a + step*i;
            }
        }
        
        // compute the number of source-mask points
        size_t N_mask = 1;
        for (size_t k=0; k<d_; ++k) N_mask *= (size_t)mask_gp;
        
        if ((size_t)mask.shape(0) != N_mask) {
            throw std::runtime_error("mask length doesn't match mask grid size");
        }

        // fast-path: mask grid exactly matches the kernel grid -> direct copy
        bool same_grid = (mask_gp == grid_points_);
        if (same_grid) {
            for (size_t k = 0; k < d_; ++k) {
                if (mask_bounds[k].lo != K_bounds_[k].lo || mask_bounds[k].hi != K_bounds_[k].hi) {
                    same_grid = false;
                    break;
                }
            }
        }
        if (same_grid) {
            const uint8_t* mask_ptr = mask.data();
            std::copy(mask_ptr, mask_ptr + N_, is_viable_.begin());
            return;
        }
        
        // build source-mask points
        PointCloud mask_cloud;
        mask_cloud.d = d_;
        mask_cloud.N = N_mask;
        mask_cloud.pts.resize(N_mask * d_);
        
        std::vector<size_t> idx(d_, 0);
        for (size_t n=0; n<N_mask; ++n) {
            for (size_t k=0; k<d_; ++k) {
                mask_cloud.pts[n*d_ + k] = mask_axes[k][idx[k]];
            }
            // increment multi-dimensional indices
            for (size_t k = d_; k-- > 0; ) {
                idx[k]++;
                if (idx[k] < (size_t)mask_gp) break;
                idx[k] = 0;
            }
        }
        
        // build a kd-tree for the source mask
        auto mask_kdtree = std::make_unique<KDTree>(
            (int)d_, mask_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)
        );
        mask_kdtree->buildIndex();

        // map with nearest-neighbor lookup
        auto r = mask.unchecked<1>();
        
        py::gil_scoped_release no_gil;
        
        // for each kernel point, nearest-neighbor sample the source mask
        std::vector<double> query(d_);
        for (size_t i=0; i<N_; ++i) {
            // kernel-grid point
            for (size_t k=0; k<d_; ++k) {
                query[k] = points_.pts[i*d_ + k];
            }
            
            // check whether the point lies inside mask bounds
            bool in_mask_bounds = true;
            for (size_t k=0; k<d_; ++k) {
                if (query[k] < mask_bounds[k].lo || query[k] > mask_bounds[k].hi) {
                    in_mask_bounds = false;
                    break;
                }
            }
            
            if (in_mask_bounds) {
                // find the nearest source-mask point
                size_t ret_index = static_cast<size_t>(-1);
                double out_dist_sqr = 0.0;
                nanoflann::KNNResultSet<double> resultSet(1);
                resultSet.init(&ret_index, &out_dist_sqr);
                mask_kdtree->findNeighbors(resultSet, query.data());
                
                if (ret_index < N_mask) {
                    is_viable_[i] = r((py::ssize_t)ret_index);
                } else {
                    is_viable_[i] = 0;
                }
            } else {
                is_viable_[i] = 0;
            }
        }
        
    }

    bool in_K(py::array_t<double, py::array::c_style | py::array::forcecast> x) const {
        if (x.ndim()!=1 || (size_t)x.shape(0)!=d_) throw std::runtime_error("x must be 1D of length d");
        auto r = x.unchecked<1>();
        for (size_t k=0;k<d_;++k){
            double v = r((py::ssize_t)k);
            if (v < K_bounds_[(size_t)k].lo || v > K_bounds_[(size_t)k].hi) return false;
        }
        return true;
    }
// compute successor table: for each state and control, nearest next grid index
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

    // worker over contiguous state blocks for openmp or serial execution
    auto worker = [&](size_t i0, size_t i1){
        std::vector<double> query(d_, 0.0);
        Vec x(d_);
        Vec xnext(d_);
        const auto* bounds = K_bounds_.data();
        nanoflann::KNNResultSet<double> resultSet(1);
        for (size_t i=i0;i<i1;++i){
            const double* point = points_ptr + i * d_;
            for (size_t k=0;k<d_;++k) x[k]=point[k];

            for (size_t j=0;j<m;++j){
                const Vec& u = U_values_[j];
                dyn->step_inplace(x, u, xnext);

                // skip transitions leaving k: successor remains -1
                bool inside_k = true;
                for (size_t k=0;k<d_;++k){
                    const double xn = xnext[k];
                    if (xn < bounds[k].lo || xn > bounds[k].hi) {
                        inside_k = false;
                        break;
                    }
                    query[k] = xn;
                }
                if (!inside_k) continue;

                // snap continuous successor to nearest grid node
                size_t ret_index = static_cast<size_t>(-1);
                double out_dist_sqr = 0.0;
                resultSet.init(&ret_index, &out_dist_sqr);
                kdtree_->findNeighbors(resultSet, query.data());

                // optional anti-artifact filter:
                // if (ret_index == i) continue;  // avoid self-loops

                succ_ptr[i * succ_stride + j] = static_cast<int32_t>(ret_index);
            }
        }
    };

#ifdef _OPENMP
    if (parallel) {
        // chunked dynamic scheduling balances heterogeneous trajectory costs
        const size_t CHUNK = (N_ < (1u<<14)) ? std::max<size_t>(1,N_/((size_t)omp_get_max_threads())) : (1u<<14);
        #pragma omp parallel for schedule(dynamic)
        for (int64_t t=0; t<(int64_t)N_; t+=(int64_t)CHUNK){
            size_t i0=(size_t)t, i1=std::min(N_, i0+CHUNK);
            worker(i0,i1);
        }
    } else {
        worker(0,N_);
    }
#else
    worker(0,N_);
#endif

    py::gil_scoped_acquire gil2;
    successors_ = succ;
    return succ;
}


    // fixed-point viability iteration using the precomputed successor graph
    py::array_t<uint8_t> iterate_viability(py::array_t<int32_t, py::array::c_style | py::array::forcecast> successors,
                                           bool parallel=true,
                                           py::object n_cores=py::none(),
                                           py::object chunk_size=py::none(),
                                           bool verbose=true)
    {
        py::gil_scoped_acquire gil;
        (void)chunk_size;
        if (successors.ndim()!=2 || (size_t)successors.shape(0)!=N_) throw std::runtime_error("successors must be (N,m)");
        const size_t m = (size_t)successors.shape(1);
        const int32_t* succ_ptr = successors.data();
        const size_t succ_stride = m;

#ifdef _OPENMP
        int threads = decide_threads(n_cores);
        if (parallel && threads>0) omp_set_num_threads(threads);
#endif

        std::vector<uint8_t> viable = is_viable_;
        std::vector<uint8_t> new_viable = viable;

        bool changed = true;
        int iteration = 0;
        const int progress_total = std::max(1, T_);
        const int progress_stride = std::max(1, progress_total / 100);

        py::gil_scoped_release no_gil;

        while (changed && iteration < T_) {
            iteration += 1;

            std::copy(viable.begin(), viable.end(), new_viable.begin());
            int changed_flag = 0;

#ifdef _OPENMP
            if (parallel) {
                #pragma omp parallel for schedule(static) reduction(|:changed_flag)
                for (int64_t idx=0; idx<(int64_t)N_; ++idx){
                    if (!viable[(size_t)idx]) continue;
                    const int32_t* succ_row = succ_ptr + (size_t)idx * succ_stride;
                    bool still=false;
                    for (size_t j=0;j<m;++j){
                        int32_t s = succ_row[j];
                        if (s>=0 && viable[(size_t)s]){ still=true; break; }
                    }
                    if (!still){
                        // viability operator: remove states with no viable successor
                        new_viable[(size_t)idx]=0;
                        changed_flag |= 1;
                    }
                }
            } else {
                for (size_t i=0;i<N_;++i){
                    if (!viable[i]) continue;
                    const int32_t* succ_row = succ_ptr + i * succ_stride;
                    bool still=false;
                    for (size_t j=0;j<m;++j){
                        int32_t s = succ_row[j];
                        if (s>=0 && viable[(size_t)s]){ still=true; break; }
                    }
                    if (!still){
                        // viability operator: remove states with no viable successor
                        new_viable[i]=0;
                        changed_flag |= 1;
                    }
                }
            }
#else
            for (size_t i=0;i<N_;++i){
                if (!viable[i]) continue;
                const int32_t* succ_row = succ_ptr + i * succ_stride;
                bool still=false;
                for (size_t j=0;j<m;++j){
                    int32_t s = succ_row[j];
                    if (s>=0 && viable[(size_t)s]){ still=true; break; }
                }
                if (!still){
                    // viability operator: remove states with no viable successor
                    new_viable[i]=0;
                    changed_flag |= 1;
                }
            }
#endif

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

        is_viable_ = viable;

        py::gil_scoped_acquire gil3;
        py::array_t<uint8_t> out((py::ssize_t)N_);
        auto r = out.mutable_unchecked<1>();
        for (py::ssize_t i=0;i<(py::ssize_t)N_;++i) r(i)=is_viable_[(size_t)i];
        return out;
    }

    // convenience entrypoint: compute successors, iterate viability, return viable points
    py::array_t<double> run(bool parallel=true,
                            py::object n_cores=py::none(),
                            py::object chunk_size=py::none(),
                            bool verbose=true)
    {
        py::gil_scoped_acquire gil;
#ifdef _OPENMP
        if (parallel){
            int threads = decide_threads(n_cores);
            if (threads>0) omp_set_num_threads(threads);
        }
#endif

        if (!dyn_) {
            throw std::runtime_error("No dynamics set. Call set_dynamics(...) before run().");
        }

        auto successors = compute_successors(parallel, n_cores, chunk_size);
        auto vmask = iterate_viability(successors, parallel, n_cores, chunk_size, verbose);
        (void)vmask;

        // export viable points as an (n_viable, d) dense array
        size_t Kcount=0; for(uint8_t v: is_viable_) if (v) ++Kcount;
        py::array_t<double> out({(py::ssize_t)Kcount, (py::ssize_t)d_});
        auto R = out.mutable_unchecked<2>();
        size_t row=0;
        for (size_t i=0;i<N_;++i){
            if (!is_viable_[i]) continue;
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

    py::array_t<uint8_t> is_viable() const {
        py::gil_scoped_acquire gil;
        py::array_t<uint8_t> out((py::ssize_t)N_);
        auto r = out.mutable_unchecked<1>();
        for (py::ssize_t i=0;i<(py::ssize_t)N_;++i) r(i)=is_viable_[(size_t)i];
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

    py::array_t<int32_t> successors() const {  
    return successors_;
    }


private:
    // dynamics used to evolve each state under candidate controls
    std::shared_ptr<viability::IDynamics> dyn_;

    // domain bounds and discrete controls
    std::vector<Bounds> K_bounds_;
    std::vector<Vec> U_values_;
    // grid metadata
    size_t d_{};
    size_t u_dim_{};
    size_t N_{};
    // grid point cloud and nn index
    PointCloud points_;
    std::vector<uint8_t> is_viable_;
    std::unique_ptr<KDTree> kdtree_;
    // cached successor table
    py::array_t<int32_t> successors_;
    // per-axis grid step and inverse step (currently informational)
    std::vector<double> grid_steps_;
    std::vector<double> inv_grid_steps_;
    bool verbose_enabled_{true};

    // build a cartesian product grid over k and flatten into point cloud storage
    void build_grid_and_points(){
        std::vector<Vec> axes(d_);
        grid_steps_.assign(d_, 0.0);
        inv_grid_steps_.assign(d_, 0.0);
        for (size_t k=0;k<d_;++k){
            axes[k].resize((size_t)grid_points_);
            double a=K_bounds_[k].lo, b=K_bounds_[k].hi;
            double step = (grid_points_==1)? 0.0 : (b-a)/double(grid_points_-1);
            grid_steps_[k] = step;
            inv_grid_steps_[k] = (step > 0.0) ? 1.0/step : 0.0;
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
void register_capture_basin(pybind11::module_& m);
void register_r_option_sets(pybind11::module_& m);
PYBIND11_MODULE(viability, m) {
    // single python extension module exposing all kernels and dynamics bindings
    m.doc() = "Saint-Pierre viability kernel (C++ + KDTree) with pluggable dynamics models";

    // abstract base class: users can pass either c++ models or python subclasses
    py::class_<viability::IDynamics, PyDynamics, std::shared_ptr<viability::IDynamics>>(m, "IDynamics")
        .def(py::init<>())
        .def("state_dim", &viability::IDynamics::state_dim)
        .def("control_dim", &viability::IDynamics::control_dim)
        .def("step", &viability::IDynamics::step, py::arg("x"), py::arg("u"))
        .def("step_inplace",
             [](const viability::IDynamics& self, const Vec& x, const Vec& u) {
                 Vec xnext;
                 self.step_inplace(x, u, xnext);
                 return xnext;
             },
             py::arg("x"), py::arg("u"));

    register_model_bindings(m);

    // main viability kernel api
    py::class_<ViabilityKernelCPU>(m, "ViabilityKernelCPU")
        .def(py::init<const std::vector<std::pair<double,double>>&,
                      const std::vector<Vec>&,
                      int,int,double,bool>(),
             py::arg("K_bounds"),
             py::arg("U_values"),
             py::arg("grid_points"),
             py::arg("T") = 100,
            py::arg("alpha") = 0.25,
            py::arg("verbose") = true)

        .def("set_dynamics", &ViabilityKernelCPU::set_dynamics,
             py::arg("model") = std::shared_ptr<viability::IDynamics>{})

        .def("set_initial_viable", &ViabilityKernelCPU::set_initial_viable,
             py::arg("mask"),
             py::arg("bounds") = py::none(),
             py::arg("grid_points") = py::none())

        .def("in_K", &ViabilityKernelCPU::in_K)
        .def("compute_successors", &ViabilityKernelCPU::compute_successors,
             py::arg("parallel")=true, py::arg("n_cores")=py::none(), py::arg("chunk_size")=py::none())
        .def("iterate_viability", &ViabilityKernelCPU::iterate_viability,
             py::arg("successors"),
             py::arg("parallel")=true, py::arg("n_cores")=py::none(), py::arg("chunk_size")=py::none(), py::arg("verbose")=true)
        .def("run", &ViabilityKernelCPU::run,
             py::arg("parallel")=true, py::arg("n_cores")=py::none(), py::arg("chunk_size")=py::none(), py::arg("verbose")=true)

        .def_readonly("grid_points", &ViabilityKernelCPU::grid_points_)
        .def_readonly("T", &ViabilityKernelCPU::T_)
        .def_property_readonly("points", &ViabilityKernelCPU::points)
        .def_property_readonly("is_viable", &ViabilityKernelCPU::is_viable)
        .def_property_readonly("K_bounds", &ViabilityKernelCPU::K_bounds)
        .def_property_readonly("U_values", &ViabilityKernelCPU::U_values)
        .def_property_readonly("successors", &ViabilityKernelCPU::successors);
        
    register_capture_basin(m);
    register_r_option_sets(m);
}
