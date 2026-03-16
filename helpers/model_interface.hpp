// dynamics interface shared by all models
// author: Xuan Tung VU

#pragma once

#include <cstddef>
#include <vector>

namespace viability {

using Vec = std::vector<double>;

struct IDynamics {
    // common interface for all dynamics models used by c++ kernels
    virtual ~IDynamics() = default;
    virtual std::size_t state_dim() const = 0;
    virtual std::size_t control_dim() const = 0;
    virtual Vec step(const Vec& x, const Vec& u) const = 0;
    virtual void step_inplace(const Vec& x, const Vec& u, Vec& xnext) const {
        xnext = step(x, u);
    }
};

}  
