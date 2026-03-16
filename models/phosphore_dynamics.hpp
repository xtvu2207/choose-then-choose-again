// lake phosphorus dynamics model
// author: Xuan Tung VU

#pragma once

#include <stdexcept>
#include <cmath>
#include <algorithm>

#include "helpers/model_interface.hpp"

class PhosphoreDynamics : public viability::IDynamics {
public:
    using Vec = viability::Vec;

    PhosphoreDynamics()
        : b_(2.2676), r_(101.96), m_(26.9), q_(2.222), dt_(0.05) {}

    PhosphoreDynamics(double b, double r, double m, double q, double dt)
        : b_(b), r_(r), m_(m), q_(q), dt_(dt) {}

    std::size_t state_dim() const override { return state_dim_; }
    std::size_t control_dim() const override { return control_dim_; }

    // single-step state update
    Vec step(const Vec& x, const Vec& u) const override {
        if (x.size() != state_dim_) {
            throw std::runtime_error("PhosphoreDynamics: x has invalid dimension");
        }
        if (u.size() != control_dim_) {
            throw std::runtime_error("PhosphoreDynamics: u has invalid dimension");
        }
        Vec xnext(2);
        dynamics_inplace(x, u, xnext);
        return xnext;
    }

    void step_inplace(const Vec& x, const Vec& u, Vec& xnext) const override {
        dynamics_inplace(x, u, xnext);
    }

    // update model parameters at runtime
    void set_params(double b, double r, double m, double q, double dt) {
        b_ = b;
        r_ = r;
        m_ = m;
        q_ = q;
        dt_ = dt;
    }

private:
    void dynamics_inplace(const Vec& x, const Vec& u, Vec& out) const {
        if (out.size() != 2) out.resize(2);
        double L = x[0];
        double P = x[1];

        double L_next = L + dt_ * u[0];

        double mq = std::pow(m_, q_);
        double Pq = std::pow(std::max(0.0, P), q_);

        double P_next = P + dt_ * (-b_ * P + L + r_ * Pq / (mq + Pq));
        out[0] = L_next;
        out[1] = P_next;
    }

    double b_, r_, m_, q_, dt_;
    std::size_t state_dim_{2};
    std::size_t control_dim_{2};
};
