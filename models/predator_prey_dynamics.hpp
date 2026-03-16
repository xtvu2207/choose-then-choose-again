// predator-prey dynamics model
// author: Xuan Tung VU

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "helpers/model_interface.hpp"

class PredPreyModel : public viability::IDynamics {
public:
    using Vec = viability::Vec;

    PredPreyModel(
        double b1 = 0.65,
        double k1 = 0.06,
        double k2 = 0.08,
        double d2 = 0.32,
        double dt = 0.01,
        bool adaptive_dt = false,
        double tol = 1e-5
    )
        : b1_(b1)
        , k1_(k1)
        , k2_(k2)
        , d2_(d2)
        , dt_(dt)
        , adaptive_dt_(adaptive_dt)
        , tol_(tol) {}

    std::size_t state_dim() const override { return state_dim_; }
    std::size_t control_dim() const override { return control_dim_; }

    // single-step state update
    Vec step(const Vec& x, const Vec& u) const override {
        validate_dims(x, u);
        Vec out(2);
        step_inplace(x, u, out);
        return out;
    }

    // in-place rk integration (adaptive optional)
    void step_inplace(const Vec& x, const Vec& u, Vec& out) const override {
        validate_dims(x, u);
        if (out.size() != 2) {
            out.resize(2);
        }

        const double X = std::max(0.0, x[0]);
        const double Y = std::max(0.0, x[1]);
        const double u1 = u[0];
        const double u2 = u[1];

        auto field = [&](double x_val, double y_val) {
            const double dx = b1_ * x_val - k1_ * x_val * y_val + u1;
            const double dy = k2_ * x_val * y_val - d2_ * y_val + u2;
            return std::pair<double, double>(dx, dy);
        };

        double dt_local = dt_;
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto rk1 = field(X, Y);

            const auto rk2 = field(
                X + dt_local * (1.0 / 5.0) * rk1.first,
                Y + dt_local * (1.0 / 5.0) * rk1.second
            );

            const auto rk3 = field(
                X + dt_local * (3.0 / 40.0 * rk1.first + 9.0 / 40.0 * rk2.first),
                Y + dt_local * (3.0 / 40.0 * rk1.second + 9.0 / 40.0 * rk2.second)
            );

            const auto rk4 = field(
                X + dt_local * (44.0 / 45.0 * rk1.first
                                - 56.0 / 15.0 * rk2.first
                                + 32.0 / 9.0 * rk3.first),
                Y + dt_local * (44.0 / 45.0 * rk1.second
                                - 56.0 / 15.0 * rk2.second
                                + 32.0 / 9.0 * rk3.second)
            );

            const auto rk5 = field(
                X + dt_local * (19372.0 / 6561.0 * rk1.first
                                - 25360.0 / 2187.0 * rk2.first
                                + 64448.0 / 6561.0 * rk3.first
                                - 212.0 / 729.0 * rk4.first),
                Y + dt_local * (19372.0 / 6561.0 * rk1.second
                                - 25360.0 / 2187.0 * rk2.second
                                + 64448.0 / 6561.0 * rk3.second
                                - 212.0 / 729.0 * rk4.second)
            );

            const auto rk6 = field(
                X + dt_local * (9017.0 / 3168.0 * rk1.first
                                - 355.0 / 33.0 * rk2.first
                                + 46732.0 / 5247.0 * rk3.first
                                + 49.0 / 176.0 * rk4.first
                                - 5103.0 / 18656.0 * rk5.first),
                Y + dt_local * (9017.0 / 3168.0 * rk1.second
                                - 355.0 / 33.0 * rk2.second
                                + 46732.0 / 5247.0 * rk3.second
                                + 49.0 / 176.0 * rk4.second
                                - 5103.0 / 18656.0 * rk5.second)
            );

            const double X5 = X + dt_local * (35.0 / 384.0 * rk1.first
                                              + 500.0 / 1113.0 * rk3.first
                                              + 125.0 / 192.0 * rk4.first
                                              - 2187.0 / 6784.0 * rk5.first
                                              + 11.0 / 84.0 * rk6.first);

            const double Y5 = Y + dt_local * (35.0 / 384.0 * rk1.second
                                              + 500.0 / 1113.0 * rk3.second
                                              + 125.0 / 192.0 * rk4.second
                                              - 2187.0 / 6784.0 * rk5.second
                                              + 11.0 / 84.0 * rk6.second);

            const double X4 = X + dt_local * (5179.0 / 57600.0 * rk1.first
                                              + 7571.0 / 16695.0 * rk3.first
                                              + 393.0 / 640.0 * rk4.first
                                              - 92097.0 / 339200.0 * rk5.first
                                              + 187.0 / 2100.0 * rk6.first
                                              + 1.0 / 40.0 * rk6.first);

            const double Y4 = Y + dt_local * (5179.0 / 57600.0 * rk1.second
                                              + 7571.0 / 16695.0 * rk3.second
                                              + 393.0 / 640.0 * rk4.second
                                              - 92097.0 / 339200.0 * rk5.second
                                              + 187.0 / 2100.0 * rk6.second
                                              + 1.0 / 40.0 * rk6.second);

            const double err = std::max(std::abs(X5 - X4), std::abs(Y5 - Y4));
            if (!adaptive_dt_) {
                out[0] = X5;
                out[1] = Y5;
                return;
            }

            if (err < tol_) {
                const double factor = std::min(2.0, 0.9 * std::pow(tol_ / (err + 1e-12), 0.2));
                dt_ = dt_local * factor;
                out[0] = X5;
                out[1] = Y5;
                return;
            }

            dt_local *= std::max(0.1, 0.8 * std::pow(tol_ / (err + 1e-12), 0.2));
        }

        throw std::runtime_error("PredPreyModel: adaptive integration did not converge");
    }

    // update model parameters at runtime
    void set_params(double b1, double k1, double k2, double d2, double dt) {
        b1_ = b1;
        k1_ = k1;
        k2_ = k2;
        d2_ = d2;
        dt_ = dt;
    }

private:
    void validate_dims(const Vec& x, const Vec& u) const {
        if (x.size() != state_dim_) {
            throw std::runtime_error("PredPreyModel: x has invalid dimension");
        }
        if (u.size() != control_dim_) {
            throw std::runtime_error("PredPreyModel: u has invalid dimension");
        }
    }

    double b1_;
    double k1_;
    double k2_;
    double d2_;
    mutable double dt_;
    bool adaptive_dt_;
    double tol_;
    std::size_t state_dim_{2};
    std::size_t control_dim_{2};
};
