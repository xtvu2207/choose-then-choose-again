// savanna trees-grass dynamics model
// author: Xuan Tung VU

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "helpers/model_interface.hpp"

class SavannaDynamics : public viability::IDynamics {
public:
    using Vec = viability::Vec;

    SavannaDynamics(
        double W = 920.0,
        double f = 0.98,
        double dt = 0.1,
        bool adaptive_dt = false,
        double tol = 1e-5
    )
        : W_(W)
        , f_(f)
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

        const double T = std::max(0.0, x[0]);
        const double G = std::max(0.0, x[1]);

        const double dT_ctrl = u[0];
        const double dG_ctrl = u[1];

        auto rhs = [&](double Gv, double Tv) {
            const double dG =
                g_G() * Gv * (1.0 - Gv / K_G())
                - dG_ctrl * Gv
                - eta_TG() * Tv * Gv
                - lambda_fG() * f_ * Gv;

            const double dT =
                g_T() * Tv * (1.0 - Tv / K_T())
                - dT_ctrl * Tv
                - f_ * varphi(Tv) * omega(Gv) * Tv;

            return std::pair<double, double>(dG, dT);
        };

        double dt_local = dt_;
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto k1 = rhs(G, T);

            const auto k2 = rhs(
                G + dt_local * (1.0 / 5.0) * k1.first,
                T + dt_local * (1.0 / 5.0) * k1.second
            );

            const auto k3 = rhs(
                G + dt_local * (3.0 / 40.0 * k1.first + 9.0 / 40.0 * k2.first),
                T + dt_local * (3.0 / 40.0 * k1.second + 9.0 / 40.0 * k2.second)
            );

            const auto k4 = rhs(
                G + dt_local * (44.0 / 45.0 * k1.first
                                - 56.0 / 15.0 * k2.first
                                + 32.0 / 9.0 * k3.first),
                T + dt_local * (44.0 / 45.0 * k1.second
                                - 56.0 / 15.0 * k2.second
                                + 32.0 / 9.0 * k3.second)
            );

            const auto k5 = rhs(
                G + dt_local * (19372.0 / 6561.0 * k1.first
                                - 25360.0 / 2187.0 * k2.first
                                + 64448.0 / 6561.0 * k3.first
                                - 212.0 / 729.0 * k4.first),
                T + dt_local * (19372.0 / 6561.0 * k1.second
                                - 25360.0 / 2187.0 * k2.second
                                + 64448.0 / 6561.0 * k3.second
                                - 212.0 / 729.0 * k4.second)
            );

            const auto k6 = rhs(
                G + dt_local * (9017.0 / 3168.0 * k1.first
                                - 355.0 / 33.0 * k2.first
                                + 46732.0 / 5247.0 * k3.first
                                + 49.0 / 176.0 * k4.first
                                - 5103.0 / 18656.0 * k5.first),
                T + dt_local * (9017.0 / 3168.0 * k1.second
                                - 355.0 / 33.0 * k2.second
                                + 46732.0 / 5247.0 * k3.second
                                + 49.0 / 176.0 * k4.second
                                - 5103.0 / 18656.0 * k5.second)
            );

            const double G5 = G + dt_local * (35.0 / 384.0 * k1.first
                                              + 500.0 / 1113.0 * k3.first
                                              + 125.0 / 192.0 * k4.first
                                              - 2187.0 / 6784.0 * k5.first
                                              + 11.0 / 84.0 * k6.first);

            const double T5 = T + dt_local * (35.0 / 384.0 * k1.second
                                              + 500.0 / 1113.0 * k3.second
                                              + 125.0 / 192.0 * k4.second
                                              - 2187.0 / 6784.0 * k5.second
                                              + 11.0 / 84.0 * k6.second);

            const auto k7 = rhs(G5, T5);

            const double G4 = G + dt_local * (5179.0 / 57600.0 * k1.first
                                              + 7571.0 / 16695.0 * k3.first
                                              + 393.0 / 640.0 * k4.first
                                              - 92097.0 / 339200.0 * k5.first
                                              + 187.0 / 2100.0 * k6.first
                                              + 1.0 / 40.0 * k7.first);

            const double T4 = T + dt_local * (5179.0 / 57600.0 * k1.second
                                              + 7571.0 / 16695.0 * k3.second
                                              + 393.0 / 640.0 * k4.second
                                              - 92097.0 / 339200.0 * k5.second
                                              + 187.0 / 2100.0 * k6.second
                                              + 1.0 / 40.0 * k7.second);

            const double err = std::max(std::abs(G5 - G4), std::abs(T5 - T4));

            if (!adaptive_dt_) {
                out[0] = T5;
                out[1] = G5;
                return;
            }

            if (err < tol_) {
                const double factor = std::min(2.0, 0.9 * std::pow(tol_ / (err + 1e-12), 0.2));
                dt_ = dt_local * factor;
                out[0] = T5;
                out[1] = G5;
                return;
            }

            dt_local *= std::max(0.1, 0.8 * std::pow(tol_ / (err + 1e-12), 0.2));
        }

        throw std::runtime_error("SavannaDynamics: adaptive integration did not converge");
    }

    // update model parameters at runtime
    void set_params(double W, double f, double dt) {
        W_ = W;
        f_ = f;
        dt_ = dt;
    }

private:
    void validate_dims(const Vec& x, const Vec& u) const {
        if (x.size() != state_dim_) {
            throw std::runtime_error("SavannaDynamics: x has invalid dimension");
        }
        if (u.size() != control_dim_) {
            throw std::runtime_error("SavannaDynamics: u has invalid dimension");
        }
    }

    double g_G() const { return gamma_G_ * W_ / (b_G_ + W_); }
    double g_T() const { return gamma_T_ * W_ / (b_T_ + W_); }

    double K_G() const { return c_G_ / (1.0 + d_G_ * std::exp(-a_G_ * W_)); }
    double K_T() const { return c_T_ / (1.0 + d_T_ * std::exp(-a_T_ * W_)); }

    double eta_TG() const { return a_eta_ * std::tanh((W_ - b_eta_) / c_eta_) + d_eta_; }

    double omega(double G) const {
        if (G <= 0.0) {
            return 0.0;
        }
        return (G * G) / (G * G + alpha_ * alpha_);
    }

    double varphi(double T) const {
        return lambda_fT_min_ + (lambda_fT_max_ - lambda_fT_min_) * std::exp(-p_ * T);
    }

    double lambda_fG() const {
        const double num = std::pow(W_, z_);
        const double den = num + std::pow(S_, z_);
        return lambda_fG_min_ + (lambda_fG_max_ - lambda_fG_min_) * (num / den);
    }

    double W_;
    double f_;
    mutable double dt_;
    bool adaptive_dt_;
    double tol_;

    double gamma_G_ = 2.7;
    double gamma_T_ = 1.5;
    double b_G_ = 500.0;
    double b_T_ = 1100.0;
    double c_G_ = 20.0;
    double c_T_ = 430.0;
    double d_G_ = 14.73;
    double d_T_ = 107.0;
    double a_G_ = 0.0029;
    double a_T_ = 0.004;

    double a_eta_ = 0.01;
    double b_eta_ = 600.0;
    double c_eta_ = 120.0;
    double d_eta_ = 0.0045;

    double alpha_ = 2.45;
    double lambda_fT_min_ = 0.05;
    double lambda_fT_max_ = 0.65;
    double p_ = 0.01;
    double lambda_fG_min_ = 0.005;
    double lambda_fG_max_ = 0.4;
    double S_ = 900.0;
    double z_ = 8.0;

    std::size_t state_dim_{2};
    std::size_t control_dim_{2};
};
