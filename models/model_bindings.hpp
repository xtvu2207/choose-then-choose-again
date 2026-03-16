// pybind registration for dynamics models
// author: Xuan Tung VU

#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include "helpers/model_interface.hpp"
#include "phosphore_dynamics.hpp"
#include "predator_prey_dynamics.hpp"
#include "savanna_dynamics.hpp"

namespace py = pybind11;

inline void register_model_bindings(py::module_& m) {
    // expose all supported dynamics models to python
    py::class_<PhosphoreDynamics, viability::IDynamics, std::shared_ptr<PhosphoreDynamics>>(m, "PhosphoreDynamics")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double>(),
             py::arg("b"), py::arg("r"), py::arg("m"), py::arg("q"), py::arg("dt"))
        .def("set_params", &PhosphoreDynamics::set_params,
             py::arg("b"), py::arg("r"), py::arg("m"), py::arg("q"), py::arg("dt"))
        .def("step", &PhosphoreDynamics::step, py::arg("x"), py::arg("u"))
        .def("state_dim", &PhosphoreDynamics::state_dim)
        .def("control_dim", &PhosphoreDynamics::control_dim);

    py::class_<PredPreyModel, viability::IDynamics, std::shared_ptr<PredPreyModel>>(m, "PredPreyModel")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double, bool, double>(),
             py::arg("b1"),
             py::arg("k1"),
             py::arg("k2"),
             py::arg("d2"),
             py::arg("dt"),
             py::arg("adaptive_dt") = false,
             py::arg("tol") = 1e-5)
        .def("set_params", &PredPreyModel::set_params,
             py::arg("b1"), py::arg("k1"), py::arg("k2"), py::arg("d2"), py::arg("dt"))
        .def("step", &PredPreyModel::step, py::arg("x"), py::arg("u"))
        .def("state_dim", &PredPreyModel::state_dim)
        .def("control_dim", &PredPreyModel::control_dim);

    py::class_<SavannaDynamics, viability::IDynamics, std::shared_ptr<SavannaDynamics>>(m, "SavannaDynamics")
        .def(py::init<>())
        .def(py::init<double, double, double, bool, double>(),
             py::arg("W"),
             py::arg("f"),
             py::arg("dt"),
             py::arg("adaptive_dt") = false,
             py::arg("tol") = 1e-5)
        .def("set_params", &SavannaDynamics::set_params,
             py::arg("W"), py::arg("f"), py::arg("dt"))
        .def("step", &SavannaDynamics::step, py::arg("x"), py::arg("u"))
        .def("state_dim", &SavannaDynamics::state_dim)
        .def("control_dim", &SavannaDynamics::control_dim);
}
