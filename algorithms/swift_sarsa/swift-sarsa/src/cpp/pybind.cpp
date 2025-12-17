#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SwiftSarsa.h"

namespace py = pybind11;

PYBIND11_MODULE(swift_sarsa, m)
{
     m.doc() = "Swift-Sarsa: A fast and robust linear control based on Sarsa(lambda)."; // Module docstring
     py::class_<SwiftSarsa>(m, "SwiftSarsa")
          .def(py::init<int, int, float, float, float, float, float, float, float>(),
               "Initialize the SwiftSarsa algorithm",
               py::arg("num_of_features"),
               py::arg("num_of_actions"),
               py::arg("lambda"),
               py::arg("alpha"),
               py::arg("meta_step_size"),
               py::arg("eta"),
               py::arg("decay"),
               py::arg("epsilon"),
               py::arg("eta_min"))
          .def("learn", &SwiftSarsa::learn,
               "Perform one step of learning. Takes as input features, reward, gamma, and action and returns the value of the chosen action.",
               py::arg("feature_indices"),
               py::arg("reward"),
               py::arg("gamma"),
               py::arg("action"))
          .def("get_action_values", &SwiftSarsa::get_action_values,
               "Get action values for given feature indices. Returns a list of values of size num_of_actions.",
               py::arg("feature_indices"))
          .def("get_weights", &SwiftSarsa::get_weights, "Return current weight vector")
          .def("get_beta", &SwiftSarsa::get_beta, "Return current beta (log step sizes)")
          .def("get_last_alpha", &SwiftSarsa::get_last_alpha, "Return last adaptive step sizes")
          .def("get_last_delta", &SwiftSarsa::get_last_delta, "Return TD error from last update")
          .def("set_weights", &SwiftSarsa::set_weights,
               "Set weights, beta, and last_alpha from saved checkpoint. Resets eligibility traces.",
               py::arg("weights"),
               py::arg("beta"),
               py::arg("last_alpha"));

     py::class_<SwiftSarsaBinaryFeatures>(m, "SwiftSarsaBinaryFeatures")
          .def(py::init<int, int, float, float, float, float, float, float, float>(),
               "Initialize the SwiftSarsa algorithm",
               py::arg("num_of_features"),
               py::arg("num_of_actions"),
               py::arg("lambda"),
               py::arg("alpha"),
               py::arg("meta_step_size"),
               py::arg("eta"),
               py::arg("decay"),
               py::arg("epsilon"),
               py::arg("eta_min"))
          .def("learn", &SwiftSarsaBinaryFeatures::learn,
               "Perform one step of learning. Takes as input features, reward, gamma, and action and returns the value of the chosen action.",
               py::arg("feature_indices"),
               py::arg("reward"),
               py::arg("gamma"),
               py::arg("action"))
          .def("get_action_values", &SwiftSarsaBinaryFeatures::get_action_values,
               "Get action values for given feature indices. Returns a list of values of size num_of_actions.",
               py::arg("feature_indices"))
          .def("get_weights", &SwiftSarsaBinaryFeatures::get_weights, "Return current weight vector")
          .def("get_beta", &SwiftSarsaBinaryFeatures::get_beta, "Return current beta (log step sizes)")
          .def("get_last_alpha", &SwiftSarsaBinaryFeatures::get_last_alpha, "Return last adaptive step sizes")
          .def("get_last_delta", &SwiftSarsaBinaryFeatures::get_last_delta, "Return TD error from last update");
}
