#include <boost/lexical_cast.hpp>
#include <cmath>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>

#include "../src/audi.hpp"
#include "../src/neural_net.hpp"

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpedantic"
    #pragma GCC diagnostic ignored "-Wshadow"
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wdeprecated"
#endif

#include "pybind11/include/pybind11/operators.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/functional.h"

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

PYBIND11_DECLARE_HOLDER_TYPE(DenseLayer, std::shared_ptr<DenseLayer>)

using namespace audi;
namespace py = pybind11;

PYBIND11_PLUGIN(_core) {
    py::module m("_core", "pyaudi's core module");

    py::class_<DenseLayer, std::shared_ptr<DenseLayer>>(m,"DenseLayer")
        .def(py::init<unsigned int, unsigned int, unsigned int, const std::function<double()>&, const std::function<double()>&, const std::function<gdual(const gdual&)>&>())
        //.def(py::init<unsigned int, unsigned int, unsigned int, const std::function<double()>&, const std::function<double()>&>())
        .def_property_readonly("num_outputs",&DenseLayer::get_outputs)
        .def("__call__",&DenseLayer::operator())
        .def("set_parameters",&DenseLayer::set_parameters)
        .def("get_parameters",&DenseLayer::get_parameters)
    ;

    py::class_<NeuralNetwork>(m,"NeuralNetwork")
        .def(py::init<int>())
        .def_property_readonly("num_inputs",&NeuralNetwork::get_inputs)
        .def_property_readonly("num_outputs",&NeuralNetwork::get_outputs)
        .def_property_readonly("num_layers",&NeuralNetwork::get_layer_count)
        .def("__call__",&NeuralNetwork::operator())
        //.def("add_layer",&NeuralNetwork::add_layer)
        .def("add_layer",[](NeuralNetwork& instance, std::shared_ptr<DenseLayer> layer) { instance.add_layer(layer); })
        .def("get_layers",&NeuralNetwork::get_layers)
    ;

    py::class_<gdual>(m,"gdual")
        .def(py::init<>())
        .def(py::init<const gdual &>())
        .def(py::init<const std::string &, unsigned int>())
        .def(py::init<double, unsigned int>())
        .def(py::init<double>())
        .def(py::init<double, const std::string &, unsigned int>())
        .def("__repr__",[](const gdual &g) -> std::string {
            std::ostringstream oss;
            oss << g;
            return oss.str();
        })
        .def("_repr_latex_",[](const gdual &g) -> std::string {
            std::ostringstream oss;
            g._poly().print_tex(oss);
            auto retval = oss.str();
            retval += std::string("+\\mathcal{O}\\left(")
                + boost::lexical_cast<std::string>(g.get_order() + 1) +  "\\right) \\]";
            return std::string("\\[ ") + retval;
        })
        .def("__getstate__", [](const gdual &p) {
            // Returns a tuple that contains the string
            // representation of a gdual as obtained
            // from the boost serialization library
            std::stringstream ss;
            boost::archive::text_oarchive oa(ss);
            oa << p;
            return py::make_tuple(ss.str());
        })
        .def("__setstate__", [](gdual &p, py::tuple t) {
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");
            // Invoke the default constructor.
            new (&p) gdual;
            // Reconstruct the gdual
            std::stringstream ss(t[0].cast<std::string>());
            boost::archive::text_iarchive ia(ss);
            ia >> p;
        })
        .def_property_readonly("symbol_set",&gdual::get_symbol_set)
        .def_property_readonly("symbol_set_size",&gdual::get_symbol_set_size)
        .def_property_readonly("degree",&gdual::degree)
        .def_property_readonly("order",&gdual::get_order)
        .def_property_readonly("constant_cf",&gdual::constant_cf)
        .def("extend_symbol_set", &gdual::extend_symbol_set, "Extends the symbol set")
        .def("integrate", &gdual::integrate, "Integrate with respect to argument")
        .def("partial", &gdual::partial, "Partial derivative with respect to argument")
        .def("evaluate",[](const gdual &g, const std::map< std::string, double> &dict) {return g.evaluate(std::unordered_map< std::string, double>(dict.begin(), dict.end()));} , "Evaluates the Taylor polynomial")
        .def("find_cf", [](const gdual &g, const std::vector<int> &v) {
            return g.find_cf(v);
        },"Find the coefficient of the Taylor expansion")
        .def("get_derivative", [](const gdual &g, const std::vector<int> &v) {
            return g.get_derivative(v);
        },"Finds the derivative (i.e. the coefficient of the Taylor expansion discounted of a factorial factor")
        .def("get_derivative", [](const gdual &g, const std::unordered_map<std::string, unsigned int> &dict) {
            return g.get_derivative(dict);
        },"Finds the derivative (i.e. the coefficient of the Taylor expansion discounted of a factorial factor")
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self + double())
        .def(py::self - double())
        .def(py::self * double())
        .def(py::self / double())
        .def(-py::self)
        .def(+py::self)
        .def(double() + py::self)
        .def(double() - py::self)
        .def(double() * py::self)
        .def(double() / py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__pow__",[](const gdual &g, double x) {return pow(g,x);} ,"Exponentiation (gdual, double).")
        .def("__pow__",[](const gdual &base, const gdual &g) {return pow(base,g);} ,"Exponentiation (gdual, gdual).")
        .def("__rpow__",[](const gdual &g, double x) {return pow(x,g);} ,"Exponentiation (double, gdual).")
    ;

    m.def("exp",[](const gdual &d) {return exp(d);},"Exponential (gdual).");
    m.def("exp",[](double x) {return std::exp(x);},"Exponential (double).");

    m.def("log",[](const gdual &d) {return log(d);},"Natural logarithm (gdual).");
    m.def("log",[](double x) {return std::log(x);},"Natural logarithm (double).");

    m.def("sqrt",[](const gdual &d) {return sqrt(d);},"Square root (gdual).");
    m.def("sqrt",[](double x) {return std::sqrt(x);},"Square root (double).");

    m.def("cbrt",[](const gdual &d) {return cbrt(d);},"Cubic root (gdual).");
    m.def("cbrt",[](double x) {return std::cbrt(x);},"Cubic root (double).");

    m.def("sin",[](const gdual &d) {return sin(d);},"Sine (gdual).");
    m.def("sin",[](double x) {return std::sin(x);},"Sine (double).");
    // m.def("sin",py::vectorize([](double x) {return std::sin(x);}),"Sine (vectorized double).");

    m.def("asin",[](const gdual &d) {return asin(d);},"Arc sine (gdual).");
    m.def("asin",[](double x) {return std::asin(x);},"Arc sine (double).");

    m.def("cos",[](const gdual &d) {return cos(d);},"Cosine (gdual).");
    m.def("cos",[](double x) {return std::cos(x);},"Cosine (double).");

    m.def("acos",[](const gdual &d) {return acos(d);},"Arc cosine (gdual).");
    m.def("acos",[](double x) {return std::acos(x);},"Arc cosine (double).");

    m.def("sin_and_cos",[](const gdual &d) {return sin_and_cos(d);}, "Sine and Cosine at once (gdual).");

    m.def("tan",[](const gdual &d) {return tan(d);},"Tangent (gdual).");
    m.def("tan",[](double x) {return std::tan(x);},"Tangent (double).");

    m.def("atan",[](const gdual &d) {return atan(d);},"Arc tangent (gdual).");
    m.def("atan",[](double x) {return std::atan(x);},"Arc tangent (double).");

    m.def("sinh",[](const gdual &d) {return sinh(d);},"Hyperbolic sine (gdual).");
    m.def("sinh",[](double x) {return std::sinh(x);},"Hyperbolic sine (double).");

    m.def("asinh",[](const gdual &d) {return asinh(d);},"Inverse hyperbolic sine (gdual).");
    m.def("asinh",[](double x) {return std::asinh(x);},"Inverse hyperbolic sine (double).");

    m.def("cosh",[](const gdual &d) {return cosh(d);},"Hyperbolic cosine (gdual).");
    m.def("cosh",[](double x) {return std::cosh(x);},"Hyperbolic cosine (double).");

    m.def("acosh",[](const gdual &d) {return acosh(d);},"Inverse hyperbolic cosine (gdual).");
    m.def("acosh",[](double x) {return std::acosh(x);},"Inverse hyperbolic cosine (double).");

    m.def("sinh_and_cosh",[](const gdual &d) {return sinh_and_cosh(d);} ,"Hyperbolic sine and hyperbolic cosine at once (gdual).");

    m.def("tanh",[](const gdual &d) {return tanh(d);},"Hyperbolic tangent (gdual).");
    m.def("tanh",[](double x) {return std::tanh(x);},"Hyperbolic tangent (double).");

    m.def("atanh",[](const gdual &d) {return atanh(d);},"Inverse hyperbolic arc tangent (gdual).");
    m.def("atanh",[](double x) {return std::atanh(x);},"Inverse hyperbolic arc tangent (double).");

    m.def("abs",[](const gdual &d) {return abs(d);},"Absolute value (gdual).");
    m.def("abs",[](double x) {return std::abs(x);},"Absolute value (double).");

    m.def("erf",[](const gdual &d) {return erf(d);},"Error function (gdual).");
    m.def("erf",[](double x) {return std::erf(x);},"Error function (double).");

    return m.ptr();
}
