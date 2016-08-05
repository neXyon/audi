#include "../src/neural_net.hpp"
#include "../src/functions.hpp"
#include "helpers.hpp"

#define BOOST_TEST_MODULE audi_neural_net_test
#include <boost/test/unit_test.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <stdexcept>
#include <vector>

using namespace audi;

BOOST_AUTO_TEST_CASE(construction)
{
	NeuralNetwork network(5);
	DenseLayer layer(0, 5, 1, []() -> double { return 0; }, []() -> double { return 1; }, nullptr);
}

BOOST_AUTO_TEST_CASE(running)
{
	NeuralNetwork network(5);
	auto layer = std::make_shared<DenseLayer>(0, 5, 1, []() -> double { return 1; }, []() -> double { return 1; }, [](const gdual& x) -> gdual {return tanh(x);});
	network.add_layer(layer);

	std::vector<gdual> inputs;
	for(int i = 0; i < 5; i++)
		inputs.push_back(gdual(i));

	const std::vector<gdual>& output = network(inputs);

	BOOST_CHECK(EPSILON_COMPARE(output[0].constant_cf(), 0.9999999994421064, 1e-14) == true);
}


