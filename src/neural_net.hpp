#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "gdual.hpp"
#include "functions.hpp"

#include <vector>
#include <memory>
#include <sstream>
#include <functional>

namespace audi
{

class Layer
{
public:
	virtual const std::vector<gdual>& operator()(const std::vector<gdual>& values)=0;
	virtual const std::vector<gdual>& get_parameters() const=0;
	virtual void set_parameters(std::vector<gdual>& values)=0;
	virtual unsigned int get_outputs() const=0;
	virtual ~Layer() {}
};

class NeuralNetwork
{
	unsigned int m_num_inputs;
	unsigned int m_num_outputs;
	std::vector<std::shared_ptr<Layer>> m_layers;

public:
	NeuralNetwork(unsigned int num_inputs) : m_num_inputs(num_inputs), m_num_outputs(num_inputs)
	{
	}

	unsigned int get_inputs() const
	{
		return m_num_inputs;
	}

	unsigned int get_outputs() const
	{
		return m_num_outputs;
	}

	unsigned int get_layer_count() const
	{
		return m_layers.size();
	}

	const std::vector<gdual>& operator()(const std::vector<gdual>& values)
	{
		const std::vector<gdual>* output = &values;
		for(auto& layer : m_layers)
		{
			output = &(*layer)(*output);
		}

		return *output;
	}

	const std::vector<std::shared_ptr<Layer>>& get_layers()
	{
		return m_layers;
	}

	void add_layer(std::shared_ptr<Layer> layer)
	{
		m_layers.push_back(layer);
		m_num_outputs = layer->get_outputs();
	}
};

class DenseLayer : public Layer
{
	const std::function<gdual(const gdual&)> m_nonlinearity;
	std::vector<gdual> m_parameters;
	std::vector<gdual> m_outputs;
	unsigned int m_layer_id;
	unsigned int m_num_inputs;
	unsigned int m_num_outputs;

public:
	DenseLayer(unsigned int layer_id, unsigned int num_inputs, unsigned int num_outputs, const std::function<double()>& W, const std::function<double()>& b, const std::function<gdual(const gdual&)>& nonlinearity) : m_nonlinearity(nonlinearity), m_layer_id(layer_id), m_num_inputs(num_inputs), m_num_outputs(num_outputs)
	{
		for(unsigned int i = 0; i < num_inputs; i++)
		{
			for(unsigned int unit = 0; unit < num_outputs; unit++)
			{
				std::ostringstream name;
				name << "w_{" << layer_id << "," << unit << "," << i << "}";
				m_parameters.push_back(gdual(W(), name.str(), 1));
			}
		}

		for(unsigned int unit = 0; unit < num_outputs; unit++)
		{
			std::ostringstream name;
			name << "b_{" << layer_id << "," << unit << "}";
			m_parameters.push_back(gdual(b(), name.str(), 1));
			m_outputs.push_back(gdual(0));
		}
	}

	DenseLayer(unsigned int layer_id, unsigned int num_inputs, unsigned int num_outputs, const std::function<double()>& W, const std::function<double()>& b) : DenseLayer(layer_id, num_inputs, num_outputs, W, b, [](const gdual& g) -> gdual {return tanh(g);})
	{
	}

	virtual const std::vector<gdual>& get_parameters() const
	{
		return m_parameters;
	}

	virtual void set_parameters(std::vector<gdual>& values)
	{
		m_parameters = values;
	}

	virtual unsigned int get_outputs() const
	{
		return m_num_outputs;
	}

	virtual const std::vector<gdual>& operator()(const std::vector<gdual>& values)
	{
		for(unsigned int unit = 0; unit < m_num_outputs; unit++)
		{
			m_outputs[unit] = m_parameters[m_num_inputs * m_num_outputs + unit];
		}

		for(unsigned int i = 0; i < m_num_inputs; i++)
		{
			for(unsigned int unit = 0; unit < m_num_outputs; unit++)
			{
				m_outputs[unit] += values[i] * m_parameters[i * m_num_outputs + unit];
			}
		}

		for(unsigned int unit = 0; unit < m_num_outputs; unit++)
		{
			m_outputs[unit] = m_nonlinearity(m_outputs[unit]);
		}

		return m_outputs;
	}
};

}

#endif
