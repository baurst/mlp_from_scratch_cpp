#include "layer.h"
#include "neuron.h"

Layer::Layer(size_t number_of_inputs, size_t number_of_neurons)
    : num_neurons(number_of_neurons) {
  for (size_t i = 0; i < number_of_inputs; ++i) {
    neurons.push_back(Neuron(number_of_inputs));
  }
}

std::vector<float> Layer::call(const std::vector<float> input) const {
  std::vector<float> output;
  output.resize(num_neurons);
  for (const auto &neuron : neurons) {
    const float tmp = neuron.call(input);
    output.push_back(tmp);
  }

  return output;
}