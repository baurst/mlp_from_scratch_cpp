#include "layer.h"
#include "neuron.h"

#include <iostream>

Layer::Layer(size_t number_of_inputs, size_t number_of_neurons)
    : num_neurons(number_of_neurons) {
  std::cout << "Creating Layer with " << num_neurons << " neurons" << std::endl;

  for (size_t i = 0; i < number_of_neurons; ++i) {
    neurons.push_back(Neuron(number_of_inputs));
  }
}

std::vector<std::vector<float>>
Layer::call(const std::vector<std::vector<float>> input) const {
  std::vector<std::vector<float>> output;
  const auto batch_size = input.size();
  output.resize(batch_size);
  // output[batch_idx].resize(num_neurons);
  for (size_t neuron_idx = 0; neuron_idx < num_neurons; ++neuron_idx) {
    const auto tmp = neurons[neuron_idx].call(input);
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      output[batch_idx].push_back(tmp[batch_idx]);
    }
  }
  return output;
}