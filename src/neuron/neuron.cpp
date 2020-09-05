#include "neuron.h"

Neuron::Neuron(size_t number_of_inputs) {
  weights.resize(number_of_inputs);
  bias = 0.0;
}

float Neuron::call(const std::vector<float> input) const {
  return bias + dot(weights, input);
}