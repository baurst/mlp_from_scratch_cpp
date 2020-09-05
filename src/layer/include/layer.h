#pragma once
#include "neuron.h"
#include <numeric>
#include <vector>

class Layer {
public:
  Layer(size_t number_of_inputs, size_t number_of_neurons);
  std::vector<float> call(const std::vector<float> input) const;

private:
  std::vector<Neuron> neurons;
  size_t num_neurons;
};