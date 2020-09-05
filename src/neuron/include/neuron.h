#pragma once

#include <vector>

class Neuron {
public:
  Neuron(size_t number_of_inputs);
  std::vector<float> call(const std::vector<std::vector<float>> input) const;

private:
  std::vector<float> weights;
  float bias;
};
