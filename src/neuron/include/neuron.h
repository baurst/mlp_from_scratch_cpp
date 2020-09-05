#pragma once

#include <numeric>
#include <vector>

class Neuron {
public:
  Neuron(size_t number_of_inputs);
  float call(const std::vector<float> input) const;

private:
  std::vector<float> weights;
  float bias;
};

template <typename T> T dot(const std::vector<T> x, const std::vector<T> y) {
  const T dp = std::inner_product(x.begin(), x.end(), y.begin(), 0);
  return dp;
};
