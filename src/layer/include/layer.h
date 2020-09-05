#pragma once
#include "utils.h"
#include <numeric>
#include <vector>

class DenseLayer {
public:
  DenseLayer(size_t number_of_inputs, size_t number_of_neurons);
  Mat2D<float> call(const Mat2D<float> &input) const;

private:
  Mat2D<float> weights;
  Mat2D<float> biases;
  size_t num_neurons;
};