#include "layer.h"
#include "utils.h"
#include <algorithm>
#include <iostream>

Layer::Layer(){};
Layer::~Layer(){};

DenseLayer::DenseLayer(size_t number_of_inputs, size_t number_of_neurons,
                       Initializer init)
    : weights(number_of_inputs, number_of_neurons, init),
      biases(1, number_of_neurons, init), num_neurons(number_of_neurons) {}

DenseLayer::~DenseLayer(){};

Mat2D<float> DenseLayer::call(const Mat2D<float> &input) const {
  return biases.add(input.dot_product(weights.transpose()));
}

ActivationLayer::~ActivationLayer(){};

ActivationLayer::ActivationLayer(){};

Mat2D<float> ActivationLayer::call(const Mat2D<float> &input) const {
  Mat2D<float> result = input;
  std::transform(result.matrix_data.begin(), result.matrix_data.end(),
                 result.matrix_data.begin(),
                 [](float x) { return std::max(static_cast<float>(0.0), x); });
  return input;
}