#include "layer.h"
#include "utils.h"

#include <iostream>

DenseLayer::DenseLayer(size_t number_of_inputs, size_t number_of_neurons,
                       Initializer init)
    : weights(number_of_inputs, number_of_neurons, init),
      biases(1, number_of_neurons, init), num_neurons(number_of_neurons) {}

Mat2D<float> DenseLayer::call(const Mat2D<float> &input) const {
  return biases.add(input.dot_product(weights.transpose()));
}