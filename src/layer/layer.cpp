#include "layer.h"
#include "utils.h"

#include <iostream>

DenseLayer::DenseLayer(size_t number_of_inputs, size_t number_of_neurons)
    : num_neurons(number_of_neurons),
      weights(number_of_inputs, number_of_neurons),
      biases(number_of_neurons, 1) {
  std::cout << "Creating DenseLayer with " << num_neurons << " neurons"
            << std::endl;
  std::cout << "TODO: Initialize weights and biases!" << std::endl;
}

Mat2D<float> DenseLayer::call(const Mat2D<float> &input) const {
  return biases.add(input.dot_product(weights.transpose()));
}