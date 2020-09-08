#include "mlp.h"
#include "layer.h"
#include "utils.h"
#include <memory>

#include <iostream>
#include <vector>

MLP::MLP(const std::vector<size_t> layer_sizes, const size_t number_of_inputs,
         const size_t number_of_targets, const Initializer init)
    : number_of_inputs(number_of_inputs), number_of_targets(number_of_targets) {
  size_t input_size = number_of_inputs;

  for (const size_t layer_size : layer_sizes) {
    layers.push_back(std::make_unique<DenseLayer>(input_size, layer_size));
    input_size = layer_size;
    layers.push_back(std::make_unique<ActivationLayer>());
  }
  layers.push_back(std::make_unique<DenseLayer>(input_size, number_of_targets));
}

Mat2D<float> MLP::call(const Mat2D<float> &input) const {
  Mat2D<float> output = input;
  for (const auto &layer : layers) {
    output = layer->call(output);
  }
  return output;
}