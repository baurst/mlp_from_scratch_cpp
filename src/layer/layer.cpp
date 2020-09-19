#include "layer.h"
#include "utils.h"
#include <algorithm>
#include <iostream>

Layer::Layer(){};
Layer::~Layer(){};

DenseLayer::DenseLayer(size_t number_of_inputs, size_t number_of_neurons,
                       Initializer init)
    : weights(number_of_inputs, number_of_neurons, init),
      biases(1, number_of_neurons, init) {
  std::cout << "DenseLayer: Inputs " << number_of_inputs << " Neurons "
            << number_of_neurons << std::endl;
}

DenseLayer::~DenseLayer(){};

Mat2D<float> DenseLayer::forward(const Mat2D<float> &input) const {
  const auto dot_prod = input.dot_product(weights);
  const auto result = dot_prod.add(biases);
  return result;
}

Mat2D<float> DenseLayer::backward(const Mat2D<float> &input,
                                  const Mat2D<float> &gradients_output) const {
  std::cout << "DenseLayer Gradient: TODO! " << std::endl;
  return input;
}

ActivationLayer::~ActivationLayer(){};

ActivationLayer::ActivationLayer(){};

Mat2D<float> ActivationLayer::forward(const Mat2D<float> &input) const {
  Mat2D<float> result = input;
  std::transform(result.matrix_data.begin(), result.matrix_data.end(),
                 result.matrix_data.begin(),
                 [](float x) { return std::max(static_cast<float>(0.0), x); });
  return result;
}
Mat2D<float>
ActivationLayer::backward(const Mat2D<float> &input,
                          const Mat2D<float> &gradient_output) const {
  Mat2D<float> result = input;
  std::transform(result.matrix_data.begin(), result.matrix_data.end(),
                 result.matrix_data.begin(),
                 [](float x) { return (x > 0.0) ? 1.0 : 0.0; });
  return result.hadamard_product(gradient_output);
}