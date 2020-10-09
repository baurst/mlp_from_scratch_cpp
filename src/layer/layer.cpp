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
                                  const Mat2D<float> &gradients_output) {
  const auto lr_mat = Mat2D<float>(1, 1, {0.001});
  const auto grad_input =
      gradients_output.dot_product(this->weights.transpose());

  const auto grad_weights = input.transpose().dot_product(gradients_output);
  const auto grad_biases = gradients_output.reduce_mean_axis(0);
  this->weights = this->weights.minus(lr_mat.hadamard_product(grad_weights));
  this->biases = this->biases.minus(lr_mat.hadamard_product(grad_biases));

  return grad_input;
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
Mat2D<float> ActivationLayer::backward(const Mat2D<float> &input,
                                       const Mat2D<float> &gradient_output) {
  Mat2D<float> result = input;
  std::transform(result.matrix_data.begin(), result.matrix_data.end(),
                 result.matrix_data.begin(),
                 [](float x) { return (x > 0.0) ? 1.0 : 0.0; });
  return result.hadamard_product(gradient_output);
}

L2Loss::~L2Loss(){};

L2Loss::L2Loss(){};

Mat2D<float> L2Loss::loss(const Mat2D<float> &predictions,
                          const Mat2D<float> &labels) const {
  const auto diff = labels.minus(predictions);
  const auto loss = diff.hadamard_product(diff);
  return loss;
}

Mat2D<float> L2Loss::loss_grad(const Mat2D<float> &predictions,
                               const Mat2D<float> &labels) const {
  return predictions.minus(labels);
}