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
                                  const Mat2D<float> &gradients_output,
                                  const Mat2D<float> &learning_rate) {
  const auto grad_input =
      gradients_output.dot_product(this->weights.transpose());

  const auto grad_weights = input.transpose().dot_product(gradients_output);
  const auto grad_biases = gradients_output.reduce_mean_axis(0);
  this->weights =
      this->weights.minus(learning_rate.hadamard_product(grad_weights));
  this->biases =
      this->biases.minus(learning_rate.hadamard_product(grad_biases));

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
                                       const Mat2D<float> &gradient_output,
                                       const Mat2D<float> &learning_rate) {
  Mat2D<float> result = input;
  std::transform(result.matrix_data.begin(), result.matrix_data.end(),
                 result.matrix_data.begin(),
                 [](float x) { return (x > 0.0) ? 1.0 : 0.0; });
  return result.hadamard_product(gradient_output);
}

MSELoss::~MSELoss(){};

MSELoss::MSELoss(){};

Mat2D<float> MSELoss::loss(const Mat2D<float> &predictions,
                           const Mat2D<float> &labels) const {
  const auto diff = predictions.minus(labels);
  const auto loss = diff.hadamard_product(diff);
  return loss;
}

Mat2D<float> MSELoss::loss_grad(const Mat2D<float> &predictions,
                                const Mat2D<float> &labels) const {
  // partial derivative of Loss for the predictions
  Mat2D<float> divisor(1, 1,
                       {1.0f / static_cast<float>(predictions.get_num_rows())});
  return predictions.minus(labels).hadamard_product(divisor);
}