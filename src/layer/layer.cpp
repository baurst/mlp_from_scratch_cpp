#include "layer.h"
#include "utils.h"
#include <algorithm>
#include <iostream>
#include <math.h>

Layer::Layer() {}
Layer::~Layer() {}

DenseLayer::DenseLayer(size_t number_of_inputs, size_t number_of_neurons,
                       Initializer weight_init, Initializer bias_init)
    : weights(number_of_inputs, number_of_neurons, weight_init),
      biases(1, number_of_neurons, bias_init) {
  std::cout << "DenseLayer: #inputs: " << number_of_inputs
            << " #neurons: " << number_of_neurons << std::endl;
}

DenseLayer::~DenseLayer() {}

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
  const auto grad_biases = gradients_output.reduce_sum_axis(0);

  const auto weight_update = learning_rate.hadamard_product(grad_weights);
  const auto bias_update = learning_rate.hadamard_product(grad_biases);
  this->weights = this->weights.minus(weight_update);
  this->biases = this->biases.minus(bias_update);

  return grad_input;
}

void DenseLayer::print_trainable_variables() const {
  std::cout << "Weight: " << this->weights.get_num_rows() << "x"
            << this->weights.get_num_cols() << std::endl;
  std::cout << this->weights << std::endl;
  std::cout << "Bias: " << this->biases.get_num_rows() << "x"
            << this->biases.get_num_cols() << std::endl;
  std::cout << this->biases << std::endl;
}

LeakyRELUActivationLayer::~LeakyRELUActivationLayer() {}

LeakyRELUActivationLayer::LeakyRELUActivationLayer(const float alpha)
    : alpha(alpha) {
  std::cout << "LeakyRELUActivationLayer: alpha: " << alpha << std::endl;
}

Mat2D<float>
LeakyRELUActivationLayer::forward(const Mat2D<float> &input) const {
  auto tmp_in = input;
  Mat2D<float> result = tmp_in.elementwise_operation(
      [this](float x) { return std::max(this->alpha * x, x); });

  return result;
}
Mat2D<float>
LeakyRELUActivationLayer::backward(const Mat2D<float> &input,
                                   const Mat2D<float> &gradient_output,
                                   const Mat2D<float> &learning_rate) {
  // learning_rate not used since no trainable parameters - silence warning:

  std::ignore = learning_rate;
  auto tmp_in = input;
  Mat2D<float> gradient = tmp_in.elementwise_operation(
      [this](float x) { return (x > 0.0) ? 1.0 : this->alpha; });
  return gradient_output.hadamard_product(gradient);
}
void LeakyRELUActivationLayer::print_trainable_variables() const {}

SigmoidActivationLayer::~SigmoidActivationLayer() {}

SigmoidActivationLayer::SigmoidActivationLayer() {
  std::cout << "SigmoidActivationLayer" << std::endl;
}

Mat2D<float> SigmoidActivationLayer::forward(const Mat2D<float> &input) const {
  auto tmp_in = input;
  Mat2D<float> result = tmp_in.elementwise_operation(
      [](float x) { return 1.0 / (1 + std::exp(-x)); });

  return result;
}
Mat2D<float>
SigmoidActivationLayer::backward(const Mat2D<float> &input,
                                 const Mat2D<float> &gradient_output,
                                 const Mat2D<float> &learning_rate) {
  // learning_rate not used since no trainable parameters - silence warning:
  std::ignore = learning_rate;

  const auto gradient =
      input.hadamard_product(Mat2D<float>(1, 1, {1.0}).minus(input));
  return gradient_output.hadamard_product(gradient);
}
void SigmoidActivationLayer::print_trainable_variables() const {}

Loss::~Loss() {}

Loss::Loss() {}

MSELoss::~MSELoss() {}

MSELoss::MSELoss() {}

Mat2D<float> MSELoss::loss(const Mat2D<float> &predictions,
                           const Mat2D<float> &labels) const {
  const auto diff = predictions.minus(labels);
  const auto loss = diff.hadamard_product(diff);
  return loss;
}

Mat2D<float> MSELoss::loss_grad(const Mat2D<float> &predictions,
                                const Mat2D<float> &labels) const {
  return predictions.minus(labels);
}

SoftmaxCrossEntropyWithLogitsLoss::~SoftmaxCrossEntropyWithLogitsLoss() {}

SoftmaxCrossEntropyWithLogitsLoss::SoftmaxCrossEntropyWithLogitsLoss() {}

Mat2D<float> softmax(const Mat2D<float> &logits) {
  const auto logits_max = logits.reduce_max_axis(1);
  auto stable_logits = logits;

  stable_logits = stable_logits.minus(logits_max);

  const auto logits_exp =
      stable_logits.elementwise_operation([](float x) { return std::exp(x); });

  auto logits_exp_sum = logits_exp.reduce_sum_axis(1);
  auto probs = logits_exp.divide_by(logits_exp_sum);
  return probs;
}
Mat2D<float>
SoftmaxCrossEntropyWithLogitsLoss::loss(const Mat2D<float> &predictions,
                                        const Mat2D<float> &labels) const {
  auto pred_probs = softmax(predictions);

  const auto log_probs =
      pred_probs.elementwise_operation([](float x) { return std::log(x); });

  const auto ce = -(labels.hadamard_product(log_probs)).reduce_sum_axis(1);

  return ce;
}

Mat2D<float> SoftmaxCrossEntropyWithLogitsLoss::loss_grad(
    const Mat2D<float> &predictions, const Mat2D<float> &labels_one_hot) const {
  auto pred_tmp = predictions;
  const auto pred_probs = softmax(pred_tmp);
  // average gradient over minibatch
  const auto grad =
      pred_probs.minus(labels_one_hot)
          .divide_by(Mat2D<float>(
              1, 1, {static_cast<float>(predictions.get_num_rows())}));
  return grad;
}