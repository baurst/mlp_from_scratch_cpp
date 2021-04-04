#include "layer.h"
#include "utils.h"
#include <algorithm>
#include <iostream>
#include <math.h>
Layer::Layer(){};
Layer::~Layer(){};

DenseLayer::DenseLayer(size_t number_of_inputs, size_t number_of_neurons,
                       Initializer init)
    : weights(number_of_inputs, number_of_neurons, init),
      biases(1, number_of_neurons, init) {
  // std::cout << "DenseLayer: Inputs " << number_of_inputs << " Neurons "
  //<< number_of_neurons << std::endl;
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

  const auto weight_update = learning_rate.hadamard_product(grad_weights);
  const auto bias_update = learning_rate.hadamard_product(grad_biases);
  this->weights = this->weights.minus(weight_update);
  this->biases = this->biases.minus(bias_update);

  return grad_input;
}

RELUActivationLayer::~RELUActivationLayer(){};

RELUActivationLayer::RELUActivationLayer(){};

Mat2D<float> RELUActivationLayer::forward(const Mat2D<float> &input) const {
  Mat2D<float> result = input;
  std::transform(result.matrix_data.begin(), result.matrix_data.end(),
                 result.matrix_data.begin(),
                 [](float x) { return std::max(static_cast<float>(0.0), x); });
  return result;
}
Mat2D<float> RELUActivationLayer::backward(const Mat2D<float> &input,
                                           const Mat2D<float> &gradient_output,
                                           const Mat2D<float> &learning_rate) {
  Mat2D<float> result = input;
  std::transform(result.matrix_data.begin(), result.matrix_data.end(),
                 result.matrix_data.begin(),
                 [](float x) { return (x > 0.0) ? 1.0 : 0.0; });
  return result.hadamard_product(gradient_output);
}

Loss::~Loss(){};

Loss::Loss(){};

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
  return predictions.minus(labels);
}

SoftmaxCrossEntropyWithLogitsLoss::~SoftmaxCrossEntropyWithLogitsLoss(){};

SoftmaxCrossEntropyWithLogitsLoss::SoftmaxCrossEntropyWithLogitsLoss(){};

Mat2D<float> softmax(const Mat2D<float> &logits) {
  auto logits_tmp = logits;
  // std::cout << "Logits:" << std::endl << logits_tmp << std::endl;

  const auto logits_max = logits_tmp.reduce_max_axis(1);

  logits_tmp = logits_tmp.minus(logits_max);

  const auto logits_exp =
      logits_tmp.elementwise_operation([](float x) { return std::exp(x); });
  // std::cout << "logits_exp:" << std::endl << logits_exp << std::endl;

  auto logits_exp_sum = logits_exp.reduce_sum_axis(1);
  auto probs = logits_exp.divide_by(logits_exp_sum);
  // std::cout << "Probs:" << std::endl << probs << std::endl;

  return probs;
}
Mat2D<float>
SoftmaxCrossEntropyWithLogitsLoss::loss(const Mat2D<float> &predictions,
                                        const Mat2D<float> &labels) const {
  auto pred_probs = softmax(predictions);
  // //std::cout << pred_probs << std::endl;
  // //std::cout << pred_probs.reduce_sum_axis(1) << std::endl;

  const auto log_probs =
      pred_probs.elementwise_operation([](float x) { return std::log(x); });

  // std::cout << log_probs << std::endl;
  const auto ce = -(labels.hadamard_product(log_probs).reduce_sum_axis(1));
  return ce;
}

Mat2D<float>
SoftmaxCrossEntropyWithLogitsLoss::loss_grad(const Mat2D<float> &predictions,
                                             const Mat2D<float> &labels) const {
  // const auto ce = -labels.add(log);
  // average gradient over minibatch
  auto pred_tmp = predictions;
  const auto pred_probs = softmax(pred_tmp);
  return pred_probs.minus(labels);
}