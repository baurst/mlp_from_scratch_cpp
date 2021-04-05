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
  // .divide_by(Mat2D<float>(
  //     1, 1, {static_cast<float>(gradients_output.get_num_rows())}));
  const auto grad_biases = gradients_output.reduce_sum_axis(0);

  const auto weight_update = learning_rate.hadamard_product(grad_weights);
  const auto bias_update = learning_rate.hadamard_product(grad_biases);
  this->weights = this->weights.minus(weight_update);
  this->biases = this->biases.minus(bias_update);

  return grad_input;
  // .divide_by(
  // Mat2D<float>(1, 1, {static_cast<float>(input.get_num_rows())}));
}

void DenseLayer::print_trainable_variables() const {
  std::cout << "Weight: " << this->weights.get_num_rows() << "x"
            << this->weights.get_num_cols() << std::endl;
  std::cout << this->weights << std::endl;
  std::cout << "Bias: " << this->biases.get_num_rows() << "x"
            << this->biases.get_num_cols() << std::endl;
  std::cout << this->biases << std::endl;
}

RELUActivationLayer::~RELUActivationLayer(){};

RELUActivationLayer::RELUActivationLayer(){};

Mat2D<float> RELUActivationLayer::forward(const Mat2D<float> &input) const {
  auto tmp_in = input;
  Mat2D<float> result = tmp_in.elementwise_operation(
      // [](float x) { return std::max(0.0f, x); });
      [](float x) { return std::max(0.1f * x, x); });

  return result;
}
Mat2D<float> RELUActivationLayer::backward(const Mat2D<float> &input,
                                           const Mat2D<float> &gradient_output,
                                           const Mat2D<float> &learning_rate) {
  auto tmp_in = input;
  Mat2D<float> result = tmp_in.elementwise_operation(
      // [](float x) { return (x > 0.0) ? 1.0 : 0.0; });
      [](float x) { return (x > 0.0) ? 1.0 : 0.1; });
  return gradient_output.hadamard_product(result);
}
void RELUActivationLayer::print_trainable_variables() const {}

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
  const auto logits_max = logits.reduce_max_axis(1);
  auto stable_logits = logits;
  // std::cout << "Logits:" << std::endl << stable_logits << std::endl;

  stable_logits = stable_logits.minus(logits_max);

  const auto logits_exp =
      stable_logits.elementwise_operation([](float x) { return std::exp(x); });
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
  // std::cout << pred_probs << std::endl;
  // std::cout << pred_probs.reduce_sum_axis(1) << std::endl;

  const auto log_probs =
      pred_probs.elementwise_operation([](float x) { return std::log(x); });

  // std::cout << log_probs << std::endl;
  const auto ce = -(labels.hadamard_product(log_probs).reduce_sum_axis(1));

  // const auto avg_loss = ce.reduce_mean();
  // if (avg_loss > 5.0) {
  //   std::cout << "about to crash!" << std::endl;
  // }

  return ce;
}

Mat2D<float> SoftmaxCrossEntropyWithLogitsLoss::loss_grad(
    const Mat2D<float> &predictions, const Mat2D<float> &labels_one_hot) const {
  // const auto ce = -labels_one_hot.add(log);
  // average gradient over minibatch
  auto pred_tmp = predictions;
  const auto pred_probs = softmax(pred_tmp);
  const auto grad =
      pred_probs.minus(labels_one_hot)
          .divide_by(Mat2D<float>(
              1, 1, {static_cast<float>(predictions.get_num_rows())}));
  return grad;
}