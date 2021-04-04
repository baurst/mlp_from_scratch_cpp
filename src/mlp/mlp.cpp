#include "mlp.h"
#include "layer.h"
#include "utils.h"
#include <iostream>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <vector>

MLP::MLP(const std::vector<size_t> layer_sizes, const size_t number_of_inputs,
         const size_t number_of_targets, const Initializer init) {
  size_t input_size = number_of_inputs;

  for (const size_t layer_size : layer_sizes) {
    layers.push_back(
        std::make_unique<DenseLayer>(input_size, layer_size, init));
    input_size = layer_size;
    layers.push_back(std::make_unique<RELUActivationLayer>());
  }
  layers.push_back(std::make_unique<DenseLayer>(input_size, number_of_targets));
}

std::vector<Mat2D<float>> MLP::forward(const Mat2D<float> &input) const {
  std::vector<Mat2D<float>> activations;
  activations.reserve(this->layers.size() + 1);
  activations.push_back(input);
  auto tmp_in = input;
  for (const auto &layer : layers) {
    const auto tmp_out = layer->forward(tmp_in);
    activations.push_back(tmp_out);
    tmp_in = tmp_out;
  }
  return activations;
}

float MLP::train(const Mat2D<float> &input, const Mat2D<float> &target_label,
                 const Loss &loss_obj, const Mat2D<float> &learning_rate) {
  const auto activations = this->forward(input);
  const auto logits = activations.back();
  const auto loss = loss_obj.loss(logits, target_label);
  const auto avg_loss = loss.reduce_mean();
  if (avg_loss > 5.0) {
    std::cout << "about to crash!" << std::endl;
  }

  auto grad = loss_obj.loss_grad(logits, target_label);

  // auto error =
  //     (loss_obj.loss_grad(logits,
  //     target_label)).hadamard_product(logits_deriv);
  std::cout << "grad: " << std::endl << grad << std::endl;

  for (int32_t layer_idx = this->layers.size() - 1; layer_idx >= 0;
       --layer_idx) {
    // std::cout << "grad: " << layer_idx << std::endl << grad << std::endl;

    grad = this->layers[layer_idx]->backward(activations[layer_idx], grad,
                                             learning_rate);
  }

  if (std::isnan(avg_loss)) {
    std::cout.flush();
    throw std::runtime_error("Encountered NAN in loss!");
  }
  return avg_loss;
}

Mat2D<size_t> MLP::predict(const Mat2D<float> &input) const {
  const auto activations = this->forward(input);
  const auto logits = activations.back();
  const auto argmax_indices = logits.argmax(1);
  return argmax_indices;
}