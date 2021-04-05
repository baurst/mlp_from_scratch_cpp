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
  size_t layer_idx = 0;
  for (const size_t layer_size : layer_sizes) {
    layers.push_back(
        std::make_unique<DenseLayer>(input_size, layer_size, init));
    std::cout << "Created DenseLayer with layer_idx " << layer_idx << std::endl;
    layer_idx++;
    input_size = layer_size;
    layers.push_back(std::make_unique<RELUActivationLayer>());
    std::cout << "Created RELUActivationLayer with layer_idx " << layer_idx
              << std::endl;
    layer_idx++;
  }
  layers.push_back(std::make_unique<DenseLayer>(input_size, number_of_targets));
  std::cout << "Created DenseLayer with layer_idx " << layer_idx << std::endl;
  layer_idx++;
}

std::vector<Mat2D<float>> MLP::forward(const Mat2D<float> &input) const {
  std::vector<Mat2D<float>> activations;
  activations.reserve(this->layers.size() + 1);
  activations.push_back(input);
  size_t layer_idx = 0;
  for (const auto &layer : layers) {
    const auto tmp_out = layer->forward(activations.back());
    // std::cout << "Layer " << layer_idx
    //           << " input shape: " << activations.back().get_num_rows() << ",
    //           "
    //           << activations.back().get_num_cols()
    //           << "--> output shape: " << tmp_out.get_num_rows() << ", "
    //           << tmp_out.get_num_cols() << std::endl;
    activations.push_back(tmp_out);

    layer_idx++;
  }
  return activations;
}

float MLP::train(const Mat2D<float> &input, const Mat2D<float> &target_label,
                 const Loss &loss_obj, const Mat2D<float> &learning_rate) {
  const auto activations = this->forward(input);
  const auto logits = activations.back();
  const auto loss = loss_obj.loss(logits, target_label);

  auto grad = loss_obj.loss_grad(logits, target_label);
  if (std::isnan(grad.reduce_mean())) {
    std::cout.flush();
    throw std::runtime_error("Encountered NAN in Gradient, we are doomed!");
  }
  // auto error =
  //     (loss_obj.loss_grad(logits,
  //     target_label)).hadamard_product(logits_deriv);
  // std::cout << "grad: " << std::endl << grad << std::endl;

  for (int32_t layer_idx = this->layers.size() - 1; layer_idx >= 0;
       --layer_idx) {
    // std::cout << "Backprop Layer Idx " << layer_idx << std::endl;
    // std::cout << "grad: " << layer_idx << std::endl << grad << std::endl;
    const auto layer_input = activations[layer_idx];

    grad = this->layers[layer_idx]->backward(layer_input, grad, learning_rate);
  }
  const auto avg_loss = loss.reduce_mean();

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